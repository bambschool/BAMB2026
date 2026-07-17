"""Pure-numpy REINFORCE trainer for the NHP meta-MDP policy.

This is the numpy reproduction of the lean repo's training path
(``Code/metamdp-lean/train_model.py`` + ``metamdp_functions.py``), which used
TensorFlow + tf-agents (PPO / REINFORCE). That stack (~2021) is not installed
and depends deeply on tf-agents; the environment itself is already ported to
numpy (:mod:`metamdp_nhp`), so the only missing piece is the *optimizer*. We
reproduce the algorithm — not the framework — with a small hand-written policy
network and REINFORCE, keeping the whole pipeline dependency-free (numpy only).

Faithful to the lean trainer's choices:
  - policy-gradient objective, **discount 1.0**, **no reward normalization**
    (the lean PPO set ``normalize_rewards=False`` "to avoid getting stuck");
  - an MLP actor with **two action heads** — ``fixated_rank`` (which
    posterior-sorted option to sample) and ``terminate`` (stop and choose);
  - lean-style **inductive biases** on the head biases: favor fixating the
    current best guess (high-posterior option) and favor *not* terminating
    (lean ``object_bias`` / ``terminate_bias``, ``load_ppo_agent``).

Differences from the lean trainer (flagged, not hidden):
  - **REINFORCE with a baseline**, not PPO — sufficient for this tiny env
    (No ≤ 3, Nf = 2, short episodes) and far simpler to read. A critic/PPO
    upgrade is possible but unnecessary for the reproduction.
  - variable option count (No = 2 or 3) is handled with a fixed ``NO_MAX`` head
    and masking of invalid ranks, rather than tf-agents' per-spec networks.

The **environment is unchanged** (``gf = 1`` — a glance measures both
attributes). That is deliberate: this trains the *object-level* model. Part 2
of the starter code makes ``gf`` attribute-selective; the trainer here is the
substrate students reuse without modification.

Usage
-----
    python train_reinforce.py --session 190430_Chip --episodes 20000
    python train_reinforce.py --grad-check      # verify analytic gradients
"""

from __future__ import annotations

from argparse import ArgumentParser

import numpy as np

from data import SessionData
from metamdp_nhp import NHPMetaMDP, NHPMetaMDPEnv, NHPMetaMDPEnvAttr, get_posterior_itrue
from task_perkins import N_FEATURES

NO_MAX = 3          # max options per trial (task pool); fixate head has this many ranks
NF = N_FEATURES     # 2 (reward level, prob level)
MAX_STEPS = 5       # hard episode cap (force terminate); episodes are short with gf=1

# Attention cost, recalibrated for the small-No regime. The lean env's penalty is
# c0 + c1·mean(Jmeas); with a fovea that puts precision ~SCALE_GO on the fixated
# option, mean(Jmeas) ≈ SCALE_GO/No, which is ~100× larger for No=2–3 than for the
# lean No=113 scenes. The lean c1=0.05 therefore makes one fixation cost ~0.26 —
# more accuracy (~0.18 gain, measured) than it buys — so any agent correctly learns
# never to look. We drop c1 to 0.01 (penalty ≈0.05–0.06) so information-gathering
# pays off. This is a documented *update* to the training path, not the lean value.
DEFAULT_COST = (0.01, 0.01)

# Observation featurization dimension (see `featurize`).
OBS_DIM = NO_MAX * NF + NO_MAX * NF + NF + NO_MAX * 3 + NO_MAX + NO_MAX


# --------------------------------------------------------------------------- #
# Observation featurization
# --------------------------------------------------------------------------- #
def featurize(obs: dict, n_options: int) -> np.ndarray:
    """Flatten one env observation into a fixed-length ``OBS_DIM`` vector.

    Per-option arrays (F, J, object_locs, posterior) are padded from ``n_options``
    up to ``NO_MAX`` rows with zeros; a validity mask tells the net which ranks
    are real. Options are already posterior-sorted by the env.
    """
    def pad(a: np.ndarray, per_row: int) -> np.ndarray:
        m = a.reshape(n_options, per_row)
        out = np.zeros((NO_MAX, per_row), dtype=np.float32)
        out[:n_options] = m
        return out.ravel()

    mask = np.zeros(NO_MAX, dtype=np.float32)
    mask[:n_options] = 1.0
    return np.concatenate([
        pad(obs["F"], NF),
        pad(obs["J"], NF),
        obs["ftarget"].astype(np.float32),
        pad(obs["object_locs"], 3),
        pad(obs["posterior"], 1),
        mask,
    ])


# --------------------------------------------------------------------------- #
# Policy network (2 hidden tanh layers, two softmax heads)
# --------------------------------------------------------------------------- #
def init_params(hidden: int = 64, seed: int = 0, n_targets_max: int = NO_MAX) -> dict:
    """Xavier-ish init, with lean-style biases on the two action heads.

    ``n_targets_max`` sizes the fixate head: ``NO_MAX`` for the object-level model
    (fixate an option), ``2·NO_MAX`` for the attribute-level model (fixate an
    (option, attribute) bar). Everything else is identical — the two models share
    the same trainer, differing only in this action-space width.
    """
    rng = np.random.default_rng(seed)

    def w(nin, nout):
        return (rng.standard_normal((nin, nout)) * np.sqrt(2.0 / nin)).astype(np.float64)

    # Favor fixating the current best guess (low target index); lean object_bias
    # = 25*exp(-0.5*arange). We use a milder version so it does not saturate.
    fix_bias = 2.0 * np.exp(-0.5 * np.arange(n_targets_max))
    # Favor NOT terminating (index 0 = continue, 1 = terminate); lean [1, 0].
    term_bias = np.array([1.0, -1.0])

    return {
        "W1": w(OBS_DIM, hidden), "b1": np.zeros(hidden),
        "W2": w(hidden, hidden), "b2": np.zeros(hidden),
        "Wf": w(hidden, n_targets_max) * 0.1, "bf": fix_bias.copy(),
        "Wt": w(hidden, 2) * 0.1, "bt": term_bias.copy(),
    }


def forward(p: dict, x: np.ndarray):
    """Return (fixate_logits[NO_MAX], terminate_logits[2], cache)."""
    z1 = x @ p["W1"] + p["b1"]; h1 = np.tanh(z1)
    z2 = h1 @ p["W2"] + p["b2"]; h2 = np.tanh(z2)
    fix_logits = h2 @ p["Wf"] + p["bf"]
    term_logits = h2 @ p["Wt"] + p["bt"]
    return fix_logits, term_logits, (x, h1, h2)


def backward(p: dict, cache, d_fix: np.ndarray, d_term: np.ndarray) -> dict:
    """Backprop gradients on the two logit vectors into parameter grads."""
    x, h1, h2 = cache
    g = {}
    g["Wf"] = np.outer(h2, d_fix); g["bf"] = d_fix
    g["Wt"] = np.outer(h2, d_term); g["bt"] = d_term
    dh2 = p["Wf"] @ d_fix + p["Wt"] @ d_term
    dz2 = dh2 * (1.0 - h2 ** 2)
    g["W2"] = np.outer(h1, dz2); g["b2"] = dz2
    dh1 = p["W2"] @ dz2
    dz1 = dh1 * (1.0 - h1 ** 2)
    g["W1"] = np.outer(x, dz1); g["b1"] = dz1
    return g


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max()
    e = np.exp(z)
    return e / e.sum()


def _masked_fix_probs(fix_logits: np.ndarray, n_valid: int) -> np.ndarray:
    """Softmax over the first ``n_valid`` fixate targets (rest masked out)."""
    masked = fix_logits.copy()
    masked[n_valid:] = -1e9
    return _softmax(masked)


# --------------------------------------------------------------------------- #
# Episode rollout
# --------------------------------------------------------------------------- #
def run_episode(p: dict, env, rng: np.random.Generator, greedy: bool = False):
    """Roll one episode; return a trajectory dict of per-step tensors + return.

    Works for both the object-level and attribute-level env: ``env.n_targets``
    gives the number of valid fixate targets, ``env.decode`` and ``env.step`` map
    a chosen target to an action. The ``targets`` list records the raw chosen
    target index each step (option rank, or (option,attribute) bar) for analysis.
    """
    ts = env.reset()
    no = env.scene.n_options
    n_head = len(p["bf"])
    xs, caches, targets, terms, rewards, p_fixes, p_terms, decoded = \
        [], [], [], [], [], [], [], []

    for t in range(MAX_STEPS):
        x = featurize(ts.observation, no)
        fix_logits, term_logits, cache = forward(p, x)
        p_term = _softmax(term_logits)
        force_term = t == MAX_STEPS - 1
        if greedy:
            terminate = force_term or (p_term[1] > 0.5)
        else:
            terminate = force_term or (rng.random() < p_term[1])

        p_fix = _masked_fix_probs(fix_logits, env.n_targets)
        if terminate:
            target = -1
            decoded.append((None, None))
        else:
            target = int(np.argmax(p_fix)) if greedy else int(rng.choice(n_head, p=p_fix))
            decoded.append(env.decode(target))

        ts = env.step(0 if terminate else target, terminate=bool(terminate))

        xs.append(x); caches.append(cache); targets.append(target)
        terms.append(int(terminate)); rewards.append(ts.reward)
        p_fixes.append(p_fix); p_terms.append(p_term)
        if terminate:
            break

    return dict(xs=xs, caches=caches, ranks=targets, terms=terms, rewards=rewards,
                p_fixes=p_fixes, p_terms=p_terms, n_options=no, n_targets=env.n_targets,
                decoded=decoded,
                ret=float(np.sum(rewards)), n_fix=int(np.sum(1 - np.array(terms))))


def _returns_to_go(rewards: list[float]) -> np.ndarray:
    """Discount 1.0 reward-to-go: G_t = sum_{k>=t} r_k."""
    g = np.cumsum(rewards[::-1])[::-1]
    return np.asarray(g, dtype=float)


# --------------------------------------------------------------------------- #
# REINFORCE gradient for a batch of episodes
# --------------------------------------------------------------------------- #
def _entropy_grad(probs: np.ndarray, coef: float) -> np.ndarray:
    """∂(−coef·H)/∂logits for a softmax head, where H = −Σ p log p.

    Used to ADD an entropy bonus to the objective (so we subtract coef·H from the
    loss). Result: ``coef · p · (log p + H)``. Pushes probs toward uniform,
    guarding against premature collapse to always-terminate.
    """
    lp = np.log(probs + 1e-12)
    H = -np.sum(probs * lp)
    return coef * probs * (lp + H)


def batch_grads(p: dict, episodes: list[dict], baseline: float,
                normalize_adv: bool = True, entropy_coef: float = 0.0):
    """Accumulate the REINFORCE policy-gradient over a batch. Return (grads, stats).

    For each step the loss is ``-A_t · logπ(action_t | s_t) - entropy_coef·H_t``
    with advantage ``A_t = G_t - baseline``. Grad on a softmax head's logits is
    ``A_t·(prob - onehot(action)) + entropy term``. The terminate head is active
    every step; the fixate head only on continuing steps.
    """
    # Collect advantages first (optionally standardize across the batch).
    all_adv = []
    for ep in episodes:
        g = _returns_to_go(ep["rewards"]) - baseline
        all_adv.append(g)
    flat = np.concatenate(all_adv) if all_adv else np.zeros(1)
    adv_std = flat.std() + 1e-8

    n_head = len(p["bf"])
    grads = {k: np.zeros_like(v) for k, v in p.items()}
    n_steps = 0
    for ep, adv in zip(episodes, all_adv):
        nv = ep["n_targets"]
        if normalize_adv:
            adv = adv / adv_std
        for t in range(len(ep["rewards"])):
            A = adv[t]
            p_term = ep["p_terms"][t]
            d_term = A * (p_term - _onehot(ep["terms"][t], 2))
            d_term += _entropy_grad(p_term, entropy_coef)
            if ep["terms"][t]:
                d_fix = np.zeros(n_head)          # fixate head unused this step
            else:
                p_fix = ep["p_fixes"][t]
                d_fix = A * (p_fix - _onehot(ep["ranks"][t], n_head))
                valid = p_fix.copy(); valid[nv:] = 0.0
                d_fix += _entropy_grad(valid, entropy_coef)
                d_fix[nv:] = 0.0                  # invalid targets carry no grad
            gstep = backward(p, ep["caches"][t], d_fix, d_term)
            for k in grads:
                grads[k] += gstep[k]
            n_steps += 1

    scale = 1.0 / max(len(episodes), 1)
    for k in grads:
        grads[k] *= scale
    return grads, dict(n_steps=n_steps)


def _onehot(i: int, n: int) -> np.ndarray:
    v = np.zeros(n)
    v[i] = 1.0
    return v


# --------------------------------------------------------------------------- #
# Adam optimizer
# --------------------------------------------------------------------------- #
class Adam:
    def __init__(self, params: dict, lr: float = 1e-3, b1: float = 0.9,
                 b2: float = 0.999, eps: float = 1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params: dict, grads: dict) -> None:
        """Gradient *descent* on the loss (grads are ∂loss/∂θ)."""
        self.t += 1
        for k in params:
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * grads[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * grads[k] ** 2
            mhat = self.m[k] / (1 - self.b1 ** self.t)
            vhat = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def greedy_episode(p: dict, env) -> dict:
    """Run the deterministic (argmax) policy once; return metrics + the scanpath.

    ``choice`` = argmax posterior over options at the moment of termination;
    ``decoded`` = the list of ``(fixated_object, attribute)`` fixations (attribute
    is None for the object-level env), for scanpath analysis.
    """
    ts = env.reset()
    no = env.scene.n_options
    n_fix, ret, choice, decoded = 0, 0.0, 0, []
    for t in range(MAX_STEPS):
        x = featurize(ts.observation, no)
        fix_logits, term_logits, _ = forward(p, x)
        terminate = (t == MAX_STEPS - 1) or (_softmax(term_logits)[1] > 0.5)
        if terminate:
            post = get_posterior_itrue(env._state, env.scene, env.mdp.Jtrue(env.scene))
            choice = int(np.argmax(post))
            ts = env.step(0, terminate=True); ret += ts.reward
            break
        target = int(np.argmax(_masked_fix_probs(fix_logits, env.n_targets)))
        decoded.append(env.decode(target))
        ts = env.step(target, terminate=False); ret += ts.reward; n_fix += 1
    return dict(n_fix=n_fix, ret=ret, choice=choice, decoded=decoded)


def evaluate(p: dict, mdp: NHPMetaMDP, env_class, scenes: list,
             rng: np.random.Generator, n: int = 400) -> dict:
    """Greedy-policy metrics over sampled scenes."""
    acc, lengths, rets, chance = [], [], [], []
    idx = rng.integers(0, len(scenes), size=min(n, len(scenes)) if scenes else 0)
    for i in idx:
        scene = scenes[i]
        g = greedy_episode(p, env_class(mdp, scene))
        acc.append(g["choice"] == scene.itrue)
        lengths.append(g["n_fix"])
        rets.append(g["ret"])
        chance.append(1.0 / scene.n_options)
    return dict(acc=float(np.mean(acc)), length=float(np.mean(lengths)),
                ret=float(np.mean(rets)), chance=float(np.mean(chance)))


# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #
def train(scenes: list, episodes: int = 20000, batch: int = 32, lr: float = 3e-3,
          hidden: int = 64, seed: int = 0, cost_params=DEFAULT_COST,
          entropy_coef: float = 0.02, env_class=NHPMetaMDPEnv, n_targets_max: int = NO_MAX,
          log_every: int = 2000, verbose: bool = True):
    """Train the policy on a set of scenes; return (params, history).

    ``env_class`` selects the model: :class:`NHPMetaMDPEnv` (object-level,
    ``n_targets_max=NO_MAX``) or :class:`NHPMetaMDPEnvAttr` (attribute-level,
    ``n_targets_max=2*NO_MAX``). The trainer body is identical for both.
    """
    rng = np.random.default_rng(seed)
    mdp = NHPMetaMDP(cost_params=cost_params)
    p = init_params(hidden=hidden, seed=seed, n_targets_max=n_targets_max)
    opt = Adam(p, lr=lr)
    baseline = 0.0
    history = []

    n_batches = episodes // batch
    for it in range(n_batches):
        eps = []
        for _ in range(batch):
            scene = scenes[int(rng.integers(len(scenes)))]
            eps.append(run_episode(p, env_class(mdp, scene), rng))
        batch_ret = float(np.mean([e["ret"] for e in eps]))
        baseline = 0.9 * baseline + 0.1 * batch_ret if it else batch_ret
        grads, _ = batch_grads(p, eps, baseline, entropy_coef=entropy_coef)
        opt.step(p, grads)

        if verbose and (it % max(n_batches // (episodes // log_every or 1), 1) == 0
                        or it == n_batches - 1):
            ev = evaluate(p, mdp, env_class, scenes, rng)
            history.append(dict(episode=(it + 1) * batch, **ev, train_ret=batch_ret))
            print(f"ep {(it+1)*batch:6d} | acc {ev['acc']:.3f} "
                  f"(chance {ev['chance']:.3f}) | n_fix {ev['length']:.2f} | "
                  f"ret {ev['ret']:+.3f} | train_ret {batch_ret:+.3f}")
    return p, history


# --------------------------------------------------------------------------- #
# Gradient check (finite differences) — trainer correctness, not a student task
# --------------------------------------------------------------------------- #
def grad_check(seed: int = 0) -> float:
    """Compare analytic REINFORCE grads to finite differences on a fixed batch.

    Uses the surrogate loss L(θ) = -mean_ep sum_t A_t·logπ_θ(a_t|s_t) with the
    actions/advantages held FIXED (as REINFORCE assumes), so finite-differencing
    L over θ must match the analytic gradient.
    """
    rng = np.random.default_rng(seed)
    mdp = NHPMetaMDP()
    from data import TrialScene
    # Two tiny synthetic scenes (No=2 and No=3).
    scenes = [
        TrialScene("s2", np.array([[5.0, 1.0], [1.0, 5.0]]),
                   np.array([[1, 0, 0], [-1, 0, 0]], float), 0, 0, False),
        TrialScene("s3", np.array([[5.0, 5.0], [1.0, 1.0], [3.0, 3.0]]),
                   np.array([[1, 0, 0], [0, 1, 0], [-1, 0, 0]], float), 0, 0, False),
    ]
    p = init_params(seed=seed)
    ecoef = 0.02
    eps = []
    for _ in range(6):
        scene = scenes[int(rng.integers(len(scenes)))]
        eps.append(run_episode(p, NHPMetaMDPEnv(mdp, scene), rng))
    baseline = float(np.mean([e["ret"] for e in eps]))

    def _entropy(probs):
        lp = np.log(probs + 1e-12)
        return -np.sum(probs * lp)

    def surrogate_loss(params):
        adv_all = [(_returns_to_go(e["rewards"]) - baseline) for e in eps]
        std = np.concatenate(adv_all).std() + 1e-8
        loss = 0.0
        for e, adv in zip(eps, adv_all):
            adv = adv / std
            no = e["n_options"]
            for t in range(len(e["rewards"])):
                fl, tl, _ = forward(params, e["xs"][t])
                p_term = _softmax(tl)
                logp = np.log(p_term[e["terms"][t]] + 1e-12)
                loss -= ecoef * _entropy(p_term)
                if not e["terms"][t]:
                    p_fix = _masked_fix_probs(fl, no)
                    logp += np.log(p_fix[e["ranks"][t]] + 1e-12)
                    loss -= ecoef * _entropy(p_fix[:no])
                loss += -adv[t] * logp
        return loss / len(eps)

    grads, _ = batch_grads(p, eps, baseline, entropy_coef=ecoef)
    max_rel = 0.0
    for k in p:
        flat = p[k].ravel()
        for _ in range(min(8, flat.size)):
            j = int(rng.integers(flat.size))
            orig = flat[j]; h = 1e-5
            flat[j] = orig + h; lp = surrogate_loss(p)
            flat[j] = orig - h; lm = surrogate_loss(p)
            flat[j] = orig
            num = (lp - lm) / (2 * h)
            ana = grads[k].ravel()[j]
            denom = max(abs(num), abs(ana), 1e-6)
            max_rel = max(max_rel, abs(num - ana) / denom)
    return max_rel


def main() -> None:
    ap = ArgumentParser(description=__doc__)
    ap.add_argument("--session", default="190430_Chip")
    ap.add_argument("--episodes", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--cost", type=float, default=DEFAULT_COST[1],
                    help="precision-scaled attention cost c1 (per-fixation penalty ≈ c0+c1·SCALE_GO/No)")
    ap.add_argument("--entropy", type=float, default=0.02)
    ap.add_argument("--level", choices=["object", "attribute"], default="object",
                    help="object-level (fixate an option, gf=1) or attribute-level "
                         "(fixate an (option,attribute) bar, gf one-hot)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grad-check", action="store_true")
    args = ap.parse_args()

    if args.grad_check:
        rel = grad_check(args.seed)
        print(f"gradient check: max relative error = {rel:.2e} "
              f"({'PASS' if rel < 1e-4 else 'FAIL'})")
        return

    env_class = NHPMetaMDPEnvAttr if args.level == "attribute" else NHPMetaMDPEnv
    n_targets_max = 2 * NO_MAX if args.level == "attribute" else NO_MAX

    data = SessionData.load(args.session)
    scenes = [data.scene(t) for t in data.trials()]
    print(f"session {args.session}: {args.level}-level model, {len(scenes)} scenes "
          f"(No distribution: "
          f"{ {k: sum(s.n_options == k for s in scenes) for k in (2, 3)} })\n")
    train(scenes, episodes=args.episodes, batch=args.batch, lr=args.lr, seed=args.seed,
          cost_params=(DEFAULT_COST[0], args.cost), entropy_coef=args.entropy,
          env_class=env_class, n_targets_max=n_targets_max)


if __name__ == "__main__":
    main()
