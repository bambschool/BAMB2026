import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Part 3 — Individual differences: the model meets the monkey

    In **Part 2** the attribute-level model, trained only to identify the best
    option, reproduced the monkey's looking on its own: **one attribute at a
    time**, and **had a bias towards the probability bar** — matching **Chip** (≈0.93 of
    within-option dwell on probability).

    But there was a catch. Probability is *objectively* the more
    diagnostic attribute (it carries weight **1.78** in the task's value rule),
    so the info-optimal model goes probability-heavy for **Dale too** — yet the
    real Dale is **balanced** (≈0.43 on probability, Part 1). A model that only
    knows the *task* can't be two different monkeys.

    > **The question for this part:** what *single* per-monkey parameter, added
    > to the model, makes it look like Chip *or* like Dale?

    The idea: give the agent a **subjective value weight** on probability, `w_p`.
    Instead of the task's fixed 1.78, *this* monkey values an option at
    **`reward + w_p · probability`**, and looks to identify *its own* best option.
    A monkey that weights probability heavily (large `w_p`) finds probability the
    diagnostic attribute and rides it; a monkey that weights the two attributes
    more equally (`w_p ≈ 1`) splits its looks. We will **fit `w_p` to each
    monkey** and read the individual difference straight off the fitted value.
    """)
    return


@app.cell
def _():
    # --- Setup: locate the project, import shared model + trace bundle --------
    import sys
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt


    def _find_root(start: Path) -> Path:
        for p in [start, *start.parents]:
            if (p / "Data" / "all_trial_options.csv").exists() and (p / "Libraries").is_dir():
                return p
        raise FileNotFoundError("Could not locate the PerkinsRich2024 project root.")

    try:
        _HERE = Path(__file__).resolve().parent
    except NameError:
        _HERE = Path.cwd()
    ROOT = _find_root(_HERE)
    sys.path.insert(0, str(ROOT / "Libraries"))

    from data import SessionData, TrialScene
    from metamdp_nhp import NHPMetaMDP, NHPMetaMDPEnvAttr, State, TimeStep
    # Alias the trainer's underscore-prefixed helpers: marimo treats leading-underscore
    # names as cell-private, so they can't be shared between cells under those names.
    from train_reinforce import (train, DEFAULT_COST, NO_MAX, featurize, forward,
                                 MAX_STEPS)
    from train_reinforce import _softmax as softmax
    from train_reinforce import _masked_fix_probs as masked_fix_probs

    return (
        DEFAULT_COST,
        MAX_STEPS,
        NHPMetaMDP,
        NHPMetaMDPEnvAttr,
        NO_MAX,
        ROOT,
        SessionData,
        State,
        TimeStep,
        TrialScene,
        featurize,
        forward,
        masked_fix_probs,
        np,
        plt,
        softmax,
        train,
    )


@app.cell
def _(mo):
    def verdict(check_fn):
        """`check_fn()` returns [(label, passed), ...]; render a callout. Unfilled
        TODOs (NotImplementedError) show a gentle 'not done yet' note."""
        try:
            checks = check_fn()
        except NotImplementedError as e:
            return mo.callout(mo.md(f"**Not done yet** — {e}"), kind="neutral")
        except Exception as e:  # pragma: no cover
            return mo.callout(mo.md(f"Your code raised **{type(e).__name__}**: {e}"), kind="danger")
        lines = [f"{'✅' if ok else '❌'} {label}" for label, ok in checks]
        return mo.callout(mo.md("\n\n".join(lines)),
                          kind="success" if all(ok for _, ok in checks) else "warn")

    return (verdict,)


@app.cell
def _(DEFAULT_COST, NHPMetaMDP, SessionData):
    # --- Given: one example session per monkey, and the shared meta-MDP -------
    CHIP_SESS, DALE_SESS = "190430_Chip", "181112_Dale"

    def _scenes(session):
        d = SessionData.from_combined(session)      # reads Data/all_trial_options.csv
        return [d.scene(t) for t in d.trials()]

    chip_scenes = _scenes(CHIP_SESS)
    dale_scenes = _scenes(DALE_SESS)
    mdp = NHPMetaMDP(cost_params=DEFAULT_COST)
    return CHIP_SESS, DALE_SESS, chip_scenes, dale_scenes, mdp


@app.cell
def _(ROOT, np):
    # --- Given: the monkeys' OBSERVED probability-dwell fraction (Part 1) ----
    # The Part 1 pipeline (assign each 500 Hz gaze sample to a reward/probability
    # bar; mean fraction of within-option dwell on the probability bar), reproduced
    # here as plumbing. Vectorised per trial, and the ragged trace arrays are pulled
    # out of the npz ONCE — indexing an NpzFile (`_Z["eye"][i]`) re-decompresses the
    # whole array on every access, which is what made the naive loop take minutes.
    # Reads only Data/example_traces.npz.
    _Z = np.load(ROOT / "Data" / "example_traces.npz", allow_pickle=True)
    _SESSION, _EYE, _DT = _Z["session"], _Z["eye"], _Z["dt"]
    _OBJPOS, _TBARS, _TCHOICE = _Z["objpos"], _Z["t_bars_on"], _Z["t_choice"]

    def observed_frac_prob(session, xhalf=1.5, yhalf=5.5):
        """Mean within-option dwell fraction on the probability bar (Part 1)."""
        fracs = []
        for i in np.where(_SESSION == session)[0]:
            eye = np.asarray(_EYE[i], float); dt = float(_DT[i])
            objpos = np.asarray(_OBJPOS[i], float)
            tb = float(_TBARS[i])
            a = max(int((0 if np.isnan(tb) else tb) / dt), 0)
            b = min(int(float(_TCHOICE[i]) / dt), len(eye))
            if b <= a:
                continue
            samples = eye[a:b]                                    # (N, 2)
            no = len(objpos)
            # bars interleaved: 2*oi = reward bar (cx-1.5), 2*oi+1 = prob bar (cx+1.5)
            bar_x = np.empty(2 * no)
            bar_x[0::2] = objpos[:, 0] - 1.5
            bar_x[1::2] = objpos[:, 0] + 1.5
            bar_y = np.repeat(objpos[:, 1], 2)
            dx = samples[:, 0:1] - bar_x[None, :]                 # (N, 2*No)
            dy = samples[:, 1:2] - bar_y[None, :]
            in_window = (np.abs(dx) < xhalf) & (np.abs(dy) < yhalf)
            dist = np.where(in_window, dx * dx + dy * dy, np.inf)
            nearest = np.argmin(dist, axis=1)                     # nearest in-window bar
            on_bar = np.isfinite(dist[np.arange(len(samples)), nearest])
            counts = np.zeros(2 * no)
            np.add.at(counts, nearest[on_bar], 1.0)              # dt cancels in the ratio
            for oi in range(no):
                r, p = counts[2 * oi], counts[2 * oi + 1]
                if r + p > 0:
                    fracs.append(p / (r + p))
        return float(np.mean(fracs))

    return (observed_frac_prob,)


@app.cell
def _(CHIP_SESS, DALE_SESS, mo, observed_frac_prob):
    OBS_CHIP = observed_frac_prob(CHIP_SESS)
    OBS_DALE = observed_frac_prob(DALE_SESS)
    mo.md(
        f"**Observed** (your Part 1 measure): Chip rides probability "
        f"**{OBS_CHIP:.0%}** of within-option dwell; Dale only **{OBS_DALE:.0%}**. "
        f"These two numbers are the targets we will fit the model to."
    )
    return OBS_CHIP, OBS_DALE


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Exercise 1 — the subjective value of an option

    The task's value rule is `reward + 1.78·probability`. We now let *each
    monkey* have its own probability weight `w_p`: the option's **subjective
    value** is

    $$V = \text{reward} + w_p \cdot \text{probability}.$$

    The agent will look to identify the option with the highest subjective value
    (its own "best option"). Fill in `subjective_value(features, w_p)`:
    `features` is an `(No, 2)` array of `[reward_level, prob_level]` per option;
    return the length-`No` vector of subjective values.
    """)
    return


@app.function
def subjective_value(features, w_p):
    """Per-option subjective value reward + w_p * probability."""
    # TODO (Exercise 1): return the length-No vector of subjective values —
    #   the reward column features[:, 0] plus w_p times the probability
    #   column features[:, 1].
    raise NotImplementedError("Exercise 1: implement subjective_value")


@app.cell
def _(np, verdict):
    def _ex1_checks():
        f = np.array([[5.0, 1.0], [1.0, 5.0]])   # option A reward-rich, B prob-rich
        return [
            ("w_p=1.78 → prob-rich option wins (task weighting)",
             int(np.argmax(subjective_value(f, 1.78))) == 1),
            ("w_p=0.5 → reward-rich option wins",
             int(np.argmax(subjective_value(f, 0.5))) == 0),
            ("values are reward + w_p·prob",
             np.allclose(subjective_value(f, 2.0), [7.0, 11.0])),
        ]
    verdict(_ex1_checks)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Exercise 2 — where the weight enters the *looking*

    The weight has to change **what the agent bothers to look at**, not just its
    final pick. It does so through the agent's belief about *which option is
    best*. That belief (the posterior over the target option) is built from
    per-attribute evidence: for each option and each attribute there is a
    log-likelihood term saying how much that attribute's measurement points at
    "this is the best option". The object's overall score sums those terms
    **across attributes**.

    The subjective weight scales each attribute's contribution to that sum. Fill
    in `weight_terms(terms, w)`: `terms` is an `(No, Nf)` array of per-attribute
    log-likelihood terms, `w` is a length-`Nf` weight vector `[1, w_p]`. Return
    the length-`No` vector where attribute `k`'s column is scaled by `w[k]` and
    the columns are summed.

    *Why this is the knob:* up-weighting probability makes probability evidence
    move the belief more, so **measuring probability becomes more valuable** —
    and a resource-rational agent spends its looks where they move the belief.
    """)
    return


@app.function
def weight_terms(terms, w):
    """Sum per-attribute log-likelihood terms, scaling attribute k by w[k]."""
    # TODO (Exercise 2): scale each column k of `terms` (shape (No, Nf)) by
    #   w[k], then sum across attributes (axis=1) to a length-No vector.
    raise NotImplementedError("Exercise 2: implement weight_terms")


@app.cell
def _(np, verdict):
    def _ex2_checks():
        terms = np.array([[2.0, 1.0], [0.0, 3.0]])
        return [
            ("equal weights [1,1] → plain row sum",
             np.allclose(weight_terms(terms, [1, 1]), [3.0, 3.0])),
            ("up-weighting probability lifts the prob-evidence row",
             np.allclose(weight_terms(terms, [1, 3]), [5.0, 9.0])),
            ("zero weight on reward drops the reward column",
             np.allclose(weight_terms(terms, [0, 1]), [1.0, 3.0])),
        ]
    verdict(_ex2_checks)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Given: wire the weight into an attribute-level environment

    `SubjEnv` below is the Part 2 attribute-level env with the subjective
    weight threaded in. Two changes, both using **your** functions:

    - the posterior over the best option scales attribute evidence with your
      `weight_terms` (via `weighted_posterior`), and
    - a scene's target option (`itrue`) is the argmax of your `subjective_value`
      (see `subjectivize`).

    Everything else — the belief update, the fovea, the (option, attribute)
    action space, the trainer — is unchanged from Part 2.
    """)
    return


@app.cell
def _(NHPMetaMDPEnvAttr, State, TimeStep, TrialScene, np):
    def weighted_posterior(state, scene, Jtrue, w):
        """Given: posterior over the best option, with your weight_terms (Exercise 2)."""
        F, J = state.F, state.J
        ft = scene.ftarget[None, :]
        terms = (np.log(1 + J / Jtrue) / 2 + 1 / (1 / J + 1 / Jtrue) / 2 * F ** 2
                 - J / 2 * (F - ft) ** 2)
        L = weight_terms(terms, w)                          # Exercise 2
        p = np.exp(L - L.max())
        return p / p.sum()

    def subjectivize(scenes, w_p):
        """Given: relabel each scene's target as the subjective-best (Exercise 1)."""
        out = []
        for s in scenes:
            itrue = int(np.argmax(subjective_value(s.features, w_p)))   # Exercise 1
            out.append(TrialScene(s.scene_id, s.features, s.object_locs,
                                  itrue, s.chosen, s.rewarded))
        return out

    def make_subj_env(w_p):
        """Given: attribute-level env whose belief uses subjective weight w=[1, w_p]."""
        w = np.array([1.0, w_p])

        class SubjEnv(NHPMetaMDPEnvAttr):
            def _encode(self):
                if self._state.is_terminal:
                    return {k: np.zeros(v, np.float32)
                            for k, v in self.observation_spec().items()}
                post = weighted_posterior(self._state, self.scene,
                                          self.mdp.Jtrue(self.scene), w)
                self._order = np.argsort(post)[::-1]
                return {
                    "F": self._state.F[self._order].flatten().astype(np.float32),
                    "J": self._state.J[self._order].flatten().astype(np.float32),
                    "ftarget": self.scene.ftarget.astype(np.float32),
                    "object_locs": self.scene.object_locs[self._order].flatten().astype(np.float32),
                    "posterior": post[self._order].astype(np.float32),
                }

            def step(self, target, terminate):
                if self._ended:
                    return self.reset()
                if terminate:
                    post = weighted_posterior(self._state, self.scene,
                                              self.mdp.Jtrue(self.scene), w)
                    reward = 1.0 if int(np.argmax(post)) == self.scene.itrue else 0.0
                    self._state = State(None, None, is_terminal=True)
                else:
                    fixated, attribute = self.decode(target)
                    go = self.mdp.fovea_go(self.scene, fixated)
                    gf = np.zeros(self.mdp.Nf); gf[attribute] = 1.0
                    Jmeas = go[:, None] * gf[None, :]
                    xj = np.random.normal(self.scene.features * Jmeas, np.sqrt(Jmeas))
                    pen = self.mdp.cost_params[0] + self.mdp.cost_params[1] * np.mean(Jmeas)
                    self._state = State(F=(self._state.F * self._state.J + xj) / (self._state.J + Jmeas),
                                        J=self._state.J + Jmeas)
                    reward = -pen
                self._ended = self._state.is_terminal
                return TimeStep(self._encode(), float(reward), self._ended)

        return SubjEnv, w

    return make_subj_env, subjectivize


@app.cell
def _(MAX_STEPS, featurize, forward, masked_fix_probs, np, softmax):
    # --- Given: read frac_prob from the STOCHASTIC policy ---------------------
    # The monkey's 0.43 is a *mixed* strategy averaged over trials, so we read the
    # model the same way — sampling fixations from the policy (not argmax), which
    # mixes gradedly near the reward/probability crossover.
    def model_frac_prob(p, env_class, mdp, scenes, w, n_samp=8, seed=7):
        rng = np.random.default_rng(seed)
        n_head = len(p["bf"]); attrs = []
        for s in scenes:
            for _ in range(n_samp):
                env = env_class(mdp, s); ts = env.reset(); no = env.scene.n_options
                for t in range(MAX_STEPS):
                    fl, tl, _ = forward(p, featurize(ts.observation, no))
                    if (t == MAX_STEPS - 1) or (rng.random() < softmax(tl)[1]):
                        break
                    tgt = int(rng.choice(n_head, p=masked_fix_probs(fl, env.n_targets)))
                    attrs.append(env.decode(tgt)[1])
                    ts = env.step(tgt, terminate=False)
        return sum(1 for a in attrs if a == 1) / len(attrs) if attrs else float("nan")

    return (model_frac_prob,)


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ### The behaviour–preference curve

    Below we sweep `w_p` and, at each value, train the model and read its
    probability-dwell fraction. This is the model's **map from preference to
    behaviour**. (Trains a handful of small policies — a few seconds each.)
    """)
    return


@app.cell
def _(
    NO_MAX,
    chip_scenes,
    dale_scenes,
    make_subj_env,
    mdp,
    model_frac_prob,
    np,
    subjectivize,
    train,
):
    # --- Given: build frac_prob(w_p) on a grid (needs Exercises 1 & 2) --------
    # Measured on a pooled sample of BOTH monkeys' option configs, so the fitted
    # preference is read against monkey-neutral behaviour. A small entropy bonus in
    # training keeps the policy mildly stochastic (real attention is not perfectly
    # optimal), which turns the reward/probability crossover into a smooth,
    # fittable curve instead of an all-or-nothing switch.
    _pool = chip_scenes + dale_scenes
    _idx = np.random.default_rng(0).choice(len(_pool), size=min(800, len(_pool)),
                                           replace=False)
    curve_scenes = [_pool[i] for i in _idx]
    W_GRID = [0.80, 0.90, 0.95, 1.00, 1.15, 1.30, 1.50, 1.80]
    try:
        curve = []
        for _wp in W_GRID:
            _env, _w = make_subj_env(_wp)
            _sc = subjectivize(curve_scenes, _wp)
            _p, _ = train(_sc, episodes=6000, seed=0, verbose=False, entropy_coef=0.05,
                          env_class=_env, n_targets_max=2 * NO_MAX)
            curve.append(model_frac_prob(_p, _env, mdp, _sc, _w))
    except NotImplementedError:
        curve = None
    return W_GRID, curve


@app.cell
def _(OBS_CHIP, OBS_DALE, W_GRID, curve, mo, plt):
    mo.stop(curve is None, mo.md("*Complete Exercises 1 & 2 to build the curve.*"))
    _fig, _ax = plt.subplots(figsize=(6, 4))
    _ax.plot(W_GRID, curve, "o-", color="#4a6fa5", label="model")
    _ax.axhline(OBS_CHIP, color="#e08a3c", ls="--", lw=1, label=f"Chip observed ({OBS_CHIP:.2f})")
    _ax.axhline(OBS_DALE, color="#5b8a72", ls="--", lw=1, label=f"Dale observed ({OBS_DALE:.2f})")
    _ax.set_xlabel("subjective probability weight  $w_p$")
    _ax.set_ylabel("model dwell fraction on probability")
    _ax.set_title("preference → looking behaviour")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Exercise 3 — fit the weight to each monkey

    The curve tells you, for any `w_p`, how probability-heavy the model looks.
    **Fitting** inverts it: given a monkey's observed dwell fraction, pick the
    swept `w_p` whose model comes closest. Fill in
    `nearest_fit(ws, fracs, target)` — `ws`/`fracs` are the swept weights and
    their model dwell fractions, `target` is the monkey's observed fraction —
    returning the `(w_p, model_frac)` pair whose `model_frac` is nearest `target`.

    (We report that model's *own* measured fraction, so the reproduction we show
    is a real trained model, not an interpolated wish.)
    """)
    return


@app.function
def nearest_fit(ws, fracs, target):
    """The swept (w_p, model_frac) whose model_frac is closest to `target`."""
    # TODO (Exercise 3): find the index i that minimizes |fracs[i] - target|
    #   (np.argmin on the absolute differences), then return (ws[i], fracs[i]).
    raise NotImplementedError("Exercise 3: implement nearest_fit")


@app.cell
def _(verdict):
    def _ex3_checks():
        ws = [0.8, 1.0, 1.2]
        fracs = [0.2, 0.5, 0.9]
        return [
            ("target near a sampled point picks that point",
             nearest_fit(ws, fracs, 0.52) == (1.0, 0.5)),
            ("high target picks the high-weight model",
             nearest_fit(ws, fracs, 0.85) == (1.2, 0.9)),
            ("low target picks the low-weight model",
             nearest_fit(ws, fracs, 0.18) == (0.8, 0.2)),
        ]
    verdict(_ex3_checks)
    return


@app.cell
def _(OBS_CHIP, OBS_DALE, W_GRID, curve):
    # --- Given: fit each monkey by matching its observed dwell to the curve ---
    try:
        if curve is None:
            raise NotImplementedError
        wp_chip, frac_chip = nearest_fit(W_GRID, curve, OBS_CHIP)
        wp_dale, frac_dale = nearest_fit(W_GRID, curve, OBS_DALE)
        fit_done = True
    except (NotImplementedError, TypeError):
        wp_chip = frac_chip = wp_dale = frac_dale = None
        fit_done = False
    return fit_done, frac_chip, frac_dale, wp_chip, wp_dale


@app.cell
def _(
    OBS_CHIP,
    OBS_DALE,
    fit_done,
    frac_chip,
    frac_dale,
    mo,
    wp_chip,
    wp_dale,
):
    mo.stop(not fit_done, mo.md("*Complete Exercises 1–3 to fit both monkeys.*"))
    _table = f"""
    | monkey | observed prob-dwell | fitted $w_p$ | model prob-dwell (fitted) |
    |---|---|---|---|
    | **Chip** | {OBS_CHIP:.2f} | **{wp_chip:.2f}** | {frac_chip:.2f} |
    | **Dale** | {OBS_DALE:.2f} | **{wp_dale:.2f}** | {frac_dale:.2f} |
    """
    mo.md("### The fitted individual difference\n" + _table)
    return


@app.cell
def _(
    OBS_CHIP,
    OBS_DALE,
    fit_done,
    frac_chip,
    frac_dale,
    verdict,
    wp_chip,
    wp_dale,
):
    def _fit_checks():
        if not fit_done:
            raise NotImplementedError("Exercises 1–3: needed to fit both monkeys")
        return [
            (f"Chip's fitted weight ({wp_chip:.2f}) exceeds Dale's ({wp_dale:.2f})",
             wp_chip > wp_dale),
            (f"model reproduces Chip's probability-riding "
             f"(fitted {frac_chip:.2f} vs observed {OBS_CHIP:.2f})",
             abs(frac_chip - OBS_CHIP) < 0.15),
            (f"model reproduces Dale's balance "
             f"(fitted {frac_dale:.2f} vs observed {OBS_DALE:.2f})",
             abs(frac_dale - OBS_DALE) < 0.15),
        ]
    verdict(_fit_checks)
    return


@app.cell
def _(fit_done, mo, wp_chip, wp_dale):
    mo.stop(not fit_done,
            mo.md("*Complete the exercises above to read the wrap-up with your numbers.*"))
    mo.md(
        rf"""
        ---
        ## What just happened

        One number per monkey — the **subjective probability weight** `w_p` — was
        enough to turn the *same* resource-rational model into either animal:

        - **Chip: `w_p ≈ {wp_chip:.2f}`** (high — probability weighs far more than
          reward). Probability dominates his subjective value, so it is the attribute
          worth measuring, and the model **rides the probability bar** — Chip.
        - **Dale: `w_p ≈ {wp_dale:.2f}`** (near 1: reward and probability matter about
          equally). Now neither attribute is decisively more diagnostic, so the model
          **splits its looks** — Dale.

        Notice how *small* the weight difference is compared to how *large* the
        behavioural difference is (≈0.93 vs ≈0.43 probability dwell). That is the
        model earning its keep: because "look at the more diagnostic attribute" is a
        sharp, near-threshold decision, a **modest difference in what a monkey values
        produces a dramatic difference in where it looks.** Individual differences in
        gaze need not mean wildly different minds — a single preference parameter,
        acting through a resource-rational bottleneck, is enough.

        **Where this sits in the arc.** Part 1 *found* the behaviour in raw gaze;
        Part 2 showed a task-trained model *reproduces* it and pins down the
        task-optimal (Chip-like) strategy; Part 3 *fits the individual* with one
        interpretable knob. The fitted `w_p` is now a per-subject parameter you could
        carry to held-out data — the natural next test being whether a monkey's
        `w_p`, fit from looking, predicts its **choices** (and, with the recordings,
        its neural signals) on sessions the model never saw.
        """
    )
    return


if __name__ == "__main__":
    app.run()
