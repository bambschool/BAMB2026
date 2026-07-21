"""
models.py: All models and likelihood functions
Prepared by Charley M. Wu (TU Darmstadt; hmc-lab.com)
for the 2026 Barcelona Summer School for Advanced Modeling of Behavior (BAMB; https://www.bambschool.org/)
"""

import random
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

from environments import GridWorldEnv


# ================================================================
# DYNA-Q  (Day 1 Block 1 / Day 2 Track 1)
# ================================================================

def q_update(
    q_values: np.ndarray,
    state: int,
    action: int,
    reward: float,
    next_state: int,
    eta: float,
    gamma: float,
) -> None:
    best_next = np.max(q_values[next_state])
    q_values[state, action] += eta * (reward + gamma * best_next - q_values[state, action])


def planning_step(
    q_values: np.ndarray,
    model: dict,
    eta: float,
    gamma: float,
    planning_steps: int,
) -> None:
    if not model:
        return
    for _ in range(planning_steps):
        state, action = random.choice(list(model.keys()))
        next_state, reward = model[(state, action)]
        q_update(q_values, state, action, reward, next_state, eta, gamma)


def run_episode_dyna_q(
    env: GridWorldEnv,
    q_values: np.ndarray,
    model: dict,
    eta: float,
    gamma: float,
    epsilon: float,
    planning_steps: int,
) -> Tuple[np.ndarray, dict, float]:
    state, _ = env.reset()
    total_return = 0.0
    done = False

    while not done:
        if np.random.rand() < epsilon:
            action = np.random.randint(env.action_space.n)
        else:
            action = int(np.argmax(q_values[state]))

        next_state, reward, terminated, truncated, _ = env.step(action) #"truncated" means max step is reached
        done = terminated or truncated

        q_update(q_values, state, action, reward, next_state, eta, gamma)
        model[(state, action)] = (next_state, reward)
        planning_step(q_values, model, eta, gamma, planning_steps)

        state = next_state
        total_return += reward

    return q_values, model, total_return


# ================================================================
# Section 2: CRP-Dyna-Q / DINER  (Day 1 Block 2 / Day 2 Track 1)
# ================================================================

def crp_assignment_probs(counts: List[int], alpha: float) -> np.ndarray:
    """Return CRP probabilities over existing clusters plus a new-cluster option.

    Parameters
    ----------
    counts : list of ints, one per existing cluster (number of episodes assigned so far)
    alpha  : concentration parameter; higher α → more new clusters

    Returns
    -------
    probs : array of length len(counts)+1. The last element is the probability
            of opening a new cluster; the rest correspond to existing clusters.
            Each existing cluster's weight is proportional to its count.
    """
    total = sum(counts) + alpha
    if total <= 0:
        return np.array([1.0])
    existing = np.array([count / total for count in counts])
    new = alpha / total
    return np.concatenate([existing, [new]])


def generate_crp_cluster_assignments(episodes: int, alpha: float, seed: int = 0) -> List[int]:
    """Generate cluster assignments for a sequence of episodes."""
    random.seed(seed)
    np.random.seed(seed)
    counts: List[int] = []
    assignments: List[int] = []

    for _ in range(episodes):
        probs = crp_assignment_probs(counts, alpha)
        cluster_idx = int(np.random.choice(len(probs), p=probs))
        if cluster_idx == len(counts):
            counts.append(0)
        counts[cluster_idx] += 1
        assignments.append(cluster_idx)

    return assignments


def run_baseline_sequence(env_sequence: List[str], envs: Dict[str, GridWorldEnv], eta: float, gamma: float, epsilon: float, planning_steps: int) -> List[float]:
    """Fresh-start DYNA-Q for each episode, no transfer."""
    episode_returns: List[float] = []
    for env_name in env_sequence:
        env = envs[env_name]
        q_values = np.zeros((env.observation_space.n, env.action_space.n))
        model: dict = {}
        _, _, total_return = run_episode_dyna_q(env, q_values, model, eta, gamma, epsilon, planning_steps)
        episode_returns.append(total_return)
    return episode_returns


def plot_episode_returns(returns: List[float], label: str, color: str) -> None:
    plt.plot(returns, label=label, color=color, linewidth=2)


def run_crp_dyna_q_sequence(env_sequence: List[str], assignments: List[int], envs: Dict[str, GridWorldEnv], eta: float, gamma: float, epsilon: float, planning_steps: int):
    """CRP"""
    clusters: List[dict] = []
    episode_returns: List[float] = []

    for env_name, cluster_idx in zip(env_sequence, assignments):
        while cluster_idx >= len(clusters):
            clusters.append({
                'env_name': None,
                'q_values': None,
                'model': {},
            })

        cluster = clusters[cluster_idx]
        if cluster['q_values'] is None:
            env = envs[env_name]
            cluster['env_name'] = env_name
            cluster['q_values'] = np.zeros((env.observation_space.n, env.action_space.n))
            cluster['model'] = {}

        env = envs[env_name]
        cluster['q_values'], cluster['model'], total_return = run_episode_dyna_q(
            env,
            cluster['q_values'],
            cluster['model'],
            eta,
            gamma,
            epsilon,
            planning_steps,
        )
        episode_returns.append(total_return)

    return episode_returns, clusters


def collect_probe_transitions(env: GridWorldEnv, n_probe: int = 20) -> list:
    """Collect probe transitions to determine cluster.

    Always starts by trying all 4 actions from the start state; these transitions
    are maximally informative because different environments place walls differently
    around the start position.  The remaining steps are random exploration.
    """
    transitions = []
    start_state, _ = env.reset()

    # Try every action from the start state
    for action in range(env.action_space.n):
        env.reset()
        next_state, reward, term, trunc, _ = env.step(action)
        transitions.append((start_state, action, next_state, reward))

    # Random exploration for additional transitions
    state, _ = env.reset()
    for _ in range(max(0, n_probe - env.action_space.n)):
        action = int(np.random.randint(env.action_space.n))
        next_state, reward, term, trunc, _ = env.step(action)
        transitions.append((state, action, next_state, reward))
        if term or trunc:
            state, _ = env.reset()
        else:
            state = next_state

    return transitions


def has_contradiction(model: dict, transitions: list) -> bool:
    """True if any probe transition contradicts the cluster's stored model."""
    for s, a, s_next, r in transitions:
        if (s, a) in model and model[(s, a)] != (s_next, r):
            return True
    return False


def has_probe_contradiction(stored_probe: list, current_probe: list) -> bool:
    """True if current probe contradicts the cluster's stored fingerprint probe."""
    stored = {(s, a): (s_next, r) for s, a, s_next, r in stored_probe}
    for s, a, s_next, r in current_probe:
        if (s, a) in stored and stored[(s, a)] != (s_next, r):
            return True
    return False


def model_match_rate(model: dict, transitions: list) -> float:
    """Fraction of predictable transitions correctly predicted by the model."""
    if not model or not transitions:
        return 0.5
    predictable = [(s, a, s_next, r) for s, a, s_next, r in transitions if (s, a) in model]
    if not predictable:
        return 0.5
    matches = sum(1 for s, a, s_next, r in predictable if model[(s, a)] == (s_next, r))
    return matches / len(predictable)


def run_persistent_dyna_q_sequence(
    env_sequence: List[str],
    envs: Dict[str, GridWorldEnv],
    eta: float,
    gamma: float,
    epsilon: float,
    planning_steps: int,
) -> List[float]:
    """DYNA-Q with a single shared Q-table and model across all episodes."""
    first_env = envs[env_sequence[0]]
    q_values = np.zeros((first_env.observation_space.n, first_env.action_space.n))
    model: dict = {}
    episode_returns: List[float] = []

    for env_name in env_sequence:
        env = envs[env_name]
        _, _, total_return = run_episode_dyna_q(env, q_values, model, eta, gamma, epsilon, planning_steps)
        episode_returns.append(total_return)

    return episode_returns


def run_diner_sequence(
    env_sequence: List[str],
    envs: Dict[str, GridWorldEnv],
    crp_alpha: float,
    eta: float,
    gamma: float,
    epsilon: float,
    planning_steps: int,
    n_probe: int = 10,
    seed: int = 0,
) -> Tuple[List[float], List[int], List[dict]]:
    """DINER: Dyna with Inferred Nonparametric Environment Recurrence."""
    np.random.seed(seed)
    random.seed(seed)

    clusters: List[dict] = []
    episode_returns: List[float] = []
    cluster_assignments: List[int] = []

    for env_name in env_sequence:
        env = envs[env_name]
        n_states = env.observation_space.n
        n_actions = env.action_space.n

        probe = collect_probe_transitions(env, n_probe)

        if not clusters:
            k = 0
            clusters.append({
                'q_values': np.zeros((n_states, n_actions)),
                'model': {},
                'count': 0,
                'fingerprint': probe,
            })
        else:
            inconsistent = [has_probe_contradiction(c['fingerprint'], probe) for c in clusters]

            if all(inconsistent):
                k = len(clusters)
                clusters.append({
                    'q_values': np.zeros((n_states, n_actions)),
                    'model': {},
                    'count': 0,
                    'fingerprint': probe,
                })
            else:
                counts = [c['count'] for c in clusters]
                total = sum(counts) + crp_alpha
                weights = np.array([
                    count / total if not ctrd else 0.0
                    for count, ctrd in zip(counts, inconsistent)
                ], dtype=float)
                weights /= weights.sum()
                k = int(np.random.choice(len(weights), p=weights))

        clusters[k]['count'] += 1
        cluster_assignments.append(k)

        q_values, model, total_return = run_episode_dyna_q(
            env,
            clusters[k]['q_values'],
            clusters[k]['model'],
            eta,
            gamma,
            epsilon,
            planning_steps,
        )
        clusters[k]['q_values'] = q_values
        clusters[k]['model'] = model
        episode_returns.append(total_return)

    return episode_returns, cluster_assignments, clusters


# ================================================================
# Section 3: GP Posterior Toolkit  (Day 1 Block 3 inline; Day 2 Track 2 imports)
# ================================================================
#
# Core GP functions: kernel and posterior.
# Day 1 Block 3 defines equivalent functions inline for pedagogical purposes.
# Day 2 Track 2 imports these directly.


def rbf_kernel(X1: np.ndarray, X2: np.ndarray, lengthscale: float = 1.0) -> np.ndarray:
    """RBF kernel: k(x,x') = exp(-‖x-x'‖²/2λ²)."""
    sq_dists = cdist(np.atleast_2d(X1), np.atleast_2d(X2), metric='sqeuclidean')
    return np.exp(-0.5 * sq_dists / lengthscale ** 2)


def gp_posterior(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    lengthscale: float = 1.0,
    noise: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """GP regression posterior (mu, std, cov) at X_test — kept for backward compatibility."""
    n = len(y_train)
    K = rbf_kernel(X_train, X_train, lengthscale) + noise ** 2 * np.eye(n)
    K_s = rbf_kernel(X_test, X_train, lengthscale)
    K_ss = rbf_kernel(X_test, X_test, lengthscale)
    mu = K_s @ np.linalg.solve(K, y_train)
    v = np.linalg.solve(K, K_s.T)
    cov = K_ss - K_s @ v
    std = np.sqrt(np.maximum(np.diag(cov), 0.0))
    return mu, std, cov


def gp_laplace_posterior(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    lengthscale: float = 1.0,
    max_iter: int = 20,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """GP classification via Laplace approximation (Rasmussen & Williams 2006, Algorithm 3.2).

    Observation model: y ~ Bernoulli(σ(f(x))), where f is a latent GP on the real line.
    Returns (f_star, std_star): posterior mean and std of the latent function at X_test.

    f > 0 predicts reward; f < 0 predicts absence; f ≈ 0 is uncertain.
    The G feature is then G(x) = σ(f_star(x) / √(1 + π·std²(x)/8)).

    This is the implementation used in Wu et al. (2025).
    """
    n = len(y_train)
    if n == 0:
        m = len(np.atleast_2d(X_test))
        return np.zeros(m), np.ones(m)

    y = y_train.astype(float)
    K = rbf_kernel(X_train, X_train, lengthscale) + 1e-6 * np.eye(n)

    f = np.zeros(n)
    for _ in range(max_iter):
        pi = 1.0 / (1.0 + np.exp(-np.clip(f, -500, 500)))
        W = pi * (1.0 - pi)
        W_sqrt = np.sqrt(np.maximum(W, 1e-12))
        B = np.eye(n) + (W_sqrt[:, None] * K) * W_sqrt[None, :]
        try:
            L = np.linalg.cholesky(B + 1e-8 * np.eye(n))
        except np.linalg.LinAlgError:
            break
        b = W * f + (y - pi)
        a_inner = np.linalg.solve(L.T, np.linalg.solve(L, W_sqrt * (K @ b)))
        f_new = K @ (b - W_sqrt * a_inner)
        if np.max(np.abs(f_new - f)) < tol:
            f = f_new
            break
        f = f_new

    pi = 1.0 / (1.0 + np.exp(-np.clip(f, -500, 500)))
    W = pi * (1.0 - pi)
    W_sqrt = np.sqrt(np.maximum(W, 1e-12))
    B = np.eye(n) + (W_sqrt[:, None] * K) * W_sqrt[None, :]
    try:
        L = np.linalg.cholesky(B + 1e-8 * np.eye(n))
    except np.linalg.LinAlgError:
        return K @ np.linalg.solve(K, f), np.ones(len(X_test))

    K_s = rbf_kernel(X_test, X_train, lengthscale)    # (m, n)
    f_star = K_s @ np.linalg.solve(K, f)              # predictive latent mean
    V = np.linalg.solve(L, W_sqrt[:, None] * K_s.T)  # (n, m)
    var_star = np.maximum(1.0 - np.sum(V ** 2, axis=0), 0.0)   # k(x*,x*)=1 for RBF
    return f_star, np.sqrt(var_star)


def _gp_mu_std(
    X_obs: np.ndarray,
    y_obs: np.ndarray,
    X_all: np.ndarray,
    lengthscale: float,
    noise: float,   # kept for API compatibility; classification has no noise term
) -> Tuple[np.ndarray, np.ndarray]:
    """Latent GP posterior mean and std via Laplace classification (Wu et al. 2025)."""
    return gp_laplace_posterior(X_obs, y_obs, X_all, lengthscale)


# ================================================================
# Section 3b: GLM Choice Model  (Day 2 Track 2; Wu et al. 2025)
# ================================================================
#
# Primary model (Wu et al. 2025 paper):
#   P(choice_i) ∝ exp( w_mean·μ(i) + w_local·loc(i) )
#   Free params: λ (lengthscale), w_mean, w_local  (k=3)
#
# GP-mean only (ablation — locality removed):
#   P(choice_i) ∝ exp( w_mean·μ(i) )
#   Free params: λ, w_mean  (k=2)
#
# Locality-only (null model — no GP terms, no reward structure):
#   P(choice_i) ∝ exp( w_local·loc(i) )
#   Free params: w_local  (k=1)
#
# Extension B — explicit uncertainty weight:
#   P(choice_i) ∝ exp( w_mean·μ(i) + w_ucb·σ(i) + w_local·loc(i) )
#   Free params: λ, w_mean, w_ucb, w_local  (k=4)
#
# Features are defined for each *unvisited* block; visited blocks are
# excluded from the choice set (_visited_mask).
# μ and σ come from GP classification via Laplace approximation (R&W 2006)
# on binary 0/1 reward observations — the same implementation as Wu et al. (2025).


def locality_feature(X_all: np.ndarray, x_current: Optional[np.ndarray]) -> np.ndarray:
    """Locality feature: negative Euclidean distance from current position.

    Closer blocks get values near 0; farther blocks are more negative.
    For the first step in a round (no current position), returns zeros.
    """
    if x_current is None:
        return np.zeros(len(X_all))
    # Changed for Excr 2 in section 6.1
    ell = 6
    diffs = X_all - x_current[np.newaxis, :]
    distances = np.sqrt((diffs ** 2).sum(axis=1))# (N, 2)
    return np.exp(-distances / ell)   # (N,), range [−√(57²+57²), 0] in raw coords


def _visited_mask(X_all: np.ndarray, X_obs: np.ndarray) -> np.ndarray:
    """Boolean mask: True = this block has already been visited."""
    if len(X_obs) == 0:
        return np.zeros(len(X_all), dtype=bool)
    dists = cdist(X_all, X_obs)
    return dists.min(axis=1) < 0.01


def glm_choice_probs(
    mu: np.ndarray,
    std: np.ndarray,
    loc: np.ndarray,
    w_mean: float,
    w_ucb: float,
    w_local: float,
    visited: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Softmax GLM choice probabilities over unvisited blocks (Wu et al. 2025).

    Feature G = μ(x)    — latent GP posterior mean from Laplace approximation (real line).
    Feature L = loc(x)  — locality (passed in directly).
    Feature U = std(x)  — explicit uncertainty; only used in the GLU extension (w_ucb term).

    P(x) ∝ exp(w_mean · μ(x) + w_ucb · std(x) + w_local · loc(x))

    For the paper model (GL): w_ucb=0.  G = μ > 0 attracts, G = μ < 0 repels.

    visited: boolean mask of already-visited blocks (excluded from choice).
    """
    g_feature = mu   # G = latent z = Laplace posterior mean (real line)
    logits = w_mean * g_feature + w_ucb * std + w_local * loc
    if visited is not None:
        logits = logits.copy()
        logits[visited] = -np.inf
    finite = logits[np.isfinite(logits)]
    if len(finite) == 0:
        probs = np.ones(len(logits)) / len(logits)
        if visited is not None:
            probs[visited] = 0.0
        return probs
    logits -= finite.max() #avoid numerical overflow
    probs = np.exp(np.clip(logits, -500, 0))
    probs[~np.isfinite(logits)] = 0.0
    s = probs.sum()
    return probs / s if s > 0 else probs


def decision_loglik_glm(
    X_all: np.ndarray,
    X_obs: np.ndarray,
    y_obs: np.ndarray,
    next_idx: int,
    lengthscale: float,
    w_mean: float,
    w_ucb: float,
    w_local: float,
    noise: float = 0.1,
) -> float:
    """Log P(choose next_idx | softmax GLM with features [GP mean, GP std, locality])."""
    n_blocks = len(X_all)
    if len(X_obs) == 0:
        mu = np.zeros(n_blocks)
        std = np.ones(n_blocks)
        x_current = None
    else:
        mu, std = _gp_mu_std(X_obs, y_obs, X_all, lengthscale, noise)
        x_current = X_obs[-1]

    visited = _visited_mask(X_all, X_obs)
    loc = locality_feature(X_all, x_current)
    probs = glm_choice_probs(mu, std, loc, w_mean, w_ucb, w_local, visited)
    return float(np.log(probs[next_idx] + 1e-300))


def round_loglik_glm(
    decisions: list,
    X_all: np.ndarray,
    lengthscale: float,
    w_mean: float,
    w_ucb: float,
    w_local: float,
    noise: float = 0.1,
) -> float:
    """Sum of softmax GLM decision log-likelihoods for one (session, round)."""
    total = 0.0
    for X_obs, y_obs, next_idx in decisions:
        total += decision_loglik_glm(
            X_all, X_obs, y_obs, next_idx, lengthscale, w_mean, w_ucb, w_local, noise
        )
    return total


def fit_participant_glm(
    df,
    name: str,
    X_all: Optional[np.ndarray] = None,
    noise: float = 0.1,
    n_restarts: int = 3,
    seed: int = 0,
    max_rounds: Optional[int] = 10,
    max_decisions: Optional[int] = 15,
    fixed_lengthscale: Optional[float] = None,
) -> dict:
    """MLE fit of softmax GLM + uncertainty parameters for one participant (GLU model).

    Free params: λ, w_mean, w_ucb, w_local  (k=4) unless fixed_lengthscale is provided,
    in which case λ is held fixed and only (w_mean, w_ucb, w_local) are optimised (k=3).
    """
    from wu2025 import grid_coords, iter_participant_decisions
    if X_all is None:
        X_all = grid_coords()

    all_rounds = list(iter_participant_decisions(df, name))
    if max_rounds is not None:
        all_rounds = all_rounds[:max_rounds]
    if max_decisions is not None:
        all_rounds = [(s, r, d[:max_decisions]) for s, r, d in all_rounds]

    n_steps = sum(len(d) for _, _, d in all_rounds)
    rng = np.random.RandomState(seed)
    best: Optional[dict] = None

    if fixed_lengthscale is not None:
        ls = float(fixed_lengthscale)

        def neg_loglik(params):
            w_mean, w_ucb, w_local = params
            if abs(w_mean) > 50.0 or w_ucb < -1.0 or w_ucb > 30.0 or abs(w_local) > 2.0:
                return 1e9
            total = 0.0
            for _, _, decisions in all_rounds:
                total += round_loglik_glm(decisions, X_all, ls, w_mean, w_ucb, w_local, noise)
            return -total

        for _ in range(n_restarts):
            x0 = [rng.uniform(0.5, 4.0), rng.uniform(0.0, 1.5), rng.uniform(0.0, 0.3)]
            res = minimize(neg_loglik, x0, method='Nelder-Mead',
                           options={'xatol': 1e-4, 'fatol': 1e-4, 'maxiter': 4000})
            if best is None or res.fun < best['negloglik']:
                best = {
                    'name': name,
                    'lengthscale': ls,
                    'w_mean': float(res.x[0]),
                    'w_ucb': float(res.x[1]),
                    'w_local': float(res.x[2]),
                    'negloglik': float(res.fun),
                    'n_steps': n_steps,
                }
    else:
        def neg_loglik(params):  # type: ignore[misc]
            lengthscale, w_mean, w_ucb, w_local = params
            if lengthscale <= 0.0 or lengthscale > 100.0:
                return 1e9
            if abs(w_mean) > 50.0 or w_ucb < -1.0 or w_ucb > 30.0 or abs(w_local) > 2.0:
                return 1e9
            total = 0.0
            for _, _, decisions in all_rounds:
                total += round_loglik_glm(decisions, X_all, lengthscale, w_mean, w_ucb, w_local, noise)
            return -total

        for _ in range(n_restarts):
            x0 = [rng.uniform(3.0, 20.0), rng.uniform(0.5, 4.0),
                  rng.uniform(0.0, 1.5), rng.uniform(0.0, 0.3)]
            res = minimize(neg_loglik, x0, method='Nelder-Mead',
                           options={'xatol': 1e-4, 'fatol': 1e-4, 'maxiter': 4000})
            if best is None or res.fun < best['negloglik']:
                best = {
                    'name': name,
                    'lengthscale': float(res.x[0]),
                    'w_mean': float(res.x[1]),
                    'w_ucb': float(res.x[2]),
                    'w_local': float(res.x[3]),
                    'negloglik': float(res.fun),
                    'n_steps': n_steps,
                }

    return best  # type: ignore[return-value]


def fit_participant_glm_no_ucb(
    df,
    name: str,
    X_all: Optional[np.ndarray] = None,
    noise: float = 0.1,
    n_restarts: int = 3,
    seed: int = 0,
    max_rounds: Optional[int] = 10,
    max_decisions: Optional[int] = 15,
    fixed_lengthscale: Optional[float] = None,
) -> dict:
    """MLE fit for GP-mean + locality model (paper model; w_ucb=0).

    Free params: λ, w_mean, w_local  (k=3) unless fixed_lengthscale is provided,
    in which case λ is held fixed and only (w_mean, w_local) are optimised (k=2).
    Pass fixed_lengthscale to test sensitivity to the assumed generalisation scale.
    """
    from wu2025 import grid_coords, iter_participant_decisions
    if X_all is None:
        X_all = grid_coords()

    all_rounds = list(iter_participant_decisions(df, name))
    if max_rounds is not None:
        all_rounds = all_rounds[:max_rounds]
    if max_decisions is not None:
        all_rounds = [(s, r, d[:max_decisions]) for s, r, d in all_rounds]

    n_steps = sum(len(d) for _, _, d in all_rounds)

    rng = np.random.RandomState(seed)
    best: Optional[dict] = None

    if fixed_lengthscale is not None:
        ls = float(fixed_lengthscale)

        def neg_loglik(params):
            w_mean, w_local = params
            if abs(w_mean) > 50.0 or abs(w_local) > 2.0:
                return 1e9
            total = 0.0
            for _, _, decisions in all_rounds:
                total += round_loglik_glm(decisions, X_all, ls, w_mean, 0.0, w_local, noise)
            return -total

        for _ in range(n_restarts):
            x0 = [rng.uniform(0.5, 4.0), rng.uniform(0.0, 0.3)]
            res = minimize(neg_loglik, x0, method='Nelder-Mead',
                           options={'xatol': 1e-4, 'fatol': 1e-4, 'maxiter': 3000})
            if best is None or res.fun < best['negloglik']:
                best = {
                    'name': name,
                    'lengthscale': ls,
                    'w_mean': float(res.x[0]),
                    'w_ucb': 0.0,
                    'w_local': float(res.x[1]),
                    'negloglik': float(res.fun),
                    'n_steps': n_steps,
                }
    else:
        def neg_loglik(params):  # type: ignore[misc]
            lengthscale, w_mean, w_local = params
            if lengthscale <= 0.0 or lengthscale > 100.0:
                return 1e9
            if abs(w_mean) > 50.0 or abs(w_local) > 2.0:
                return 1e9
            total = 0.0
            for _, _, decisions in all_rounds:
                total += round_loglik_glm(decisions, X_all, lengthscale, w_mean, 0.0, w_local, noise)
            return -total

        for _ in range(n_restarts):
            x0 = [rng.uniform(3.0, 20.0), rng.uniform(0.5, 4.0), rng.uniform(0.0, 0.3)]
            res = minimize(neg_loglik, x0, method='Nelder-Mead',
                           options={'xatol': 1e-4, 'fatol': 1e-4, 'maxiter': 3000})
            if best is None or res.fun < best['negloglik']:
                best = {
                    'name': name,
                    'lengthscale': float(res.x[0]),
                    'w_mean': float(res.x[1]),
                    'w_ucb': 0.0,
                    'w_local': float(res.x[2]),
                    'negloglik': float(res.fun),
                    'n_steps': n_steps,
                }

    return best  # type: ignore[return-value]


def fit_participant_gp_mean_only(
    df,
    name: str,
    X_all: Optional[np.ndarray] = None,
    noise: float = 0.1,
    n_restarts: int = 3,
    seed: int = 0,
    max_rounds: Optional[int] = 10,
    max_decisions: Optional[int] = 15,
    fixed_lengthscale: Optional[float] = None,
) -> dict:
    """MLE fit for GP-mean only model (no locality; w_ucb=w_local=0).

    P(x) ∝ exp(w_mean · μ(x))   — tests whether reward structure alone drives choices.
    Free params: λ (lengthscale), w_mean  (k=2) unless fixed_lengthscale is provided,
    in which case only w_mean is optimised (k=1).
    """
    from wu2025 import grid_coords, iter_participant_decisions
    if X_all is None:
        X_all = grid_coords()

    all_rounds = list(iter_participant_decisions(df, name))
    if max_rounds is not None:
        all_rounds = all_rounds[:max_rounds]
    if max_decisions is not None:
        all_rounds = [(s, r, d[:max_decisions]) for s, r, d in all_rounds]

    n_steps = sum(len(d) for _, _, d in all_rounds)

    rng = np.random.RandomState(seed)
    best: Optional[dict] = None

    if fixed_lengthscale is not None:
        ls = float(fixed_lengthscale)

        def neg_loglik(params):
            (w_mean,) = params
            if abs(w_mean) > 50.0:
                return 1e9
            total = 0.0
            for _, _, decisions in all_rounds:
                total += round_loglik_glm(decisions, X_all, ls, w_mean, 0.0, 0.0, noise)
            return -total

        for _ in range(n_restarts):
            x0 = [rng.uniform(0.5, 4.0)]
            res = minimize(neg_loglik, x0, method='Nelder-Mead',
                           options={'xatol': 1e-4, 'fatol': 1e-4, 'maxiter': 2000})
            if best is None or res.fun < best['negloglik']:
                best = {
                    'name': name,
                    'lengthscale': ls,
                    'w_mean': float(res.x[0]),
                    'w_ucb': 0.0,
                    'w_local': 0.0,
                    'negloglik': float(res.fun),
                    'n_steps': n_steps,
                }
    else:
        def neg_loglik(params):  # type: ignore[misc]
            lengthscale, w_mean = params
            if lengthscale <= 0.0 or lengthscale > 100.0:
                return 1e9
            if abs(w_mean) > 50.0:
                return 1e9
            total = 0.0
            for _, _, decisions in all_rounds:
                total += round_loglik_glm(decisions, X_all, lengthscale, w_mean, 0.0, 0.0, noise)
            return -total

        for _ in range(n_restarts):
            x0 = [rng.uniform(3.0, 20.0), rng.uniform(0.5, 4.0)]
            res = minimize(neg_loglik, x0, method='Nelder-Mead',
                           options={'xatol': 1e-4, 'fatol': 1e-4, 'maxiter': 2000})
            if best is None or res.fun < best['negloglik']:
                best = {
                    'name': name,
                    'lengthscale': float(res.x[0]),
                    'w_mean': float(res.x[1]),
                    'w_ucb': 0.0,
                    'w_local': 0.0,
                    'negloglik': float(res.fun),
                    'n_steps': n_steps,
                }

    return best  # type: ignore[return-value]


def fit_participant_locality_only(
    df,
    name: str,
    X_all: Optional[np.ndarray] = None,
    n_restarts: int = 3,
    seed: int = 0,
    max_rounds: Optional[int] = 10,
    max_decisions: Optional[int] = 15,
) -> dict:
    """MLE fit of locality-only model (single free parameter w_local, no GP terms).

    P(x) ∝ exp(w_local · loc(x))  where loc(x) = -‖x - x_curr‖

    This is the null model for the GP comparison: it captures spatial proximity
    preference but ignores all reward structure. If the GP-based GLM models
    outperform this by BIC, reward structure demonstrably drives choices.
    """
    from wu2025 import grid_coords, iter_participant_decisions
    if X_all is None:
        X_all = grid_coords()

    all_rounds = list(iter_participant_decisions(df, name))
    if max_rounds is not None:
        all_rounds = all_rounds[:max_rounds]
    if max_decisions is not None:
        all_rounds = [(s, r, d[:max_decisions]) for s, r, d in all_rounds]

    n_steps = sum(len(d) for _, _, d in all_rounds)

    def neg_loglik(params):
        w_local, = params
        if abs(w_local) > 2.0:
            return 1e9
        total = 0.0
        for _, _, decisions in all_rounds:
            for X_obs, y_obs, next_idx in decisions:
                x_current = X_obs[-1] if len(X_obs) > 0 else None
                loc = locality_feature(X_all, x_current)
                visited = _visited_mask(X_all, X_obs)
                logits = w_local * loc
                if visited is not None:
                    logits = logits.copy()
                    logits[visited] = -np.inf
                finite = logits[np.isfinite(logits)]
                if len(finite) == 0:
                    continue
                logits -= finite.max()
                probs = np.exp(np.clip(logits, -500, 0))
                probs[~np.isfinite(logits)] = 0.0
                s = probs.sum()
                if s > 0:
                    probs /= s
                total += float(np.log(probs[next_idx] + 1e-300))
        return -total

    rng = np.random.RandomState(seed)
    best: Optional[dict] = None

    for _ in range(n_restarts):
        x0 = [rng.uniform(0.01, 0.15)]   # raw coords; distances ∈ [0, ~81]
        res = minimize(neg_loglik, x0, method='Nelder-Mead',
                       options={'xatol': 1e-4, 'fatol': 1e-4, 'maxiter': 500})
        if best is None or res.fun < best['negloglik']:
            best = {
                'name': name,
                'w_local': float(res.x[0]),
                'negloglik': float(res.fun),
                'n_steps': n_steps,
            }

    return best  # type: ignore[return-value]


def compute_bic_glm(negloglik: float, n_params: int, n_steps: int) -> float:
    """BIC for GLM models: k·ln(n) − 2·ln(L̂). Lower = better."""
    return n_params * np.log(n_steps) + 2.0 * negloglik
