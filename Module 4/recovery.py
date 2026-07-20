"""
recovery.py: softmax DINER simulation and likelihood for Day 2, Track 1.
Prepared by Charley M. Wu (TU Darmstadt; hmc-lab.com) 
for the 2026 Barcelona Summer School for Advanced Modeling of Behavior (BAMB; https://www.bambschool.org/)



Provides three models (DINER, fresh-start, persistent) each with a softmax policy,
alongside MLE fitting routines for Wilson & Collins (2019) parameter and model recovery.
"""

import random
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

from environments import GridWorldEnv
from models import collect_probe_transitions, has_probe_contradiction


# ---------------------------------------------------------------------------
# Softmax policy
# ---------------------------------------------------------------------------

def softmax_probs(q_row: np.ndarray, tau: float) -> np.ndarray:
    """Softmax action distribution over Q-values with temperature τ."""
    logits = q_row / max(tau, 1e-8)
    logits = logits - logits.max() #prevent numerical overflow
    probs = np.exp(logits)
    return probs / probs.sum()


# ---------------------------------------------------------------------------
# Single-episode helpers
# ---------------------------------------------------------------------------

def _run_episode_softmax(
    env: GridWorldEnv,
    q_values: np.ndarray,
    model: dict,
    eta: float,
    gamma: float,
    tau: float,
    planning_steps: int,
) -> Tuple[np.ndarray, dict, List[Tuple[int, int]], float]:
    """Run one episode with softmax policy. Returns (q_values, model, trajectory, total_return).

    trajectory is a list of (state, action) pairs, one per time step.
    """
    state, _ = env.reset()
    trajectory: List[Tuple[int, int]] = []
    total_return = 0.0
    done = False

    while not done:
        probs = softmax_probs(q_values[state], tau)
        action = int(np.random.choice(len(probs), p=probs))

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        trajectory.append((state, action))

        # Q-update
        best_next = float(np.max(q_values[next_state]))
        q_values[state, action] += eta * (reward + gamma * best_next - q_values[state, action])
        model[(state, action)] = (next_state, reward)

        # Planning sweeps
        if model and planning_steps > 0:
            keys = list(model.keys())
            for _ in range(planning_steps):
                s, a = random.choice(keys)
                ns, r = model[(s, a)]
                best_p = float(np.max(q_values[ns]))
                q_values[s, a] += eta * (r + gamma * best_p - q_values[s, a])

        total_return += reward
        state = next_state

    return q_values, model, trajectory, total_return


def _loglik_episode_softmax(
    env: GridWorldEnv,
    q_values: np.ndarray,
    model: dict,
    trajectory: List[Tuple[int, int]],
    eta: float,
    gamma: float,
    tau: float,
    planning_steps: int,
) -> float:
    """Teacher-forced log-likelihood for one episode.

    Re-plays the environment using the observed (state, action) sequence,
    accumulating log P(observed action | Q-values at each step).
    """
    loglik = 0.0
    state, _ = env.reset()

    for obs_state, obs_action in trajectory:
        probs = softmax_probs(q_values[state], tau)
        loglik += np.log(float(probs[obs_action]) + 1e-300)

        next_state, reward, terminated, truncated, _ = env.step(obs_action)
        done = terminated or truncated

        best_next = float(np.max(q_values[next_state]))
        q_values[state, obs_action] += eta * (reward + gamma * best_next - q_values[state, obs_action])
        model[(state, obs_action)] = (next_state, reward)

        if model and planning_steps > 0:
            keys = list(model.keys())
            for _ in range(planning_steps):
                s, a = random.choice(keys)
                ns, r = model[(s, a)]
                best_p = float(np.max(q_values[ns]))
                q_values[s, a] += eta * (r + gamma * best_p - q_values[s, a])

        if done:
            break
        state = next_state

    return loglik


# ---------------------------------------------------------------------------
# CRP cluster assignment (shared by simulate and loglik)
# ---------------------------------------------------------------------------

def _assign_cluster(
    clusters: List[dict],
    probe: list,
    crp_alpha: float,
    n_states: int,
    n_actions: int,
    rng: np.random.RandomState,
) -> int:
    """Return cluster index for this episode using probe fingerprinting + CRP prior."""
    if not clusters:
        clusters.append({
            'q_values': np.zeros((n_states, n_actions)),
            'model': {},
            'count': 0,
            'fingerprint': probe,
        })
        return 0

    inconsistent = [has_probe_contradiction(c['fingerprint'], probe) for c in clusters]

    if all(inconsistent):
        k = len(clusters)
        clusters.append({
            'q_values': np.zeros((n_states, n_actions)),
            'model': {},
            'count': 0,
            'fingerprint': probe,
        })
        return k

    counts = [c['count'] for c in clusters]
    total = sum(counts) + crp_alpha
    weights = np.array(
        [count / total if not ctrd else 0.0 for count, ctrd in zip(counts, inconsistent)],
        dtype=float,
    )
    weights /= weights.sum()
    return int(rng.choice(len(weights), p=weights))


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate_diner(
    env_sequence: List[str],
    envs: Dict[str, GridWorldEnv],
    crp_alpha: float,
    eta: float,
    gamma: float,
    tau: float,
    planning_steps: int,
    n_probe: int = 20,
    seed: int = 0,
) -> Tuple[List[float], List[int], List[List[Tuple[int, int]]]]:
    """Simulate DINER (Dyna with Inferred Nonparametric Environment Recurrence) with softmax policy.

    Returns
    -------
    episode_returns     : total reward per episode
    cluster_assignments : cluster index used in each episode
    trajectories        : list of (state, action) sequences per episode
    """
    rng = np.random.RandomState(seed)
    np.random.seed(seed)
    random.seed(seed)

    clusters: List[dict] = []
    episode_returns: List[float] = []
    cluster_assignments: List[int] = []
    trajectories: List[List[Tuple[int, int]]] = []

    for env_name in env_sequence:
        env = envs[env_name]
        n_states = env.observation_space.n
        n_actions = env.action_space.n

        probe = collect_probe_transitions(env, n_probe)
        k = _assign_cluster(clusters, probe, crp_alpha, n_states, n_actions, rng)
        clusters[k]['count'] += 1
        cluster_assignments.append(k)

        q, m, traj, ret = _run_episode_softmax(
            env, clusters[k]['q_values'], clusters[k]['model'],
            eta, gamma, tau, planning_steps,
        )
        clusters[k]['q_values'] = q
        clusters[k]['model'] = m
        episode_returns.append(ret)
        trajectories.append(traj)

    return episode_returns, cluster_assignments, trajectories


# ---------------------------------------------------------------------------
# True model simulators (Fresh-start and Persistent)
# ---------------------------------------------------------------------------

def simulate_fresh_start(
    env_sequence: List[str],
    envs: Dict[str, GridWorldEnv],
    eta: float,
    gamma: float,
    tau: float,
    planning_steps: int,
    seed: int = 0,
) -> Tuple[List[float], List[List[Tuple[int, int]]]]:
    """True fresh-start simulator: Q-table resets at the start of every episode.

    This is NOT equivalent to overriding DINER's cluster assignments with
    asgn = list(range(n_episodes)). That trick still generates trajectories from a
    CRP agent that reuses Q-values internally. This function generates trajectories
    from an agent that genuinely has no cross-episode memory.
    """
    np.random.seed(seed)
    random.seed(seed)

    episode_returns: List[float] = []
    trajectories: List[List[Tuple[int, int]]] = []

    for env_name in env_sequence:
        env = envs[env_name]
        n_states = env.observation_space.n
        n_actions = env.action_space.n
        q_values = np.zeros((n_states, n_actions))
        model: dict = {}
        q_values, model, traj, ret = _run_episode_softmax(
            env, q_values, model, eta, gamma, tau, planning_steps
        )
        episode_returns.append(ret)
        trajectories.append(traj)

    return episode_returns, trajectories


def simulate_persistent(
    env_sequence: List[str],
    envs: Dict[str, GridWorldEnv],
    eta: float,
    gamma: float,
    tau: float,
    planning_steps: int,
    seed: int = 0,
) -> Tuple[List[float], List[List[Tuple[int, int]]]]:
    """True persistent simulator: one shared Q-table accumulates across all episodes.

    This is NOT equivalent to overriding DINER's cluster assignments with
    asgn = [0]*n_episodes. That trick generates trajectories from a CRP agent
    (which runs the probe and uses its own Q-table logic). This function generates
    trajectories from an agent that truly never resets or distinguishes environments.
    """
    np.random.seed(seed)
    random.seed(seed)

    first_env = envs[env_sequence[0]]
    n_states = first_env.observation_space.n
    n_actions = first_env.action_space.n
    q_values = np.zeros((n_states, n_actions))
    model: dict = {}

    episode_returns: List[float] = []
    trajectories: List[List[Tuple[int, int]]] = []

    for env_name in env_sequence:
        env = envs[env_name]
        q_values, model, traj, ret = _run_episode_softmax(
            env, q_values, model, eta, gamma, tau, planning_steps
        )
        episode_returns.append(ret)
        trajectories.append(traj)

    return episode_returns, trajectories


# ---------------------------------------------------------------------------
# Log-likelihood functions
# ---------------------------------------------------------------------------

def loglik_diner(
    env_sequence: List[str],
    envs: Dict[str, GridWorldEnv],
    trajectories: List[List[Tuple[int, int]]],
    eta: float,
    gamma: float,
    tau: float,
    planning_steps: int = 5,
    cluster_assignments: Optional[List[int]] = None,
) -> float:
    """Teacher-forced log-likelihood under DINER.

    Uses pre-computed cluster_assignments (from simulate_diner or
    compute_cluster_assignments) so that fitting eta/tau doesn't change cluster structure.
    cluster_assignments must be provided; it is keyword-only to keep the API
    consistent with loglik_fresh_start / loglik_persistent.
    """
    if cluster_assignments is None:
        raise ValueError("cluster_assignments must be provided for CRP model")
    random.seed(0)
    n_clusters = max(cluster_assignments) + 1
    first_env = envs[env_sequence[0]]
    n_states = first_env.observation_space.n
    n_actions = first_env.action_space.n

    cluster_q = [np.zeros((n_states, n_actions)) for _ in range(n_clusters)]
    cluster_models: List[dict] = [{} for _ in range(n_clusters)]

    total_loglik = 0.0
    for env_name, traj, k in zip(env_sequence, trajectories, cluster_assignments):
        env = envs[env_name]
        ep_ll = _loglik_episode_softmax(
            env, cluster_q[k], cluster_models[k], traj, eta, gamma, tau, planning_steps
        )
        total_loglik += ep_ll

    return total_loglik


def loglik_fresh_start(
    env_sequence: List[str],
    envs: Dict[str, GridWorldEnv],
    trajectories: List[List[Tuple[int, int]]],
    eta: float,
    gamma: float,
    tau: float,
    planning_steps: int = 5,
    cluster_assignments: Optional[List[int]] = None,  # ignored, present for uniform API
) -> float:
    """Teacher-forced log-likelihood under fresh-start (new Q-table each episode)."""
    random.seed(0)
    total_loglik = 0.0
    for env_name, traj in zip(env_sequence, trajectories):
        env = envs[env_name]
        n_states = env.observation_space.n
        n_actions = env.action_space.n
        q_values = np.zeros((n_states, n_actions))
        model: dict = {}
        ep_ll = _loglik_episode_softmax(env, q_values, model, traj, eta, gamma, tau, planning_steps)
        total_loglik += ep_ll
    return total_loglik


def loglik_persistent(
    env_sequence: List[str],
    envs: Dict[str, GridWorldEnv],
    trajectories: List[List[Tuple[int, int]]],
    eta: float,
    gamma: float,
    tau: float,
    planning_steps: int = 5,
    cluster_assignments: Optional[List[int]] = None,  # ignored, present for uniform API
) -> float:
    """Teacher-forced log-likelihood under persistent (shared Q-table across all episodes)."""
    random.seed(0)
    first_env = envs[env_sequence[0]]
    n_states = first_env.observation_space.n
    n_actions = first_env.action_space.n
    q_values = np.zeros((n_states, n_actions))
    model: dict = {}

    total_loglik = 0.0
    for env_name, traj in zip(env_sequence, trajectories):
        env = envs[env_name]
        ep_ll = _loglik_episode_softmax(env, q_values, model, traj, eta, gamma, tau, planning_steps)
        total_loglik += ep_ll
    return total_loglik


# ---------------------------------------------------------------------------
# MLE fitting
# ---------------------------------------------------------------------------

def fit_model(
    loglik_fn,
    env_sequence: List[str],
    envs: Dict[str, GridWorldEnv],
    trajectories: List[List[Tuple[int, int]]],
    gamma: float = 0.95,
    planning_steps: int = 5,
    n_restarts: int = 5,
    seed: int = 0,
    free_gamma: bool = True,
    maxiter: int = 500,
    eta_range: Tuple[float, float] = (0.02, 0.60),
    tau_range: Tuple[float, float] = (0.10, 2.00),
    gamma_range: Tuple[float, float] = (0.70, 0.99),
    **loglik_kwargs,
) -> dict:
    """MLE for any model with softmax policy.

    Parameters
    ----------
    loglik_fn   : one of loglik_diner, loglik_fresh_start, loglik_persistent
    gamma       : discount factor. Used as fixed value when free_gamma=False;
                  ignored when free_gamma=True (gamma is optimised jointly).
    free_gamma  : if True (default), fit (eta, tau, gamma) jointly — k=3 free parameters.
                  if False, fix gamma at the provided value and fit (eta, tau) — k=2.
    maxiter     : maximum Nelder-Mead iterations per restart.
                  Default 500 compiles quickly; increase to 2000–5000 for production fits.
    eta_range   : (lo, hi) uniform sampling range for η restart initialisation.
    tau_range   : (lo, hi) uniform sampling range for τ restart initialisation.
    gamma_range : (lo, hi) uniform sampling range for γ restart initialisation
                  (only used when free_gamma=True).
    loglik_kwargs : extra keyword arguments forwarded to loglik_fn
                   (e.g. cluster_assignments=... for CRP model)

    Returns
    -------
    dict with keys: eta, tau, gamma, negloglik, n_params, n_steps
    """
    rng = np.random.RandomState(seed)
    n_steps = sum(len(t) for t in trajectories)

    if free_gamma:
        def neg_loglik(params):
            eta, tau, gam = params
            if not (0.0 < eta <= 1.0) or tau <= 0.0 or not (0.0 < gam < 1.0):
                return 1e9
            return -loglik_fn(
                env_sequence, envs, trajectories, eta, gam, tau, planning_steps,
                **loglik_kwargs,
            )

        best: Optional[dict] = None
        for _ in range(n_restarts):
            x0 = [rng.uniform(*eta_range), rng.uniform(*tau_range), rng.uniform(*gamma_range)]
            res = minimize(neg_loglik, x0, method='Nelder-Mead',
                           options={'xatol': 1e-4, 'fatol': 1e-4, 'maxiter': maxiter})
            if best is None or res.fun < best['negloglik']:
                best = {'eta': float(res.x[0]), 'tau': float(res.x[1]),
                        'gamma': float(res.x[2]),
                        'negloglik': float(res.fun), 'n_params': 3,
                        'n_steps': n_steps}
    else:
        def neg_loglik(params):
            eta, tau = params
            if not (0.0 < eta <= 1.0) or tau <= 0.0:
                return 1e9
            return -loglik_fn(
                env_sequence, envs, trajectories, eta, gamma, tau, planning_steps,
                **loglik_kwargs,
            )

        best: Optional[dict] = None
        for _ in range(n_restarts):
            x0 = [rng.uniform(*eta_range), rng.uniform(*tau_range)]
            res = minimize(neg_loglik, x0, method='Nelder-Mead',
                           options={'xatol': 1e-4, 'fatol': 1e-4, 'maxiter': maxiter})
            if best is None or res.fun < best['negloglik']:
                best = {'eta': float(res.x[0]), 'tau': float(res.x[1]),
                        'gamma': gamma,
                        'negloglik': float(res.fun), 'n_params': 2,
                        'n_steps': n_steps}

    return best  # type: ignore[return-value]


def fit_diner_full(
    env_sequence: List[str],
    envs: Dict[str, GridWorldEnv],
    trajectories: List[List[Tuple[int, int]]],
    planning_steps: int = 5,
    n_probe: int = 20,
    n_restarts: int = 5,
    seed: int = 0,
    maxiter: int = 1000,
    eta_range: Tuple[float, float] = (0.02, 0.60),
    tau_range: Tuple[float, float] = (0.10, 2.00),
    gamma_range: Tuple[float, float] = (0.70, 0.99),
    alpha_range: Tuple[float, float] = (0.50, 3.00),
) -> dict:
    """MLE for DINER with (eta, tau, gamma, crp_alpha) all free.

    Cluster assignments are recomputed from probing for each candidate crp_alpha,
    so this is slower than fit_model but correctly accounts for the CRP prior.

    Parameters
    ----------
    maxiter     : maximum Nelder-Mead iterations per restart.
                  Default 1000 compiles quickly; increase to 3000–5000 for production fits.
    eta_range   : (lo, hi) uniform sampling range for η restart initialisation.
    tau_range   : (lo, hi) uniform sampling range for τ restart initialisation.
    gamma_range : (lo, hi) uniform sampling range for γ restart initialisation.
    alpha_range : (lo, hi) uniform sampling range for α (CRP concentration) initialisation.

    Returns
    -------
    dict with keys: eta, tau, gamma, crp_alpha, negloglik, n_params, n_steps
    """
    rng = np.random.RandomState(seed)
    n_steps = sum(len(t) for t in trajectories)

    def neg_loglik(params):
        eta, tau, gamma, crp_alpha = params
        if not (0.0 < eta <= 1.0) or tau <= 0.0 or not (0.0 < gamma < 1.0) or crp_alpha <= 0.0:
            return 1e9
        asgn = compute_cluster_assignments(
            env_sequence, envs, crp_alpha=crp_alpha, n_probe=n_probe, seed=0
        )
        ll = loglik_diner(
            env_sequence, envs, trajectories, eta, gamma, tau, planning_steps,
            cluster_assignments=asgn,
        )
        return -ll

    best: Optional[dict] = None
    for _ in range(n_restarts):
        x0 = [
            rng.uniform(*eta_range),    # eta
            rng.uniform(*tau_range),    # tau
            rng.uniform(*gamma_range),  # gamma
            rng.uniform(*alpha_range),  # crp_alpha
        ]
        res = minimize(neg_loglik, x0, method='Nelder-Mead',
                       options={'xatol': 1e-4, 'fatol': 1e-4, 'maxiter': maxiter})
        if best is None or res.fun < best['negloglik']:
            best = {
                'eta':       float(res.x[0]),
                'tau':       float(res.x[1]),
                'gamma':     float(res.x[2]),
                'crp_alpha': float(res.x[3]),
                'negloglik': float(res.fun),
                'n_params':  4,
                'n_steps':   n_steps,
            }

    return best  # type: ignore[return-value]


def compute_bic(negloglik: float, n_params: int, n_steps: int) -> float:
    """BIC = k·ln(n) − 2·ln(L̂). Lower BIC = better model."""
    return n_params * np.log(n_steps) + 2.0 * negloglik


def compute_cluster_assignments(
    env_sequence: List[str],
    envs: Dict[str, GridWorldEnv],
    crp_alpha: float = 1.0,
    n_probe: int = 20,
    seed: int = 0,
) -> List[int]:
    """Compute DINER cluster assignments from probing (no learning required).

    Useful for obtaining the assignment vector before fitting lr/tau.
    """
    rng = np.random.RandomState(seed)
    np.random.seed(seed)
    random.seed(seed)

    clusters: List[dict] = []
    assignments: List[int] = []

    for env_name in env_sequence:
        env = envs[env_name]
        n_states = env.observation_space.n
        n_actions = env.action_space.n
        probe = collect_probe_transitions(env, n_probe)
        k = _assign_cluster(clusters, probe, crp_alpha, n_states, n_actions, rng)
        clusters[k]['count'] += 1
        assignments.append(k)

    return assignments
