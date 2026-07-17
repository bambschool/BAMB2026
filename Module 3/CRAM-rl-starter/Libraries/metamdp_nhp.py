"""Meta-MDP environment for the Perkins–Rich NHP task.

A numpy adaptation of the lean visual-search meta-MDP
(``Code/metamdp-lean/metamdp.py`` and ``metamdp_functions.py``, Radulescu et al.
2026) for the macaque multiattribute-choice data. The belief-update and reward
math are carried over verbatim; only the inputs change (2 attributes instead of
VGG features, 2–3 options instead of ≤113 scene objects, screen positions
instead of 3-D scene rays).

Deviations from the lean spec (flagged, not silently reinterpreted):
  1. **Reward semantics (plan §5.1-A).** The task is value-based choice, not
     target search. We keep the lean *identification* reward unchanged and set
     the target to the *ideal option*: ``ftarget`` = ideal attribute profile
     ``[5, 5]`` and ``itrue`` = the highest-value option. Terminal reward is 1
     iff the argmax-posterior option is the truly-best option.
  2. **Features.** ``Nf = 2`` (reward level, probability level); no VGG /
     shape / color embeddings, no occlusion (all options are valid targets).
  3. **Locations.** 2-D screen positions in degrees embedded as ``(x, y, 0)``
     unit rays (see ``data._locs_to_rays``); the fovea filter is otherwise the
     lean ``go = scale_go·exp((⟨loc, loc_fix⟩ − 1)·200)``.

Pure numpy: runs without TensorFlow/tf-agents. A thin
``tf_agents.py_environment.PyEnvironment`` wrapper (parallel to the lean
``MetaMDPEnv``) can sit on top of :class:`NHPMetaMDPEnv` once TF is installed.
"""

from __future__ import annotations

from collections import namedtuple
from dataclasses import dataclass

import numpy as np

from data import TrialScene
from task_perkins import N_FEATURES

# Fovea-filter constants, matching the lean env's _decode_action_from_tensor.
SCALE_GO = 10.0
FOVEA_SHARPNESS = 200.0

# Lightweight TimeStep so a rollout mirrors the tf-agents loop without importing tf.
TimeStep = namedtuple("TimeStep", ["observation", "reward", "is_last"])


@dataclass
class State:
    """Belief state: per-option feature means ``F`` and precisions ``J``."""

    F: np.ndarray  # (No, Nf)
    J: np.ndarray  # (No, Nf)
    is_terminal: bool = False


def get_posterior_itrue(state: State, scene: TrialScene, Jtrue: np.ndarray) -> np.ndarray:
    """Posterior over which option is the target, given the belief state.

    Identical to ``metamdp.get_posterior_itrue`` (all options are valid, so the
    occlusion / Nvalid masking is a no-op here).
    """
    F, J = state.F, state.J
    ftarget = scene.ftarget[None, :]
    L = np.sum(
        np.log(1 + J / Jtrue) / 2
        + 1 / (1 / J + 1 / Jtrue) / 2 * F ** 2
        - J / 2 * (F - ftarget) ** 2,
        axis=1,
    )
    p = np.exp(L - np.max(L))
    return p / np.sum(p)


class NHPMetaMDP:
    """Core meta-MDP dynamics for a single scene (one trial).

    Parameters
    ----------
    jtrue : float
        Precision of the true feature distribution (lean default 1.0).
    cost_params : (float, float)
        ``(c0, c1)`` fixed + precision-scaled attention cost per fixation.
    initial_precision : float
        Prior precision on every belief entry (lean uses 0.01).
    """

    def __init__(self, jtrue: float = 1.0, cost_params=(0.01, 0.05),
                 initial_precision: float = 0.01) -> None:
        self.Nf = N_FEATURES
        self.jtrue = float(jtrue)
        self.cost_params = np.asarray(cost_params, dtype=float)
        self.initial_precision = float(initial_precision)

    def Jtrue(self, scene: TrialScene) -> np.ndarray:
        return np.full((scene.n_options, self.Nf), self.jtrue)

    def get_initial_state(self, scene: TrialScene) -> State:
        """Diffuse prior belief (lean ``get_initial_state``)."""
        no = scene.n_options
        return State(
            F=np.random.normal(0, 0.01, size=(no, self.Nf)),
            J=np.full((no, self.Nf), self.initial_precision),
        )

    def fovea_go(self, scene: TrialScene, fixated_object: int) -> np.ndarray:
        """Object-attention filter ``go`` for fixating one option (lean fovea)."""
        loc_fix = scene.object_locs[fixated_object]
        cos_sim = scene.object_locs @ loc_fix
        return SCALE_GO * np.exp((cos_sim - 1) * FOVEA_SHARPNESS)

    def transition(self, state: State, scene: TrialScene, fixated_object: int | None,
                   terminate: bool, attribute: int | None = None):
        """Advance one step; return ``(next_state, reward)``.

        Mirrors ``metamdp.MetaMDP.transition``: on terminate, reward = 1 iff the
        argmax-posterior option is the target (``itrue``); otherwise sample the
        fixated option's features, Bayesian-update ``(F, J)``, and pay the
        attention cost.

        Feature filter ``gf`` (the meta-MDP's second action component):
          - ``attribute is None`` → **object-level** model, ``gf = 1`` — one glance
            measures BOTH attributes (the lean default).
          - ``attribute in {0, 1}`` → **attribute-level** model, ``gf`` is a
            one-hot so a glance measures reward XOR probability. This is the
            data-motivated change (Part 1: the monkey looks one attribute at a
            time). ``Jmeas = go ⊗ gf`` either way — the paper's outer product.
        """
        Jtrue = self.Jtrue(scene)
        if terminate:
            posterior = get_posterior_itrue(state, scene, Jtrue)
            reward = 1.0 if int(np.argmax(posterior)) == scene.itrue else 0.0
            return State(None, None, is_terminal=True), reward

        go = self.fovea_go(scene, fixated_object)
        if attribute is None:
            gf = np.ones(self.Nf)
        else:
            gf = np.zeros(self.Nf); gf[attribute] = 1.0
        Jmeas = go[:, None] * gf[None, :]  # precision allocated this fixation
        # get_observation returns a sample of Ftrue*Jmeas with precision 1/Jmeas.
        x_times_j = np.random.normal(scene.features * Jmeas, scale=np.sqrt(Jmeas))
        penalty = self.cost_params[0] + self.cost_params[1] * np.mean(Jmeas)
        next_state = State(
            F=(state.F * state.J + x_times_j) / (state.J + Jmeas),
            J=state.J + Jmeas,
        )
        return next_state, -penalty


class NHPMetaMDPEnv:
    """Episode wrapper over one scene — parallels the lean ``MetaMDPEnv``.

    One episode = one trial. Actions are ``(fixated_object, terminate)`` where
    ``fixated_object`` indexes options **sorted by descending posterior** (as in
    the lean env, so action 0 is always "fixate the current best guess"). The
    observation dict matches the lean keys, flattened.
    """

    def __init__(self, mdp: NHPMetaMDP, scene: TrialScene) -> None:
        self.mdp = mdp
        self.scene = scene
        self.reset()

    def observation_spec(self) -> dict:
        no, nf = self.scene.n_options, self.mdp.Nf
        return {
            "F": (no * nf,), "J": (no * nf,), "ftarget": (nf,),
            "object_locs": (3 * no,), "posterior": (no,),
        }

    def action_spec(self) -> dict:
        return {"fixated_object": (0, self.scene.n_options - 1), "terminate": (0, 1)}

    @property
    def n_targets(self) -> int:
        """Number of fixate targets: one per option (object-level)."""
        return self.scene.n_options

    def decode(self, target: int) -> tuple[int, int | None]:
        """Map a fixate target to ``(fixated_object, attribute)``; attribute None."""
        return int(self._order[target]), None

    def reset(self) -> TimeStep:
        self._state = self.mdp.get_initial_state(self.scene)
        self._ended = False
        return TimeStep(self._encode(), 0.0, False)

    def _encode(self) -> dict:
        if self._state.is_terminal:
            no, nf = self.scene.n_options, self.mdp.Nf
            return {k: np.zeros(v, np.float32) for k, v in self.observation_spec().items()}
        posterior = get_posterior_itrue(self._state, self.scene, self.mdp.Jtrue(self.scene))
        self._order = np.argsort(posterior)[::-1]  # options sorted by posterior desc
        return {
            "F": self._state.F[self._order].flatten().astype(np.float32),
            "J": self._state.J[self._order].flatten().astype(np.float32),
            "ftarget": self.scene.ftarget.astype(np.float32),
            "object_locs": self.scene.object_locs[self._order].flatten().astype(np.float32),
            "posterior": posterior[self._order].astype(np.float32),
        }

    def step(self, target: int, terminate: bool) -> TimeStep:
        """Take one action. ``target`` indexes the posterior-sorted options."""
        if self._ended:
            return self.reset()
        fixated_object = None if terminate else int(self._order[target])
        self._state, reward = self.mdp.transition(
            self._state, self.scene, fixated_object, terminate)
        self._ended = self._state.is_terminal
        return TimeStep(self._encode(), float(reward), self._ended)


class NHPMetaMDPEnvAttr(NHPMetaMDPEnv):
    """Attribute-level env: a fixation targets an ``(option, attribute)`` bar.

    The **attribute-level** model motivated by Part 1: the monkey looks at one
    attribute at a time, so a fixation measures reward XOR probability. Identical
    to :class:`NHPMetaMDPEnv` except the action space is the ``2·No`` bar targets
    (option × attribute) instead of ``No`` whole-option targets. A target encodes
    ``option_rank = target // 2`` (posterior-sorted) and ``attribute = target % 2``
    (0 = reward, 1 = probability); the env passes that ``attribute`` to
    ``transition`` so ``gf`` becomes a one-hot. The observation is unchanged
    (still per-option), so the same policy network features feed both models.
    """

    @property
    def n_targets(self) -> int:
        """Two fixate targets per option: (reward bar, probability bar)."""
        return 2 * self.scene.n_options

    def decode(self, target: int) -> tuple[int, int | None]:
        """Map a bar target to ``(fixated_object, attribute)``."""
        option_rank, attribute = divmod(int(target), 2)
        return int(self._order[option_rank]), attribute

    def step(self, target: int, terminate: bool) -> TimeStep:
        """Take one action. ``target`` indexes posterior-sorted ``(option, attr)`` bars."""
        if self._ended:
            return self.reset()
        if terminate:
            fixated_object, attribute = None, None
        else:
            fixated_object, attribute = self.decode(target)
        self._state, reward = self.mdp.transition(
            self._state, self.scene, fixated_object, terminate, attribute=attribute)
        self._ended = self._state.is_terminal
        return TimeStep(self._encode(), float(reward), self._ended)
