"""Task constants and trial/fixation extraction for the Perkins–Rich NHP task.

Paradigm: **Variable, 2 Attributes, 3 Options** (Perkins, Gillis & Rich 2024,
*Multiattribute Decision-making in Macaques Relies on Direct Attribute
Comparisons*, JoCN 36(9):1879–1897). Monkeys **Chip** and **Dale**.

Source specs:
  - ``Info/Variable_2_Attributes_3_Options_EventMarkers.txt`` — behavioral codes
  - Empirical data: ``Data/*_Variable_2_Attributes_3_Options.bhv2`` (MonkeyLogic)
  - Meta-MDP target framework: ``Code/metamdp-lean/metamdp.py`` (Radulescu et al. 2026)

Each trial shows **2 or 3 option stimuli** (bars) at fixed screen positions.
Every option carries **two attributes on an ordinal 1–5 scale**: reward
magnitude and reward probability. The monkey freely fixates the options (gaze
coded per fixation) and then chooses one; reward is delivered probabilistically.

This module extracts two structurally-parallel tables from a raw session
(see ``build_csv.py``), matching the meta-MDP's expected inputs:
  - **option-level** rows (one per option per trial) — the meta-MDP feature space
    (``Ftrue``/``object_locs``); parallels ``metamdp_feature_space.csv``.
  - **fixation-level** rows (one per fixation) — the empirical scanpath the
    policy is compared against.

Mapping to the meta-MDP (see plan §3; deviations flagged):
  - ``No``  = options per trial (2 or 3, variable)
  - ``Nf``  = 2 (reward level, probability level)
  - ``Ftrue[o]``      = [reward_level[o], prob_level[o]]
  - ``object_locs[o]``= ObjPos[o] in degrees, embedded as an (x, y, 0) ray
  - ``ftarget``/``itrue`` = DERIVED (no direct counterpart): the "ideal option"
    reinterpretation — target = the highest-value option, ideal profile [5, 5].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

TASK_ID = "perkins_variable_2attr_3opt"

# --- Behavioral event codes (Info/…_EventMarkers.txt) -----------------------
CODE_START_TRIAL = 1
CODE_BARS_ON = 6
CODE_ACQUIRED_FIXATION = 24
CODE_REWARD_DELIVERED = 8

# Fixate-on-stimulus onset codes, one per option slot -> option index (0-based).
FIXATE_CODE_TO_OPTION = {27: 0, 28: 1, 32: 2}
# Choose-stimulus codes -> option index (0-based).
CHOOSE_CODE_TO_OPTION = {29: 0, 30: 1, 33: 2}
CODE_CHOOSE_NOTHING = 34

# --- Attribute structure ----------------------------------------------------
N_FEATURES = 2  # (reward level, probability level) == meta-MDP Nf
FEATURE_NAMES = ("reward_level", "prob_level")
ATTRIBUTE_LEVELS = (1, 2, 3, 4, 5)  # ordinal 1–5, both attributes
MAX_LEVEL = 5

# Value / "best option" rule, from the task script
# (Info/tasks/Variable_2_Attributes_3_Options_*.m, lines ~319–322):
#   selected = reward_level + 1.78 * prob_level ; correct = argmax over options.
# This ADDITIVE, probability-weighted rule on the ordinal levels is the task's
# own definition of the best option — it supersedes the earlier provisional
# multiplicative proxy (reward_level * prob_level), which disagreed on ~8% of
# trials and fit the monkey's choices worse. ev_ordinal is retained only for
# comparison; is_best / the meta-MDP itrue use the additive rule.
PROB_WEIGHT = 1.78
# Physical probability of reward per level (task script `PROBS`, line 43).
PROB_VALUES = (0.1, 0.3, 0.5, 0.7, 0.9)
# Per-attribute fill DIRECTION (task script): 1 = ascending (bar height grows
# with level), 2 = descending (height *inverts* — a tall bar is a LOW level).
# Stored per option in UserVars.RewardAttributes / ProbabilityAttributes and
# signaled to the monkey by hue. Needed to read value from bar height, and to
# separate visual salience from value. 0 marks "unavailable" for that trial.
DIR_ASCENDING, DIR_DESCENDING = 1, 2

# The "ideal" attribute profile used for the ideal-option-identification mapping
# (plan §5.1-A): the target feature vector ftarget the agent searches for.
IDEAL_PROFILE = np.array([MAX_LEVEL, MAX_LEVEL], dtype=float)

# TrialError == 0 marks a completed trial with a valid choice; others are aborts.
TRIALERROR_COMPLETED = 0


@dataclass(frozen=True)
class OptionRow:
    """One option within one trial (option-level / feature-space row)."""

    option_idx: int
    reward_level: int
    prob_level: int
    prob_value: float  # physical reward probability for prob_level (0.1–0.9)
    reward_dir: int  # fill direction: 1=ascending, 2=descending, 0=unavailable
    prob_dir: int
    ev_ordinal: int  # reward_level * prob_level (retained for comparison only)
    ev_additive: float  # reward_level + 1.78*prob_level (task's value rule)
    x_dva: float
    y_dva: float
    is_best: bool  # argmax ev_additive within trial -> meta-MDP itrue
    chosen: bool  # monkey's choice
    rewarded: bool  # outcome; only True on the chosen row


@dataclass(frozen=True)
class FixationRow:
    """One fixation onto an option (fixation-level / scanpath row)."""

    fixation_idx: int
    option_idx: int  # 0..No-1 (fixate codes 27/28/32)
    onset_ms: float
    dwell_ms: float
    is_choice: bool  # coincides with a choose code (29/30/33)


def _as_1d(x: Any) -> np.ndarray:
    return np.atleast_1d(np.asarray(x)).ravel()


def extract_options(trial: dict[str, Any]) -> list[OptionRow]:
    """Option-level rows for one trial, or ``[]`` if not a completed choice trial.

    Reads ``UserVars`` (``Rewards``/``Probabilities``/``ObjPos``/``ChosenPic``/
    ``RewardAttributes``/``ProbabilityAttributes``) and ``BehavioralCodes``
    (reward delivery). ``is_best`` uses the task's own value rule
    ``ev_additive = reward_level + 1.78*prob_level`` (see PROB_WEIGHT).
    """
    if int(_as_1d(trial["TrialError"])[0]) != TRIALERROR_COMPLETED:
        return []
    uv = trial.get("UserVars")
    if not isinstance(uv, dict) or "Rewards" not in uv:
        return []

    reward = _as_1d(uv["Rewards"]).astype(int)
    prob = _as_1d(uv["Probabilities"]).astype(int)
    pos_flat = np.asarray(uv["ObjPos"]).ravel()
    # Displayed options == rows of ObjPos. In a few EARLY sessions (Sept 2018),
    # Rewards/Probabilities carry the full 3-option *pool* while only 2 options
    # are shown (ObjPos has 2 rows). We cannot know which pool entries were the
    # displayed pair without the task config, so such trials are skipped rather
    # than guessed. See Data/Intermediary/README.md (schema caveat). Candidate
    # disambiguators for a later pass: UserVars.ObjPosIndex / PossibleRewards.
    if pos_flat.size % 2 != 0:
        return []
    n_shown = pos_flat.size // 2
    if len(reward) != n_shown or len(prob) != n_shown:
        return []
    n_opt = n_shown
    pos = pos_flat.reshape(n_opt, 2)
    ev_ordinal = reward * prob                       # retained for comparison only
    ev_additive = reward + PROB_WEIGHT * prob        # task's value rule
    best = int(np.argmax(ev_additive))               # meta-MDP itrue
    chosen = int(_as_1d(uv["ChosenPic"])[0]) - 1 if "ChosenPic" in uv else -1

    # Per-option fill directions (reordered to match displayed options in the
    # task script). Absent or length-mismatched -> 0 ("unavailable"), never guessed.
    def _dirs(key: str) -> np.ndarray:
        vals = _as_1d(uv[key]).astype(int) if key in uv else np.array([], dtype=int)
        return vals if vals.size == n_opt else np.zeros(n_opt, dtype=int)
    reward_dir = _dirs("RewardAttributes")
    prob_dir = _dirs("ProbabilityAttributes")

    codes = _as_1d(trial["BehavioralCodes"]["CodeNumbers"]).astype(int)
    rewarded = CODE_REWARD_DELIVERED in codes.tolist()

    rows = []
    for o in range(n_opt):
        rows.append(
            OptionRow(
                option_idx=o,
                reward_level=int(reward[o]),
                prob_level=int(prob[o]),
                prob_value=PROB_VALUES[int(prob[o]) - 1],
                reward_dir=int(reward_dir[o]),
                prob_dir=int(prob_dir[o]),
                ev_ordinal=int(ev_ordinal[o]),
                ev_additive=float(ev_additive[o]),
                x_dva=float(pos[o, 0]),
                y_dva=float(pos[o, 1]),
                is_best=(o == best),
                chosen=(o == chosen),
                rewarded=(o == chosen and rewarded),
            )
        )
    return rows


def extract_fixations(trial: dict[str, Any]) -> list[FixationRow]:
    """Fixation-level rows for one trial from behavioral codes 27/28/32.

    Each fixate code marks the *onset* of a gaze transition onto an option;
    ``dwell_ms`` is approximated as the interval to the next behavioral code
    (refined from the 500 Hz eye trace in a later pass — see plan §2.2). A
    fixation is flagged ``is_choice`` when the immediately following code is a
    choose code (29/30/33) for that same option.
    """
    if int(_as_1d(trial["TrialError"])[0]) != TRIALERROR_COMPLETED:
        return []
    bc = trial.get("BehavioralCodes")
    if not isinstance(bc, dict):
        return []
    codes = _as_1d(bc["CodeNumbers"]).astype(int).tolist()
    times = _as_1d(bc["CodeTimes"]).astype(float)

    rows = []
    fix_idx = 0
    for i, code in enumerate(codes):
        if code not in FIXATE_CODE_TO_OPTION:
            continue
        option = FIXATE_CODE_TO_OPTION[code]
        onset = float(times[i])
        nxt = float(times[i + 1]) if i + 1 < len(times) else onset
        next_code = codes[i + 1] if i + 1 < len(codes) else None
        is_choice = (
            next_code in CHOOSE_CODE_TO_OPTION
            and CHOOSE_CODE_TO_OPTION[next_code] == option
        )
        rows.append(
            FixationRow(
                fixation_idx=fix_idx,
                option_idx=option,
                onset_ms=onset,
                dwell_ms=nxt - onset,
                is_choice=is_choice,
            )
        )
        fix_idx += 1
    return rows


def parse_session_filename(name: str) -> tuple[str, str]:
    """``'190430_Chip_Variable_..bhv2'`` -> ``(session='190430_Chip', monkey='Chip')``."""
    stem = name.split("_Variable")[0]  # e.g. '190430_Chip'
    parts = stem.split("_")
    monkey = parts[1] if len(parts) > 1 else ""
    return stem, monkey
