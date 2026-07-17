"""Load and validate the extracted Perkins–Rich CSVs.

Consumes the two tables produced by ``build_csv.py`` and exposes them as
per-trial "scenes" for the meta-MDP environment. A *scene* is one trial: a small
set (2 or 3) of options, each with two attribute values and a screen location —
structurally parallel to a scene of objects in ``metamdp_feature_space.csv``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from task_perkins import IDEAL_PROFILE

ROOT = Path(__file__).resolve().parents[1]
INTERMEDIARY = ROOT / "Data" / "Intermediary"


@dataclass(frozen=True)
class TrialScene:
    """One trial as a meta-MDP scene (the analogue of a lean-repo scene).

    Attributes
    ----------
    scene_id : str
        ``{session}_t{trial}`` identifier.
    features : np.ndarray, shape (No, 2)
        Per-option attribute values ``[reward_level, prob_level]`` == ``Ftrue``.
        Uses z-scored columns when ``standardized=True`` was requested.
    object_locs : np.ndarray, shape (No, 3)
        Per-option unit ray ``(x, y, 0)`` from the degree-space screen position;
        drives the meta-MDP fovea filter ``go``.
    itrue : int
        Index of the highest-value option (argmax ev_additive = reward_level +
        1.78*prob_level, the task's own value rule) — the derived "target" for
        the ideal-option-identification mapping (plan §5.1-A).
    chosen : int
        Option the monkey actually chose (0-based); ``-1`` if unavailable.
    rewarded : bool
        Whether the chosen option was rewarded on this trial.
    """

    scene_id: str
    features: np.ndarray
    object_locs: np.ndarray
    itrue: int
    chosen: int
    rewarded: bool

    @property
    def n_options(self) -> int:
        return self.features.shape[0]

    @property
    def ftarget(self) -> np.ndarray:
        """Target feature vector: the ideal attribute profile (plan §5.1-A)."""
        return IDEAL_PROFILE.copy()


def _locs_to_rays(xy_dva: np.ndarray) -> np.ndarray:
    """Embed 2-D degree positions as unit ``(x, y, 0)`` rays (plan §5.3).

    The meta-MDP fovea filter takes inner products of unit object-location rays,
    so each option's screen position is embedded on the z=0 plane and normalised.
    """
    rays = np.column_stack([xy_dva, np.zeros(len(xy_dva))]).astype(float)
    norms = np.linalg.norm(rays, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return rays / norms


class SessionData:
    """In-memory view of one session's option and fixation tables."""

    def __init__(self, options: pd.DataFrame, fixations: pd.DataFrame,
                 standardized: bool = False) -> None:
        self.options = options
        self.fixations = fixations
        self.standardized = standardized
        self._validate()

    @classmethod
    def load(cls, session: str, intermediary: Path | str = INTERMEDIARY,
             standardized: bool = False) -> "SessionData":
        intermediary = Path(intermediary)
        options = pd.read_csv(intermediary / f"{session}_trial_options.csv")
        fixations = pd.read_csv(intermediary / f"{session}_fixations.csv")
        return cls(options, fixations, standardized=standardized)

    @classmethod
    def from_combined(cls, session: str, data_dir: Path | str | None = None,
                      standardized: bool = False) -> "SessionData":
        """Load one session from the combined ``all_*.csv`` tables (Data/).

        The starter code ships only the two combined CSVs (not the per-session
        Intermediary files), so notebooks use this. Filters both tables to the
        given ``session``.
        """
        data_dir = Path(data_dir) if data_dir is not None else INTERMEDIARY.parent
        options = pd.read_csv(data_dir / "all_trial_options.csv")
        options = options[options["session"] == session].reset_index(drop=True)
        if options.empty:
            raise ValueError(f"session {session!r} not found in all_trial_options.csv")
        try:
            fixations = pd.read_csv(data_dir / "all_fixations.csv")
            fixations = fixations[fixations["session"] == session].reset_index(drop=True)
        except FileNotFoundError:
            fixations = options.iloc[:0]
        return cls(options, fixations, standardized=standardized)

    def _validate(self) -> None:
        """Assert the invariants the environment relies on."""
        opt = self.options
        assert not opt.empty, "empty option table"
        # Attribute levels in range.
        for col in ("reward_level", "prob_level"):
            assert opt[col].between(1, 5).all(), f"{col} outside 1–5"
        # Positions finite.
        assert np.isfinite(opt[["x_dva", "y_dva"]].to_numpy()).all(), "non-finite ObjPos"
        # One row per (trial, option) and n_options consistent with row count.
        counts = opt.groupby("trial")["option_idx"].agg(["count", "nunique", "max"])
        assert (counts["count"] == counts["nunique"]).all(), "duplicate option rows"
        assert (counts["count"] == opt.groupby("trial")["n_options"].first()).all(), \
            "n_options != number of option rows"
        # Exactly one chosen option per trial (when a choice was recorded).
        chosen_per = opt.groupby("trial")["chosen"].sum()
        assert chosen_per.isin((0, 1)).all(), "trial with >1 chosen option"
        # Fixations reference valid options.
        if not self.fixations.empty:
            assert self.fixations["option_idx"].between(0, 2).all(), "fixation option out of range"

    def trials(self) -> list[int]:
        return sorted(self.options["trial"].unique().tolist())

    def scene(self, trial: int) -> TrialScene:
        """Build the :class:`TrialScene` for one trial."""
        rows = self.options[self.options["trial"] == trial].sort_values("option_idx")
        if self.standardized:
            feats = rows[["reward_z", "prob_z"]].to_numpy(dtype=float)
        else:
            feats = rows[["reward_level", "prob_level"]].to_numpy(dtype=float)
        locs = _locs_to_rays(rows[["x_dva", "y_dva"]].to_numpy(dtype=float))
        # Target = the task's own best option (additive value rule); ev_additive
        # supersedes the multiplicative ev_ordinal proxy (see task_perkins).
        value = rows["ev_additive"] if "ev_additive" in rows else rows["ev_ordinal"]
        itrue = int(np.argmax(value.to_numpy()))
        chosen_mask = rows["chosen"].to_numpy(dtype=bool)
        chosen = int(np.argmax(chosen_mask)) if chosen_mask.any() else -1
        rewarded = bool(rows["rewarded"].any())
        return TrialScene(
            scene_id=str(rows["scene_id"].iloc[0]),
            features=feats, object_locs=locs, itrue=itrue,
            chosen=chosen, rewarded=rewarded,
        )

    def fixation_sequence(self, trial: int) -> np.ndarray:
        """Ordered array of fixated option indices for a trial (the scanpath)."""
        rows = self.fixations[self.fixations["trial"] == trial].sort_values("fixation_idx")
        return rows["option_idx"].to_numpy(dtype=int)
