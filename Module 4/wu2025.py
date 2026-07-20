"""
wu2025.py: data loading and preprocessing for the Wu et al. (NatComms 2025) foraging dataset.
code + data: https://github.com/charleywu/minecraftforaging
paper: https://charleywu.github.io/downloads/wu2025adaptive.pdf

Prepared by Charley M. Wu (TU Darmstadt; hmc-lab.com)
for the 2026 Barcelona Summer School for Advanced Modeling of Behavior (BAMB; https://www.bambschool.org/)


The experiment: participants explore a 20×20 grid of melons/pumpkins in a Minecraft world.
We will focus on the solo + smooth environment condition. However, participants also completed group rounds (4 participants)
and searched in random environments.
Each block has coordinates in {2, 5, 8, …, 59} for both x and z axes (step=3, 20 values).
All model code works directly in these raw coordinates. The paper's GP kernel lengthscale
λ = √48 ≈ 6.93 is expressed in this same coordinate system.
The raw data is in a feather file, which is loaded into a pandas DataFrame.
The data is then filtered to the solo/smooth condition with block indices and integer rewards added.
Finally, there are iterators that yield the sequence of decisions for each participant, where at each step the model sees all previous observations and must predict the next block index.

Column notes
------------
time       : elapsed game time in seconds since the start of the round; used for ARS (Area-Restricted Search) analyses where Δt is the elapsed time in seconds since the last reward.
x, z       : raw block coordinates in {2, 5, …, 59} (integer multiples of 3, 20 values each axis).
block_idx  : flat grid index in [0, 399] = x_index * 20 + z_index (added by tidy()).
reward_int : reward as 0/1 integer; raw column 'reward' is bool (added by tidy()).
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Grid geometry constants
# ---------------------------------------------------------------------------

_BLOCK_RANGE = np.arange(start=2, stop=60, step=3)   # [2, 5, 8, ..., 59], 20 values
N_GRID = len(_BLOCK_RANGE)                            # 20
N_BLOCKS = N_GRID * N_GRID                            # 400

def grid_coords() -> np.ndarray:
    """Return all 400 block locations as a (400, 2) array in raw coordinates {2,5,...,59}.

    Ordering matches the Cartesian product of x × z (x varies slowest),
    so block index = x_index * 20 + z_index.
    """
    coords = np.array([[x, z] for x in _BLOCK_RANGE for z in _BLOCK_RANGE], dtype=float)
    return coords                                      # (400, 2)


def coord_to_idx(x_raw: float, z_raw: float) -> int:
    """Convert raw (x, z) block coordinates to a flat block index in [0, 399]."""
    xi = int(round((x_raw - 2.0) / 3.0))
    zi = int(round((z_raw - 2.0) / 3.0))
    return xi * N_GRID + zi


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

DATA_PATH = 'data/wu2025minecraft.feather' #this is a duplicate of data from https://github.com/charleywu/minecraftforaging


def load_blocks(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the raw feather file. Returns the full DataFrame (all conditions)."""
    return pd.read_feather(path)


def tidy(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to solo, smooth-environment trials and add derived columns.

    Adds columns
    ----------
    block_idx  : flat grid index 0–399
    reward_int : reward as 0/1 integer (raw column is bool)

    Raw x, z coordinates are kept in their original {2,5,...,59} form.
    """
    mask = (df['type'] == 'solo') & (df['env'] == 'smooth')
    out = df[mask].copy()
    out['block_idx'] = [coord_to_idx(x, z) for x, z in zip(out['x'].values, out['z'].values)]
    out['reward_int'] = out['reward'].astype(int)
    out = out.sort_values(['name', 'session', 'round', 'time']).reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Decision iterators
# ---------------------------------------------------------------------------

def iter_decisions(round_df: pd.DataFrame):
    """Yield (X_obs, y_obs, next_idx) tuples for a single (name, session, round) chunk.

    At each step t the model sees all observations from steps 0..t-1 and must
    predict which block is visited at step t.

    The first yielded tuple has X_obs of shape (0, 2) and y_obs of shape (0,),
    since the model has no history before the first block is destroyed.
    Model fitting functions (iter_participant_decisions) skip this first step
    because the GP has nothing to condition on — predictions start from step 2,
    using step 1 as the first observation.

    Yields
    ------
    X_obs    : (t, 2) array of previously visited locations in raw coords {2,5,...,59}  (empty on first step)
    y_obs    : (t,) array of {0,1} rewards at those locations
    next_idx : integer block index of the decision made at step t
    """
    rows = round_df.sort_values('time').reset_index(drop=True)
    X_so_far: list = []
    y_so_far: list = []

    for _, row in rows.iterrows():
        X_obs = np.array(X_so_far, dtype=float).reshape(-1, 2)
        y_obs = np.array(y_so_far, dtype=float)
        next_idx = int(row['block_idx'])
        yield X_obs, y_obs, next_idx

        X_so_far.append([row['x'], row['z']])
        y_so_far.append(row['reward_int'])


def iter_decisions_with_time(round_df: pd.DataFrame):
    """Yield (X_obs, y_obs, t_obs, next_idx, t_next) tuples including real timestamps.

    Same as iter_decisions but also returns the elapsed game time (in seconds)
    for each observation and for the upcoming decision.  Used for ARS analyses
    where Δt is measured in real seconds rather than decision steps.

    Yields
    ------
    X_obs    : (t, 2) array of previously visited locations in raw coords {2,5,...,59}
    y_obs    : (t,) array of {0,1} rewards at those locations
    t_obs    : (t,) array of game-time timestamps (seconds) for past observations
    next_idx : integer block index of the decision made at step t
    t_next   : game-time timestamp (seconds) of the upcoming decision
    """
    rows = round_df.sort_values('time').reset_index(drop=True)
    X_so_far: list = []
    y_so_far: list = []
    t_so_far: list = []

    for _, row in rows.iterrows():
        X_obs = np.array(X_so_far, dtype=float).reshape(-1, 2)
        y_obs = np.array(y_so_far, dtype=float)
        t_obs = np.array(t_so_far, dtype=float)
        next_idx = int(row['block_idx'])
        t_next = float(row['time'])
        yield X_obs, y_obs, t_obs, next_idx, t_next

        X_so_far.append([row['x'], row['z']])
        y_so_far.append(row['reward_int'])
        t_so_far.append(float(row['time']))


def iter_participant_decisions(df: pd.DataFrame, name: str):
    """Yield (session, round, decisions) for one participant.

    decisions is a list of (X_obs, y_obs, next_idx) tuples for that round,
    starting from the second observation (first has no GP history).
    """
    p_df = df[df['name'] == name]
    for (session, rnd), grp in p_df.groupby(['session', 'round']):
        grp_sorted = grp.sort_values('time').reset_index(drop=True)
        if len(grp_sorted) < 2:
            continue
        decisions = list(iter_decisions(grp_sorted))
        # Skip the very first step (X_obs is empty, so the GP has nothing to condition on).
        # Predictions start at step 2: the model sees step 1 as the first observation
        # and predicts which block is destroyed at step 2.
        decisions_with_data = [(X, y, idx) for X, y, idx in decisions if len(X) > 0]
        if decisions_with_data:
            yield session, rnd, decisions_with_data
