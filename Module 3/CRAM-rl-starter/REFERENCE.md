# Reference — data structure & code
---

## Data

### `Data/all_trial_options.csv` 

One row per option shown on a trial: the meta-MDP "feature space" (each option's
attribute levels and screen location). Load one session with
`SessionData.from_combined("190430_Chip")` (see `Libraries/data.py`).

| Column | Type | Units / range | Notes |
|--------|------|---------------|-------|
| `source_file` | str | — | full raw file stem (provenance) |
| `session` | str | — | `{date}_{monkey}`, e.g. `190430_Chip` |
| `monkey` | str | Chip / Dale | |
| `scene_id` | str | — | `{session}_t{trial}` — one trial = one meta-MDP scene |
| `trial` | int | 1-based | non-contiguous (aborted trials dropped) |
| `block`, `trial_in_block`, `condition` | int | — | task structure (condition 1–20, balanced attribute combos) |
| `n_options` | int | 2 or 3 | = meta-MDP `No` (mostly 2, sometimes 3) |
| `option_idx` | int | 0…n_options−1 | option slot |
| `reward_level` | int | 1–5 | reward magnitude → `Ftrue[o,0]` |
| `prob_level` | int | 1–5 | reward probability → `Ftrue[o,1]` |
| `prob_value` | float | 0.1–0.9 | physical probability for `prob_level` (`PROBS = [.1 .3 .5 .7 .9]`) |
| `reward_dir`, `prob_dir` | int | 1/2/0 | bar fill direction: 1=ascending, 2=descending (**tall bar = low level**), 0=n/a |
| `reward_z`, `prob_z` | float | z-score | levels z-scored within the source file |
| `ev_ordinal` | int | 1–25 | multiplicative `reward_level × prob_level` — **comparison only** (superseded) |
| `ev_additive` | float | — | the task's value rule `reward_level + 1.78 × prob_level` |
| `x_dva`, `y_dva` | float | degrees | option center (Reward bar at x−1.5°, Probability bar at x+1.5°) |
| `is_best` | bool | — | option with max `ev_additive` on the trial → meta-MDP target `itrue` |
| `chosen` | bool | — | the option the monkey chose (exactly one per trial) |
| `rewarded` | bool | — | reward delivered (chosen option only) |
| `trial_error` | int | 0 | always 0 — only completed trials are kept |

**Value rule.** The task defines the best option as `argmax(reward_level + 1.78 ×
prob_level)` — additive and probability-weighted, verified against the task-generation
script. `is_best` / `itrue` use `ev_additive`; `ev_ordinal` is kept only for comparison.

**Bar geometry.** Each option is a **Reward bar (left, x−1.5°)** and a **Probability
bar (right, x+1.5°)**, 3° apart. Level is drawn as bar height, but on *descending*
options (`dir==2`) a tall bar means a **low** level — so read value from the level (or
`ev_additive`), not the rendered height.

### `Data/example_traces.npz` — raw 500 Hz eye traces (Part 1)

Coded fixations can't resolve *within-option, within-attribute* looking; that needs the
raw trace. This bundle carries just the fields Part 1 uses, for two example sessions
(`190430_Chip`, `181112_Dale`). Parallel arrays, one entry per completed trial:
`session`, `trial`, `eye` (Ni×2 float32, deg), `dt` (ms/sample), `objpos` (No×2 deg),
`reward`/`prob` (No, 1–5), `t_acquire`/`t_bars_on`/`t_choice` (ms; first two may be
NaN). Load with `np.load(..., allow_pickle=True)`.

---

## Code — `Libraries/`

The four library files the notebooks import. (`data` / `metamdp_nhp` / `train_reinforce`
are imported directly; they pull in `task_perkins`.)

| File | Role |
|--------|------|
| `data.py` | Loader + validator. `SessionData.from_combined(session)` reads `all_trial_options.csv`, filters to one session, and exposes each trial as a meta-MDP `TrialScene`. |
| `task_perkins.py` | Task constants (event codes, attribute layout, bar geometry) shared by the loader and the environment. |
| `metamdp_nhp.py` | The meta-MDP environment (numpy). `NHPMetaMDPEnv` is **object-level** (`gf=1`, fixate a whole option); `NHPMetaMDPEnvAttr` is **attribute-level** (`gf` one-hot, fixate one (option, attribute) bar). Only the action space differs. |
| `train_reinforce.py` | Pure-numpy REINFORCE trainer (policy gradient, gradient-checked). The same trainer serves both environments; the starter code uses it to train the policy and read out its looking behavior. |

### The mapping (meta-MDP ← monkey data)

`No` = options/trial (2 or 3) · `Nf` = 2 (reward level, probability level) ·
`Ftrue[o]` = `[reward_level, prob_level]` · `object_locs` = option positions (deg) ·
target `itrue` = highest-`ev_additive` option (the "identify the best option"
reinterpretation of the meta-MDP's target-search reward).
