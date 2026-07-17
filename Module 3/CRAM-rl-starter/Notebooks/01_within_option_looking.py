import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    # Enlarge markdown body text.
    mo.Html(
        """
        <style>
          .markdown .paragraph,
          .markdown li,
          .markdown blockquote { font-size: 1.15rem; line-height: 1.65; }
        </style>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # CRAM: Curriculum-based deep RL modeling of multi-Attribute decision-making in Macaques

    **Goal**: adapt a meta-MDP model of eye movements
    ([Radulescu, van Opheusden et al., 2026](https://direct.mit.edu/opmi/article/doi/10.1162/OPMI.a.322/135355/A-Resource-Rational-Account-of-Human-Eye-Movements), *Open Mind*) to a macaque multiattribute-choice task
    ([Perkins, Gillis & Rich, 2024](https://direct.mit.edu/jocn/article-lookup/doi/10.1162/jocn_a_02208), *Journal of Cognitive Neuroscience*). We want to model how the monkey learns this task using the same curriculum that the monkeys were given during training (*JoCN* shaping and paper data). Behavior during neural recordings and neural data are held out ([Perkins & Rich, 2025](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3003281), *PLOS Computational Biology*).

    In the task, two monkeys **Chip** and **Dale** see 2–3 options on every trial; each option has two attributes — **reward magnitude** and **reward probability** — drawn as two vertical bars (*JoCN* **Fig. 2** below):

    - **Reward** bar on the **left**
    - **Probability** bar on the **right**

    ## Part 1 — Does the monkey look at whole options, or one attribute at a time?

    The model from *OPMI*, as written, assumes that a single glance at an option measures
    **both** of its attributes at once. However, Perkins & Rich found that macaques rely on "direct attribute comparisons" to solve the task. That finding suggests a different model!

    Before we build anything, let's check whether we can reproduce the paper's finding.

    > **The question:** when the monkey looks at an option, does
    > it take in the whole option (both bars) — or does it sample **one attribute
    > at a time**? The answer decides how we structure the model.

    We have the **raw 500 Hz eye trace** for every trial, so we can look directly.
    """)
    return


@app.function
def asset_path(name: str):
    """Path to a file in Notebooks/assets/, searching upward from this notebook."""
    from pathlib import Path

    try:
        here = Path(__file__).resolve().parent
    except NameError:
        here = Path.cwd()
    for base in [here, *here.parents]:
        for cand in (base / "assets" / name, base / "Notebooks" / "assets" / name):
            if cand.exists():
                return cand
    return None


@app.function
def show_asset(mo, name, alt, caption, width=None, center=False):
    """Render an asset image with a caption, or a 'not found' note."""
    path = asset_path(name)
    if path is None:
        return mo.md(f"*`Notebooks/assets/{name}` not found.*")
    img = mo.image(src=path, alt=alt, width=width) if width else mo.image(src=path, alt=alt)
    cap = mo.md(
        f'<p style="font-size:14px; font-style:italic; line-height:1.5; '
        f'max-width:640px; margin-top:0.4rem;">{caption}</p>'
    )
    block = mo.vstack([img, cap], align="center" if center else "start")
    return mo.center(block) if center else block


@app.cell
def _(mo):
    # --- Trial structure (JoCN Fig. 2) ---------------------------
    show_asset(
        mo,
        "task.png",
        alt="task trial structure: fixation, options on, choice, reward",
        caption=(
            "The four-phase trial: hold central fixation, options "
            "appear, the monkey chooses one, then reward is delivered. Each option "
            "is a pair of colored bars."
        ),
        width=600,
        center=True,
    )
    return


@app.cell
def _(mo):
    # --- Given: how bar height maps to attribute value ----------------------
    show_asset(
        mo,
        "reward_mapping.png",
        alt="mapping from bar height to reward magnitude and probability",
        caption=(
            "Each bar's height encodes its attribute. Under the direct, "
            "mapping taller = more; under the indirect mapping, taller = less — "
            "for both reward magnitude (sweetness) and probability."
        ),
        width=500,
        center=True,
    )
    return


@app.cell
def _(mo):
    # --- Given: the raw-gaze demo gif for the motivating trial --------------
    show_asset(
        mo,
        "raw_gaze_demo_190430_Chip_t33.gif",
        alt="raw gaze demo, 190430_Chip trial 33",
        caption="Gaze data (500Hz) in an example trial.",
        width=600,
        center=True,
    )
    return


@app.cell
def _():
    # --- Setup: locate the example traces ---------------------------
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt


    def _find_traces(start: Path) -> Path:
        for p in [start, *start.parents]:
            cand = p / "Data" / "example_traces.npz"
            if cand.exists():
                return cand
        raise FileNotFoundError(
            "Data/example_traces.npz not found — run Libraries/make_example_traces.py."
        )

    try:
        _HERE = Path(__file__).resolve().parent
    except NameError:
        _HERE = Path.cwd()
    TRACES = _find_traces(_HERE)
    return TRACES, np, pd, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How the exercises work

    This is a reactive notebook: when you edit a cell, the notebook re-runs every cell that depends on it. Each `TODO` starts by raising
    `NotImplementedError`, so before you fill it in you'll see that message in the exercise's own cell and a grey **"not done yet"** note in its check. Fill the blank and the check turns **green** (or red, with a hint) and the figures downstream populate automatically. Work top to bottom.
    """)
    return


@app.cell
def _(mo):
    def verdict(check_fn):
        """`check_fn()` returns a list of (label, passed) pairs; render a callout.

        All calls to your exercise code happen *inside* `check_fn`, so an unfilled
        `TODO` (NotImplementedError) shows a gentle "not done yet" note rather than
        crashing the check cell.
        """
        try:
            checks = check_fn()
        except NotImplementedError as e:
            return mo.callout(mo.md(f"**Not done yet** — {e}"), kind="neutral")
        except Exception as e:  # pragma: no cover - student feedback
            return mo.callout(
                mo.md(f"Your code raised **{type(e).__name__}**: {e}"), kind="danger"
            )
        lines = [f"{'✅' if ok else '❌'} {label}" for label, ok in checks]
        kind = "success" if all(ok for _, ok in checks) else "warn"
        return mo.callout(mo.md("\n\n".join(lines)), kind=kind)

    return (verdict,)


@app.cell
def _(TRACES, np):
    # --- Given: load one session's trials from the trace bundle --------------
    # Each returned trial has: eye (Nx2 deg, 500 Hz), dt (ms/sample), objpos
    # (No x 2 option centers, deg), reward/prob levels, and the key event times
    # t_acquire (central fixation), t_bars_on, t_choice. (t_acquire/t_bars_on are
    # None if that event wasn't recorded on the trial.)
    _Z = np.load(TRACES, allow_pickle=True)

    def load_trials(session: str) -> list[dict]:
        out = []
        for i in np.where(_Z["session"] == session)[0]:
            ta, tb = float(_Z["t_acquire"][i]), float(_Z["t_bars_on"][i])
            out.append(
                dict(
                    trial=int(_Z["trial"][i]),
                    eye=np.asarray(_Z["eye"][i], dtype=float),
                    dt=float(_Z["dt"][i]),
                    objpos=np.asarray(_Z["objpos"][i], dtype=float),
                    reward=np.asarray(_Z["reward"][i], dtype=int),
                    prob=np.asarray(_Z["prob"][i], dtype=int),
                    t_acquire=None if np.isnan(ta) else ta,
                    t_bars_on=None if np.isnan(tb) else tb,
                    t_choice=float(_Z["t_choice"][i]),
                )
            )
        return out

    return (load_trials,)


@app.cell
def _(load_trials, mo):
    SESSION = "190430_Chip"
    trials = load_trials(SESSION)
    mo.md(
        f"Loaded **{len(trials)}** completed choice trials from session `{SESSION}`. "
    )
    return (trials,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Single trial visualization

    Below is the raw 500 Hz gaze trace for one trial, drawn over the option geometry. Reward bars (left of each option center) are blue outlines, probability bars (right) are orange. The gaze path is colored dark => light over time. **Where does the monkey tend to look — at whole options, or at particular bars?**
    """)
    return


@app.cell
def _(np, plt, trials):
    # --- Given: draw one trial's raw gaze over the bar geometry -------------
    def plot_trial(tr, ax=None):
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 5))
        for cx, cy in tr["objpos"]:
            for bx, col, lab in ((cx - 1.5, "#4a90d9", "R"), (cx + 1.5, "#e08a3c", "P")):
                ax.add_patch(
                    plt.Rectangle((bx - 1.0, cy - 5), 2.0, 10, fill=False, ec=col, lw=1.5)
                )
                ax.text(bx, cy - 5.6, lab, color=col, ha="center", va="top", fontsize=9)
        a = max(int((tr["t_bars_on"] or 0) / tr["dt"]), 0)
        b = min(int(tr["t_choice"] / tr["dt"]), len(tr["eye"]))
        seg = tr["eye"][a:b]
        ax.scatter(seg[:, 0], seg[:, 1], c=np.arange(len(seg)), cmap="cividis", s=6)
        ax.set_xlim(-15, 15)
        ax.set_ylim(-13, 13)
        ax.set_aspect("equal")
        ax.set_title(f"trial {tr['trial']} — raw gaze over the bars")
        ax.set_xlabel("x (deg)")
        ax.set_ylabel("y (deg)")
        return ax

    _t33 = next(t for t in trials if t["trial"] == 33)
    plot_trial(_t33)
    plt.gcf()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Exercise 1 — bar geometry

    To measure which bar the gaze is on, we first need each bar's screen location. An option is centered at `(cx, cy)` (a row of `objpos`). By the task design:

    - the **reward** bar sits at `x = cx − 1.5`, `y = cy`
    - the **probability** bar sits at `x = cx + 1.5`, `y = cy`

    Fill in `option_bars(objpos)` to return a list of `(option_index, attribute, x, y)` tuples — two per option — where `attribute` is the string `"reward"` or `"prob"`.
    """)
    return


@app.function
def option_bars(objpos):
    """Return [(option_idx, 'reward'|'prob', bar_x, bar_y), ...] for all options."""
    # TODO (Exercise 1): loop over the rows of objpos. Each row is an option
    #   center (cx, cy). Append TWO tuples per option:
    #     (option_idx, "reward", cx - 1.5, cy)  and
    #     (option_idx, "prob",   cx + 1.5, cy)
    raise NotImplementedError("Exercise 1: implement option_bars")


@app.cell
def _(np, verdict):
    def _ex1_checks():
        objpos = np.array([[8.5, 5.0], [-8.5, -5.0]])
        bars = option_bars(objpos)
        d = {(o, a): (x, y) for o, a, x, y in bars}
        return [
            ("returns 2 bars per option", len(bars) == 4),
            ("reward bar is on the left (cx-1.5)", np.allclose(d[(0, "reward")], [7.0, 5.0])),
            ("probability bar is on the right (cx+1.5)", np.allclose(d[(0, "prob")], [10.0, 5.0])),
        ]

    verdict(_ex1_checks)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Exercise 2 — classify a gaze sample

    Now write `classify_sample(gx, gy, objpos)`: given one gaze sample `(gx, gy)`, decide **which bar (if any) it is on**. Return `(option_idx, "reward"|"prob")`, or `None` if the sample is not on any bar.

    A sample counts as "on" a bar when it lies within a small window of that bar's center: **`|gx − bar_x| < 1.5`** (half the 3° gap between an option's two bars — so the window splits the option at its center) **and `|gy − bar_y| < 5.5`** (near the 10°-tall bar). If several bars qualify, pick the **nearest** one.

    *Hint:* `option_bars` already gives you every bar's location.
    """)
    return


@app.function
def classify_sample(gx, gy, objpos, xhalf=1.5, yhalf=5.5):
    """Return (option_idx, 'reward'|'prob') for the nearest bar the sample is on, else None."""
    # TODO (Exercise 2): go through every bar from option_bars(objpos). A bar
    #   qualifies if |gx - bar_x| < xhalf AND |gy - bar_y| < yhalf. Among the
    #   bars that qualify, return the (option_idx, attribute) of the NEAREST
    #   (smallest squared distance). Return None if no bar qualifies.
    raise NotImplementedError("Exercise 2: implement classify_sample")


@app.cell
def _(np, verdict):
    def _ex2_checks():
        op = np.array([[0.0, 0.0], [8.5, 5.0]])
        return [
            ("right bar -> probability", classify_sample(1.4, 0.0, op) == (0, "prob")),
            ("left bar -> reward", classify_sample(-1.4, 0.0, op) == (0, "reward")),
            ("far above the bars -> None", classify_sample(0.0, 12.0, op) is None),
            ("in the gap between options -> None", classify_sample(4.0, 0.0, op) is None),
        ]

    verdict(_ex2_checks)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Exercise 3 - checking for confounds

    Good modeling practice: first check that a boring artifact can't produce your effect. The scariest artifact here is a
    **horizontal calibration offset** — if the recorded gaze were shifted a degree to the right, option-centered looking would *masquerade* as looking at the right-hand (probability) bar.

    We can measure the offset directly. Right before the bars appear, the monkey must **hold central fixation** — its gaze should be at `(0, 0)`. So: average the gaze between `t_acquire` and `t_bars_on` and see how far it is from the origin.

    Fill in `central_hold_gaze(tr)` to return the mean `(x, y)` gaze during that window (or `None` if the window is missing/too short). We'll average it over the session; if it's near `(0, 0)`, calibration is clean.
    """)
    return


@app.function
def central_hold_gaze(tr, min_samples=5):
    """Mean (x, y) gaze during the central-fixation hold (t_acquire -> t_bars_on)."""
    # TODO (Exercise 3): if either t_acquire or t_bars_on is None, return None.
    #   Convert those times to sample indices by dividing by tr["dt"], slice
    #   tr["eye"] between them, and return the mean over axis 0. Return None if
    #   the window has fewer than min_samples samples.
    raise NotImplementedError("Exercise 3: implement central_hold_gaze")


@app.cell
def _(mo, np, trials, verdict):
    try:
        _raw = [central_hold_gaze(t) for t in trials]
        _holds = np.array([h for h in _raw if h is not None])
        _mean_offset = _holds.mean(axis=0) if len(_holds) else np.array([np.nan, np.nan])
        _done = True
    except NotImplementedError:
        _holds, _mean_offset, _done = np.empty((0, 2)), np.array([np.nan, np.nan]), False

    def _ex3_checks():
        if not _done:
            raise NotImplementedError("Exercise 3: implement central_hold_gaze")
        return [
            ("recovered a central hold for most trials", len(_holds) > 100),
            ("horizontal offset is small (< 0.5°)", abs(_mean_offset[0]) < 0.5),
        ]

    _msg = (
        "*Complete Exercise 3 to measure the calibration offset.*"
        if not _done
        else (
            f"**Mean central-hold gaze:** x = `{_mean_offset[0]:+.3f}°`, "
            f"y = `{_mean_offset[1]:+.3f}°`  (over {len(_holds)} trials).  "
            f"A clean ~0° x means any left/right result below is **real**, not a "
            f"calibration shift."
        )
    )
    mo.vstack([mo.md(_msg), verdict(_ex3_checks)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Exercise 4 — how is dwell time split between the two bars?

    For each **option-visit** (an option that received any bar-dwell on a trial), we want the time spent on its **reward** bar vs its
    **probability** bar. Sum dwell over the *free-viewing* window only (`t_bars_on` → `t_choice`); each 500 Hz sample contributes `dt` ms to whichever bar `classify_sample` assigns it to.

    Fill in `dwell_by_attribute(tr)` to return a dict mapping `(option_idx, attribute)` → **milliseconds** of dwell, for one trial.
    """)
    return


@app.function
def dwell_by_attribute(tr):
    """{(option_idx, 'reward'|'prob'): dwell_ms} over the free-viewing window."""
    # TODO (Exercise 4): the free-viewing window runs from t_bars_on to
    #   t_choice; convert both to sample indices (clip to [0, len(eye)]).
    #   For each sample in that window, call classify_sample(...). If it
    #   returns a label, add tr["dt"] ms to that label's running total in a
    #   dict. Return the dict.
    raise NotImplementedError("Exercise 4: implement dwell_by_attribute")


@app.cell
def _(pd, trials):
    # --- Given: turn per-trial dwell dicts into a per-option-visit table -----
    def visit_table(trial_list):
        rows = []
        for tr in trial_list:
            dwell = dwell_by_attribute(tr)
            for oi in range(len(tr["objpos"])):
                r = dwell.get((oi, "reward"), 0.0)
                p = dwell.get((oi, "prob"), 0.0)
                if r + p <= 0:
                    continue
                rows.append(
                    dict(
                        trial=tr["trial"],
                        option=oi,
                        reward_level=int(tr["reward"][oi]),
                        prob_level=int(tr["prob"][oi]),
                        dwell_reward=r,
                        dwell_prob=p,
                        frac_prob=p / (r + p),
                        one_attr_dominance=max(r, p) / (r + p),
                    )
                )
        return pd.DataFrame(rows)

    try:
        visits = visit_table(trials)
        summary = dict(
            n_visits=len(visits),
            frac_prob=visits["frac_prob"].mean(),
            one_attr=visits["one_attr_dominance"].mean(),
            pct90=float((visits["one_attr_dominance"] >= 0.90).mean() * 100),
        )
    except NotImplementedError:
        visits, summary = None, None
    summary
    return summary, visits


@app.cell
def _(summary, verdict):
    def _ex4_checks():
        if summary is None:
            raise NotImplementedError("Exercise 4: implement dwell_by_attribute")
        return [
            (f"~{summary['n_visits']} option-visits measured", summary["n_visits"] > 1500),
            (f"mean dwell fraction on probability bar = {summary['frac_prob']:.2f}",
             0.85 <= summary["frac_prob"] <= 1.0),
            (f"{summary['pct90']:.0f}% of visits are ≥90% a single attribute",
             summary["pct90"] > 70),
        ]

    verdict(_ex4_checks)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### See it: the distribution of within-option looking

    Two views of the `visits` table. **Left:** how one-sided each visit is (`one_attr_dominance` = fraction of the visit spent on whichever single bar got more time). **Right:** which bar that time went to (`frac_prob`).
    """)
    return


@app.cell
def _(mo, plt, visits):
    mo.stop(visits is None, mo.md("*Complete Exercise 4 to draw these histograms.*"))
    _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(10, 4))
    _ax1.hist(visits["one_attr_dominance"], bins=20, color="#5b8a72")
    _ax1.axvline(0.9, color="k", ls="--", lw=1)
    _ax1.set_xlabel("one-attribute dominance\n(1.0 = looked at only ONE bar)")
    _ax1.set_ylabel("option-visits")
    _ax1.set_title("How single-attribute is each visit?")

    _ax2.hist(visits["frac_prob"], bins=20, color="#e08a3c")
    _ax2.set_xlabel("fraction of dwell on the PROBABILITY bar\n(0 = all reward, 1 = all probability)")
    _ax2.set_ylabel("option-visits")
    _ax2.set_title("Which attribute gets looked at?")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Spoiler alert:** You should see the left histogram concentrated near 1: when the monkey visits an option, it overwhelmingly looks at *one* of the two bars, not both. What does that mean for the model?

    ---
    ## Exercise 5 — is the bias universal, or per-monkey?

    **Spoiler alert**: Chip (above) is biased towards the probability bar. Is that a fact about the *task* (an artifact of geometry would apply to every monkey) or about this *individual's strategy*? Test it: run the same pipeline on a **Dale** session and compare the mean `frac_prob`.

    Fill in `session_frac_prob(session)` to load that session and return its mean `frac_prob` across option-visits. (Reuse `load_trials` and `visit_table`.)
    """)
    return


@app.function
def session_frac_prob(session):
    """Mean fraction of within-option dwell on the probability bar, for a session."""
    # TODO (Exercise 5): load the session's trials with load_trials(session),
    #   build the visit table with visit_table(...), and return the mean of
    #   its "frac_prob" column.
    raise NotImplementedError("Exercise 5: implement session_frac_prob")


@app.cell
def _(mo, summary, verdict):
    try:
        _dale = session_frac_prob("181112_Dale")
        _chip = summary["frac_prob"] if summary else None
        _done5 = _chip is not None
    except NotImplementedError:
        _dale, _chip, _done5 = None, None, False

    def _ex5_checks():
        if not _done5:
            raise NotImplementedError("Exercises 4 & 5: needed to compare the monkeys")
        return [
            (f"Chip 190430 frac_prob = {_chip:.2f}", True),
            (f"Dale 181112 frac_prob = {_dale:.2f}", True),
            ("the probability bias is NOT universal — it differs by monkey",
             _dale < _chip - 0.2),
        ]

    _msg5 = (
        "*Complete Exercises 4 & 5 to compare Chip and Dale.*"
        if not _done5
        else (
            f"**Chip** puts **{_chip:.0%}** of within-option dwell on probability; "
            f"**Dale** only **{_dale:.0%}**. Same task, same geometry — so this is a "
            f"*strategy* difference, not an artifact. (It also independently confirms "
            f"the effect is real: an artifact would hit both monkeys identically.)"
        )
    )
    mo.vstack([mo.md(_msg5), verdict(_ex5_checks)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Write up a summary of your findings, and discuss what should the model do about it.

    1. **Finding 1.**
    2. **Finding 2.**
    3. **Finding 3.**

    Now recall the modeling assumption we started with:

    > the model measures **both** attributes on every glance (`gf = 1`).

    **Reflect (no code):**

    - If one glance already delivers both attributes, how many glances does the model need per option to decide? How does that compare to the number of fixations the monkey actually makes?
    - The monkey's gaze is *attribute-selective*. Which piece of the model would have to change for a fixation to measure reward **xor** probability?
    - Where in the model could individual differences (Chip-vs-Dale probability-biased vs uniform) be captured as a single fitted parameter?

    These questions set up **Part 2**, where we make the feature filter `gf` real so that a fixation targets an `(option, attribute)` pair — the *attribute-level* model — and leave the original whole-option model as a baseline to beat.
    """)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
