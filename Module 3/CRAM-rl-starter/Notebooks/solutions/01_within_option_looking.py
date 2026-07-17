import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
        # Part 1 — Does the monkey look at whole options, or one attribute at a time?

        We are adapting a **resource-rational meta-MDP** model of eye movements
        (Radulescu et al., 2026) to a macaque multiattribute-choice task
        (Perkins, Gillis & Rich, 2024). In that task, monkeys **Chip** and **Dale**
        see 2–3 options; each option has two attributes — **reward magnitude** and
        **reward probability** — drawn as two vertical bars:

        - **Reward** bar on the **left**  (option center − 1.5°)
        - **Probability** bar on the **right** (option center + 1.5°)

        The model, as written, assumes that a single glance at an option measures
        **both** of its attributes at once (in the code, the feature filter
        `gf = 1`). Before we build anything, we should ask the data whether that
        assumption is true.

        > **The question for this part:** when the monkey looks at an option, does
        > it take in the whole option (both bars) — or does it sample **one attribute
        > at a time**? The answer decides how we structure the model's *fixations*.

        We have the **raw 500 Hz eye trace** for every trial, so we can look directly.
        Work through the exercises (`TODO` cells) in order. Each has a self-check.
        """
    )
    return


@app.cell
def _():
    # --- Setup: locate the bundled example traces ---------------------------
    # The only raw dependency in this starter code is a small 500 Hz eye-trace bundle
    # (Data/example_traces.npz) for two example sessions — one per monkey. It is
    # generated once from the raw .bhv2 by Libraries/make_example_traces.py; the
    # full raw sessions are too large to ship. Everything else is derived.
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


@app.cell
def _(mo):
    mo.md(
        r"""
        ## How the exercises work

        This is a **reactive** notebook: when you edit a cell, marimo re-runs every
        cell that depends on it. Each `TODO` starts by raising
        `NotImplementedError`, so before you fill it in you'll see that message in
        the exercise's own cell and a grey **"not done yet"** note in its check.
        Fill the blank and the check turns **green** (or red, with a hint) and the
        figures downstream populate automatically. Work top to bottom.
        """
    )
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
        f"Loaded **{len(trials)}** completed choice trials from `{SESSION}`. "
        f"Trial 33 (below) is our motivating example."
    )
    return SESSION, trials


@app.cell
def _(mo):
    mo.md(
        r"""
        ## First, just look

        Below is the raw 500 Hz gaze trace for one trial, drawn over the option
        geometry. Reward bars (left of each option center) are blue outlines,
        probability bars (right) are orange. The gaze path is colored dark→light
        over time. **Where does the gaze spend its time — on whole options, or on
        particular bars?** Form a hypothesis before you measure.
        """
    )
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
    return (plot_trial,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ---
        ## Exercise 1 — bar geometry

        To measure *which bar* the gaze is on, we first need each bar's screen
        location. An option is centered at `(cx, cy)` (a row of `objpos`). By the
        task design:

        - the **reward** bar sits at `x = cx − 1.5`, `y = cy`
        - the **probability** bar sits at `x = cx + 1.5`, `y = cy`

        Fill in `option_bars(objpos)` to return a list of
        `(option_index, attribute, x, y)` tuples — two per option — where
        `attribute` is the string `"reward"` or `"prob"`.
        """
    )
    return


@app.cell
def _():
    def option_bars(objpos):
        """Return [(option_idx, 'reward'|'prob', bar_x, bar_y), ...] for all options."""
        # SOLUTION
        bars = []
        for oi, (cx, cy) in enumerate(objpos):
            bars.append((oi, "reward", cx - 1.5, cy))
            bars.append((oi, "prob", cx + 1.5, cy))
        return bars
    return (option_bars,)


@app.cell
def _(np, option_bars, verdict):
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


@app.cell
def _(mo):
    mo.md(
        r"""
        ---
        ## Exercise 2 — classify a gaze sample

        Now write `classify_sample(gx, gy, objpos)`: given one gaze sample
        `(gx, gy)`, decide **which bar (if any) it is on**. Return
        `(option_idx, "reward"|"prob")`, or `None` if the sample is not on any bar.

        A sample counts as "on" a bar when it lies within a small window of that
        bar's center: **`|gx − bar_x| < 1.5`** (half the 3° gap between an option's
        two bars — so the window splits the option at its center) **and
        `|gy − bar_y| < 5.5`** (near the 10°-tall bar). If several bars qualify,
        pick the **nearest** one.

        *Hint:* `option_bars` already gives you every bar's location.
        """
    )
    return


@app.cell
def _(option_bars):
    def classify_sample(gx, gy, objpos, xhalf=1.5, yhalf=5.5):
        """Return (option_idx, 'reward'|'prob') for the nearest bar the sample is on, else None."""
        # SOLUTION
        best, best_d = None, float("inf")
        for oi, attr, bx, by in option_bars(objpos):
            if abs(gx - bx) < xhalf and abs(gy - by) < yhalf:
                d = (gx - bx) ** 2 + (gy - by) ** 2
                if d < best_d:
                    best, best_d = (oi, attr), d
        return best
    return (classify_sample,)


@app.cell
def _(classify_sample, np, verdict):
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


@app.cell
def _(mo):
    mo.md(
        r"""
        ---
        ## Exercise 3 — a skeptic's guard (do this *before* trusting any result)

        You are about to compute a surprising number. Good practice: first check
        that a boring artifact can't produce it. The scariest artifact here is a
        **horizontal calibration offset** — if the recorded gaze were shifted a
        degree to the right, option-centered looking would *masquerade* as looking
        at the right-hand (probability) bar, and we'd "discover" something that
        isn't real.

        We can measure the offset directly. Right before the bars appear, the
        monkey must **hold central fixation** — its gaze should be at `(0, 0)`. So:
        average the gaze between `t_acquire` and `t_bars_on` and see how far it is
        from the origin.

        Fill in `central_hold_gaze(tr)` to return the mean `(x, y)` gaze during
        that window (or `None` if the window is missing/too short). We'll average
        it over the session; if it's near `(0, 0)`, calibration is clean.
        """
    )
    return


@app.cell
def _():
    def central_hold_gaze(tr, min_samples=5):
        """Mean (x, y) gaze during the central-fixation hold (t_acquire -> t_bars_on)."""
        # SOLUTION
        if tr["t_acquire"] is None or tr["t_bars_on"] is None:
            return None
        a = int(tr["t_acquire"] / tr["dt"])
        b = int(tr["t_bars_on"] / tr["dt"])
        if b - a < min_samples:
            return None
        return tr["eye"][a:b].mean(axis=0)
    return (central_hold_gaze,)


@app.cell
def _(central_hold_gaze, mo, np, trials, verdict):
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


@app.cell
def _(mo):
    mo.md(
        r"""
        ---
        ## Exercise 4 — how is dwell time split between the two bars?

        Now the payoff. For each **option-visit** (an option that received any
        bar-dwell on a trial), we want the time spent on its **reward** bar vs its
        **probability** bar. Sum dwell over the *free-viewing* window only
        (`t_bars_on` → `t_choice`); each 500 Hz sample contributes `dt` ms to
        whichever bar `classify_sample` assigns it to.

        Fill in `dwell_by_attribute(tr)` to return a dict mapping
        `(option_idx, attribute)` → **milliseconds** of dwell, for one trial.
        """
    )
    return


@app.cell
def _(classify_sample):
    def dwell_by_attribute(tr):
        """{(option_idx, 'reward'|'prob'): dwell_ms} over the free-viewing window."""
        # SOLUTION
        a = max(int((tr["t_bars_on"] or 0) / tr["dt"]), 0)
        b = min(int(tr["t_choice"] / tr["dt"]), len(tr["eye"]))
        dwell = {}
        for i in range(a, b):
            lab = classify_sample(tr["eye"][i, 0], tr["eye"][i, 1], tr["objpos"])
            if lab is not None:
                dwell[lab] = dwell.get(lab, 0.0) + tr["dt"]
        return dwell
    return (dwell_by_attribute,)


@app.cell
def _(dwell_by_attribute, np, pd, trials):
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
    return summary, visit_table, visits


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


@app.cell
def _(mo):
    mo.md(
        r"""
        ### See it: the distribution of within-option looking

        Two views of the `visits` table. **Left:** how one-sided each visit is
        (`one_attr_dominance` = fraction of the visit spent on whichever single bar
        got more time). **Right:** which bar that time went to (`frac_prob`).
        """
    )
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


@app.cell
def _(mo):
    mo.md(
        r"""
        You should see the left histogram **piled up near 1.0**: when the monkey
        visits an option, it overwhelmingly looks at *one* of the two bars, not
        both. That is the empirical fact the model must respect.

        ---
        ## Exercise 5 — is the *probability* bias universal, or per-monkey?

        Chip (above) parks on the probability bar. Is that a fact about the *task*
        (an artifact of geometry would apply to every monkey) or about this
        *individual's strategy*? Test it: run the same pipeline on a **Dale**
        session and compare the mean `frac_prob`.

        Fill in `session_frac_prob(session)` to load that session and return its
        mean `frac_prob` across option-visits. (Reuse `load_trials` and
        `visit_table`.)
        """
    )
    return


@app.cell
def _(load_trials, visit_table):
    def session_frac_prob(session):
        """Mean fraction of within-option dwell on the probability bar, for a session."""
        # SOLUTION
        return visit_table(load_trials(session))["frac_prob"].mean()
    return (session_frac_prob,)


@app.cell
def _(mo, session_frac_prob, summary, verdict):
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


@app.cell
def _(mo):
    mo.md(
        r"""
        ---
        ## What did we find, and what should the model do about it?

        Your own measurements (not our say-so):

        1. **One attribute at a time.** Most option-visits are ≥90% a single bar —
           the monkey samples reward *or* probability on a given look, rarely both.
        2. **Attribute choice is a strategy.** Chip rides probability; Dale is far
           more balanced. That's a per-individual knob, and it proves the effect is
           real rather than a calibration artifact (you checked calibration in
           Exercise 3).
        3. **Sampling is sparse.** ~2 bar-visits per trial against 4–6 available
           attribute-cells — the monkey decides after seeing a *fraction* of the
           information. (That sparsity is the whole point of a *resource-rational*
           model.)

        Now recall the modeling assumption we started with:

        > the model measures **both** attributes on every glance (`gf = 1`).

        **Reflect (no code):**

        - If one glance already delivers both attributes, how many glances does the
          model *need* per option to decide? How does that compare to the ~2 the
          monkey actually makes?
        - The monkey's looks are *attribute-selective*. Which piece of the model —
          the object filter `go`, or the feature filter `gf` — would have to change
          for a fixation to measure reward **XOR** probability?
        - Where in the model could Chip-vs-Dale (probability-heavy vs balanced)
          live as a single fitted parameter?

        These questions set up **Part 2**, where we make the feature filter `gf`
        real so that a fixation targets an `(option, attribute)` pair — the
        *attribute-level* model — and leave the original whole-option model as a
        baseline to beat.
        """
    )
    return


if __name__ == "__main__":
    app.run()
