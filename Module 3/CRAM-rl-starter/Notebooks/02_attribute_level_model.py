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
    # Part 2 — Building the attribute-level model

    In **Part 1** you measured, from the raw gaze, that the monkeys sample
    **one attribute at a time** (reward *or* probability per look), and that
    Chip is biased to sample the **probability** bar (~93% of within-option dwell). The
    meta-MDP as written can't do this: its fixations measure **both** attributes
    at once — in the code, the feature filter `gf = 1` (all-ones).

    In this part we will **adapt** the model by making `gf` real and learnable, so that a fixation measures one
    attribute, then ask whether a policy *trained only to identify the best
    option* reproduces the monkey's looking — without ever being told to.

    We compare two models, sharing everything except the action:

    | | fixation targets | a look measures |
    |---|---|---|
    | **object-level** (Part 1's model) | an option (`No` targets) | both attributes (`gf=1`) |
    | **attribute-level** (this part) | an (option, attribute) **bar** (`2·No` targets) | one attribute (`gf` one-hot) |

    The training code is **unchanged** — you only change the environment.
    """)
    return


@app.cell
def _():
    # --- Setup: locate the project and import the shared model code ---------
    import sys
    from pathlib import Path

    import numpy as np
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

    from data import SessionData
    from metamdp_nhp import (NHPMetaMDP, NHPMetaMDPEnv, State, TimeStep,
                             get_posterior_itrue)
    from train_reinforce import train, greedy_episode, DEFAULT_COST, NO_MAX

    return (
        DEFAULT_COST,
        NHPMetaMDP,
        NHPMetaMDPEnv,
        NO_MAX,
        SessionData,
        State,
        TimeStep,
        greedy_episode,
        np,
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
def _(SessionData):
    SESSION = "190430_Chip"
    _data = SessionData.from_combined(SESSION)   # reads Data/all_trial_options.csv
    scenes = [_data.scene(t) for t in _data.trials()]
    return SESSION, scenes


@app.cell
def _(DEFAULT_COST, NHPMetaMDP):
    # One shared meta-MDP (belief math + cost). Both models use it.
    mdp = NHPMetaMDP(cost_params=DEFAULT_COST)
    return (mdp,)


@app.cell
def _(SESSION, mo, scenes):
    mo.md(f"""
    Loaded **{len(scenes)}** scenes from `{SESSION}` "
        f"(No = {sorted(set(s.n_options for s in scenes))}). "
        f"Both models train on these real option configurations.
    """)
    return


@app.cell
def _(NHPMetaMDPEnv, NO_MAX, mo, scenes, train):
    # --- Given: the object-level BASELINE (Part 1's model, gf=1) -----------
    # Trained here so we have a reference. This is the model you already have;
    # it fixates whole options and measures both attributes per look.
    p_object, _ = train(scenes, episodes=8000, seed=0, verbose=False,
                        env_class=NHPMetaMDPEnv, n_targets_max=NO_MAX)
    mo.md("Trained the **object-level baseline** (≈8k episodes).")
    return (p_object,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Exercise 1 — make the feature filter `gf` real

    A fixation allocates measurement precision `Jmeas = go ⊗ gf` (outer product
    of an object filter `go` and a **feature filter** `gf`). The object-level
    model uses `gf = [1, 1]` — it measures both features. The **attribute-level**
    model uses a **one-hot** `gf`: a fixation on the *reward* bar measures only
    reward, on the *probability* bar only probability.

    Fill in `feature_filter(attribute, nf)`:

    - `attribute is None` → return all-ones `[1, 1]` (the object-level filter),
    - `attribute == 0` → `[1, 0]` (reward only),
    - `attribute == 1` → `[0, 1]` (probability only).
    """)
    return


@app.function
def feature_filter(attribute, nf):
    """The meta-MDP feature filter gf: all-ones (object-level) or one-hot."""
    # TODO (Exercise 1): return np.ones(nf) when attribute is None; otherwise
    #   a length-nf one-hot vector with a 1.0 at index `attribute`.
    raise NotImplementedError("Exercise 1: implement feature_filter")


@app.cell
def _(np, verdict):
    def _ex1_checks():
        return [
            ("attribute=None → both features [1,1]", np.allclose(feature_filter(None, 2), [1, 1])),
            ("attribute=0 → reward only [1,0]", np.allclose(feature_filter(0, 2), [1, 0])),
            ("attribute=1 → probability only [0,1]", np.allclose(feature_filter(1, 2), [0, 1])),
        ]
    verdict(_ex1_checks)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Exercise 2 — the (option, attribute) action space

    Object-level fixations index `No` options. Attribute-level fixations index
    `2·No` **bars**: for each option (posterior-sorted) there is a reward bar
    and a probability bar. We enumerate targets as
    `target = option_rank*2 + attribute`, so:

    | target | 0 | 1 | 2 | 3 | 4 | 5 |
    |---|---|---|---|---|---|---|
    | (option_rank, attribute) | (0,R) | (0,P) | (1,R) | (1,P) | (2,R) | (2,P) |

    Fill in `decode_target(target)` returning `(option_rank, attribute)` with
    `attribute` 0 = reward, 1 = probability. *Hint:* `divmod`.
    """)
    return


@app.function
def decode_target(target):
    """Map a flat bar-target index to (option_rank, attribute)."""
    # TODO (Exercise 2): target = option_rank*2 + attribute. Recover both with
    #   divmod(int(target), 2) and return (option_rank, attribute).
    raise NotImplementedError("Exercise 2: implement decode_target")


@app.cell
def _(verdict):
    def _ex2_checks():
        return [
            ("target 0 → (option 0, reward)", decode_target(0) == (0, 0)),
            ("target 1 → (option 0, probability)", decode_target(1) == (0, 1)),
            ("target 3 → (option 1, probability)", decode_target(3) == (1, 1)),
            ("target 4 → (option 2, reward)", decode_target(4) == (2, 0)),
        ]
    verdict(_ex2_checks)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Given: wire your two functions into an attribute-level environment

    `AttrEnv` below is the object-level env with three overrides: `n_targets`
    is now `2·No`, `decode` uses **your Exercise 2**, and `step` measures with
    **your Exercise 1**'s one-hot `gf`. The belief update (`_measure`) is the
    same precision-weighted Bayesian update as the object-level model — only
    `gf` differs. Nothing else about the model or trainer changes.
    """)
    return


@app.cell
def _(NHPMetaMDPEnv, State, TimeStep, np):
    def _measure(mdp, state, scene, fixated_object, gf):
        """Given: one non-terminal fixation's Bayesian belief update with filter gf."""
        go = mdp.fovea_go(scene, fixated_object)
        Jmeas = go[:, None] * gf[None, :]                      # go ⊗ gf
        x_times_j = np.random.normal(scene.features * Jmeas, np.sqrt(Jmeas))
        penalty = mdp.cost_params[0] + mdp.cost_params[1] * np.mean(Jmeas)
        nxt = State(F=(state.F * state.J + x_times_j) / (state.J + Jmeas),
                    J=state.J + Jmeas)
        return nxt, -penalty

    class AttrEnv(NHPMetaMDPEnv):
        """Attribute-level env built from Exercises 1 & 2 (bar-level fixations)."""

        @property
        def n_targets(self):
            return 2 * self.scene.n_options

        def decode(self, target):
            option_rank, attribute = decode_target(target)      # Exercise 2
            return int(self._order[option_rank]), attribute

        def step(self, target, terminate):
            if self._ended:
                return self.reset()
            if terminate:
                self._state, reward = self.mdp.transition(self._state, self.scene, None, True)
            else:
                fixated, attribute = self.decode(target)
                gf = feature_filter(attribute, self.mdp.Nf)     # Exercise 1
                self._state, reward = _measure(self.mdp, self._state, self.scene, fixated, gf)
            self._ended = self._state.is_terminal
            return TimeStep(self._encode(), float(reward), self._ended)

    return (AttrEnv,)


@app.cell
def _(AttrEnv, NO_MAX, mo, scenes, train):
    # Train the attribute-level model (re-runs if you edit Exercise 1 or 2).
    try:
        p_attr, _ = train(scenes, episodes=8000, seed=0, verbose=False,
                          env_class=AttrEnv, n_targets_max=2 * NO_MAX)
        _msg2 = ("Trained the **attribute-level model** (≈8k episodes). "
                 "Edit Exercise 1 or 2 and this retrains automatically.")
    except NotImplementedError:
        p_attr = None
        _msg2 = "*Complete Exercises 1 & 2 to build and train the attribute-level model.*"
    mo.md(_msg2)
    return (p_attr,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Exercise 3 — measure what the model looks at

    Now score the trained policy's scanpaths the way you scored the monkey's in
    Part 1. `greedy_episode` returns a `decoded` list of `(option, attribute)`
    fixations. Fill in `prob_fraction(decoded)`: of the fixations that carry an
    attribute (attribute is not `None`), what fraction are on the **probability**
    bar (attribute == 1)? Return `nan` if there are no such fixations.
    """)
    return


@app.function
def prob_fraction(decoded):
    """Fraction of attribute-bearing fixations that are on the probability bar."""
    # TODO (Exercise 3): from `decoded` (list of (option, attribute)), keep the
    #   fixations whose attribute is not None; return the fraction with
    #   attribute == 1 (probability). Return float("nan") if there are none.
    raise NotImplementedError("Exercise 3: implement prob_fraction")


@app.cell
def _(np, verdict):
    def _ex3_checks():
        return [
            ("all-probability scanpath → 1.0",
             prob_fraction([(0, 1), (1, 1)]) == 1.0),
            ("half reward / half prob → 0.5",
             prob_fraction([(0, 0), (1, 1)]) == 0.5),
            ("object-level (attribute None) → nan",
             np.isnan(prob_fraction([(0, None), (1, None)]))),
        ]
    verdict(_ex3_checks)
    return


@app.cell
def _(
    AttrEnv,
    NHPMetaMDPEnv,
    greedy_episode,
    mdp,
    np,
    p_attr,
    p_object,
    scenes,
):
    # --- Given: run both trained policies over all scenes, collect stats ------
    def _stats(p, env_class):
        acc, nfix, decoded_all, per_visit1 = [], [], [], []
        for scene in scenes:
            g = greedy_episode(p, env_class(mdp, scene))
            acc.append(g["choice"] == scene.itrue)
            nfix.append(g["n_fix"])
            decoded_all += g["decoded"]
            # distinct attributes per consecutive-same-option visit
            if g["decoded"] and g["decoded"][0][1] is not None:
                cur, seen = None, set()
                for opt, a in g["decoded"]:
                    if opt != cur:
                        if seen:
                            per_visit1.append(len(seen) == 1)
                        cur, seen = opt, {a}
                    else:
                        seen.add(a)
                if seen:
                    per_visit1.append(len(seen) == 1)
        return dict(acc=np.mean(acc), nfix=np.mean(nfix),
                    prob_frac=prob_fraction(decoded_all),
                    one_attr=(np.mean(per_visit1) if per_visit1 else np.nan))

    try:
        obj_stats = _stats(p_object, NHPMetaMDPEnv)
        attr_stats = _stats(p_attr, AttrEnv) if p_attr is not None else None
    except NotImplementedError:
        obj_stats = attr_stats = None       # Exercise 3 (prob_fraction) not done yet
    return attr_stats, obj_stats


@app.cell
def _(attr_stats, mo, obj_stats):
    mo.stop(obj_stats is None or attr_stats is None,
            mo.md("*Complete Exercises 1–3 above to populate the comparison.*"))

    def _fmt(x):
        return "—" if x != x else f"{x:.2f}"   # x!=x tests nan

    _table = f"""
    | metric | object-level | attribute-level | monkey (Chip, Part 1) |
    |---|---|---|---|
    | accuracy (picks best) | {obj_stats['acc']:.2f} | {attr_stats['acc']:.2f} | — |
    | fixations / trial | {obj_stats['nfix']:.2f} | {attr_stats['nfix']:.2f} | ~2 |
    | measures 1 attribute / look | **no** (gf=1) | **yes** (gf one-hot) | yes |
    | visits sampling ONE attribute | {_fmt(obj_stats['one_attr'])} | {attr_stats['one_attr']:.0%} | ~mode 1 |
    | fixations on the **probability** bar | {_fmt(obj_stats['prob_frac'])} | {attr_stats['prob_frac']:.2f} | 0.93 |
    """
    mo.md("### The comparison\n" + _table)
    return


@app.cell
def _(attr_stats, mo):
    mo.stop(attr_stats is None,
            mo.md("*Complete the exercises above to read the wrap-up with your numbers.*"))
    mo.md(
        rf"""
        ---
        ## What just happened

        You changed **only the environment** — `gf` from all-ones to one-hot, and the
        action space from options to (option, attribute) bars — and retrained the
        *same* policy with the *same* objective (identify the best option). The
        attribute-level model then, on its own:

        1. **looks one attribute at a time** ({attr_stats['one_attr']:.0%} of option-visits
           sample a single attribute) — the object-level model *cannot* do this, because
           every look measures both. This is the Part 1 fact, now reproduced by a model.
        2. **is biased towards the probability bar** (fixation fraction {attr_stats['prob_frac']:.2f}),
           matching Chip's 0.93 — and it sampled probability *across options*, i.e. it
           learned **direct attribute comparison** (the Perkins–Rich finding). Nobody
           told it to prefer probability: probability carries weight 1.78 in the value
           rule, so it is the more *diagnostic* attribute, and the resource-rational
           policy spends its looks there.

        ---
        ### 🎁 Bonus question — what about Dale?

        Everything above is a **Chip** session. Before reading on, *predict*: if you
        train this same attribute-level model on a **Dale** session, what will its
        probability-bar fixation fraction be — closer to Chip's ~0.93, or to Dale's
        **balanced ~0.43** from Part 1?

        Try it — change `SESSION` at the top of the notebook to a Dale session
        (e.g. `"181112_Dale"`) and re-run. Then think about:

        - Does the model reproduce Dale's balance, or does it *still* go
          probability-heavy? Why?
        - The objective and the value rule are **identical** for both monkeys. So what,
          if anything, could make the model weigh the two attributes differently for
          Dale than for Chip?
        - What would you have to *add* to the model to capture an individual monkey's
          style, rather than only the task-optimal one?

        Jot down your prediction and what you find. We pick this thread up in the next
        part — it's the point where the *model meets the individual monkey.*
        """
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
