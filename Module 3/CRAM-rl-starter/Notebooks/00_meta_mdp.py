import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    # Enlarge markdown body text (matches the other starter code notebooks).
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
    # Introduction to the meta-MDP model

    This notebook lays out the meta-MDP model specified in

    > Radulescu, van Opheusden, Callaway, Griffiths & Hillis (2026),
    > *A Resource-Rational Account of Human Eye Movements During Immersive Visual
    > Search*, **Open Mind** 10, 91–117.
    > [doi:10.1162/OPMI.a.322](https://direct.mit.edu/opmi/article/doi/10.1162/OPMI.a.322/135355/A-Resource-Rational-Account-of-Human-Eye-Movements)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. What is a meta-MDP?

    Visual search is cast as a **sequential decision problem**: a sequence of
    decisions about where to fixate next, where each decision takes into account the
    information gained from previous fixations. To capture the trade-off between the
    *utility* and the *cost* of gathering information, the paper formalizes search as
    a **meta-level Markov decision process** (meta-MDP; Hay et al. 2012; Russell &
    Wefald 1991).

    Like a standard MDP, a meta-MDP has a set of states, a set of actions, a
    transition function, and a reward function. What makes it *meta* is what those
    pieces refer to:

    | Standard MDP | meta-MDP |
    |---|---|
    | states = configurations of the world | **states = beliefs** about the world |
    | actions = physical moves | **actions = computations** (here: fixations) |
    | transition = how the world changes | **transition = how a computation updates beliefs** |
    | reward = task payoff | **reward = payoff for accurate beliefs − cost of computing** |

    Crucially, the environment itself does not change — *the agent's beliefs about
    it do*. The per-computation cost implements the "cost" of gathering information
    and pushes the agent toward policies that balance **utility** (fixating objects
    similar to the target) against **complexity** (making only as many fixations as
    necessary). This is what makes the resulting policy *resource-rational*.

    /// admonition | Relation to POMDPs
        type: note
    A meta-MDP resembles a POMDP, but the uncertainty is framed as arising from the
    agent's *internal* state (its beliefs), and the actions are *mental* actions
    (computations) rather than moves in a partially observed world.
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Model components

    For a general version of visual search, the paper describes the meta-MDP through
    five components:

    1. **Latent state.** A scene is a set of objects at known locations, each an
       (unknown) feature vector in a low-dimensional space. At the start of an
       episode the agent perceives *which* objects are present and *where*, but not
       their features.
    2. **Belief state.** A distribution over latent states. For every object the
       belief is two vectors: the current **mean** and **precision** of that
       object's features.
    3. **Computations.** A computation = fixating one object and sampling
       information about its features. A special computation terminates search.
    4. **Transition function.** What information a computation samples (features of
       objects near the center of gaze) and how it is folded into beliefs (Bayesian
       cue combination), through a fovea-like filter whose weight decays with
       distance from the fixated object.
    5. **Reward.** Each computation costs the agent. On termination it forms a
       posterior over which object is the target, reports the argmax, and is
       rewarded iff correct.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Latent state

    The latent (unknown) state is a matrix $A$ of dimensionality $N_o \times N_f$,
    where the value of feature $f$ for object $o$ is $A_{of}$; $N_o$ is the number
    of objects and $N_f$ the number of features in the chosen representational
    space. The agent knows the $N_o$ objects present in each scene, but not their
    features. $A$ is a **fixed** property of the environment (it does not change
    within an episode).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Beliefs

    The agent represents the environment as beliefs over the features in $A$. For
    tractability these are **independent Gaussian** beliefs, held in a mean matrix
    $F$ and a precision matrix $J$, both $N_o \times N_f$:

    $$p(A_{of}) = \mathcal{N}\!\left(F_{of},\; 1/J_{of}\right) \tag{1}$$

    So $F_{of}$ is the mean belief for feature $f$ of object $o$, and $J_{of}$ its
    precision (variance $1/J_{of}$).

    **Prior.** At the start of each episode, beliefs are initialized broad and
    uninformative:

    $$F_{of} \sim \mathcal{N}(0,\; 0.01^2), \qquad J^{(0)}_{of} = 0.01.$$

    Feature means start near zero with tiny jitter, and the low precision
    $J_{of} = 0.01$ corresponds to high uncertainty (variance $1/J_{of} = 100$) —
    an effectively uninformative prior over feature values.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Computations and actions

    - A **computation** corresponds to looking at the center of an object $o$.
    - An **action** corresponds to performing a computation.

    (A special computation, $\perp$, terminates search — see §7.)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Transition function

    A computation takes a noisy measurement $X$ of the features of objects near the
    center of gaze, and folds it into the belief via **Bayesian cue combination**.
    In four steps:

    **(a) Object attentional mask (the "fovea").** Compute a mask $g_o$ in which
    attention to every object decays exponentially with its distance to the fixated
    object. Using a von Mises–Fisher kernel that scores how aligned each object's
    location $x_o$ (in 3-D) is to the current fixation location $\mu$:

    $$g_o = \text{scale}_{g_o} \cdot \exp\!\left[\left(x_o^{\mathsf{T}}\mu - 1\right)\cdot \kappa\right] \tag{2}$$

    Here $\text{scale}_{g_o}$ sets the overall strength of the output and $\kappa$
    controls how sharply it responds to differences in location. Both are free
    parameters.

    **(b) Measurement precision.** Two filters combine to say how precisely each
    *(object, feature)* pair is measured: the object mask $g_o$ (an $N_o$-vector,
    from step a) that decays with distance from fixation, and a **feature-level
    filter** $g_f$ (an $N_f$-vector) that says how much precision each feature
    receives. Their outer product gives the measurement-precision matrix:

    $$J_{\text{meas}} = g_o\, g_f^{\mathsf{T}}$$

    In the original model, $g_f = \mathbf{1}$ (a vector of ones): every feature of
    an object is measured with equal precision, so a single fixation takes in
    **all** of an object's features at once. A fixated object is perfectly aligned
    and sampled at full precision; misalignment causes exponential decay in sampling
    precision. Writing the ones vector out as $g_f$ (rather than the paper's
    $\mathbf{1}^{\mathsf{T}}$) makes this "measure everything" assumption explicit.

    **(c) Sample the measurement** independently from the true latent state:

    $$X \sim \mathcal{N}\!\left(A,\; 1/J_{\text{meas}}\right) \tag{3}$$

    **(d) Bayesian update** of mean and precision ($\odot$ is the element-wise
    product):

    $$F \leftarrow \frac{F \odot J + X \odot J_{\text{meas}}}{J + J_{\text{meas}}}, \qquad J \leftarrow J + J_{\text{meas}} \tag{4}$$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Reward, target readout, and termination

    **Cost.** The agent incurs a cost $-c$ for each computation, where $c$ is a free
    parameter governing the penalty per additional fixation.

    **Posterior over the target.** When the agent terminates, it computes a
    posterior over which object $o$ is the target, given its beliefs:

    $$p_o \propto \exp\!\left(\frac{1}{2}\sum_f \left[\, \log\!\left(1 + J_{of}\right) + \frac{J_{of}}{1 + J_{of}}\,F_{of}^2 - J_{of}\left(F_{of} - f^{\text{target}}_f\right)^2 \,\right]\right) \tag{5}$$

    This assumes the true feature values are mutually independent and Gaussian with
    mean 0 and variance 1. The agent reports $\arg\max_o\, p_o$ and receives a final
    reward $R = 1$ if this is the true target and $R = 0$ otherwise.

    **Termination.** A special computation $\perp$ terminates search whenever
    $\max_o p_o$ exceeds a threshold $\theta$ — a free parameter setting how
    confident the agent must be before it stops.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Parameters

    The paper's summary of model parameters and the values used in the experiment:

    | Parameter | Symbol | Value |
    |---|---|---|
    | Initial mean belief | $F_{of}$ | $\mathcal{N}(0,\,0.01^2)$ |
    | Initial belief precision | $J^{(0)}_{of}$ | $0.01$ |
    | Mask scale | $\text{scale}_{g_o}$ | $3$ |
    | Mask sharpness | $\kappa$ | $200$ |
    | Feature-level filter | $g_f$ | $\mathbf{1}$ (all features measured; fixed in the original model) |
    | Cost of computation | $c$ | $0.01$ |
    | Termination threshold | $\theta$ | optimized (0.998 when predicting human behavior) |
    | Lapse rate | $\epsilon$ | fit (when comparing to human policies) |

    The initialization values (mean, precision), mask scale/sharpness, and
    computation cost are treated as fixed hyperparameters — **not** tuned during
    deep-RL training. Only $\theta$ (and $\epsilon$, for the human comparison) are
    optimized/fit.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. The optimal policy: Fixate_MAP

    Given this meta-MDP, what is the *resource-rational* (optimal) policy — the one
    that maximizes performance while minimizing search cost? The paper trains deep-RL
    agents (PPO) to solve it, and finds that independent agents all converge to the
    **same** policy, up to their representational constraints:

    > **Fixate_MAP** — at each step, fixate the object currently *most likely to be
    > the target* (the maximum-a-posteriori object under $p_o$), and terminate when
    > $\max_o p_o > \theta$.

    Because all agents converge to it regardless of representation, Fixate_MAP is
    taken as a computationally tractable approximation of the optimal policy. It
    also **contains the classic 1-D ideal-observer detector as a special case**,
    giving a rational basis for that model in structured, naturalistic scenes.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10. Adapting the meta-MDP to a new task

    The meta-MDP is a general formalism for active information gathering. Training a meta-MDP with deep reinforcement learning allows us to discover resource rational information gathering policies that balance expected reward with the cost of computation. Before adapting this model to a new task, take a moment to consider what needs to change, and what can stay the same.

    Reflect on the following questions:

    1. Which components of the model do you think should remain **task-independent**, and which components are likely to depend on the specifics of a task?

    2. If the goal changes from searching for a target in a scene to choosing the best option from a small set of alternatives, what aspects of the model's **inputs** would need to be redefined?

    3. Can the same belief-update and reward framework be reused in a different task? If so, what would need to be reinterpreted?

    4. How might the information acquired during a single fixation depend on the task? For example, could a glance reveal all of an option's attributes, or might it reveal only part of the available information? What consequences would each assumption have for the model?

    As you work through the starter code, revisit your answers and compare your initial expectations with the modeling decisions you encounter.
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
