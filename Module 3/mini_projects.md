# Module 3 mini-project

There is one project, and everyone does it.

> **In the session, you watched a policy trained on fifty real demonstrations fail on the
> very arm that produced them. Fix it.**
> **On the last day, every group's policy runs on the real arm, live.**

## What you inherit

- **Two datasets**, both recorded on the demo arm, both loading through `LeRobotDataset`
  exactly as in the tutorial (the "cube" is a paper ball; the physics of imitation
  learning are indifferent to this):
  - [`nisheet0/so101_blind_pickplace`](https://huggingface.co/datasets/nisheet0/so101_blind_pickplace)
    — 50 camera-free episodes, instructor-teleoperated, the ball starting from **one
    fixed, taped position** every time. This is the dataset behind the failing baseline.
  - [`nisheet0/bamb2026_so101_pickplace_vision`](https://huggingface.co/datasets/nisheet0/bamb2026_so101_pickplace_vision)
    — 50 episodes recorded **by you** in the session, with a wrist camera
    (`observation.images.wrist`, 480×640 RGB) and **varied** ball start positions.
- **The baseline.**
  [`train_blind_chunked.py`](./part2_imitation_learning/train_blind_chunked.py) — the
  tutorial's clone with action chunking (Section 4.1's upgrade #2) already applied, plus a
  normalized-time input. This exact script trained the policy that failed in front of you.
- **The harness.**
  [`deploy_blind_chunked.py`](./part2_imitation_learning/deploy_blind_chunked.py) — how any
  submission gets run on the arm: ramp to home pose, clamping to training ranges, rate
  limiting, and (for vision policies) live wrist-camera frames. Your submission will be run
  by this harness, unchanged — read its docstring for the exact interface.

The baseline's training loss is *excellent* — it predicts the demonstrated actions to about
a degree of error, on held-out data too. Sit with that fact until it bothers you: the policy
imitates the demonstrations nearly perfectly, frame by frame, and still cannot do the task.
That contradiction is the whole project.

## Dataset notes and quirks (read before training)

**Blind dataset:** the ball never moves between episodes, so position is not the problem —
timing and inputs are. One episode (16) does something atypical mid-carry; decide for
yourself whether to keep it. The wrist_roll joint never moves in any episode (a hardware
quirk — the harness handles it; your policy can treat that joint as constant).

**Vision dataset:** wrist camera only — no external view, so at the home pose the ball is
visible, and keeping it in view during the reach is part of what a policy must learn. The
scene differs from the blind dataset: the ball is deposited on the **opposite side**, and
the ball is slightly **bigger** — so nothing learned from one dataset transfers to the
other's scene; pick one per experiment. Housekeeping, all indices 0-based: episodes
**18, 28, and 31** contain some stray motion after the drop and deserve end-trimming;
episodes **42 and 49** are exemplary reference trajectories worth eyeballing first.
Trimming and dropping episodes is easy (everything is keyed by `episode_index`), and
deciding which demonstrations deserve imitation is legitimate, gradeable project work.

## Choosing your track

Free choices, all of them legitimate:

- **Either dataset.** The blind one is the cleaner scientific puzzle (the failure you
  watched, and its fix, live entirely in states and time). The vision one is the fuller
  robot-learning experience — your own demonstrations, varied ball positions, and a camera
  to feed in. You may ignore the images in the vision dataset and treat it as a second
  blind dataset if you prefer.
- **Imitation learning is the recommended route** — it is what the module taught, the
  baseline gives you a running start, and it is what we know can work. But it is not
  mandatory: anything that maps the harness's inputs to actions is a valid submission, and
  a defensible negative result beats an inscrutable positive one.
- The public tutorial dataset
  ([`lerobot/svla_so101_pickplace`](https://huggingface.co/datasets/lerobot/svla_so101_pickplace))
  remains available for prototyping, but it was recorded on someone else's rig —
  demonstrations do not transfer across scenes, so train your submission on our data.

## Diagnose before you fix

Questions worth taking seriously, in roughly the order we would ask them:

- Mid-task, the arm passes through very similar joint angles reaching *down* to the ball and
  lifting *back up* with it. What, in the policy's input, distinguishes those two moments?
  Is it enough?
- Pick two episodes and compare them at the same elapsed time. Are they doing the same thing
  seven seconds in? Should "seven seconds in" mean anything to a policy at all? Is there
  something in the data that says where an episode *actually* is in the task?
- Are all fifty demonstrations equally good? Equally alike? What does an MSE-trained network
  produce where demonstrations disagree?
- At deployment, the policy's own actions produce its next inputs, so its small mistakes
  feed back. The training data contains no such states. Can you measure that gap without
  touching a robot?

**Semi-concretely, if you take the imitation route**, expect to change
`train_blind_chunked.py` in three places, described here in words — the implementation is
yours:

1. **Re-time the episodes.** Demonstrators pause and vary pace, so "40% through the
   episode" is not the same moment of the *task* in any two episodes. The data tells you
   where each episode truly is: find, per episode, the timepoint where the gripper closes
   on the ball and the timepoint where it releases, and stretch/scale each episode's
   timeline so those events line up across all of them. The gripper channel of
   `observation.state` is where to look.
2. **Rethink the inputs.** One of the baseline's inputs is so close to the correct output,
   frame by frame, that the network can get an excellent loss by nearly copying it — and a
   policy that copies its input is a policy that stays put when deployed. Ask what the
   network *needs* to know to produce the trajectory, and give it only that.
3. **Curate the episodes.** Not all fifty demonstrations are equally alike; decide which
   ones deserve to be imitated, and defend the decision in your write-up.

For the vision dataset the same three ideas apply, plus a small convolutional encoder for
the wrist image (Section 4.1's upgrade #1) — and vision is what makes *varied* ball
positions learnable at all.

## One trap worth hunting on purpose

Try adding **the previous action** to your policy's input. Your held-out error will improve,
and your policy will get *worse* — because copying your own last action is an excellent way
to predict the next one and a terrible way to do a task.

That is [causal confusion](https://arxiv.org/abs/1905.11979), and it is exactly the
over-imitating child from Section 1 of the tutorial: faithfully copying the part of the
demonstration that had nothing to do with why it worked. And the previous action is not the
only input with this property. At 30 frames per second, ask yourself how different the
correct output can possibly be from things the policy is already handed — and what gradient
descent does with an input that nearly equals the label.

It is also a warning about your scoreboard, one the session already demonstrated on real
hardware. **Held-out prediction error is not competence.** It grades your policy frame by
frame while handing it the true state every time, so its mistakes never come home to roost —
the one thing that cannot happen on a real robot. You watched a policy with superb held-out
error grasp thin air.

## The scoreboard that actually predicts the robot

Better than held-out error: a **closed-loop rollout on your laptop**. Start from the
demonstrations' home pose, feed the policy its own predicted actions as if the servos
executed them, roll out a full episode, and compare the resulting trajectory to the
demonstrations — above all, *where is the gripper when it closes?* A policy that survives
its own feedback loop offline has earned its shot at the arm. One that only looks good with
the true state spoon-fed to it has not.

## Scope and deliverable

Four groups, twelve hours. One clear figure answering one clear question, plus a short
presentation. A negative result you can explain beats a positive result you cannot.

## Deployment day

**Every group demos on the real arm.** Submit **one `.pt` file** to Nisheet (Slack DM) by
the deadline announced on Slack, and say **which dataset you trained on** — that determines
the action clamps, the home pose, the episode length, and whether the camera is attached
for your run.

Two accepted formats:

1. **Baseline format** — the checkpoint dict that `train_blind_chunked.py` saves. You may
   change `d_in`, `hidden`, and `chunk_size`; keep the rest of the format intact.
2. **TorchScript** — required for vision policies or custom architectures. Save with
   `torch.jit.trace`; the harness calls your module as

       forward(image, state, phase) -> actions

   with `image` float32 `(1, 3, 480, 640)` RGB in [0, 1] from the live wrist camera
   (zeros if your run has no camera — blind policies must *accept* the argument and are
   free to ignore it), `state` float32 `(1, 6)` joint angles in degrees, `phase` float32
   `(1, 1)` running linearly 0 → 1 over the episode. Return float32 `(1, K, 6)`, your next
   `K ≥ 10` actions in degrees. The harness queries your module every 10 frames (~0.33 s)
   and executes the first 10 actions at 30 Hz. Do any image resizing *inside* your module.

   ```python
   example = (torch.zeros(1, 3, 480, 640), torch.zeros(1, 6), torch.zeros(1, 1))
   torch.jit.save(torch.jit.trace(model.eval(), example), "group<N>.pt")
   ```

Three rules, and they exist because a policy that has drifted off the demonstrations will
happily command a pose that damages a real motor:

- your policy sees **only what the harness provides** — the signature above is the whole
  world;
- your forward pass must fit the **~0.33 s CPU budget** — the arm cannot wait for you;
- your actions will be **clamped** to the joint ranges seen in your training dataset, and
  rate-limited. Do not rely on the clamp: a policy that needs it is not a policy that works.

Evaluation scene: blind-dataset policies get the ball at its taped spot; vision-dataset
policies get the ball at one of the taped start marks, chosen by us on the day.

**Test before you submit**: run your `.pt` through your own closed-loop rollout, and check
that `torch.jit.load("group<N>.pt")` followed by a call with the example inputs above
returns the right shape. A submission that crashes on load demos nothing.

Come and find us early with what you are trying, and we will help you scope it.
