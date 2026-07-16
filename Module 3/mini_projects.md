# Module 3 mini-project

There is one project, and everyone does it.

> **In the session, you watched a policy trained on fifty real demonstrations fail on the
> very arm that produced them. Fix it.**
> **On the last day, the best fixes get deployed on the real arm, live.**

## What you inherit

- **The dataset.** ~50 camera-free pick-and-place episodes we teleoperated on our arm, the
  cube starting from one fixed, taped position every time. The Hub id is announced in the
  session. Everything loads through `LeRobotDataset`, exactly as in the tutorial.
- **The baseline.**
  [`train_blind_chunked.py`](./part2_imitation_learning/train_blind_chunked.py) — the
  tutorial's clone with action chunking (Section 4.1's upgrade #2) already applied, plus a
  normalized-time input. This exact script trained the policy that failed in front of you.
- **The harness.**
  [`deploy_blind_chunked.py`](./part2_imitation_learning/deploy_blind_chunked.py) — how any
  checkpoint gets run on the arm: ramp to home pose, clamping to training ranges, rate
  limiting. Your submission will be run by this harness, unchanged.

The baseline's training loss is *excellent* — it predicts the demonstrated actions to about
a degree of error, on held-out data too. Sit with that fact until it bothers you: the policy
imitates the demonstrations nearly perfectly, frame by frame, and still cannot do the task.
That contradiction is the whole project.

## Diagnose before you fix

Questions worth taking seriously, in roughly the order we would ask them:

- Mid-task, the arm passes through very similar joint angles reaching *down* to the cube and
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

Any one of these, followed carefully, leads to a real fix. Several of them compose.

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
its own feedback loop offline has earned a shot at the arm. One that only looks good with
the true state spoon-fed to it has not.

## Optional second front: vision

Our dataset has no images (the deployment rig has no camera), so the fix above is a
state-and-time problem. If your group wants the vision upgrade from Section 4.1, use the
public tutorial dataset
[`lerobot/svla_so101_pickplace`](https://huggingface.co/datasets/lerobot/svla_so101_pickplace)
— it has camera frames and *varied* cube positions, which is exactly when vision earns its
keep. A vision policy cannot be deployed on our rig, so it is judged on held-out episodes
and on the quality of your analysis; a careful figure showing *when* eyes help and when they
do not is a first-rate deliverable.

## Scope and deliverable

Twelve hours, three of you. One clear figure answering one clear question, plus a short
presentation. A negative result you can explain beats a positive result you cannot.

## Deployment day

Submit a trained policy by the deadline announced in the session. We cannot run thirty
policies on one arm, so we will pick **the best few** — judged by your offline rollouts —
and run those in front of everyone on the last day.

Three rules, and they exist because a policy that has drifted off the demonstrations will
happily command a pose that damages a real motor:

- your policy's input is the **six joint angles plus phase** — that is everything the
  deployment harness can provide (there is no camera on the rig);
- your policy must run at **15 Hz or better on CPU** — the arm cannot wait for you;
- your actions will be **clamped** to the joint ranges seen in the training data, and
  rate-limited. Do not rely on the clamp: a policy that needs it is not a policy that works.

The easiest interface to hand us is a checkpoint the harness already understands: state in,
chunk of actions out, like the baseline's. If yours differs, come talk to us before the
deadline — early, with what you are trying, and we will help you scope it.
