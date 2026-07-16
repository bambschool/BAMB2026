# SO-101 prep for BAMB Module 3

Everything here is against **LeRobot 0.5.1**, which is the version pinned for the school. All
flags below have been checked against that version.

> **What we actually ran (July 2026): the camera-free variant.** No camera could be attached
> (two USB-C ports, two arms), so the finale policy is *blind*: the tutorial's clone plus
> action chunking, trained by [`train_blind_chunked.py`](./train_blind_chunked.py) and deployed
> by [`deploy_blind_chunked.py`](./deploy_blind_chunked.py), on ~50 camera-free episodes with
> the cube at a **single taped position**. The baseline **fails on the arm by design** — that
> failure is the session's finale and fixing it is the mini-project, so the instructors'
> working reference solution lives outside the repository (see the `.gitignore` in this
> folder) and must never be committed. Facts worth keeping: `lerobot-record` happily records
> without cameras, but **`lerobot-train` with `--policy.type=act` refuses an image-free
> dataset** ("You must provide at least one image..."), hence the custom training script; and
> if a follower joint sits pinned tens of degrees from its commanded value for a whole
> session (ours: wrist_roll, 42.5° — a leader/follower offset against a mechanical stop),
> the motor strains the entire time and eventually trips its **overload protection**, which
> latches until a power cycle. The deploy script pins such never-moved joints to their
> observed state for exactly this reason. The rest of this document is the with-camera
> pipeline, kept for a future edition with a webcam and a GPU. Calibration, ports, teleop
> and replay sections apply to both variants unchanged.

## What we need, and why

**One deliverable: ~50 teleoperated demonstrations recorded on *our* arm, pushed to the Hub.**

It does double duty:

1. It trains the policy for the **live finale** of the session — imitation learning driving the
   arm by itself, no human touching the leader.
2. It is the dataset **all 30 students train on** for the mini-project, whose prize is that the
   best few policies get deployed on the arm on the last day.

There is no way around recording it ourselves. A policy trained on a public dataset cannot work
on our arm — see [Why we cannot just use data from the web](#why-we-cannot-just-use-data-from-the-web)
at the bottom, which is worth reading before deciding anything.

## Decisions we need to make

| | |
|---|---|
| **Hub repo id for the dataset** | proposed: `bambschool/so101_pickplace` |
| **Hub repo id for the policy** | proposed: `bambschool/act_so101_pickplace` |
| **The task** | pick-and-place matches the tutorial and looks great, but **grasping is where imitation policies fail most often**. A push-into-a-zone task is far more forgiving. Suggestion: try pick-and-place, and downgrade if the trained policy turns out flaky. |
| **Cameras** | we currently have **only a wrist camera**. That is workable, but it has one specific failure mode — read [Working with only a wrist camera](#working-with-only-a-wrist-camera) before recording. Getting a second camera is the cheapest risk reduction available to us. |
| **GPU for training** | free Colab works but is slow; a lab GPU is much better |
| **Deadline** | dataset on the Hub before the school; policy trained and verified before the Module 3 session |

---

## Step 0 — Hardware setup (once)

```bash
lerobot-find-port          # run twice, unplugging one arm in between, to learn which is which

lerobot-setup-motors --robot.type=so101_follower --robot.port=/dev/ttyACM0

lerobot-calibrate --robot.type=so101_follower  --robot.port=/dev/ttyACM0 --robot.id=bamb_follower
lerobot-calibrate --teleop.type=so101_leader   --teleop.port=/dev/ttyACM1 --teleop.id=bamb_leader

lerobot-find-cameras       # note the camera index
```

> **Use the ids `bamb_follower` and `bamb_leader`.** They are hardcoded in the tutorial notebook.
> Calibrate once and **keep the calibration files**. If you run any command with an id that has
> never been calibrated, LeRobot cheerfully starts a full recalibration — which is a bad thing to
> discover in front of thirty people.

## Step 1 — Sanity-check teleoperation

```bash
lerobot-teleoperate \
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=bamb_follower \
  --teleop.type=so101_leader  --teleop.port=/dev/ttyACM1 --teleop.id=bamb_leader \
  --robot.cameras="{ wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
  --display_data=true
```

## Step 2 — Record the dataset (~25 minutes of actual work)

```bash
lerobot-record \
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=bamb_follower \
  --teleop.type=so101_leader  --teleop.port=/dev/ttyACM1 --teleop.id=bamb_leader \
  --robot.cameras="{ wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
  --dataset.repo_id=bambschool/so101_pickplace \
  --dataset.single_task="Pick up the cube and place it in the bin" \
  --dataset.num_episodes=50 \
  --dataset.episode_time_s=20 \
  --dataset.reset_time_s=10 \
  --dataset.push_to_hub=true
```

> **Set `episode_time_s` and `reset_time_s` explicitly.** They both **default to 60 seconds**,
> which would turn this into a 100-minute slog. At 20 s / 10 s it is about 25 minutes.

### The one thing that is easy to get wrong and is fatal

**Vary the cube's start position across episodes.**

If every demonstration starts with the cube in the same spot, the policy never learns to *look* —
it just memorises one trajectory. It will appear to work, and then collapse the instant we move
the cube in front of the room. Spread the cube across the workspace, and **mark the positions with
tape** so they are reproducible on demo day.

With a wrist-only camera there is a constraint on *how* you spread them — see the next section.

## Working with only a wrist camera

A wrist camera is real vision, not a consolation prize. Wrist-only imitation learning is a
standard, workable setup, and for the grasp itself it is arguably the *best* single camera: it
sees the cube up close exactly as the gripper closes on it.

It has one specific weakness, and it is the thing to design around:

> **At the home pose, can the wrist camera see the cube?**

If the arm starts somewhere the cube is outside the wrist camera's field of view, then at the
first frame the policy has **no information about where the cube is**. It cannot choose which way
to reach, so it reaches for the average cube and misses. The wrist camera only begins to help once
the arm is already pointed roughly the right way — which is too late.

### The fix, which is free

1. **Pick a home pose where the wrist camera looks out over the workspace** — arm raised, gripper
   angled down at the table — so the cube is in view from the very first frame.
2. **Keep the cube's start positions inside that initial field of view.** Still vary them (see
   above, this is non-negotiable), but vary them *within the region the camera can see at home*.
3. **Test this before recording anything.** Run `lerobot-teleoperate --display_data=true`, park
   the arm at the intended home pose, and put the cube at each extreme of where you plan to place
   it. Is it visible in the wrist feed every time?

That third step takes five minutes and **it determines whether the whole thing works.** Do it first.

### Seriously consider a second camera

Any USB webcam is just another `--robot.cameras` entry. Two options:

- **A £15–25 USB webcam.** The highest-leverage money in this project. Wrist + front is the
  standard robust configuration and it removes the failure mode above completely.
- **Free, today: use the laptop's built-in webcam as the front camera**, with the laptop propped so
  it sees the workspace. Janky but real. The only hard requirement is that it **cannot move**
  between recording and deployment — so tape or clamp it, and do not close the lid.

```bash
# two cameras, if we get a second one
--robot.cameras="{ wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, \
                   front: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}"
```

**Decide this before recording.** Dropping a camera later is trivial; adding one means recording
everything again. If there is any chance of a second camera, record with it from the start.

## Step 3 — Train ACT

```bash
lerobot-train \
  --dataset.repo_id=bambschool/so101_pickplace \
  --policy.type=act \
  --policy.repo_id=bambschool/act_so101_pickplace \
  --policy.device=cuda \
  --output_dir=outputs/train/act_so101 \
  --job_name=act_so101 \
  --steps=100000 \
  --batch_size=8 \
  --policy.push_to_hub=true
```

> **Do not wait for step 100,000 to find out whether it works.** Checkpoints are saved every
> 20,000 steps. Test the 20k and 40k checkpoints on the arm — ACT on 50 episodes is often decent
> well before the end, and we need to know *early* whether the task is too hard, because the fix
> (an easier task) costs another hour of recording.

## Step 4 — The finale: the policy drives the arm

This is what gets run live in the session.

```bash
lerobot-record \
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=bamb_follower \
  --robot.cameras="{ wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
  --dataset.repo_id=bambschool/eval_act_so101 \
  --dataset.single_task="Pick up the cube and place it in the bin" \
  --dataset.num_episodes=5 \
  --policy.path=bambschool/act_so101_pickplace
```

Note there is **no `--teleop`** here. That is the whole point: nobody is touching the leader arm.

---

## Things that reliably go wrong

- **Do not let the cameras move between recording and deployment.** If the camera pose shifts,
  every vision policy — ours and all thirty students' — dies at once, and nobody will be able to
  work out why. This is the single most likely thing to ruin the demo.
  - For the **wrist camera**: make sure it is *rigidly* mounted and cannot rotate or slip on its
    bracket. Check the screws before recording, and again before the session.
  - For any **external camera**: tape or clamp it down and never touch it again.
  - Also keep the **table, lighting and background** the same. The policy sees all of it.
- **USB ports renumber between reboots.** `/dev/ttyACM0` and `/dev/ttyACM1` are handed out in
  enumeration order, so leader and follower can swap. Do not hardcode them: check every time, or
  use the stable `/dev/serial/by-id/` paths. (On macOS they are `/dev/tty.usbmodem*`.)
- **Calibration is per-id.** See the warning in Step 0.

---

## Why we cannot just use data from the web

The obvious shortcut is: skip recording, grab a public SO-101 dataset (or a policy trained on
one), and just place our cube where the arm expects it. It is worth being precise about why this
works for one thing and not the other, because the two cases are completely different.

**What `lerobot-calibrate` fixes:** it sets the homing offsets and joint ranges so that raw motor
ticks map onto the standard SO-101 joint-angle convention. Two properly calibrated SO-101s should
therefore agree, roughly, about what "joint 2 at 45°" means. Calibration fixes the
**proprioceptive** mismatch between their arm and ours.

**What calibration does *not* fix:** the camera position, the camera angle, the lighting, the
table, the background, the cube, the bin. Calibration does nothing whatsoever about the
**visual** mismatch.

That single distinction decides both cases.

### Replaying a recorded episode: yes, this works

`lerobot-replay` plays back a recorded joint trajectory open-loop. **It ignores the camera
entirely.** So the visual mismatch is irrelevant, and calibration is exactly enough.

```bash
lerobot-replay --robot.type=so101_follower --robot.port=/dev/ttyACM0 \
  --robot.id=bamb_follower --dataset.repo_id=lerobot/svla_so101_pickplace --dataset.episode=0
```

And yes — **you can absolutely place the cube where it will be grabbed.** Do not try to reproduce
their scene. Just run the replay, watch where *our* gripper closes, put the cube there, and run it
again. Ten minutes of fiddling and it will pick up the cube.

Be careful the first time: run it with a hand near the e-stop and nothing fragile on the table.
Residual calibration differences (servo horns mount on a splined shaft, so they are quantised to a
few degrees) can shift the trajectory enough to hit the table.

**But be honest about what it is.** This is a *recorded human demonstration played back through
the motors*. It is not a learned policy — it does not see anything, it cannot adapt, and if you
move the cube one centimetre it will grasp thin air. That last part is actually a *great* teaching
moment, and the tutorial already uses it: it is the cleanest possible demonstration of why the
students need a dataset from our arm.

### Deploying a policy trained on public data: no, this cannot work

A trained ACT policy takes **camera images** as input. Trained on their scene, it will see our
table, our lighting, our camera angle — all far outside anything in its training data — and emit
garbage. You cannot fix this by moving the cube, because **the problem is not where the cube is,
it is that the entire image is from a different room.**

In our case it is worse than that, and not even close. The tutorial's public dataset
(`lerobot/svla_so101_pickplace`) was recorded on an **SO-100**, with **two external cameras**
(`observation.images.up` and `observation.images.side`) and **no wrist camera at all**. We have a
wrist camera and no external one. The observation spaces do not even have the same keys, so a
policy trained on their data cannot physically be fed our data. This is not a tuning problem.

There is no cheap repair. Recreating their camera rig, pose and lighting well enough to fool a CNN
is far harder than just recording 50 episodes ourselves.

### So the ranking is

1. **Record on our arm and train ACT** (Steps 2–3). Real imitation learning, robust, works, and it
   is the only option that also serves the students' mini-project. **This is the plan.**
2. **Fallback: teleoperate live, then `lerobot-replay` a public episode** with the cube placed
   empirically. Costs nothing, no GPU, no recording, and it will grab the cube. Weaker — it is
   playback, not learning — but it is a guaranteed-safe demo and the tutorial is already written
   to accommodate it.
3. **Teleoperation alone.** Always works, zero risk, least impressive.

We should do (1), and have (2) tested and ready as insurance regardless.
