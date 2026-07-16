# Part 2: Imitation learning on a real robot arm

Welcome to the second part of Module 3. In part 1, our agent learned to act by trial and error, and it did so by failing: it fell into the frozen lake and walked off the cliff hundreds of times before the value of not doing that finally sank in. That is a luxury. A chimpanzee cannot afford to learn which branch holds by falling out of the tree, and a robot arm that costs real money cannot afford to learn to grasp by flinging itself at the table ten thousand times. Neither of them usually gets a reward signal to learn from either.

So in this part we take the reward away. Instead, a human demonstrates the task, and we fit a policy to the demonstrated behaviour. This is **imitation learning**, and its simplest form, **behavioural cloning**, turns out to be something you already know how to do: it is maximum-likelihood model fitting on behavioural data, exactly as in Module 2, except that the participant is a robot and the choice is a six-dimensional continuous action.

You will be working with [`tutorial_3b.ipynb`](./tutorial_3b.ipynb), the jupyter notebook with all instructions and code. It is short and simple: look at real demonstration data, fit a policy to it on your laptop, and then watch what happens when a policy trained exactly that way drives the real arm at the front of the room. What happens is instructive.

The [mini-project](../mini_projects.md) picks up from there: fix the policy that just failed in front of you, and the best few fixes in the class get deployed on the arm on the last day.

## What you need

Nothing but a laptop and a browser. The demonstration data streams from the [Hugging Face Hub](https://huggingface.co/datasets/lerobot/svla_so101_pickplace), everything in the tutorial trains on CPU in a couple of minutes, and the arm itself only ever appears at the front of the room. You do not need hardware to do the tutorial or the mini-project; if your policy is picked for deployment, we run it on the arm for you.

## The SO-101

The [SO-101](https://huggingface.co/docs/lerobot/so101) is a small, open-source, 3D-printable robot arm with six degrees of freedom, developed by [The Robot Studio](https://github.com/TheRobotStudio/SO-ARM100) together with the [Hugging Face LeRobot](https://github.com/huggingface/lerobot) team. It is cheap enough that a lab can own several, which is exactly why it has become the standard arm for teaching and for open robot learning research.

It comes as a pair:

- the **leader** arm, which you move by hand to demonstrate the task, and
- the **follower** arm, which mirrors the leader while you record, and which is driven by a learned policy once you have one.

That mirroring is the whole trick, and it is worth pausing on. While a human demonstrates, the follower is producing precisely the `(observation, action)` pairs that a policy needs to learn from: the observation is what the camera and the joint encoders report, and the action is whatever the leader just commanded. Teleoperation is a data collection device disguised as a toy.

The software that ties all of this together is [LeRobot](https://huggingface.co/docs/lerobot), Hugging Face's library for real-world robot learning. It gives you a common dataset format, a set of policies (behavioural cloning, ACT, diffusion policies, and others), and command line tools to teleoperate, record, train, and deploy.

### Resources

- [LeRobot documentation](https://huggingface.co/docs/lerobot), the place to start
- [Getting started with the SO-101](https://huggingface.co/docs/lerobot/so101), including assembly and calibration
- [Imitation learning on real robots](https://huggingface.co/docs/lerobot/il_robots), the full pipeline from teleoperation to a deployed policy, and the guide to follow if you ever get your hands on an arm
- [Setting up cameras](https://huggingface.co/docs/lerobot/cameras)
- [SO-ARM100 hardware repository](https://github.com/TheRobotStudio/SO-ARM100), with the bill of materials and the parts to print
- [The LeRobot dataset format](https://huggingface.co/blog/lerobot-datasets), and what makes a dataset a good one
- [ACT, the Action Chunking Transformer](https://arxiv.org/abs/2304.13705) (Zhao et al., 2023), the policy most people reach for on this arm
- [Causal confusion in imitation learning](https://arxiv.org/abs/1905.11979) (de Haan et al., 2019), on what goes wrong when a policy copies the wrong thing
- [The LeRobot Discord](https://discord.com/invite/s3KuuzsPFb), which is friendly and unusually responsive

## Running the notebook

We recommend Colab. Nothing here needs a GPU, the demonstration data streams from the Hub, and the first cell installs what it needs.

If you would rather run it locally, LeRobot is an **optional dependency group** of the Module 3 environment. It is not installed by default, because it depends on torch, OpenCV, rerun and a video stack, roughly a gigabyte of it, and nobody doing part 1 needs any of that. Ask for it by name:

```sh
# from the "Module 3/" folder, not this one
uv sync --group robot
uv run jupyter lab part2_imitation_learning/tutorial_3b.ipynb
```

That installs the exact LeRobot version this tutorial is tested against, on top of the environment you already built for part 1. It takes well under a minute.

> [!WARNING]
> If somebody tells you to install `lerobot==0.5.2`, it will fail: that version has never been published to PyPI. It is the in-development version number reported by a git checkout, so an announced 0.5.2 always came from source. The published releases go 0.5.0, 0.5.1, then 0.6.0. We pin **0.5.1**, the last release of the 0.5 series, and that is the version everything here has been run against.

### A note on video decoding

The notebook passes `video_backend="pyav"` when it loads the dataset, and that is not incidental. LeRobot's default decoder, torchcodec, links against FFmpeg's shared libraries, which have to be installed system-wide and are simply absent on most laptops. pyav brings its own copy of FFmpeg and ships with LeRobot, so this one keyword argument is what lets the notebook run on thirty machines without anybody installing anything.

## For instructors

On the day, the tutorial needs three things: an assembled and calibrated SO-101 pair on the demo laptop, both arms clamped to the table, wifi for the students, and one pinned LeRobot version announced to everybody. Test the arm the day before, not on the morning.

Four commands are worth having in your fingers, all documented in the [LeRobot walkthrough](https://huggingface.co/docs/lerobot/il_robots): `lerobot-teleoperate`, `lerobot-record` (which both records demonstrations and, with `--policy.path`, runs a trained policy), `lerobot-train`, and `lerobot-replay`.

### Prepare this before the school

The finale of the session (Section 5) is a policy trained by imitation learning on our own arm — and it **fails on purpose**. The pipeline is deliberately camera-free (no webcam, no USB hub, no camera pose to protect), and the blind policy's confident, wrong grasp is the cliffhanger the mini-project resolves: the students' job is to fix it. This makes the live demo nearly risk-free — failure is the expected outcome, and the working contrast is `lerobot-replay` of a recorded episode, guaranteed by construction. Four jobs, done the day before:

1. **Clamp the arm base and tape one cube start position and the bin position.** Geometry relative to the base is the only thing a blind policy knows, so the tape is the whole contract between recording and the demo.
2. **Record ~50 pick-and-place episodes on our arm** with `lerobot-record` and no `--robot.cameras` flag, the cube starting on its tape every episode — about half an hour. Push to the Hub afterwards and announce the id: it is the mini-project dataset.
3. **Train the baseline** with [`train_blind_chunked.py`](./train_blind_chunked.py) (the notebook's blind clone plus action chunking — minutes on a laptop CPU, no GPU), then verify with [`deploy_blind_chunked.py`](./deploy_blind_chunked.py): dry-run first, then one live run so you know its exact failure before the room does.
4. **Verify the project is solvable.** Keep a working reference solution *outside the repository* — never commit it, inventing it is the project — and confirm it beats the baseline in closed-loop rollout.

The original camera-and-ACT pipeline is documented in [`robot_setup.md`](./robot_setup.md), for a future edition with a webcam and a GPU. Note for that path: lerobot 0.5.1's ACT refuses to train on an image-free dataset, which is why the blind pipeline has its own training script.

**Fallback**, if the trained policy is not ready or misbehaves on the day: teleoperate live, then `lerobot-replay` a recorded episode. Less spectacular, still makes the point. Replaying an episode from the *public* dataset is a nice bonus lesson in its own right — the arm reproduces the motion perfectly and grasps thin air, because their cube was somewhere else.

Two more things reliably go wrong:

- **The USB ports.** Linux hands out `/dev/ttyACM0` and `/dev/ttyACM1` in the order the devices happen to enumerate, so which one is the leader and which is the follower can change between reboots and will certainly differ on somebody else's laptop. Do not trust the numbering, and do not hardcode it: check every time, or use the stable `/dev/serial/by-id/` paths. On macOS the ports are `/dev/tty.usbmodem*` instead.
- **Calibration.** Calibration lives in a file per arm, keyed by the id you pass. If you run a command with an id that has never been calibrated, LeRobot will cheerfully start a full recalibration in front of the room. Calibrate both arms once, under the `bamb_leader` and `bamb_follower` ids that the notebook uses, and keep those files.

## Further reading: learning by watching

Section 1 of the notebook is a prompt to talk about how animals actually learn, rather than how our Q-learning agent does. These are the studies behind it.

- **Chimpanzees of the Bugoma Forest, Uganda.** The video shown in the session. Preprint: <https://www.biorxiv.org/content/10.64898/2025.12.01.691567v1>
- **Boesch (1991)**, [Teaching among wild chimpanzees](https://doi.org/10.1016/S0003-3472(05)80857-7). Mothers in the Tai forest occasionally leave a hammer on the anvil for an infant, and very occasionally correct its grip. Nut cracking takes years to learn and is learned by watching.
- **Kawai (1965)**, [Newly acquired pre-cultural behavior of the natural troop of Japanese monkeys on Koshima islet](https://doi.org/10.1007/BF01794457). Imo, a young macaque, starts washing sand off sweet potatoes in 1953. The habit spreads through the troop along social ties, youngsters first.
- **Allen, Weinrich, Hoppitt and Rendell (2013)**, [Network-based diffusion analysis reveals cultural transmission of lobtail feeding in humpback whales](https://doi.org/10.1126/science.1231976). A feeding technique spreads through a whale population over 27 years, and it travels along the social network rather than by genes or by chance. As quantitative a demonstration of social learning in the wild as you will find.
- **Thornton and McAuliffe (2006)**, [Teaching in wild meerkats](https://doi.org/10.1126/science.1128727). Adults give pups dead scorpions, then live ones with the sting removed, then intact ones. A curriculum, adjusted to the age of the pup.
- **Horner and Whiten (2005)**, [Causal knowledge and imitation/emulation switching in chimpanzees and children](https://doi.org/10.1007/s10071-004-0239-6), and **Lyons, Young and Keil (2007)**, [The hidden structure of overimitation](https://doi.org/10.1073/pnas.0704452104). Children copy the causally irrelevant steps of a demonstration; chimpanzees skip them. Over-imitation looks like a bug and is probably a feature, and it has a direct analogue in what goes wrong with behavioural cloning.
