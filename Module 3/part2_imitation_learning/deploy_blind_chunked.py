"""Run a trained policy on the SO-101 follower arm.

Two checkpoint formats are accepted:

1. **Baseline format** — what `train_blind_chunked.py` saves: a dict with the
   BlindChunkedClone state_dict plus metadata. You may change `d_in`,
   `hidden`, and `chunk_size`; the harness reconstructs the model from the
   checkpoint.

2. **TorchScript** (required for vision or custom architectures) — a module
   saved with `torch.jit.trace`/`torch.jit.save`, with the fixed signature

       forward(image, state, phase) -> actions

   - image: float32 (1, 3, 480, 640), RGB in [0, 1], the live wrist camera
     (zeros if no camera is attached). Ignore it if your policy is blind.
   - state: float32 (1, 6), joint angles in degrees.
   - phase: float32 (1, 1), linear 0 -> 1 over the episode.
   - returns: float32 (1, K, 6), the next K >= 10 actions in degrees.

   The harness calls the module every 10 frames (~0.33 s at 30 fps) and
   executes the first 10 returned actions. TorchScript checkpoints carry no
   metadata, so `--dataset` is required: action clamps, home pose, and
   episode length are derived from the training dataset.

Safety and robustness, applied to every policy:
- the run starts with a 3 s ramp to the demonstrations' home pose;
- joints whose demonstrations never moved are commanded at their observed
  state (a leader/follower offset can sit beyond a mechanical stop — our
  wrist_roll does, by 42 degrees — and the motor would strain all run);
- actions are clamped to the joint ranges seen in training, and lerobot's
  max_relative_target rate-limits every motor step;
- Ctrl+C stops cleanly and releases torque.

Usage:
    # Baseline checkpoint, dry run (full loop, prints actions, sends nothing)
    uv run python deploy_blind_chunked.py --checkpoint blind_chunked.pt \
        --port /dev/tty.usbmodem5B610326911 --dry-run

    # TorchScript submission with live wrist camera
    uv run python deploy_blind_chunked.py --checkpoint group1.pt \
        --port /dev/tty.usbmodem5B610326911 \
        --dataset nisheet0/bamb2026_so101_pickplace_vision --camera-index 0
"""

import argparse
import time

import torch
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
from lerobot.robots.so_follower.so_follower import SOFollower
from train_blind_chunked import BlindChunkedClone

FPS = 30
IMAGE_SHAPE = (1, 3, 480, 640)
JOINTS = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
]


def dataset_contract(repo_id):
    """Derive deployment metadata from a training dataset: action clamps,
    home pose, per-joint state stats, and median episode length."""
    import numpy as np
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id, video_backend="pyav")
    hf = ds.hf_dataset.with_format("numpy")
    states = np.stack(hf["observation.state"])
    actions = np.stack(hf["action"])
    ep = np.array(hf["episode_index"])
    starts = np.stack([states[ep == e][0] for e in np.unique(ep)])
    lens = [int((ep == e).sum()) for e in np.unique(ep)]
    t = lambda a: torch.tensor(a, dtype=torch.float32)  # noqa: E731
    return {
        "action_low": t(actions.min(0)), "action_high": t(actions.max(0)),
        "home": t(starts.mean(0)),
        "state_mean": t(states.mean(0)), "state_std": t(states.std(0)),
        "ep_len": int(np.median(lens)),
    }


def read_state(robot):
    obs = robot.get_observation()
    state = torch.tensor([obs[f"{j}.pos"] for j in JOINTS], dtype=torch.float32)
    image = torch.zeros(IMAGE_SHAPE)
    if "wrist" in obs:
        frame = torch.from_numpy(obs["wrist"])  # (H, W, 3) uint8 RGB
        image = frame.permute(2, 0, 1).float().div(255).unsqueeze(0)
    return state, image


def send(robot, target):
    robot.send_action({f"{j}.pos": float(target[i]) for i, j in enumerate(JOINTS)})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--port", required=True)
    p.add_argument("--robot-id", default="bamb_follower")
    p.add_argument("--seconds", type=float, default=60,
                   help="hard time limit; the run also ends by itself when "
                        "the episode length is reached")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--camera-index", type=int, default=None,
                   help="OpenCV index of the wrist camera; omit to feed zeros")
    p.add_argument("--dataset", default=None,
                   help="training dataset repo_id; required for TorchScript "
                        "checkpoints (supplies clamps, home pose, episode length)")
    args = p.parse_args()

    # Load the checkpoint, in either format.
    try:
        scripted = torch.jit.load(args.checkpoint)
        scripted.eval()
    except RuntimeError:
        scripted = None

    if scripted is not None:
        if args.dataset is None:
            raise SystemExit("TorchScript checkpoints need --dataset for the "
                             "deployment contract (clamps, home, episode length).")
        print(f"TorchScript checkpoint; contract from {args.dataset}")
        c = dataset_contract(args.dataset)
        low, high, home = c["action_low"], c["action_high"], c["home"]
        s_mean, s_std, ep_len = c["state_mean"], c["state_std"], c["ep_len"]
        n_exec = 10

        def query(state, image, frame):
            phase = torch.tensor([[min(frame / (ep_len - 1), 1.0)]])
            with torch.no_grad():
                out = scripted(image, state.unsqueeze(0), phase)
            if out.ndim != 3 or out.shape[2] != 6 or out.shape[1] < n_exec:
                raise SystemExit(f"policy returned {tuple(out.shape)}, "
                                 f"expected (1, K>={n_exec}, 6)")
            return out[0]
    else:
        ckpt = torch.load(args.checkpoint, weights_only=True)
        stats = {k: torch.zeros(ckpt["d_in"] if k.startswith("x") else ckpt["d_out"])
                 for k in ("x_mean", "x_std", "y_mean", "y_std")}
        policy = BlindChunkedClone(ckpt["d_in"], ckpt["d_out"], ckpt["chunk_size"],
                                   stats, hidden=ckpt.get("hidden", 256))
        policy.load_state_dict(ckpt["model"])
        policy.eval()
        low, high, home = ckpt["action_low"], ckpt["action_high"], ckpt["home"]
        if "state_mean" in ckpt:
            s_mean, s_std = ckpt["state_mean"], ckpt["state_std"]
        else:
            s_mean, s_std = policy.x_mean[:6], policy.x_std[:6]
        schedule = ckpt["phase_schedule"]
        ep_len = len(schedule)
        n_exec = ckpt["chunk_size"] // 2

        def query(state, image, frame):
            phase = schedule[min(frame, ep_len - 1)]
            # Available inputs are [6 joint angles, phase]; a checkpoint may
            # declare a smaller d_in to use only the trailing ones.
            x = torch.cat([state, phase.reshape(1)])[-ckpt["d_in"]:].unsqueeze(0)
            with torch.no_grad():
                return policy(x)[0]

    # Joints the demonstrations never moved: command their observed state.
    pinned = {i: s_mean[i].item() for i in range(6) if s_std[i].item() < 0.5}
    if pinned:
        print("pinned joints:", {JOINTS[i]: round(v, 1) for i, v in pinned.items()})

    cameras = {}
    if args.camera_index is not None:
        cameras["wrist"] = OpenCVCameraConfig(
            index_or_path=args.camera_index, width=640, height=480, fps=FPS
        )
    robot = SOFollower(SO101FollowerConfig(
        port=args.port, id=args.robot_id, max_relative_target=10.0,
        cameras=cameras,
    ))
    robot.connect()
    print(f"connected ({'DRY RUN — no actions sent' if args.dry_run else 'LIVE'})")

    try:
        # Ramp to the demonstrations' home pose before handing over.
        cur, _ = read_state(robot)
        print("home pose delta:", [f"{v:6.1f}" for v in (home - cur).tolist()])
        if not args.dry_run:
            n_ramp = int(3 * FPS)
            for k in range(n_ramp):
                tgt = cur + (home - cur) * (k + 1) / n_ramp
                for i, v in pinned.items():
                    tgt[i] = v
                send(robot, tgt)
                time.sleep(1 / FPS)

        frame = 0
        t_end = time.perf_counter() + args.seconds
        while time.perf_counter() < t_end and frame < ep_len:
            state, image = read_state(robot)
            chunk = torch.clamp(query(state, image, frame), low, high)
            for i, v in pinned.items():
                chunk[:, i] = v

            for step in range(n_exec):
                t_next = time.perf_counter() + 1 / FPS
                if args.dry_run:
                    if step == 0:
                        print(f"frame {frame:4d}",
                              "state ", [f"{v:7.1f}" for v in state.tolist()],
                              "action", [f"{v:7.1f}" for v in chunk[0].tolist()])
                else:
                    send(robot, chunk[step])
                frame += 1
                if time.perf_counter() > t_end or frame >= ep_len:
                    break
                time.sleep(max(0.0, t_next - time.perf_counter()))
        print(f"episode finished ({frame}/{ep_len} frames)")
    except KeyboardInterrupt:
        print("\nstopped by user")
    finally:
        robot.disconnect()
        print("disconnected, torque released")


if __name__ == "__main__":
    main()
