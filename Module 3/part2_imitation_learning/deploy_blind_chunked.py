"""Run a trained BlindChunkedClone on the SO-101 follower arm. No camera needed.

The policy predicts a chunk of future actions from the current joint angles
and the task phase; we execute the first half of each chunk, then re-plan from
the live state. Phase advances along the schedule saved at training time
(median pace of each task segment).

Safety and robustness:
- the run starts with a 3 s ramp to the demonstrations' home pose, so the
  policy never starts from a state it has not seen;
- joints whose demonstrations never moved are commanded at their observed
  state (the leader's value for such a joint can sit beyond a mechanical stop
  — our wrist_roll does, by 42 degrees — and the motor would strain against
  it for the whole run, eventually tripping its overload protection);
- actions are clamped to the joint ranges seen in training, and lerobot's
  max_relative_target rate-limits every motor step;
- Ctrl+C stops cleanly and releases torque.

Usage:
    # First: full loop, prints actions, arm holds still
    uv run python deploy_blind_chunked.py --checkpoint blind_chunked.pt \
        --port /dev/tty.usbmodem5B610326911 --dry-run

    # Then: the real thing (hand near the power switch the first time)
    uv run python deploy_blind_chunked.py --checkpoint blind_chunked.pt \
        --port /dev/tty.usbmodem5B610326911
"""

import argparse
import time

import torch
from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
from lerobot.robots.so_follower.so_follower import SOFollower
from train_blind_chunked import BlindChunkedClone

FPS = 30
JOINTS = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
]


def read_state(robot):
    obs = robot.get_observation()
    return torch.tensor([obs[f"{j}.pos"] for j in JOINTS], dtype=torch.float32)


def send(robot, target):
    robot.send_action({f"{j}.pos": float(target[i]) for i, j in enumerate(JOINTS)})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--port", required=True)
    p.add_argument("--robot-id", default="bamb_follower")
    p.add_argument("--seconds", type=float, default=60,
                   help="hard time limit; the run also ends by itself when "
                        "the phase schedule completes")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, weights_only=True)
    stats = {k: torch.zeros(ckpt["d_in"] if k.startswith("x") else ckpt["d_out"])
             for k in ("x_mean", "x_std", "y_mean", "y_std")}
    policy = BlindChunkedClone(ckpt["d_in"], ckpt["d_out"], ckpt["chunk_size"], stats)
    policy.load_state_dict(ckpt["model"])
    policy.eval()
    low, high = ckpt["action_low"], ckpt["action_high"]
    schedule = ckpt["phase_schedule"]
    home = ckpt["home"]

    # Joints the demonstrations never moved: command their observed state.
    x_mean, x_std = policy.x_mean[:6], policy.x_std[:6]
    pinned = {i: x_mean[i].item() for i in range(6) if x_std[i].item() < 0.5}
    if pinned:
        print("pinned joints:", {JOINTS[i]: round(v, 1) for i, v in pinned.items()})

    robot = SOFollower(SO101FollowerConfig(
        port=args.port, id=args.robot_id, max_relative_target=10.0,
    ))
    robot.connect()
    print(f"connected ({'DRY RUN — no actions sent' if args.dry_run else 'LIVE'})")

    try:
        # Ramp to the demonstrations' home pose before handing over.
        cur = read_state(robot)
        print("home pose delta:", [f"{v:6.1f}" for v in (home - cur).tolist()])
        if not args.dry_run:
            n_ramp = int(3 * FPS)
            for k in range(n_ramp):
                tgt = cur + (home - cur) * (k + 1) / n_ramp
                for i, v in pinned.items():
                    tgt[i] = v
                send(robot, tgt)
                time.sleep(1 / FPS)

        ep_len = len(schedule)
        n_exec = ckpt["chunk_size"] // 2  # execute half the chunk, then re-plan
        frame = 0
        t_end = time.perf_counter() + args.seconds
        while time.perf_counter() < t_end and frame < ep_len:
            state = read_state(robot)
            phase = schedule[min(frame, ep_len - 1)]
            x = torch.cat([state, phase.reshape(1)]).unsqueeze(0)
            with torch.no_grad():
                chunk = policy(x)[0]  # (chunk_size, 6)
            chunk = torch.clamp(chunk, low, high)
            for i, v in pinned.items():
                chunk[:, i] = v

            for step in range(n_exec):
                t_next = time.perf_counter() + 1 / FPS
                if args.dry_run:
                    if step == 0:
                        print(f"phase {phase:.2f}",
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
