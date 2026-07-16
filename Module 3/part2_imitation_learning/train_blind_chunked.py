"""Train a chunked blind clone on a (camera-free) LeRobot dataset.

This is the tutorial's BlindClone with the mini-project's "upgrade #2" applied:
instead of one action per frame, the policy predicts a chunk of the next
CHUNK_SIZE actions, so per-frame errors do not compound. Its inputs are the
six joint angles plus the episode's elapsed time, normalized 0 to 1.

Trains in minutes on CPU. This exact script trained the policy you watched in
the session. Watch it closely before you trust it — it is the mini-project's
starting point, not its answer.

Usage:
    uv run python train_blind_chunked.py --repo-id <user>/so101_blind_pickplace \
        --epochs 300 --out blind_chunked.pt
"""

import argparse

import numpy as np
import torch
import torch.nn as nn
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import DataLoader, TensorDataset

SEED = 0
CHUNK_SIZE = 20  # actions predicted per query, ~0.7 s at 30 fps


class BlindChunkedClone(nn.Module):
    """MLP from (joint angles, phase) to a chunk of future actions."""

    def __init__(self, d_in, d_out, chunk_size, stats, hidden=256):
        super().__init__()
        self.chunk_size = chunk_size
        self.d_out = d_out
        for name, val in stats.items():
            self.register_buffer(name, val)
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, d_out * chunk_size),
        )

    def forward(self, s):
        out = self.net((s - self.x_mean) / self.x_std)
        out = out.view(-1, self.chunk_size, self.d_out)
        return out * self.y_std + self.y_mean


def load_chunked_arrays(repo_id, chunk_size, episodes=None):
    """Return (inputs, action_chunks, phase_schedule, home).

    Inputs are the joint state plus normalized elapsed time ("phase"). The
    phase schedule is what deployment replays phase from — here, simply the
    median episode length. Home is the mean starting pose of the episodes.
    """
    ds = LeRobotDataset(repo_id, video_backend="pyav")
    hf = ds.hf_dataset.with_format("numpy")
    states = np.stack(hf["observation.state"])
    actions = np.stack(hf["action"])
    ep_idx = np.array(hf["episode_index"])

    X, Y, ep_lens, ep_starts = [], [], [], []
    keep = np.unique(ep_idx) if episodes is None else np.array(episodes)
    for ep in keep:
        s, a = states[ep_idx == ep], actions[ep_idx == ep]
        ep_lens.append(len(s))
        ep_starts.append(s[0])
        ph = np.linspace(0, 1, len(s))
        for t in range(len(s) - chunk_size):
            X.append(np.append(s[t], ph[t]))
            Y.append(a[t : t + chunk_size])

    schedule = np.linspace(0, 1, int(np.median(ep_lens)))
    home = torch.tensor(np.mean(ep_starts, axis=0), dtype=torch.float32)
    X = torch.tensor(np.array(X), dtype=torch.float32)
    Y = torch.tensor(np.array(Y))
    return X, Y, torch.tensor(schedule, dtype=torch.float32), home


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", required=True)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--out", default="blind_chunked.pt")
    p.add_argument("--episodes", default=None,
                   help="comma-separated episode indices to train on (default: all)")
    args = p.parse_args()

    episodes = None if args.episodes is None else [
        int(e) for e in args.episodes.split(",")
    ]
    torch.manual_seed(SEED)
    X, Y, schedule, home = load_chunked_arrays(args.repo_id, CHUNK_SIZE, episodes)
    print(
        f"{len(X)} training pairs: "
        f"state {tuple(X.shape[1:])} -> chunk {tuple(Y.shape[1:])}"
    )

    stats = {
        "x_mean": X.mean(0), "x_std": X.std(0).clamp(min=1e-4),
        "y_mean": Y.mean((0, 1)), "y_std": Y.std((0, 1)).clamp(min=1e-4),
    }
    model = BlindChunkedClone(X.shape[1], Y.shape[2], CHUNK_SIZE, stats)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    loader = DataLoader(TensorDataset(X, Y), batch_size=256, shuffle=True)

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(xb)
        if epoch % 20 == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch:4d}  loss {epoch_loss / len(loader.dataset):.5f}")

    # Save the action ranges seen in training: deployment clamps to these.
    action_low = Y.reshape(-1, Y.shape[2]).min(0).values
    action_high = Y.reshape(-1, Y.shape[2]).max(0).values
    torch.save(
        {
            "model": model.state_dict(),
            "d_in": X.shape[1], "d_out": Y.shape[2], "chunk_size": CHUNK_SIZE,
            "action_low": action_low, "action_high": action_high,
            "phase_schedule": schedule,
            # Mean starting pose of the training episodes: deployment ramps
            # to it first, so the policy never starts off-distribution.
            "home": home,
        },
        args.out,
    )
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
