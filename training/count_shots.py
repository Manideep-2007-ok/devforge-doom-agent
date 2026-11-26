"""Count wasted shots: how often a brain fires with no enemy in its crosshair.

    .venv/bin/python training/count_shots.py submission/weights/model.pt runs/run2/ckpt/ckpt_X.pt

Each brain plays the same held-out maps with the same dice. Uses the same "enemy ahead" test
as the training reward (rewards.enemy_ahead).
"""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "submission"), str(ROOT / "training"), str(ROOT / "env")]

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)

from agent import DoomAgent  # noqa: E402
from doom_env import make_env  # noqa: E402
from features import ACTIONS  # noqa: E402
from ppo import map_ticks  # noqa: E402
from rewards import enemy_ahead  # noqa: E402
from watch import load_brain  # noqa: E402

ATTACK = 6


def fired(action) -> bool:
    a = np.asarray(action)
    row = ACTIONS[int(a)] if a.ndim == 0 else a
    return bool(row[ATTACK])


def run(ckpt: str, maps: list[str]) -> dict:
    tot = dict(steps=0, shots=0, wasted=0)
    for i, wad in enumerate(maps):
        env = make_env(wad, "MAP01", episode_ticks=map_ticks(wad, 2100), render_buffers=True)
        obs, _ = env.reset(seed=i)
        agent = DoomAgent(env.observation_space, env.action_space, "cpu")
        agent.model = load_brain(ckpt)
        agent.reset()
        torch.manual_seed(i)
        np.random.seed(i)
        while True:
            ahead = enemy_ahead(obs)
            action = agent.act(obs)
            if fired(action):
                tot["shots"] += 1
                tot["wasted"] += not ahead
            obs, _, term, trunc, _ = env.step(action)
            tot["steps"] += 1
            if term or trunc:
                break
        env.close()
    return tot


if __name__ == "__main__":
    maps = sorted(glob.glob(str(ROOT / "maps/gen/heldout/*/*.wad")))[:12]
    for ck in sys.argv[1:]:
        t = run(ck, maps)
        per100 = 100 * t["wasted"] / max(t["steps"], 1)
        print(f"{Path(ck).stem:22s} wasted {t['wasted']:4d} of {t['shots']:4d} shots "
              f"({100 * t['wasted'] / max(t['shots'], 1):3.0f}%)  = {per100:4.1f} wasted per 100 moves", flush=True)
