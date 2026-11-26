"""Watch the brain play, in a window, while training runs.

Each round it loads the NEWEST saved checkpoint, so you see it get better over time.
It runs on the CPU with 1 thread so it does not slow training down much.

    .venv/bin/python training/watch.py                 # our held-out test maps
    .venv/bin/python training/watch.py --outside       # real Freedoom levels
    .venv/bin/python training/watch.py --ckpt runs/run1/ckpt/best.pt

Keys: N = next map, Space = pause, Esc = quit.
"""

from __future__ import annotations

import argparse
import glob
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "submission"), str(ROOT / "training"), str(ROOT / "env")]

import numpy as np  # noqa: E402
import pygame  # noqa: E402
import torch  # noqa: E402

from agent import DoomAgent  # noqa: E402
from doom_env import make_env  # noqa: E402
from doom_net import DoomNet  # noqa: E402
from ppo import map_ticks  # noqa: E402
from score import outside_maps  # noqa: E402

SCALE = 5


def newest(run: str) -> str:
    cks = sorted(glob.glob(str(ROOT / run / "ckpt" / "ckpt_*.pt")))
    return cks[-1] if cks else str(ROOT / "submission/weights/model.pt")


def load_brain(path: str) -> DoomNet:
    state = torch.load(path, map_location="cpu", weights_only=False)
    weights = state["net"] if "net" in state else state["model"]
    net = DoomNet()
    net.load_state_dict({k: v for k, v in weights.items() if not k.startswith("value.")}, strict=False)
    return net.eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/run1", help="training run whose newest save to watch")
    ap.add_argument("--ckpt", default="", help="watch this exact save instead of the newest")
    ap.add_argument("--outside", action="store_true", help="real Freedoom levels instead of our test maps")
    ap.add_argument("--fps", type=int, default=9, help="decisions per second (9 = real game speed)")
    args = ap.parse_args()
    torch.set_num_threads(1)

    maps = outside_maps() if args.outside else sorted(glob.glob(str(ROOT / "maps/gen/heldout/*/*.wad")))
    random.shuffle(maps)
    pygame.init()
    screen = pygame.display.set_mode((160 * SCALE, 120 * SCALE + 40))
    font = pygame.font.SysFont("monospace", 18)
    clock = pygame.time.Clock()

    for i in range(10**9):
        spec = maps[i % len(maps)]
        wad, _, map_name = spec.partition(":")
        map_name = map_name or "MAP01"
        ckpt = args.ckpt or newest(args.run)
        ticks = 4200 if args.outside else map_ticks(wad, 2100)
        env = make_env(wad, map_name, episode_ticks=ticks, render_buffers=True)
        obs, _ = env.reset(seed=i)
        agent = DoomAgent(env.observation_space, env.action_space, "cpu")
        agent.model = load_brain(ckpt)
        agent.reset()
        label = f"{Path(wad).stem}:{map_name}" if args.outside else Path(wad).stem
        step, paused, skip = 0, False, False
        while True:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                    env.close()
                    return
                if ev.type == pygame.KEYDOWN and ev.key == pygame.K_n:
                    skip = True
                if ev.type == pygame.KEYDOWN and ev.key == pygame.K_SPACE:
                    paused = not paused
            if skip:
                break
            if paused:
                clock.tick(20)
                continue
            obs, _, term, trunc, _ = env.step(agent.act(obs))
            step += 1
            ps = env.privileged_state()
            frame = np.ascontiguousarray(obs["rgb"]).swapaxes(0, 1)
            img = pygame.transform.scale(pygame.surfarray.make_surface(frame), (160 * SCALE, 120 * SCALE))
            screen.fill((0, 0, 0))
            screen.blit(img, (0, 0))
            hud = (f"{label}  |  brain {Path(ckpt).stem}  |  step {step}  "
                   f"hp {max(ps['health'], 0):.0f}  kills {ps['kills']}  keys {int(np.sum(ps['keys']))}")
            screen.blit(font.render(hud, True, (230, 230, 230)), (8, 120 * SCALE + 10))
            pygame.display.flip()
            clock.tick(args.fps)
            if term or trunc:
                result = "EXIT REACHED" if ps["exit_reached"] else ("died" if ps["dead"] else "time up")
                screen.blit(font.render(result, True, (255, 210, 60)), (8, 8))
                pygame.display.flip()
                pygame.time.wait(1500)
                break
        env.close()


if __name__ == "__main__":
    main()
