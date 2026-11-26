"""Watch the brain play, in a window, while training runs.

Each round it loads the NEWEST saved checkpoint, so you see it get better over time.
It runs on the CPU with 1 thread so it does not slow training down much.

    .venv/bin/python training/watch.py                 # our held-out test maps
    .venv/bin/python training/watch.py --outside       # real Freedoom levels
    .venv/bin/python training/watch.py --ckpt runs/run1/ckpt/best.pt

On the right: the agent's OWN memory map (memory_map.py), drawn only from what it has seen and
its own self-tracking, never from the game's real map. Brains without a map still get it drawn
here, so you can see what the next run will be able to remember.

On the game view: a coloured overlay of what the agent SENSES (rays: doors blue, locked doors red;
visible things: health/ammo green, keys yellow, enemies red), drawn only from its observation.

Keys: N = next map, O = overlay on/off, Space = pause, Esc = quit.
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
from doom_net import DoomNet, has_map  # noqa: E402
from memory_map import DOOR, ENEMY, FREE, KEY, LOCKED, TRAIL, WALL, MemoryMap  # noqa: E402
from ppo import map_ticks  # noqa: E402
from score import outside_maps  # noqa: E402

SCALE = 5
PANEL = 440  # map panel size in pixels
# layer -> colour, drawn in this order (later ones on top)
COLOURS = [(FREE, (55, 58, 70)), (TRAIL, (70, 130, 255)), (WALL, (225, 225, 225)), (DOOR, (60, 210, 90)),
           (LOCKED, (255, 60, 60)), (KEY, (255, 220, 40)), (ENEMY, (230, 70, 230))]
LEGEND = [("open floor", (55, 58, 70)), ("its path", (70, 130, 255)), ("wall", (225, 225, 225)),
          ("door", (60, 210, 90)), ("locked door", (255, 60, 60)), ("key", (255, 220, 40)), ("enemy", (230, 70, 230))]


# ------------------------------------------------------------------ "what the agent senses" overlay
W, H = 160 * SCALE, 120 * SCALE
FOCAL = W / 2  # 90 degree horizontal field of view
EYE, WALL_H = 41.0, 128.0  # Doom eye height; a typical wall height (only for drawing)
RAY_COLOURS = {2: (60, 140, 255), 3: (255, 50, 50)}  # door: blue, locked door: red (walls left plain)
THING = {1: ("enemy", (255, 40, 40), 56), 2: ("enemy", (255, 40, 40), 56), 3: ("enemy", (255, 40, 40), 56),
         5: ("health", (40, 230, 90), 20), 6: ("armor", (40, 230, 90), 20), 7: ("ammo", (40, 230, 90), 16),
         8: ("weapon", (40, 230, 90), 16), 9: ("KEY", (255, 220, 40), 20), 10: ("barrel", (255, 150, 40), 32),
         4: ("shot", (230, 70, 230), 12)}


def draw_sight(screen, font, obs: dict) -> None:
    """Colour the 3D view using ONLY the agent's own sensors: its 64 rays (doors, locked doors) and
    its list of visible things. Nothing here comes from the game's internal state."""
    over = pygame.Surface((W, H), pygame.SRCALPHA)
    dist = np.asarray(obs["rays"]["distance"]) * 1024.0
    hit = np.asarray(obs["rays"]["hit_type"])
    ray_ang = np.arange(64) * (2 * np.pi / 64)
    ray_ang = (ray_ang + np.pi) % (2 * np.pi) - np.pi  # left is positive
    for col in range(0, W, 4):
        ang = np.arctan((W / 2 - col - 2) / FOCAL)
        i = int(np.argmin(np.abs((ray_ang - ang + np.pi) % (2 * np.pi) - np.pi)))
        c = RAY_COLOURS.get(int(hit[i]))
        if c is None or dist[i] >= 1023 or dist[i] < 40:  # < 40: standing inside a doorway
            continue
        alpha = int(np.clip(30 + dist[i] / 3, 30, 120))  # faint up close, so the view stays readable
        perp = max(dist[i] * np.cos(ang), 16.0)
        top = H / 2 - (WALL_H - EYE) * FOCAL / perp
        bottom = H / 2 + EYE * FOCAL / perp
        pygame.draw.rect(over, c + (alpha,), (col, max(top, 0), 4, min(bottom, H) - max(top, 0)))
    ve = obs["visible_entities"]
    for f, t, m in zip(ve["features"], ve["entity_type"], ve["mask"]):
        if not m or int(t) not in THING:
            continue
        name, c, size = THING[int(t)]
        d, ang = f[0] * 1024.0, np.arctan2(f[1], f[2])
        if abs(ang) > np.pi / 4 + 0.1:
            continue
        perp = max(d * np.cos(ang), 16.0)
        x = W / 2 - np.tan(ang) * FOCAL
        h = size * FOCAL / perp
        bottom = H / 2 + EYE * FOCAL / perp
        w = max(h * 0.7, 10)
        rect = pygame.Rect(int(x - w / 2), int(bottom - h), int(w), int(h))
        pygame.draw.rect(over, c + (60,), rect)
        pygame.draw.rect(over, c + (255,), rect, 2)
        over.blit(font.render(f"{name} {d:.0f}", True, c), (rect.x, max(rect.y - 20, 0)))
    screen.blit(over, (0, 0))


def draw_map(screen, font, mm: MemoryMap, x0: int) -> None:
    """Top-down view (north up) of everything the agent has drawn so far, with an arrow for itself."""
    g = mm.grids[0]  # the near grid: 64 map units per cell
    seen = g.max(0) > 0
    pygame.draw.rect(screen, (12, 12, 16), (x0, 0, PANEL, PANEL))
    if seen.any():
        ys, xs = np.nonzero(seen)
        cx = int(np.floor(mm.x / 64.0)) + g.shape[2] // 2
        cy = int(np.floor(mm.y / 64.0)) + g.shape[1] // 2
        lo_x, hi_x = min(xs.min(), cx) - 2, max(xs.max(), cx) + 3
        lo_y, hi_y = min(ys.min(), cy) - 2, max(ys.max(), cy) + 3
        span = max(hi_x - lo_x, hi_y - lo_y, 24)
        mx, my = (lo_x + hi_x) // 2 - span // 2, (lo_y + hi_y) // 2 - span // 2
        px = PANEL / span
        img = np.zeros((span, span, 3), np.uint8)
        sub = g[:, max(my, 0):my + span, max(mx, 0):mx + span]
        oy, ox = max(my, 0) - my, max(mx, 0) - mx
        for ch, col in COLOURS:
            layer = sub[ch] > (0.05 if ch == ENEMY else 0.5)
            img[oy:oy + layer.shape[0], ox:ox + layer.shape[1]][layer] = col
        img = img[::-1]  # world y points up, screen y points down
        surf = pygame.surfarray.make_surface(img.swapaxes(0, 1))
        screen.blit(pygame.transform.scale(surf, (PANEL, PANEL)), (x0, 0))
        ax = x0 + (cx - mx + 0.5) * px
        ay = (span - (cy - my) - 0.5) * px
        h = mm.heading
        tip = (ax + 14 * np.cos(h), ay - 14 * np.sin(h))
        l = (ax + 8 * np.cos(h + 2.5), ay - 8 * np.sin(h + 2.5))
        r = (ax + 8 * np.cos(h - 2.5), ay - 8 * np.sin(h - 2.5))
        pygame.draw.polygon(screen, (255, 140, 0), [tip, l, r])
    screen.blit(font.render("the agent's own memory map", True, (230, 230, 230)), (x0 + 8, PANEL + 6))
    for k, (name, col) in enumerate(LEGEND):
        x, y = x0 + 8 + (k % 2) * 210, PANEL + 32 + (k // 2) * 22
        pygame.draw.rect(screen, col, (x, y + 3, 14, 14))
        screen.blit(font.render(name, True, (200, 200, 200)), (x + 22, y))


def newest(run: str) -> str:
    cks = sorted(glob.glob(str(ROOT / run / "ckpt" / "ckpt_*.pt")))
    return cks[-1] if cks else str(ROOT / "submission/weights/model.pt")


def load_brain(path: str) -> DoomNet:
    state = torch.load(path, map_location="cpu", weights_only=False)
    weights = state["net"] if "net" in state else state["model"]
    net = DoomNet(use_map=has_map(weights))
    net.load_state_dict({k: v for k, v in weights.items() if not k.startswith("value.")}, strict=False)
    return net.eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/run1", help="training run whose newest save to watch")
    ap.add_argument("--ckpt", default="", help="watch this exact save instead of the newest")
    ap.add_argument("--outside", action="store_true", help="real Freedoom levels instead of our test maps")
    ap.add_argument("--plain", action="store_true", help="hide the coloured 'what the agent senses' overlay")
    ap.add_argument("--fps", type=int, default=9, help="decisions per second (9 = real game speed)")
    args = ap.parse_args()
    torch.set_num_threads(1)

    maps = outside_maps() if args.outside else sorted(glob.glob(str(ROOT / "maps/gen/heldout/*/*.wad")))
    random.shuffle(maps)
    pygame.init()
    screen = pygame.display.set_mode((160 * SCALE + PANEL, max(120 * SCALE + 40, PANEL + 130)))
    font = pygame.font.SysFont("monospace", 18)
    small = pygame.font.SysFont("monospace", 15, bold=True)
    sight = not args.plain
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
        own = agent.memory is None  # map-less brain: draw a map here just for display
        mm = MemoryMap() if own else agent.memory
        if own:
            mm.observe(obs)
        label = f"{Path(wad).stem}:{map_name}" if args.outside else Path(wad).stem
        step, paused, skip = 0, False, False
        while True:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                    env.close()
                    return
                if ev.type == pygame.KEYDOWN and ev.key == pygame.K_n:
                    skip = True
                if ev.type == pygame.KEYDOWN and ev.key == pygame.K_o:
                    sight = not sight
                if ev.type == pygame.KEYDOWN and ev.key == pygame.K_SPACE:
                    paused = not paused
            if skip:
                break
            if paused:
                clock.tick(20)
                continue
            action = agent.act(obs)
            obs, _, term, trunc, _ = env.step(action)
            if own:
                mm.observe(obs, action)
            step += 1
            ps = env.privileged_state()
            frame = np.ascontiguousarray(obs["rgb"]).swapaxes(0, 1)
            img = pygame.transform.scale(pygame.surfarray.make_surface(frame), (160 * SCALE, 120 * SCALE))
            screen.fill((0, 0, 0))
            screen.blit(img, (0, 0))
            if sight:
                draw_sight(screen, small, obs)
            hud = (f"{label}  |  brain {Path(ckpt).stem}  |  step {step}  "
                   f"hp {max(ps['health'], 0):.0f}  kills {ps['kills']}  keys {int(np.sum(ps['keys']))}")
            screen.blit(font.render(hud, True, (230, 230, 230)), (8, 120 * SCALE + 10))
            draw_map(screen, font, mm, 160 * SCALE)
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
