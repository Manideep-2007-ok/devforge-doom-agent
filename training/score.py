"""Mini-scorer: play a saved brain on held-out maps and give ONE number, so versions can be compared.

Uses the real submission/agent.py (Sai's DoomAgent) on 1 CPU thread with fixed seeds, the same
way the judges run it. The real weights are hidden; ours copy their categories:

  exit 100 (+10 x health left, +10 x time left)   keys 20 each   locked doors 20   doors 5
  new 128x128 areas 0.5 each   kills 3 each   damage -0.05 per point   early death -20 x time left

Each map's points are multiplied by its tier (1-4), then summed.

    .venv/bin/python training/score.py runs/run1/ckpt/ckpt_00012345k.pt
    .venv/bin/python training/score.py submission/weights/model.pt --maps "maps/gen/heldout/*/*.wad"
    .venv/bin/python training/score.py runs/run1/ckpt --best    # score every checkpoint, copy the best
"""

from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import os
import re
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAPS = "maps/gen/heldout/*/*.wad"


def tier_of(wad: str) -> int:
    meta = Path(wad).with_suffix(".json")
    if meta.exists():
        return int(json.loads(meta.read_text()).get("tier", 1))
    m = re.search(r"/t(\d)/", wad)
    return int(m.group(1)) if m else 1


def outside_maps() -> list[str]:
    """Real handmade levels (Freedoom, BSD licence) that ship inside the vizdoom package."""
    import vizdoom
    d = os.path.dirname(vizdoom.__file__)
    return ([f"{d}/freedoom2.wad:MAP{i:02d}" for i in range(1, 11)]
            + [f"{d}/freedoom1.wad:E1M{i}" for i in range(1, 10)])


def play_one(args) -> dict:
    ckpt, spec, seed, ticks = args
    wad, _, map_name = spec.partition(":")
    map_name = map_name or "MAP01"
    os.environ["OMP_NUM_THREADS"] = "1"
    sys.path[:0] = [str(ROOT / "submission"), str(ROOT / "training")]
    import numpy as np
    import torch

    from agent import DoomAgent
    from doom_env import make_env
    from doom_net import DoomNet
    from ppo import map_ticks

    torch.set_num_threads(1)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    weights = state["net"] if "net" in state else state["model"]
    env = make_env(wad, map_name, episode_ticks=ticks or map_ticks(wad, 2100), render_buffers=False)
    obs, _ = env.reset(seed=seed)
    agent = DoomAgent(env.observation_space, env.action_space, "cpu")
    net = DoomNet()
    net.load_state_dict({k: v for k, v in weights.items() if not k.startswith("value.")}, strict=False)
    agent.model = net.eval()
    agent.reset()

    ps = env.privileged_state()
    cells = {(int(ps["x"] // 128), int(ps["y"] // 128))}
    kills0, opened, keys_seen = ps["kills"], set(), np.zeros(6)
    doors = locked = 0
    worst_ms = 0.0
    total_ms = 0.0
    steps = 0
    while True:
        t = time.perf_counter()
        action = agent.act(obs)
        ms = (time.perf_counter() - t) * 1000
        worst_ms, total_ms, steps = max(worst_ms, ms), total_ms + ms, steps + 1
        obs, _, term, trunc, _ = env.step(action)
        ps = env.privileged_state()
        cells.add((int(ps["x"] // 128), int(ps["y"] // 128)))
        keys_seen = np.maximum(keys_seen, ps["keys"])
        lock, ceil, floor = ps["door_lock"], ps["sector_ceiling"], ps["sector_floor"]
        for s in np.flatnonzero((lock >= 0) & (ceil - floor > 32.0)):
            if s not in opened:
                opened.add(int(s))
                if lock[s] > 0:
                    locked += 1
                else:
                    doors += 1
        if term or trunc:
            break
    env.close()

    time_left = max(0.0, 1.0 - ps["tic"] / ps["episode_ticks"])
    out = {
        "map": os.path.relpath(wad, ROOT) + (":" + map_name if map_name != "MAP01" else ""),
        "tier": 4 if ticks else tier_of(wad), "exit": int(ps["exit_reached"]),
        "dead": int(ps["dead"]), "keys": int(keys_seen.sum()), "doors": doors, "locked": locked,
        "cells": len(cells), "kills": ps["kills"] - kills0, "damage": ps["damage_taken"],
        "health": max(ps["health"], 0.0), "time_left": time_left, "errors": agent.errors,
        "avg_ms": total_ms / max(steps, 1), "worst_ms": worst_ms,
    }
    pts = 0.0
    if out["exit"]:
        pts += 100 + 10 * out["health"] / 100 + 10 * time_left
    pts += 20 * out["keys"] + 20 * locked + 5 * doors + 0.5 * out["cells"] + 3 * out["kills"]
    pts -= 0.05 * out["damage"]
    if out["dead"]:
        pts -= 20 * time_left
    out["points"] = pts
    out["weighted"] = pts * out["tier"]
    return out


def score(ckpt: str, maps: list[str], procs: int, seed: int = 0, ticks: int = 0) -> dict:
    ctx = mp.get_context("spawn")
    with ctx.Pool(procs) as pool:
        rows = pool.map(play_one, [(ckpt, m, seed, ticks) for m in maps])
    total = sum(r["weighted"] for r in rows)
    by_tier = {}
    for r in rows:
        t = by_tier.setdefault(r["tier"], {"maps": 0, "exit": 0, "keys": 0, "locked": 0, "doors": 0, "cells": 0, "kills": 0, "dead": 0})
        t["maps"] += 1
        for k in ("exit", "keys", "locked", "doors", "cells", "kills", "dead"):
            t[k] += r[k]
    return {"ckpt": ckpt, "score": total, "by_tier": by_tier, "rows": rows,
            "errors": sum(r["errors"] for r in rows), "worst_ms": max(r["worst_ms"] for r in rows),
            "avg_ms": sum(r["avg_ms"] for r in rows) / len(rows)}


def print_report(res: dict) -> None:
    print(f"\n{res['ckpt']}\n  SCORE {res['score']:.1f}   errors {res['errors']}   "
          f"decision avg {res['avg_ms']:.1f} ms, worst {res['worst_ms']:.0f} ms")
    for tier in sorted(res["by_tier"]):
        t = res["by_tier"][tier]
        n = t["maps"]
        print(f"  tier {tier}: exits {t['exit']}/{n}  keys {t['keys']}  locked {t['locked']}  doors {t['doors']}  "
              f"cells {t['cells'] / n:.0f}/map  kills {t['kills']}  deaths {t['dead']}/{n}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("target", help="a checkpoint .pt, or a run's ckpt/ folder with --best")
    ap.add_argument("--maps", default=DEFAULT_MAPS)
    ap.add_argument("--procs", type=int, default=max(1, min(10, (os.cpu_count() or 2) - 2)))
    ap.add_argument("--outside", action="store_true", help="test on real Freedoom levels that ship with ViZDoom (not our generated maps)")
    ap.add_argument("--ticks", type=int, default=0, help="episode length override (35 ticks = 1 second)")
    ap.add_argument("--best", action="store_true", help="score every checkpoint in the folder; copy the best to best.pt")
    args = ap.parse_args()
    if args.outside:
        maps, args.ticks = outside_maps(), args.ticks or 4200
    else:
        maps = sorted(glob.glob(str(ROOT / args.maps) if not os.path.isabs(args.maps) else args.maps))
    if not maps:
        raise SystemExit(f"no maps matched {args.maps}")

    targets = sorted(glob.glob(os.path.join(args.target, "ckpt_*.pt"))) if args.best else [args.target]
    history_path = Path(args.target if args.best else os.path.dirname(args.target)) / (
        "scores_outside.jsonl" if args.outside else "scores.jsonl")
    done = {}
    if history_path.exists():
        for line in history_path.read_text().splitlines():
            r = json.loads(line)
            done[r["ckpt"]] = r
    best = None
    for t in targets:
        res = done.get(t) or score(t, maps, args.procs, ticks=args.ticks)
        if t not in done:
            with history_path.open("a") as f:
                f.write(json.dumps({k: v for k, v in res.items() if k != "rows"}) + "\n")
            print_report(res)
        else:
            print(f"{t}: SCORE {res['score']:.1f} (already scored)")
        # never pick a version that crashed or was too slow
        if res["errors"] == 0 and res["worst_ms"] < 1000 and (best is None or res["score"] > best["score"]):
            best = res
    if args.best and best:
        dst = Path(args.target) / "best.pt"
        shutil.copy(best["ckpt"], dst)
        print(f"\nBEST: {best['ckpt']}  score {best['score']:.1f}  -> copied to {dst}")


if __name__ == "__main__":
    main()
