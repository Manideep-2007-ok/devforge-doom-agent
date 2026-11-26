"""Phase 0 speed test: how many game steps per second can this laptop simulate?

Runs N separate processes, each playing random actions on the test map with no
screen rendering, and reports total steps/second and RAM per game copy.
This number sets the training budget (steps/hour) for the whole night.

    uv run python training/bench_speed.py --copies 1 4 8 11 --seconds 20
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time


def run_copy(seconds: float, wad: str, out) -> None:
    import numpy as np
    from doom_env import make_env

    env = make_env(wad, "MAP01", episode_ticks=2100, render_buffers=False)
    rng = np.random.default_rng(os.getpid())
    env.reset(seed=int(rng.integers(1 << 30)))
    steps = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        _, _, term, trunc, _ = env.step(rng.integers(0, 2, size=9, dtype=np.int8))
        steps += 1
        if term or trunc:
            env.reset(seed=int(rng.integers(1 << 30)))
    rss_mb = 0.0
    with open(f"/proc/{os.getpid()}/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                rss_mb = int(line.split()[1]) / 1024
    out.put((steps, time.perf_counter() - t0, rss_mb))
    env.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--copies", type=int, nargs="+", default=[1, 4, 8, 11])
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--wad", default="maps/two_rooms.wad")
    args = ap.parse_args()

    ctx = mp.get_context("spawn")
    print(f"{'copies':>6} {'steps/s':>9} {'per copy':>9} {'RAM/copy MB':>12} {'steps/hour':>12}")
    for n in args.copies:
        q = ctx.Queue()
        procs = [ctx.Process(target=run_copy, args=(args.seconds, args.wad, q)) for _ in range(n)]
        for p in procs:
            p.start()
        res = [q.get() for _ in procs]
        for p in procs:
            p.join()
        total = sum(s / t for s, t, _ in res)
        ram = sum(r for _, _, r in res) / n
        print(f"{n:>6} {total:>9.0f} {total / n:>9.0f} {ram:>12.0f} {total * 3600 / 1e6:>10.1f} M", flush=True)


if __name__ == "__main__":
    main()
