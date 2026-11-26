"""Pick the brain to submit, judged on FOUR test sets instead of one.

    .venv/bin/python training/final_pick.py --baseline runs/submitted_before_finale.pt --run runs/run2/ckpt

Candidates: the 3 best Run 2 saves on our held-out test (from its score history) + the 2 newest saves.
Every candidate and the baseline (the brain on GitHub) play the same games on:

    heldout  : our 20 held-out maps x 3 games
    fresh    : 20 freshly generated maps (seed 4242) x 2 games
    freedoom : the 19 Freedoom levels that ship with ViZDoom, 2 minutes each
    packs    : Scythe + 1024:ABC MAP01-10 (community maps from the idgames archive), 2 minutes each

Each set's score is divided by the baseline's score on that set, so all four count the same, and the
four ratios are averaged. A candidate wins only if that average is above 1.0 (better than GitHub) and
it made no errors and stayed under the 40 ms rule (each set ratio is capped at 2x). Prints a table and, on the last line,
"PICK <path>" or "KEEP".
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "training"), str(ROOT / "submission"), str(ROOT / "env")]

from score import outside_maps, score  # noqa: E402

PACKS = Path("/mnt/Linux_Home/manideep/devforge/data/outside_maps")


def test_sets() -> dict[str, dict]:
    packs = []
    for i in range(1, 11):
        packs += [f"{PACKS}/scythe/SCYTHE.WAD:MAP{i:02d}", f"{PACKS}/1024abc/1024abc.wad:MAP{i:02d}"]
    return {
        "heldout": dict(maps=sorted(glob.glob(str(ROOT / "maps/gen/heldout/*/*.wad"))), seeds=3, ticks=0),
        "fresh": dict(maps=sorted(glob.glob(str(ROOT / "maps/gen/fresh_test/*/*.wad"))), seeds=2, ticks=0),
        "freedoom": dict(maps=outside_maps(), seeds=1, ticks=4200),
        "packs": dict(maps=packs, seeds=1, ticks=4200),
    }


def candidates(run: Path) -> list[str]:
    hist = run / "scores_x3.jsonl"
    rows = [json.loads(l) for l in hist.read_text().splitlines() if l.strip()] if hist.exists() else []
    best = [r["ckpt"] for r in sorted(rows, key=lambda r: -r["score"])[:3]]
    newest = sorted(glob.glob(str(run / "ckpt_*.pt")))[-2:]
    out = []
    for c in best + newest:
        c = str(Path(c).resolve()) if Path(c).is_absolute() else str((ROOT / c).resolve())
        if c not in out and Path(c).exists():
            out.append(c)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--run", default="runs/run2/ckpt")
    ap.add_argument("--procs", type=int, default=10)
    ap.add_argument("--only", default="", help="comma list of checkpoints instead of the automatic candidates")
    args = ap.parse_args()

    sets = test_sets()
    brains = [args.baseline] + (args.only.split(",") if args.only else candidates(ROOT / args.run))
    table: dict[str, dict] = {}
    for b in brains:
        table[b] = {}
        for name, s in sets.items():
            r = score(b, s["maps"], args.procs, ticks=s["ticks"], seeds=s["seeds"])
            table[b][name] = r
            print(f"{Path(b).stem:22s} {name:9s} {r['score']:8.1f}  errors {r['errors']}  worst {r['worst_ms']:.0f} ms", flush=True)

    base = table[args.baseline]
    print("\nbrain                   " + "  ".join(f"{n:>9s}" for n in sets) + "   vs GitHub")
    best, best_ratio = None, 1.0
    for b, res in table.items():
        ratios = [min(2.0, res[n]["score"] / max(base[n]["score"], 1e-6)) for n in sets]  # capped: one lucky set cannot decide
        avg = sum(ratios) / len(ratios)
        ok = all(res[n]["errors"] == 0 and res[n]["worst_ms"] < 40 for n in sets)
        tag = "" if ok else "  (rejected: errors or slow)"
        print(f"{Path(b).stem:22s}  " + "  ".join(f"{res[n]['score']:9.1f}" for n in sets) + f"   x{avg:.2f}{tag}", flush=True)
        if b != args.baseline and ok and avg > best_ratio:
            best, best_ratio = b, avg
    print(f"PICK {best}" if best else "KEEP")


if __name__ == "__main__":
    main()
