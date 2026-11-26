"""Where is the exit? Read from the map file, for the TRAINING reward only (the agent never sees it).

UDMF maps (TEXTMAP): linedefs with special 243/244 (Exit_Normal / Exit_Secret).
Doom-format maps (LINEDEFS): specials 11, 51, 52, 124, 197, 198.
Returns the average midpoint of the exit lines, or None if the map has none (e.g. a switch we can't find).
"""

from __future__ import annotations

import re
import struct
import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "env"))
from doom_env.wad import map_lumps, read_wad  # noqa: E402

UDMF_EXIT = {243, 244}
DOOM_EXIT = {11, 51, 52, 124, 197, 198}
_BLOCK = re.compile(r"(\w+)\s*(?:\/\/[^\n]*)?\s*\{([^}]*)\}", re.S)
_KV = re.compile(r"(\w+)\s*=\s*([^;]+);")


@lru_cache(maxsize=None)
def exit_point(wad: str, map_name: str = "MAP01") -> tuple[float, float] | None:
    lumps = map_lumps(read_wad(wad), map_name)
    mids = []
    if "TEXTMAP" in lumps:
        verts, lines = [], []
        for kind, body in _BLOCK.findall(lumps["TEXTMAP"].decode("latin-1")):
            kv = {k.lower(): v.strip() for k, v in _KV.findall(body)}
            if kind.lower() == "vertex":
                verts.append((float(kv["x"]), float(kv["y"])))
            elif kind.lower() == "linedef":
                lines.append(kv)
        for kv in lines:
            if int(kv.get("special", 0)) in UDMF_EXIT:
                a, b = verts[int(kv["v1"])], verts[int(kv["v2"])]
                mids.append(((a[0] + b[0]) / 2, (a[1] + b[1]) / 2))
    elif "LINEDEFS" in lumps:
        vb = lumps["VERTEXES"]
        verts = [struct.unpack_from("<hh", vb, i) for i in range(0, len(vb) - 3, 4)]
        lb = lumps["LINEDEFS"]
        for i in range(0, len(lb) - 13, 14):
            v1, v2, _flags, special = struct.unpack_from("<HHHH", lb, i)
            if special in DOOM_EXIT:
                a, b = verts[v1], verts[v2]
                mids.append(((a[0] + b[0]) / 2, (a[1] + b[1]) / 2))
    if not mids:
        return None
    return sum(m[0] for m in mids) / len(mids), sum(m[1] for m in mids) / len(mids)


if __name__ == "__main__":
    import glob
    files = [f for g in sys.argv[1:] for f in sorted(glob.glob(g))]
    found = sum(exit_point(f) is not None for f in files)
    print(f"exit found in {found} of {len(files)} maps; example {files[0]}: {exit_point(files[0])}")
