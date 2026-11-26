"""Random map generator for training (UDMF, loads straight into doom_env.make_env).

Builds on Atharva's tile-to-UDMF approach in build_curriculum.py: the map is a grid of 128-unit
tiles; floor tiles share one sector, each door tile is its own sector, the exit is a tile whose
edges trigger "exit level" when crossed.

Every map is random (rooms, corridors, doors, monsters, items, start angle), and generated maps
are CHECKED before saving (flood fill over tiles):
  - the exit is reachable,
  - each locked door really blocks the way (exit unreachable without its key),
  - each key is reachable before its door.

Four tiers, matching the hidden test suite:
  t1 combat       1-2 rooms, 3-10 monsters, health/ammo around, exit
  t2 explore      4-8 rooms, normal doors, dead ends, exit far from start
  t3 key-door     key in a side area, locked door on the way to the exit
  t4 gauntlet     2 keys, 2 locked doors in a chain, more monsters, less ammo
  t5 key school   small: key close to the start, locked door right after it (practice for t3/t4)

    .venv/bin/python training/mapgen.py --out maps/gen/train --per-tier 100 --seed 1
    .venv/bin/python training/mapgen.py --out maps/gen/heldout --per-tier 5 --seed 999 --check

Each map also gets <name>.json with its tier and tick cap (2100-4200), which ppo.py reads.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import deque
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "env"))
from doom_env.wad import Lump, write_wad  # noqa: E402

TILE = 128
WALL, FLOOR, EXIT = "#", ".", "E"
DOOR = "D"
LOCKED = {"R": 129, "B": 130, "Y": 131}  # ZDoom lock numbers: any red / blue / yellow key
KEY_THING = {"R": 13, "B": 5, "Y": 6}  # red, blue, yellow keycard
MONSTERS = {  # Freedoom 2 thing types (same numbers as Doom II)
    "zombieman": 3004, "shotgunner": 9, "imp": 3001, "demon": 3002, "chaingunner": 65,
    "lost_soul": 3006, "cacodemon": 3005,
}
ITEMS = {
    "stimpack": 2011, "medikit": 2012, "armor": 2018, "clip": 2007, "shells": 2008,
    "bullet_box": 2048, "shotgun": 2001,
}
WALL_TEX = ["STARTAN2", "STARTAN3", "BROWN1", "BROWNGRN", "GRAY1", "GRAY4", "STONE2", "METAL1", "ICKWALL1"]
TIER_TICKS = {1: 2100, 2: 3150, 3: 3675, 4: 4200, 5: 2100}


# ===================================================================== layout
class Layout:
    def __init__(self, w: int, h: int):
        self.w, self.h = w, h
        self.g = [[WALL] * w for _ in range(h)]
        self.rooms: list[tuple[int, int, int, int]] = []  # x, y, w, h
        self.corridors: list[list[tuple[int, int]]] = []
        self.things: list[tuple[float, float, int, int]] = []  # tile x, tile y, type, angle
        self.start = (0, 0)
        self.start_angle = 0

    def __getitem__(self, p):
        x, y = p
        return self.g[y][x] if 0 <= x < self.w and 0 <= y < self.h else WALL

    def __setitem__(self, p, v):
        self.g[p[1]][p[0]] = v

    def passable(self, p, blocked=()):
        return self[p] != WALL and p not in blocked

    def reach(self, start, blocked=()) -> dict[tuple[int, int], int]:
        """Flood fill: tile -> steps from start, never entering `blocked` tiles."""
        dist = {start: 0}
        q = deque([start])
        while q:
            x, y = q.popleft()
            for n in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if n not in dist and self.passable(n, blocked):
                    dist[n] = dist[(x, y)] + 1
                    q.append(n)
        return dist

    def door_ok(self, p) -> bool:
        """A door needs a 1-tile-wide passage: floor on two opposite sides, walls on the other two."""
        x, y = p
        if self[p] != FLOOR:
            return False
        lr = self[(x - 1, y)] == FLOOR and self[(x + 1, y)] == FLOOR
        ud = self[(x, y - 1)] == FLOOR and self[(x, y + 1)] == FLOOR
        wl = self[(x - 1, y)] == WALL and self[(x + 1, y)] == WALL
        wu = self[(x, y - 1)] == WALL and self[(x, y + 1)] == WALL
        return (lr and wu) or (ud and wl)

    def room_of(self, p):
        for i, (rx, ry, rw, rh) in enumerate(self.rooms):
            if rx <= p[0] < rx + rw and ry <= p[1] < ry + rh:
                return i
        return None

    def room_tiles(self, i):
        rx, ry, rw, rh = self.rooms[i]
        return [(x, y) for y in range(ry, ry + rh) for x in range(rx, rx + rw)]


def carve_rooms(L: Layout, rng: random.Random, n_rooms: int, min_size=3, max_size=7) -> None:
    tries = 0
    while len(L.rooms) < n_rooms and tries < 400:
        tries += 1
        rw, rh = rng.randint(min_size, max_size), rng.randint(min_size, max_size)
        rx, ry = rng.randint(1, L.w - rw - 1), rng.randint(1, L.h - rh - 1)
        if any(rx - 2 < ox + ow and ox - 2 < rx + rw and ry - 2 < oy + oh and oy - 2 < ry + rh
               for ox, oy, ow, oh in L.rooms):
            continue  # keep a 2-tile gap so corridors (and their doors) fit between rooms
        L.rooms.append((rx, ry, rw, rh))
        for p in L.room_tiles(len(L.rooms) - 1):
            L[p] = FLOOR


def center(room, rng):
    rx, ry, rw, rh = room
    return rng.randint(rx + 1, rx + rw - 2) if rw > 2 else rx, rng.randint(ry + 1, ry + rh - 2) if rh > 2 else ry


def carve_corridor(L: Layout, a, b, rng) -> list[tuple[int, int]]:
    (x0, y0), (x1, y1) = a, b
    path = []
    if rng.random() < 0.5:
        for x in range(min(x0, x1), max(x0, x1) + 1):
            path.append((x, y0))
        for y in range(min(y0, y1), max(y0, y1) + 1):
            path.append((x1, y))
    else:
        for y in range(min(y0, y1), max(y0, y1) + 1):
            path.append((x0, y))
        for x in range(min(x0, x1), max(x0, x1) + 1):
            path.append((x, y1))
    for p in path:
        L[p] = FLOOR
    return path


def connect_rooms(L: Layout, rng: random.Random, extra_loops: int) -> None:
    """Spanning tree over room centres (nearest unconnected room each time), plus a few loops."""
    centers = [center(r, rng) for r in L.rooms]
    connected = [0]
    edges = []
    while len(connected) < len(L.rooms):
        best = None
        for i in connected:
            for j in range(len(L.rooms)):
                if j in connected:
                    continue
                d = abs(centers[i][0] - centers[j][0]) + abs(centers[i][1] - centers[j][1])
                if best is None or d < best[0]:
                    best = (d, i, j)
        _, i, j = best
        connected.append(j)
        edges.append((i, j))
    for _ in range(extra_loops):
        i, j = rng.sample(range(len(L.rooms)), 2)
        edges.append((i, j))
    for i, j in edges:
        L.corridors.append(carve_corridor(L, centers[i], centers[j], rng))


def corridor_door_spots(L: Layout) -> list[tuple[int, int]]:
    spots = set()
    for c in L.corridors:
        for p in c:
            if L.room_of(p) is None and L.door_ok(p):
                spots.add(p)
    return sorted(spots)


def floor_tiles(L: Layout):
    return [(x, y) for y in range(L.h) for x in range(L.w) if L[(x, y)] == FLOOR]


# ===================================================================== tiers
class Retry(Exception):
    pass


def place_normal_doors(L: Layout, rng, prob: float) -> None:
    for c in L.corridors:
        spots = [p for p in c if L.room_of(p) is None and L.door_ok(p)]
        if spots and rng.random() < prob:
            L[rng.choice(spots)] = DOOR


def separators(L: Layout, start, goal, blocked=()) -> list[tuple[int, int]]:
    """Door-able tiles (corridor tiles, or existing normal doors) that cut `goal` off from `start`."""
    out = []
    reach = L.reach(start, blocked)
    for p in reach:
        if p == start or p == goal:
            continue
        if not (L[p] == DOOR or L.door_ok(p)) or L.room_of(p) is not None:
            continue
        if goal not in L.reach(start, set(blocked) | {p}):
            out.append(p)
    return out


def pick_far(rng, dist: dict, tiles, frac=0.5):
    """Random tile among the farther `frac` (by steps from start)."""
    tiles = sorted(tiles, key=lambda p: dist.get(p, 0))
    if not tiles:
        raise Retry("no tiles")
    return rng.choice(tiles[int(len(tiles) * (1 - frac)):] or tiles)


def finish_layout(L: Layout, rng, tier: int, n_monsters: int, n_items: int, keys: list) -> dict:
    """Place exit (far from start), locked doors and keys, monsters and items. Returns the map's stats."""
    dist = L.reach(L.start)
    room_floor = [p for p in floor_tiles(L) if L.room_of(p) is not None and p != L.start]
    exit_tile = pick_far(rng, dist, [p for p in room_floor if L.room_of(p) != L.room_of(L.start)] or room_floor, 0.25)
    L[exit_tile] = EXIT

    blocked: list = []
    placed_keys = []
    colors = rng.sample(list(LOCKED), len(keys))
    # Chain: key 1 before door 1, key 2 between door 1 and door 2, exit after the last door.
    all_seps = separators(L, L.start, exit_tile)  # every chokepoint between start and exit
    prev_area: set = set()
    for k, color in enumerate(colors):
        # door k must lie beyond door k-1 (outside the area reachable while door k-1 is locked)
        seps = [s for s in all_seps if s not in blocked and s not in prev_area]
        if not seps:
            raise Retry("no chokepoint for a locked door")
        rng.shuffle(seps)
        if len(colors) == 1:
            # one key: prefer doors that leave a big area before them (the key hides there)
            seps.sort(key=lambda s: -len(L.reach(L.start, blocked + [s])))
            door = seps[rng.randint(0, min(2, len(seps) - 1))]
        else:
            # a chain: the first door must leave room beyond it for the next door and key
            seps.sort(key=lambda s: len(L.reach(L.start, blocked + [s])))
            door = seps[rng.randint(0, len(seps) // 2)] if k == 0 else seps[-1 - rng.randint(0, min(2, len(seps) - 1))]
        area = L.reach(L.start, blocked[k:] + [door])  # earlier doors open (their keys are held)
        area = {p: d for p, d in area.items() if p not in prev_area}  # key k lies beyond door k-1
        candidates = [p for p in area if L[p] == FLOOR and L.room_of(p) is not None and p != L.start]
        if not candidates:
            raise Retry("no room for the key")
        if tier == 5:  # key school: the key is among the closest half of the tiles before the door
            candidates.sort(key=lambda p: area[p])
            key = rng.choice(candidates[: max(1, len(candidates) // 2)])
        else:
            key = pick_far(rng, area, candidates, 0.5)
        prev_area = set(L.reach(L.start, blocked[k:] + [door]))
        L[door] = color
        placed_keys.append((key, color))
        blocked.append(door)

    # verify the chain with flood fills: nothing reachable early that should not be
    for k, (key, color) in enumerate(placed_keys):
        locked_now = [p for p in blocked[k:]]
        r = L.reach(L.start, locked_now)
        if key not in r:
            raise Retry("key unreachable")
        if exit_tile in r:
            raise Retry("exit reachable without key")
    if exit_tile not in L.reach(L.start):
        raise Retry("exit unreachable")
    for key, color in placed_keys:
        L.things.append((key[0] + 0.5, key[1] + 0.5, KEY_THING[color], 0))

    # monsters away from the start, items anywhere
    free = [p for p in floor_tiles(L) if p != L.start and p not in {k for k, _ in placed_keys}]
    far = [p for p in free if dist.get(p, 99) >= 4]
    pool = {1: ["zombieman", "shotgunner", "imp", "demon"],
            2: ["zombieman", "shotgunner", "imp"],
            3: ["zombieman", "shotgunner", "imp", "demon", "chaingunner"],
            4: ["zombieman", "shotgunner", "imp", "demon", "chaingunner", "lost_soul", "cacodemon"],
            5: ["zombieman", "imp"]}[tier]
    for p in rng.sample(far, min(n_monsters, len(far))):
        L.things.append((p[0] + rng.uniform(0.3, 0.7), p[1] + rng.uniform(0.3, 0.7),
                         MONSTERS[rng.choice(pool)], rng.randrange(0, 360, 45)))
    item_pool = ["stimpack", "medikit", "armor", "clip", "shells", "bullet_box", "shotgun"]
    for p in rng.sample(free, min(n_items, len(free))):
        L.things.append((p[0] + rng.uniform(0.3, 0.7), p[1] + rng.uniform(0.3, 0.7), ITEMS[rng.choice(item_pool)], 0))
    return {"tier": tier, "ticks": TIER_TICKS[tier], "rooms": len(L.rooms), "keys": len(placed_keys),
            "monsters": n_monsters, "exit_steps": L.reach(L.start)[exit_tile]}


def make_layout(tier: int, rng: random.Random) -> tuple[Layout, dict]:
    for _ in range(200):
        try:
            if tier == 1:
                L = Layout(rng.randint(14, 20), rng.randint(14, 20))
                carve_rooms(L, rng, rng.randint(1, 2), 5, 9)
                n_mon, n_items, keys, door_p, loops = rng.randint(3, 10), rng.randint(4, 8), [], 0.3, 0
            elif tier == 2:
                L = Layout(rng.randint(24, 34), rng.randint(24, 34))
                carve_rooms(L, rng, rng.randint(4, 8))
                n_mon, n_items, keys, door_p, loops = rng.randint(2, 6), rng.randint(3, 7), [], 0.6, rng.randint(0, 1)
            elif tier == 3:
                L = Layout(rng.randint(24, 34), rng.randint(24, 34))
                carve_rooms(L, rng, rng.randint(4, 8))
                n_mon, n_items, keys, door_p, loops = rng.randint(3, 8), rng.randint(3, 6), [1], 0.4, 0
            elif tier == 5:
                L = Layout(rng.randint(16, 24), rng.randint(16, 24))
                carve_rooms(L, rng, rng.randint(3, 4), 3, 6)
                n_mon, n_items, keys, door_p, loops = rng.randint(1, 3), rng.randint(3, 5), [1], 0.2, 0
            else:
                L = Layout(rng.randint(30, 40), rng.randint(30, 40))
                carve_rooms(L, rng, rng.randint(6, 10))
                n_mon, n_items, keys, door_p, loops = rng.randint(8, 15), rng.randint(2, 4), [1, 2], 0.4, 0
            if len(L.rooms) < (1 if tier == 1 else 3):
                raise Retry("not enough rooms")
            if len(L.rooms) > 1:
                connect_rooms(L, rng, loops)
            place_normal_doors(L, rng, door_p)
            L.start = center(L.rooms[0], rng)
            L.start_angle = rng.randrange(0, 360, 45)
            info = finish_layout(L, rng, tier, n_mon, n_items, keys)
            return L, info
        except Retry:
            continue
    raise RuntimeError(f"could not generate a tier {tier} map")


# ===================================================================== UDMF (after Atharva's build_textmap)
def build_textmap(L: Layout, rng: random.Random) -> bytes:
    wall_tex = rng.choice(WALL_TEX)
    ceil_h = rng.choice([128, 160, 192])
    light = rng.choice([144, 160, 192, 224])

    def sector_of(p):
        c = L[p]
        if c == WALL:
            return None
        if c == DOOR or c in LOCKED:
            return p  # each door is its own sector
        return c  # "." floor or "E" exit

    fields = {"vertex": [], "sector": [], "sidedef": [], "linedef": [], "thing": []}
    sectors, vertices, seen = {}, {}, set()

    def add(kind, props):
        fields[kind].append(f"{kind} {{ {props} }}")
        return len(fields[kind]) - 1

    def vertex(pt):
        if pt not in vertices:
            vertices[pt] = add("vertex", f"x = {TILE * pt[0]}.0; y = {-TILE * pt[1]}.0;")
        return vertices[pt]

    def side(sec, mid="-", top="-"):
        return add("sidedef", f'sector = {sectors[sec]}; texturemiddle = "{mid}"; texturetop = "{top}"; texturebottom = "-";')

    tiles = [(x, y) for y in range(L.h) for x in range(L.w) if L[(x, y)] != WALL]
    for sec in dict.fromkeys(sector_of(p) for p in tiles):
        is_door = isinstance(sec, tuple)
        floor_tex = "GATE3" if sec == EXIT else "FLOOR0_1"
        sectors[sec] = add("sector", f"heightfloor = 0; heightceiling = {0 if is_door else ceil_h}; "
                                     f'texturefloor = "{floor_tex}"; textureceiling = "CEIL1_1"; lightlevel = {light};')

    sx, sy = L.start
    add("thing", f"x = {TILE * sx + TILE // 2}.0; y = {-TILE * sy - TILE // 2}.0; angle = {L.start_angle}; type = 1; "
                 "skill1 = true; skill2 = true; skill3 = true; skill4 = true; skill5 = true; single = true;")
    for tx, ty, ttype, ang in L.things:
        add("thing", f"x = {TILE * tx:.1f}; y = {-TILE * ty:.1f}; angle = {ang}; type = {ttype}; "
                     "skill1 = true; skill2 = true; skill3 = true; skill4 = true; skill5 = true; single = true;")

    for (x, y) in tiles:
        corners = [(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)]
        neighbors = [(x, y - 1), (x + 1, y), (x, y + 1), (x - 1, y)]
        for a, b, n in zip(corners, corners[1:] + corners[:1], neighbors):
            front, back = sector_of((x, y)), sector_of(n)
            edge = frozenset((a, b))
            if front == back or edge in seen:
                continue
            seen.add(edge)
            special = ""
            if back is None:
                sides = f'sidefront = {side(front, mid=wall_tex)}; blocking = true;'
            else:
                if isinstance(front, tuple) or front == EXIT:
                    front, back, a, b = back, front, b, a
                if isinstance(back, tuple):
                    c = L[back]
                    lock = LOCKED.get(c, 0)
                    # 12 = Door_Raise, 13 = Door_LockedRaise (args: tag 0 = this door, speed, delay, lock)
                    special = (f"special = {13 if lock else 12}; arg1 = 16; arg2 = 150; arg3 = {lock}; "
                               "playeruse = true; repeatspecial = true;")
                    top = "DOORRED" if c == "R" else "DOORBLU" if c == "B" else "DOORYEL" if c == "Y" else "BIGDOOR2"
                elif back == EXIT:
                    special = "special = 243; playercross = true;"  # 243 = Exit_Normal
                    top = "-"
                else:
                    top = "-"
                sides = f"sidefront = {side(front, top=top)}; sideback = {side(back)}; twosided = true;"
            add("linedef", f"v1 = {vertex(a)}; v2 = {vertex(b)}; {sides} {special}")

    blocks = [b for group in fields.values() for b in group]
    return "\n".join(['namespace = "zdoom";', *blocks, ""]).encode("ascii")


def ascii_art(L: Layout) -> str:
    g = [row[:] for row in L.g]
    g[L.start[1]][L.start[0]] = "P"
    for tx, ty, t, _ in L.things:
        if t in KEY_THING.values():
            g[int(ty)][int(tx)] = {13: "r", 5: "b", 6: "y"}[t]
    return "\n".join("".join(r) for r in g)


def write_map(path: Path, L: Layout, info: dict, rng: random.Random) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_wad(path, [Lump("MAP01", b""), Lump("TEXTMAP", build_textmap(L, rng)), Lump("ENDMAP", b"")])
    path.with_suffix(".json").write_text(json.dumps(info))


def check_map(path: Path, info: dict) -> str:
    """Load the map in the real engine and confirm doors/locks match what we built."""
    from doom_env import make_env

    env = make_env(str(path), "MAP01", episode_ticks=info["ticks"], render_buffers=False)
    try:
        env.reset(seed=0)
        ps = env.privileged_state()
        locked = int((ps["door_lock"] > 0).sum())
        if locked != info["keys"]:
            raise RuntimeError(f"{path.name}: engine sees {locked} locked doors, built {info['keys']}")
        return f"ok ({len(ps['door_lock'])} sectors, {int((ps['door_lock'] >= 0).sum())} doors, {locked} locked)"
    finally:
        env.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-tier", type=int, default=50)
    ap.add_argument("--tiers", default="1,2,3,4")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--check", action="store_true", help="load every map in ViZDoom (slower)")
    ap.add_argument("--show", action="store_true", help="print each layout as ASCII")
    args = ap.parse_args()
    out = Path(args.out)
    for tier in [int(t) for t in args.tiers.split(",")]:
        for i in range(args.per_tier):
            seed = args.seed * 100_000 + tier * 10_000 + i
            rng = random.Random(seed)
            L, info = make_layout(tier, rng)
            info["seed"] = seed
            path = out / f"t{tier}" / f"t{tier}_{i:04d}.wad"
            write_map(path, L, info, rng)
            msg = check_map(path, info) if args.check else ""
            print(f"{path}  rooms {info['rooms']} keys {info['keys']} monsters {info['monsters']} "
                  f"exit {info['exit_steps']} tiles away  {msg}", flush=True)
            if args.show:
                print(ascii_art(L) + "\n")


if __name__ == "__main__":
    main()
