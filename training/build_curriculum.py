"""Build the original curriculum maps for the DevForge training environment."""

from pathlib import Path

from doom_env.wad import Lump, write_wad


MAP_DIR = Path(__file__).resolve().parents[1] / "maps" / "curriculum"
TILE = 128
DOOR_LOCKS = {"D": 0, "R": 129, "B": 130}
THINGS = {"P": 1, "r": 13, "b": 5}

# # is solid, a is the second room, E is the exit, and r/b are red/blue keys.
# D is an ordinary door, and R/B require the matching key.
LAYOUTS = {
    "train/01_straight": (
        "#######",
        "#P...E#",
        "#######",
    ),
    "train/02_turn": (
        "######",
        "#P...#",
        "####.#",
        "#E...#",
        "######",
    ),
    "train/03_door": (
        "###########",
        "#...#aaaaa#",
        "#P..DaaaaE#",
        "#...#aaaaa#",
        "###########",
    ),
    "train/04_red_key": (
        "############",
        "#....#aaaaa#",
        "#P.r.RaaaaE#",
        "#....#aaaaa#",
        "############",
    ),
    "holdout/01_blue_key_turn": (
        "############",
        "#P..#aaEaaa#",
        "#...#aaaaaa#",
        "#..bBaaaaaa#",
        "#...#aaaaaa#",
        "############",
    ),
}


def build_textmap(layout):
    tiles = {
        (x, y): char
        for y, row in enumerate(layout)
        for x, char in enumerate(row)
        if char != "#"
    }
    # A door has its own sector. Player and key markers use the first room's sector.
    tile_sectors = {
        tile: tile if char in DOOR_LOCKS else "." if char in THINGS else char
        for tile, char in tiles.items()
    }
    sectors = {}
    fields = {"vertex": [], "sector": [], "sidedef": [], "linedef": [], "thing": []}
    vertices = {}
    seen_edges = set()

    def add(kind, properties):
        index = len(fields[kind])
        fields[kind].append(f"{kind} {{ {properties} }}")
        return index

    def vertex(point):
        if point not in vertices:
            x, y = point
            vertices[point] = add("vertex", f"x = {TILE * x}; y = {-TILE * y};")
        return vertices[point]

    def side(sector, mid="-", top="-"):
        return add("sidedef", f'sector = {sectors[sector]}; texturemiddle = "{mid}"; '
                   f'texturetop = "{top}"; texturebottom = "-";')

    for sector in dict.fromkeys(tile_sectors.values()):
        ceiling = 0 if isinstance(sector, tuple) else 128
        floor = "GATE3" if sector == "E" else "FLOOR0_1"
        sectors[sector] = add("sector", f'heightfloor = 0; heightceiling = {ceiling}; '
                              f'texturefloor = "{floor}"; textureceiling = "CEIL1_1"; lightlevel = 192;')

    for (x, y), char in tiles.items():
        if char in THINGS:
            add("thing", f"x = {TILE * x + TILE // 2}; y = {-TILE * y - TILE // 2}; angle = 0; "
                f"type = {THINGS[char]}; skill1 = true; skill2 = true; skill3 = true; "
                "skill4 = true; skill5 = true; single = true;")

        # Clockwise edges keep the tile interior on the linedef's front (right) side.
        corners = [(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)]
        neighbors = [(x, y - 1), (x + 1, y), (x, y + 1), (x - 1, y)]
        for start, end, neighbor in zip(corners, corners[1:] + corners[:1], neighbors):
            front = tile_sectors[x, y]
            back = tile_sectors.get(neighbor)
            edge = frozenset((start, end))
            if front == back or edge in seen_edges:
                continue
            seen_edges.add(edge)

            special = ""
            if back is None:
                sides = f'sidefront = {side(front, mid="STARTAN2")}; blocking = true;'
            else:
                if isinstance(front, tuple) or front == "E":
                    front, back = back, front
                    start, end = end, start
                if isinstance(back, tuple):
                    lock = DOOR_LOCKS[tiles[back]]
                    special = (f"special = {13 if lock else 12}; arg1 = 16; arg2 = 150; "
                               f"arg3 = {lock}; playeruse = true; repeatspecial = true;")
                elif back == "E":
                    special = "special = 243; playercross = true;"
                top = "BIGDOOR2" if isinstance(back, tuple) else "-"
                sides = f"sidefront = {side(front, top=top)}; sideback = {side(back)}; twosided = true;"
            add("linedef", f"v1 = {vertex(start)}; v2 = {vertex(end)}; {sides} {special}")

    blocks = [block for group in fields.values() for block in group]
    return '\n'.join(['namespace = "zdoom";', *blocks, '']).encode('ascii')


def main():
    for name, layout in LAYOUTS.items():
        path = MAP_DIR / f"{name}.wad"
        path.parent.mkdir(parents=True, exist_ok=True)
        write_wad(path, [Lump("MAP01", b""), Lump("TEXTMAP", build_textmap(layout)), Lump("ENDMAP", b"")])
        print(path.relative_to(MAP_DIR.parent.parent))


if __name__ == "__main__":
    main()
