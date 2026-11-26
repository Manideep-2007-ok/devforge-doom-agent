"""The agent's own memory map: a top-down sketch it draws from what it has seen.

The official observation has no map and no position, so the agent keeps track of itself the
way you would with your eyes closed: it counts its own turns (it knows which buttons it pressed)
and adds up its own speed (vitals vel_forward / vel_right). That estimate is its "pose".

Every step, the 64 rays are painted into a grid around that pose: cells a ray passed through are
free, the cell it hit is a wall, door or locked door. Keys and enemies from visible_entities are
painted too, and so is the trail of where the agent has walked. The network then reads two
crops of this grid, turned so "forward" is always up:

    near : 32 x 32 cells of 64 units  (2 km across, the same reach as the rays)
    far  : 32 x 32 cells of 256 units (8 km across: "the locked door I saw a minute ago")

NumPy only, about 0.2 ms per step. Built only from the agent's own observations and actions,
never from the game engine, so it runs the same in training and in evaluation.
"""

from __future__ import annotations

import numpy as np

# grid channels
FREE, WALL, DOOR, LOCKED, KEY, ENEMY, TRAIL = range(7)
CHANNELS = 7
CROP = 32  # crop is CROP x CROP cells
SCALES = (64.0, 256.0)  # map units per cell for the near and far crop
GRID = 256  # cells per side of each stored grid (near: 16k units, far: 65k units)

RAY_RANGE = 1024.0
NUM_RAYS = 64
RAY_ANGLES = np.arange(NUM_RAYS) * (2.0 * np.pi / NUM_RAYS)  # ray i is i*5.6 degrees left of forward
HIT_WALL, HIT_DOOR, HIT_LOCKED = 1, 2, 3
ENEMY_TYPES = (1, 2, 3)
KEY_TYPE = 9

# Measured in the engine (training/check_memory_map.py): degrees turned in one 4-tic step,
# by how many steps in a row a turn button has been held. Doom starts turns slowly.
TURN_DEG = np.array([7.03125, 12.3046875, 14.0625])
TURN_DEG_FIRST = np.array([0.0, 3.515625, 14.0625])  # a turn held from the very first step of a map
TURN_LEFT_COL, TURN_RIGHT_COL = 4, 5  # columns in features.ACTIONS
FRICTION = 0.90625  # Doom's per-tic ground friction
TICS = 4  # frame skip
VEL_F, VEL_R = 14, 15  # vitals indices

ENEMY_FADE = 0.97  # enemy marks fade (they move); everything else is kept
_SAMPLES = np.linspace(0.0, 1.0, 33)[:, None]  # points along each ray, as a fraction of its length


def _crop_offsets() -> np.ndarray:
    # cell centres of the crop in ego coordinates (forward, left), forward pointing "up"
    i = (np.arange(CROP) - CROP / 2 + 0.5)
    fwd = -i[:, None] * np.ones(CROP)[None, :]  # row 0 is the far front
    left = -i[None, :] * np.ones(CROP)[:, None]  # column 0 is the far left
    return np.stack([fwd, left], -1).reshape(-1, 2)


_OFFS = _crop_offsets()


class MemoryMap:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.grids = [np.zeros((CHANNELS, GRID, GRID), np.float32) for _ in SCALES]
        self.x = self.y = 0.0  # estimated position, starting at 0
        self.heading = 0.0  # estimated heading in radians, starting at 0
        self.turn_held = 0
        self.turn_from_start = False
        self.steps = 0
        self.vel = np.zeros(2)  # (forward, right) speed from the last observation

    # ------------------------------------------------------------ pose
    def move(self, action_row: np.ndarray, obs: dict) -> None:
        """Update the pose after taking `action_row` (one row of features.ACTIONS) and seeing `obs`."""
        turn = int(action_row[TURN_LEFT_COL]) - int(action_row[TURN_RIGHT_COL])
        if turn:
            if self.turn_held == 0:
                self.turn_from_start = self.steps == 0
            table = TURN_DEG_FIRST if self.turn_from_start else TURN_DEG
            d = np.deg2rad(table[min(self.turn_held, 2)]) * turn
            self.turn_held += 1
        else:
            d, self.turn_held = 0.0, 0
        h0 = self.heading
        self.heading = (h0 + d + np.pi) % (2 * np.pi) - np.pi

        # Distance moved in 4 tics from the speed before and after (Doom: v <- f * (v + thrust)).
        v4 = np.array([obs["vitals"][VEL_F], obs["vitals"][VEL_R]], np.float64)
        v0 = self.vel
        f = FRICTION
        thrust = (v4 - f**TICS * v0) / (f * (1 - f**TICS) / (1 - f))
        v, disp = v0.copy(), np.zeros(2)
        for _ in range(TICS):
            v = v + thrust
            disp += v
            v = v * f
        self.vel = v4
        hm = h0 + d / 2  # the turn happens during the step: use the halfway heading
        c, s = np.cos(hm), np.sin(hm)
        self.x += disp[0] * c + disp[1] * s
        self.y += disp[0] * s - disp[1] * c
        self.steps += 1

    # ------------------------------------------------------------ drawing
    def _cells(self, k: int, wx: np.ndarray, wy: np.ndarray):
        sc = SCALES[k]
        cx = np.floor(wx / sc).astype(np.int64) + GRID // 2
        cy = np.floor(wy / sc).astype(np.int64) + GRID // 2
        ok = (cx >= 0) & (cx < GRID) & (cy >= 0) & (cy < GRID)
        return cx[ok], cy[ok], ok

    def draw(self, obs: dict) -> None:
        rays = obs["rays"]
        dist = np.asarray(rays["distance"], np.float64) * RAY_RANGE
        hit = np.asarray(rays["hit_type"])
        ang = self.heading + RAY_ANGLES
        dx, dy = np.cos(ang), np.sin(ang)
        # free space: points along each ray up to just before what it hit
        reach = np.where(hit == 0, dist, np.maximum(dist - 24.0, 0.0))
        fx = (self.x + _SAMPLES * reach * dx).ravel()
        fy = (self.y + _SAMPLES * reach * dy).ravel()
        ex, ey = self.x + dist * dx, self.y + dist * dy  # where each ray ended
        ve = obs["visible_entities"]
        m = np.asarray(ve["mask"]) > 0
        et = np.asarray(ve["entity_type"])
        feat = np.asarray(ve["features"], np.float64)
        ed = feat[:, 0] * RAY_RANGE
        ea = self.heading + np.arctan2(feat[:, 1], feat[:, 2])
        nx, ny = self.x + ed * np.cos(ea), self.y + ed * np.sin(ea)

        for k, g in enumerate(self.grids):
            g[ENEMY] *= ENEMY_FADE
            cx, cy, _ = self._cells(k, fx, fy)
            g[FREE, cy, cx] = 1.0
            g[WALL:LOCKED + 1, cy, cx] = 0.0  # seen through: not blocked (any more)
            for ch, code in ((WALL, HIT_WALL), (DOOR, HIT_DOOR), (LOCKED, HIT_LOCKED)):
                sel = hit == code
                if sel.any():
                    hx, hy, _ = self._cells(k, ex[sel], ey[sel])
                    g[ch, hy, hx] = 1.0
            for ch, types in ((ENEMY, ENEMY_TYPES), (KEY, (KEY_TYPE,))):
                sel = m & np.isin(et, types)
                if sel.any():
                    hx, hy, _ = self._cells(k, nx[sel], ny[sel])
                    g[ch, hy, hx] = 1.0
            px, py, _ = self._cells(k, np.array([self.x]), np.array([self.y]))
            g[TRAIL, py, px] = 1.0

    def crops(self) -> np.ndarray:
        """(2, CHANNELS, CROP, CROP) float32: near and far crop, forward is up, left is left."""
        c, s = np.cos(self.heading), np.sin(self.heading)
        out = np.zeros((len(SCALES), CHANNELS, CROP, CROP), np.float32)
        for k, g in enumerate(self.grids):
            sc = SCALES[k]
            f, l = _OFFS[:, 0] * sc, _OFFS[:, 1] * sc
            wx = self.x + f * c - l * s
            wy = self.y + f * s + l * c
            cx, cy, ok = self._cells(k, wx, wy)
            flat = np.zeros((CHANNELS, CROP * CROP), np.float32)
            flat[:, ok] = g[:, cy, cx]
            out[k] = flat.reshape(CHANNELS, CROP, CROP)
        return out

    def observe(self, obs: dict, action_row: np.ndarray | None = None) -> np.ndarray:
        """Call with the first observation of a map (no action), then after every action."""
        if action_row is None:
            self.vel = np.array([obs["vitals"][VEL_F], obs["vitals"][VEL_R]], np.float64)
        else:
            self.move(action_row, obs)
        self.draw(obs)
        return self.crops()
