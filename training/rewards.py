"""Training reward: copies the scorer's priorities (exit > keys/doors > exploring > combat).

Reads env.privileged_state() (ground truth: position, kills, keys, door heights). This is used
ONLY to compute rewards during training; the agent's network never sees it.

Every "once per" rule stops reward farming (e.g. opening the same door again and again).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RewardWeights:
    exit: float = 10.0
    exit_health_bonus: float = 2.0  # x health/100 on exit
    key: float = 3.0  # once per key
    locked_door: float = 3.0  # once per locked door opened
    door: float = 0.5  # once per normal door opened
    new_cell: float = 0.05  # once per 64x64 map cell entered
    kill: float = 0.5
    damage: float = -0.01  # per health point lost
    heal: float = 0.01  # per health point gained (medkits, stimpacks): the mirror of `damage`
    death: float = -3.0
    timeout: float = -1.0
    step: float = -0.001
    wasted_shot: float = -0.02  # firing with no enemy in front (25 wasted shots = one kill)
    cell_size: float = 64.0


@dataclass
class RewardTracker:
    """Per-episode memory of what has already been rewarded."""

    w: RewardWeights = field(default_factory=RewardWeights)

    def reset(self, ps: dict) -> None:
        self.cells: set[tuple[int, int]] = {self._cell(ps)}
        self.opened: set[int] = set()
        self.kills = ps["kills"]
        self.damage = ps["damage_taken"]
        self.health = ps["health"]
        self.keys = np.asarray(ps["keys"]).copy()
        self.stats = dict(ret=0.0, kills=0, keys=0, doors=0, locked=0, cells=1, exit=0, death=0, length=0, wasted=0, healed=0.0)
        self.parts = dict(exit=0.0, key=0.0, door=0.0, cell=0.0, kill=0.0, damage=0.0, death=0.0, time=0.0, waste=0.0, heal=0.0)

    def shot(self, obs: dict, fired: bool) -> float:
        """Small cost for pulling the trigger when no enemy is roughly straight ahead."""
        if not fired or enemy_ahead(obs):
            return 0.0
        self.parts["waste"] += self.w.wasted_shot
        self.stats["wasted"] += 1
        self.stats["ret"] += self.w.wasted_shot
        return self.w.wasted_shot

    def _cell(self, ps: dict) -> tuple[int, int]:
        return int(ps["x"] // self.w.cell_size), int(ps["y"] // self.w.cell_size)

    def __call__(self, ps: dict) -> float:
        w, p, s = self.w, self.parts, self.stats
        r = w.step
        p["time"] += w.step

        c = self._cell(ps)
        if c not in self.cells:
            self.cells.add(c)
            r += w.new_cell
            p["cell"] += w.new_cell
            s["cells"] += 1

        dk = ps["kills"] - self.kills
        if dk > 0:
            r += w.kill * dk
            p["kill"] += w.kill * dk
            s["kills"] += dk
        self.kills = ps["kills"]

        dd = ps["damage_taken"] - self.damage
        if dd > 0:
            r += w.damage * dd
            p["damage"] += w.damage * dd
        self.damage = ps["damage_taken"]

        # Health only goes up by picking something up, so this cannot be farmed. A medkit is worth
        # more when hurt (it restores more), which teaches "low health -> go to the medkit".
        dh = ps["health"] - self.health
        if dh > 0 and not ps["dead"]:
            r += w.heal * dh
            p["heal"] += w.heal * dh
            s["healed"] += dh
        self.health = ps["health"]

        keys = np.asarray(ps["keys"])
        new_keys = int(((keys > 0.5) & (self.keys < 0.5)).sum())
        if new_keys:
            r += w.key * new_keys
            p["key"] += w.key * new_keys
            s["keys"] += new_keys
        self.keys = keys.copy()

        lock, ceil, floor = ps["door_lock"], ps["sector_ceiling"], ps["sector_floor"]
        if lock is not None:
            open_now = np.flatnonzero((lock >= 0) & (ceil - floor > 32.0))
            for sec in open_now:
                if sec not in self.opened:
                    self.opened.add(int(sec))
                    if lock[sec] > 0:
                        r += w.locked_door
                        p["door"] += w.locked_door
                        s["locked"] += 1
                    else:
                        r += w.door
                        p["door"] += w.door
                        s["doors"] += 1

        if ps["finished"]:
            if ps["exit_reached"]:
                bonus = w.exit + w.exit_health_bonus * max(ps["health"], 0.0) / 100.0
                r += bonus
                p["exit"] += bonus
                s["exit"] = 1
            elif ps["dead"]:
                r += w.death
                p["death"] += w.death
                s["death"] = 1
            else:
                r += w.timeout
                p["time"] += w.timeout

        s["ret"] += r
        s["length"] += 1
        return r

    def critic_features(self, ps: dict) -> np.ndarray:
        """Hidden facts for the asymmetric critic (training only)."""
        return np.array(
            [
                ps["tic"] / max(ps["episode_ticks"], 1),  # how much time is used up
                max(ps["health"], 0.0) / 100.0,
                self.stats["kills"] / 10.0,
                float(np.sum(np.asarray(ps["keys"]) > 0.5)) / 3.0,
                (self.stats["doors"] + self.stats["locked"]) / 5.0,
                self.stats["cells"] / 200.0,
            ],
            np.float32,
        )


PRIV_DIM = 6

ENEMY_TYPES = (1, 2, 3)  # enemy_melee, enemy_hitscan, enemy_projectile (doom_env.spaces.ENTITY_TYPES)
AIM_SIN = 0.26  # about 15 degrees either side of straight ahead


def enemy_ahead(obs: dict) -> bool:
    """True if the agent can see an enemy roughly in its crosshair (from the agent's own observation)."""
    ve = obs["visible_entities"]
    f, t, m = ve["features"], ve["entity_type"], ve["mask"]
    hit = (m > 0) & np.isin(t, ENEMY_TYPES) & (np.abs(f[:, 1]) < AIM_SIN) & (f[:, 2] > 0)
    return bool(hit.any())
