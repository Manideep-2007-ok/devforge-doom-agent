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
    exit_time_bonus: float = 2.0  # x fraction of time left on exit (the judges reward fast exits too)
    key: float = 3.0  # once per key
    locked_door: float = 3.0  # once per locked door opened
    door: float = 0.5  # once per normal door opened
    new_cell: float = 0.05  # once per 64x64 map cell entered
    kill: float = 0.5
    damage: float = -0.01  # per health point lost
    damage_low_extra: float = 1.0  # damage costs up to 2x when health is low: fight when healthy, back off when hurt
    heal: float = 0.01  # per health point gained (medkits, stimpacks): the mirror of `damage`
    death: float = -3.0
    death_time: float = 2.0  # extra cost x fraction of time left: an early death costs more (like the judges' -20 x time left)
    timeout: float = -1.0
    step: float = -0.001
    wasted_shot: float = -0.02  # firing with no enemy in front (25 wasted shots = one kill)
    cell_size: float = 64.0
    revisit: float = -0.02  # entering a cell for the 3rd+ time (going back once is free): stops circling
    key_pull: float = 0.001  # per map unit closer to the nearest VISIBLE key (walking away takes it back)
    too_close: float = -0.01  # per step with an enemy within arm's reach: shoot from a distance
    close_range: float = 128.0
    exit_pull: float = 0.004  # per map unit closer to the exit, within exit_radius (training-only knowledge)
    exit_radius: float = 512.0


@dataclass
class RewardTracker:
    """Per-episode memory of what has already been rewarded."""

    w: RewardWeights = field(default_factory=RewardWeights)
    exit_xy: tuple[float, float] | None = None  # set by the trainer from the map file (exit_finder.py)

    def reset(self, ps: dict) -> None:
        self.cells: set[tuple[int, int]] = {self._cell(ps)}
        self.entries: dict[tuple[int, int], int] = {self._cell(ps): 1}
        self.last_cell = self._cell(ps)
        self.key_d: float | None = None
        self.exit_phi = self._exit_phi(ps)
        self.opened: set[int] = set()
        self.kills = ps["kills"]
        self.damage = ps["damage_taken"]
        self.health = ps["health"]
        self.keys = np.asarray(ps["keys"]).copy()
        self.stats = dict(ret=0.0, kills=0, keys=0, doors=0, locked=0, cells=1, exit=0, death=0, length=0, wasted=0, healed=0.0, revisits=0, close=0, exit_d=-1.0)
        self.parts = dict(exit=0.0, key=0.0, door=0.0, cell=0.0, kill=0.0, damage=0.0, death=0.0, time=0.0, waste=0.0, heal=0.0, revisit=0.0, pull=0.0, close=0.0, exit_pull=0.0)

    def shot(self, obs: dict, fired: bool) -> float:
        """Small cost for pulling the trigger when no enemy is roughly straight ahead."""
        if not fired or enemy_ahead(obs):
            return 0.0
        self.parts["waste"] += self.w.wasted_shot
        self.stats["wasted"] += 1
        self.stats["ret"] += self.w.wasted_shot
        return self.w.wasted_shot

    def look(self, obs: dict) -> float:
        """Rewards from what the agent can see (its own observation): key pull and enemy too close."""
        ve = obs["visible_entities"]
        f, t, m = ve["features"], ve["entity_type"], ve["mask"]
        d = np.asarray(f[:, 0], np.float64) * 1024.0
        r = 0.0
        keys = (m > 0) & (t == KEY_TYPE)
        kd = float(d[keys].min()) if keys.any() else None
        if kd is not None and self.key_d is not None:  # potential-based: back-and-forth adds up to zero
            pull = self.w.key_pull * (self.key_d - kd)
            r += pull
            self.parts["pull"] += pull
        self.key_d = kd
        if ((m > 0) & np.isin(t, ENEMY_TYPES) & (d < self.w.close_range)).any():
            r += self.w.too_close
            self.parts["close"] += self.w.too_close
            self.stats["close"] += 1
        self.stats["ret"] += r
        return r

    def _exit_d(self, ps: dict) -> float | None:
        if self.exit_xy is None:
            return None
        return float(np.hypot(ps["x"] - self.exit_xy[0], ps["y"] - self.exit_xy[1]))

    def _exit_phi(self, ps: dict) -> float | None:
        d = self._exit_d(ps)
        return None if d is None else max(0.0, self.w.exit_radius - d)

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
        if c != self.last_cell:
            n = self.entries.get(c, 0) + 1
            self.entries[c] = n
            if n > 2:
                r += w.revisit
                p["revisit"] += w.revisit
                s["revisits"] += 1
            self.last_cell = c

        dk = ps["kills"] - self.kills
        if dk > 0:
            r += w.kill * dk
            p["kill"] += w.kill * dk
            s["kills"] += dk
        self.kills = ps["kills"]

        dd = ps["damage_taken"] - self.damage
        if dd > 0:
            low = 1.0 - min(max(self.health, 0.0), 100.0) / 100.0  # 0 at full health, 1 near death
            cost = w.damage * dd * (1.0 + w.damage_low_extra * low)
            r += cost
            p["damage"] += cost
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

        phi = self._exit_phi(ps)
        if phi is not None and self.exit_phi is not None and not ps["finished"]:
            pull = w.exit_pull * (phi - self.exit_phi)  # potential-based: circling near the exit adds up to zero
            r += pull
            p["exit_pull"] += pull
        self.exit_phi = phi

        if ps["finished"]:
            if ps["exit_reached"]:
                time_left = max(0.0, 1.0 - ps["tic"] / max(ps["episode_ticks"], 1))
                bonus = w.exit + w.exit_health_bonus * max(ps["health"], 0.0) / 100.0 + w.exit_time_bonus * time_left
                r += bonus
                p["exit"] += bonus
                s["exit"] = 1
                d = self._exit_d(ps)
                s["exit_d"] = -1.0 if d is None else d
            elif ps["dead"]:
                time_left = max(0.0, 1.0 - ps["tic"] / max(ps["episode_ticks"], 1))
                cost = w.death - w.death_time * time_left
                r += cost
                p["death"] += cost
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
KEY_TYPE = 9
AIM_SIN = 0.26  # about 15 degrees either side of straight ahead


def enemy_ahead(obs: dict) -> bool:
    """True if the agent can see an enemy roughly in its crosshair (from the agent's own observation)."""
    ve = obs["visible_entities"]
    f, t, m = ve["features"], ve["entity_type"], ve["mask"]
    hit = (m > 0) & np.isin(t, ENEMY_TYPES) & (np.abs(f[:, 1]) < AIM_SIN) & (f[:, 2] > 0)
    return bool(hit.any())
