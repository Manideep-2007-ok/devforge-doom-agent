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
    too_close: float = -0.02  # per step with an enemy within arm's reach: shoot from a distance
    projectile_near: float = -0.05  # per step with a fireball/rocket within projectile_range: step aside
    projectile_range: float = 160.0
    gunner_exposed: float = -0.01  # per step a bullet-firing enemy sees you and you are not aiming at it:
    gunner_range: float = 1024.0
    chaingunner_extra: float = -0.02  # on top of gunner_exposed when that gunner is a chaingunner (3x in total)  #   shoot it or get out of its sight (chaingunner, shotgunner, zombieman)
    close_range: float = 128.0
    exit_pull: float = 0.004  # per map unit closer to the exit, within exit_radius (training-only knowledge)
    exit_radius: float = 512.0
    door_pull: float = 0.001  # with a key in hand: per map unit closer to the nearest unopened locked door
    door_radius: float = 2048.0
    wrong_door: float = -0.02  # per step next to (within 112 units of) a locked door it has NO key for
    wrong_door_range: float = 112.0
    use_door: float = 0.05  # pressing USE while a closed door is right in front (doors open only with USE
    use_door_range: float = 80.0  #   and shut again after ~4 s); max once per 12 steps, 8 times per game
    use_exit: float = 0.1  # pressing USE within 96 units of the exit (exit SWITCHES need USE); max 5 per game
    use_exit_range: float = 96.0
    find_attacker: float = 0.15  # hurt with no enemy in view, then an enemy comes into view within 12 steps
    find_attacker_steps: int = 12  #   (turn round to face what is shooting you from behind)
    chaingunner_kill: float = 1.0  # extra on top of `kill` when a chaingunner dies (3x a normal kill)
    stuck: float = -0.01  # per step after moving less than 32 units in the last 24 steps (pinned at a door, a wall)
    stuck_steps: int = 24
    stuck_dist: float = 32.0


@dataclass
class RewardTracker:
    """Per-episode memory of what has already been rewarded."""

    w: RewardWeights = field(default_factory=RewardWeights)
    exit_xy: tuple[float, float] | None = None  # set by the trainer from the map file (exit_finder.py)
    doors_xy: tuple = ()  # (sector, x, y, key flags) of each locked door, from the map file (exit_finder.locked_doors)

    def reset(self, ps: dict) -> None:
        self.cells: set[tuple[int, int]] = {self._cell(ps)}
        self.entries: dict[tuple[int, int], int] = {self._cell(ps): 1}
        self.last_cell = self._cell(ps)
        self.key_d: float | None = None
        self.exit_phi = self._exit_phi(ps)
        self.door_phi, self.door_target = None, None
        self.trail = []
        self.length_now, self.last_use = 0, -100
        self.enemy_in_view, self.hurt_unseen, self.cg_n = False, 0, None
        self.used_now = False
        self.opened: set[int] = set()
        self.kills = ps["kills"]
        self.damage = ps["damage_taken"]
        self.health = ps["health"]
        self.keys = np.asarray(ps["keys"]).copy()
        self.stats = dict(ret=0.0, kills=0, keys=0, doors=0, locked=0, cells=1, exit=0, death=0, length=0, wasted=0, healed=0.0, revisits=0, close=0, exit_d=-1.0, dodge=0, exposed=0, chaingun=0, wrong_door=0, stuck=0, use_door=0, found=0, cg_kills=0, use_exit=0)
        self.parts = dict(exit=0.0, key=0.0, door=0.0, cell=0.0, kill=0.0, damage=0.0, death=0.0, time=0.0, waste=0.0, heal=0.0, revisit=0.0, pull=0.0, close=0.0, exit_pull=0.0, dodge=0.0, door_pull=0.0, exposed=0.0, wrong_door=0.0, stuck=0.0, use_door=0.0, found=0.0, cg_kill=0.0, use_exit=0.0)

    def shot(self, obs: dict, fired: bool) -> float:
        """Small cost for pulling the trigger when no enemy is roughly straight ahead."""
        if not fired or enemy_ahead(obs):
            return 0.0
        self.parts["waste"] += self.w.wasted_shot
        self.stats["wasted"] += 1
        self.stats["ret"] += self.w.wasted_shot
        return self.w.wasted_shot

    def look(self, obs: dict, chaingunners=(), used: bool = False) -> float:
        self.used_now = used
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
        if ((m > 0) & (t == PROJECTILE_TYPE) & (d < self.w.projectile_range)).any():
            r += self.w.projectile_near
            self.parts["dodge"] += self.w.projectile_near
            self.stats["dodge"] += 1
        gun = (m > 0) & (t == HITSCAN_TYPE) & (d < self.w.gunner_range)
        if gun.any() and not (gun & (np.abs(f[:, 1]) < AIM_SIN) & (f[:, 2] > 0)).any():
            r += self.w.gunner_exposed
            self.parts["exposed"] += self.w.gunner_exposed
            self.stats["exposed"] += 1
            if len(chaingunners) and (np.abs(d[gun][:, None] - np.asarray(chaingunners)[None, :]) < 16).any():
                r += self.w.chaingunner_extra
                self.parts["exposed"] += self.w.chaingunner_extra
                self.stats["chaingun"] += 1
        if used and self.stats["use_door"] < 8 and self.length_now - self.last_use >= 12:
            ht, rd = obs["rays"]["hit_type"], obs["rays"]["distance"]
            mid = [-2, -1, 0, 1]  # rays go all the way round, ray 0 = straight ahead: these 4 = +-11 degrees
            if ((ht[mid] == RAY_DOOR) & (rd[mid] * 1024.0 < self.w.use_door_range)).any():
                r += self.w.use_door
                self.parts["use_door"] += self.w.use_door
                self.stats["use_door"] += 1
                self.last_use = self.length_now
        self.enemy_in_view = bool(((m > 0) & np.isin(t, ENEMY_TYPES)).any())
        if self.hurt_unseen > 0:
            self.hurt_unseen -= 1
            if self.enemy_in_view:
                r += self.w.find_attacker
                self.parts["found"] += self.w.find_attacker
                self.stats["found"] += 1
                self.hurt_unseen = 0
        n = len(chaingunners)
        if self.cg_n is not None and n < self.cg_n:
            r += self.w.chaingunner_kill * (self.cg_n - n)
            self.parts["cg_kill"] += self.w.chaingunner_kill * (self.cg_n - n)
            self.stats["cg_kills"] += self.cg_n - n
        self.cg_n = n
        self.length_now += 1
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
            if not self.enemy_in_view:
                self.hurt_unseen = w.find_attacker_steps
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

        # Door pull: once a key is held, lead it back to a locked door it has not opened yet.
        # Potential-based, and restarted whenever the target door changes, so it cannot be farmed.
        held = np.asarray(ps["keys"]) > 0.5
        if self.doors_xy and held.any() and not ps["finished"]:
            # only doors that a key it HOLDS can open: the blue key leads to the blue door
            todo = [(sec, x, y) for sec, x, y, ks in self.doors_xy if sec not in self.opened and held[list(ks)].any()]
            if todo:
                ds = [float(np.hypot(ps["x"] - x, ps["y"] - y)) for _, x, y in todo]
                i = int(np.argmin(ds))
                dphi = max(0.0, w.door_radius - ds[i])
                if self.door_target == todo[i][0] and self.door_phi is not None:
                    pull = w.door_pull * (dphi - self.door_phi)
                    r += pull
                    p["door_pull"] += pull
                self.door_phi, self.door_target = dphi, todo[i][0]
            else:
                self.door_phi = self.door_target = None

        # Waiting at a locked door without its key never helps: go and find the key.
        if self.doors_xy and not ps["finished"]:
            for sec, x, y, ks in self.doors_xy:
                if sec not in self.opened and not held[list(ks)].any() and np.hypot(ps["x"] - x, ps["y"] - y) < w.wrong_door_range:
                    r += w.wrong_door
                    p["wrong_door"] += w.wrong_door
                    s["wrong_door"] += 1
                    break

        # Exit switches: USE pressed right next to the exit.
        if self.used_now and self.exit_xy is not None and s["use_exit"] < 5 and not ps["finished"]:
            if np.hypot(ps["x"] - self.exit_xy[0], ps["y"] - self.exit_xy[1]) < w.use_exit_range:
                r += w.use_exit
                p["use_exit"] += w.use_exit
                s["use_exit"] += 1

        # Stuck: barely moved for a while (door that will not open, wall, corner).
        self.trail.append((ps["x"], ps["y"]))
        if len(self.trail) > w.stuck_steps:
            self.trail.pop(0)
            if not ps["finished"] and np.hypot(ps["x"] - self.trail[0][0], ps["y"] - self.trail[0][1]) < w.stuck_dist:
                r += w.stuck
                p["stuck"] += w.stuck
                s["stuck"] += 1

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
PROJECTILE_TYPE = 4
RAY_DOOR = 2  # rays hit_type "door" (a normal, closed door)
HITSCAN_TYPE = 2  # enemy_hitscan: zombieman, shotgunner, chaingunner (their bullets cannot be dodged)  # fireballs, rockets (doom_env.spaces.ENTITY_TYPES)
AIM_SIN = 0.26  # about 15 degrees either side of straight ahead


def enemy_ahead(obs: dict) -> bool:
    """True if the agent can see an enemy roughly in its crosshair (from the agent's own observation)."""
    ve = obs["visible_entities"]
    f, t, m = ve["features"], ve["entity_type"], ve["mask"]
    hit = (m > 0) & np.isin(t, ENEMY_TYPES) & (np.abs(f[:, 1]) < AIM_SIN) & (f[:, 2] > 0)
    return bool(hit.any())
