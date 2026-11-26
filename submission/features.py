"""Observation features and the action set. NumPy only (no torch), so game worker processes stay small.

Shared by the network (model.py), the submission (agent.py) and training (training/ppo.py).
"""

from __future__ import annotations

import numpy as np

NUM_RAYS = 64
NUM_HIT_TYPES = 6  # none, wall, door, locked door, enemy, item
MAX_ENTITIES = 32
NUM_ENTITY_TYPES = 12
NUM_VITALS = 17

# ---------------------------------------------------------------- action set
# Instead of 2^9 = 512 raw button combinations, the agent picks one of 13 useful ones.
# Same order as Sai's original list in doom_net.py; index 0 must stay idle.
# Order: FWD, BACK, STRAFE_L, STRAFE_R, TURN_L, TURN_R, ATTACK, USE, NEXT_WEAPON
ACTIONS = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0 idle
        [1, 0, 0, 0, 0, 0, 0, 0, 0],  # 1 forward
        [0, 1, 0, 0, 0, 0, 0, 0, 0],  # 2 back
        [0, 0, 1, 0, 0, 0, 0, 0, 0],  # 3 strafe left
        [0, 0, 0, 1, 0, 0, 0, 0, 0],  # 4 strafe right
        [0, 0, 0, 0, 1, 0, 0, 0, 0],  # 5 turn left
        [0, 0, 0, 0, 0, 1, 0, 0, 0],  # 6 turn right
        [1, 0, 0, 0, 1, 0, 0, 0, 0],  # 7 forward + turn left
        [1, 0, 0, 0, 0, 1, 0, 0, 0],  # 8 forward + turn right
        [0, 0, 0, 0, 0, 0, 1, 0, 0],  # 9 attack
        [1, 0, 0, 0, 0, 0, 1, 0, 0],  # 10 forward + attack
        [0, 0, 0, 0, 0, 0, 0, 1, 0],  # 11 use (doors, switches)
        [0, 0, 0, 0, 0, 0, 0, 0, 1],  # 12 next weapon
    ],
    dtype=np.int8,
)
NUM_ACTIONS = len(ACTIONS)
# Mirror trick: flipping left/right in the rays means swapping these actions.
MIRROR_ACTION = np.array([0, 1, 2, 4, 3, 6, 5, 8, 7, 9, 10, 11, 12])

# Rough scales so every vitals number is about 0..1 (networks learn badly from raw 0..200 values).
VITALS_SCALE = np.array(
    [100, 100, 7, 50, 200, 50, 50, 300, 1, 1, 1, 1, 1, 1, 16, 16, 16], dtype=np.float32
)


# ---------------------------------------------------------------- observation -> arrays
def preprocess(obs: dict, prev_action: int | np.ndarray) -> dict[str, np.ndarray]:
    """Turn one raw observation (or a batch with a leading dim) into the compact float arrays the net reads."""
    return {
        "ray_dist": np.asarray(obs["rays"]["distance"], np.float32),
        "ray_hit": np.asarray(obs["rays"]["hit_type"], np.int64),
        "ent_feat": np.asarray(obs["visible_entities"]["features"], np.float32),
        "ent_type": np.asarray(obs["visible_entities"]["entity_type"], np.int64),
        "ent_mask": np.asarray(obs["visible_entities"]["mask"], np.float32),
        "vitals": np.asarray(obs["vitals"], np.float32) / VITALS_SCALE,
        "prev_action": np.asarray(prev_action, np.int64),
    }


def mirror(x: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Left-right mirror of a preprocessed obs. Ray i points i*5.6 deg to the LEFT, so ray i <-> ray (64-i)."""
    idx = (-np.arange(NUM_RAYS)) % NUM_RAYS
    y = dict(x)
    y["ray_dist"] = x["ray_dist"][..., idx]
    y["ray_hit"] = x["ray_hit"][..., idx]
    f = x["ent_feat"].copy()
    f[..., 1] *= -1  # sin(rel_angle): left <-> right
    f[..., 4] *= -1  # rel_vel_right
    y["ent_feat"] = f
    v = x["vitals"].copy()
    v[..., 15] *= -1  # vel_right
    y["vitals"] = v
    y["prev_action"] = MIRROR_ACTION[x["prev_action"]]
    if "map" in x:  # memory map crops: (..., 2, 7, rows, cols), column 0 = far left
        y["map"] = x["map"][..., ::-1].copy()
    return y
