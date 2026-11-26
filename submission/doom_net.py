"""Network, observation encoding and action list, shared by training and submission/agent.py.

This is a first version for wiring and speed tests. Change the design here (not in a copy), so that
training and evaluation always use the same code.

Training use:
    inputs = encode_observation(obs, prev_action)       # one step, numpy arrays
    batch each key to shape (time, batch, ...)
    logits, value, hidden = model(batch, hidden)
A sequence passed to forward() must belong to one episode: the memory is not reset inside it.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from doom_env import spaces as S

# Each choice the network can make, as the buttons it holds down. Index 0 must stay "idle".
# USE and next weapon fire only when the button goes down, so the network has to pick something
# else for a step before pressing them again. It sees its previous choice, so it can learn that.
ACTION_NAMES = (
    "idle",
    "forward",
    "back",
    "strafe_left",
    "strafe_right",
    "turn_left",
    "turn_right",
    "forward_turn_left",
    "forward_turn_right",
    "attack",
    "forward_attack",
    "use",
    "next_weapon",
)
_PRESSED = {
    "idle": (),
    "forward": ("MOVE_FORWARD",),
    "back": ("MOVE_BACKWARD",),
    "strafe_left": ("MOVE_LEFT",),
    "strafe_right": ("MOVE_RIGHT",),
    "turn_left": ("TURN_LEFT",),
    "turn_right": ("TURN_RIGHT",),
    "forward_turn_left": ("MOVE_FORWARD", "TURN_LEFT"),
    "forward_turn_right": ("MOVE_FORWARD", "TURN_RIGHT"),
    "attack": ("ATTACK",),
    "forward_attack": ("MOVE_FORWARD", "ATTACK"),
    "use": ("USE",),
    "next_weapon": ("SELECT_NEXT_WEAPON",),
}
NUM_ACTIONS = len(ACTION_NAMES)
IDLE = 0
ACTION_TABLE = np.array(
    [[int(button in _PRESSED[name]) for button in S.BUTTONS] for name in ACTION_NAMES],
    dtype=np.int8,
)

NUM_HIT_TYPES = len(S.RAY_HIT_TYPES)
NUM_ENTITY_TYPES = len(S.ENTITY_TYPES)
NUM_ENTITY_FEATURES = len(S.ENTITY_FEATURES)
NUM_VITALS = len(S.VITALS_FIELDS)

# Fixed divisors that bring each input to roughly -1..1. Order of VITALS_FIELDS.
_VITALS_SCALE = np.array(
    [100, 100, 7, 100, 200, 50, 50, 300, 1, 1, 1, 1, 1, 1, 20, 20, 20], dtype=np.float32
)
# [distance, sin, cos, rel_vel_forward, rel_vel_right]
_ENTITY_SCALE = np.array([1, 1, 1, 20, 20], dtype=np.float32)

HIDDEN_SIZE = 256
CHECKPOINT_FORMAT = 1


def encode_observation(observation: dict, prev_action: int) -> dict[str, np.ndarray]:
    """Turn one environment observation into the network's inputs. Ignores rgb and depth."""
    distance = np.asarray(observation["rays"]["distance"], dtype=np.float32)
    hit_type = np.asarray(observation["rays"]["hit_type"], dtype=np.int64)
    rays = np.zeros((1 + NUM_HIT_TYPES, S.NUM_RAYS), dtype=np.float32)
    rays[0] = distance
    rays[1 + hit_type, np.arange(S.NUM_RAYS)] = 1.0

    entities = observation["visible_entities"]
    previous = np.zeros(NUM_ACTIONS, dtype=np.float32)
    previous[prev_action] = 1.0
    vitals = np.asarray(observation["vitals"], dtype=np.float32) / _VITALS_SCALE

    return {
        "rays": rays,
        "entity_features": np.asarray(entities["features"], dtype=np.float32) / _ENTITY_SCALE,
        "entity_type": np.asarray(entities["entity_type"], dtype=np.int64),
        "entity_mask": np.asarray(entities["mask"], dtype=bool),
        "state": np.concatenate([vitals, previous]),
    }


class DoomNet(nn.Module):
    """Circular ray convolution + attention over visible entities + GRU memory."""

    def __init__(self):
        super().__init__()

        def conv(in_ch: int, out_ch: int, size: int, stride: int) -> nn.Conv1d:
            # Circular padding: ray 63 sits next to ray 0, as the directions do.
            return nn.Conv1d(in_ch, out_ch, size, stride, padding=size // 2, padding_mode="circular")

        self.ray_encoder = nn.Sequential(
            conv(1 + NUM_HIT_TYPES, 32, 5, 1), nn.ReLU(),
            conv(32, 32, 5, 2), nn.ReLU(),
            conv(32, 32, 3, 2), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * (S.NUM_RAYS // 4), 128), nn.ReLU(),
        )
        self.state_encoder = nn.Sequential(nn.Linear(NUM_VITALS + NUM_ACTIONS, 64), nn.ReLU())
        self.type_embedding = nn.Embedding(NUM_ENTITY_TYPES, 8)
        self.entity_encoder = nn.Sequential(
            nn.Linear(NUM_ENTITY_FEATURES + 8, 64), nn.ReLU(), nn.Linear(64, 64)
        )
        # An extra "nothing" entry that is always visible, so zero entities is a normal input.
        self.no_entity = nn.Parameter(torch.zeros(64))
        self.query = nn.Linear(64, 64)
        self.fuse = nn.Sequential(nn.Linear(128 + 64 + 64, HIDDEN_SIZE), nn.ReLU())
        self.memory = nn.GRU(HIDDEN_SIZE, HIDDEN_SIZE)
        self.policy_head = nn.Linear(HIDDEN_SIZE, NUM_ACTIONS)
        self.value_head = nn.Linear(HIDDEN_SIZE, 1)  # used only in training

    def initial_state(self, batch: int, device: torch.device | str = "cpu") -> torch.Tensor:
        return torch.zeros(1, batch, HIDDEN_SIZE, device=device)

    def forward(self, inputs: dict[str, torch.Tensor], hidden: torch.Tensor):
        steps, batch = inputs["state"].shape[:2]
        flat = {key: value.flatten(0, 1) for key, value in inputs.items()}
        count = steps * batch

        rays = self.ray_encoder(flat["rays"])
        state = self.state_encoder(flat["state"])

        types = self.type_embedding(flat["entity_type"])
        entities = self.entity_encoder(torch.cat([flat["entity_features"], types], dim=-1))
        nothing = self.no_entity.expand(count, 1, -1)
        entities = torch.cat([nothing, entities], dim=1)
        visible = torch.cat(
            [flat["entity_mask"].new_ones(count, 1), flat["entity_mask"]], dim=1
        )
        scores = torch.einsum("nd,ned->ne", self.query(state), entities) / 8.0
        weights = scores.masked_fill(~visible, float("-inf")).softmax(dim=1)
        pooled = torch.einsum("ne,ned->nd", weights, entities)

        fused = self.fuse(torch.cat([rays, state, pooled], dim=-1))
        output, hidden = self.memory(fused.view(steps, batch, -1), hidden)
        return self.policy_head(output), self.value_head(output).squeeze(-1), hidden


def save_checkpoint(model: DoomNet, path) -> None:
    """Save weights in the format agent.py loads."""
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    torch.save({"format": CHECKPOINT_FORMAT, "model": state}, path)


def load_model(path, device: torch.device | str) -> DoomNet:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"unknown checkpoint format in {path}")
    model = DoomNet()
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval()
