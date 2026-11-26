r"""The brain: network, observation encoding and action list, shared by training and agent.py.

agent.py (Sai) only calls: ACTION_TABLE, IDLE, encode_observation, load_model.
Training (training/ppo.py) also uses DoomNet and save_checkpoint.

Data flow for one decision:

    rays (64 directions)  --> RayEncoder (circular conv)    --\
    visible entities (32) --> EntityEncoder (attention)      ---> fuse --> GRU memory --> policy head (13 actions)
    vitals + last action  --> small MLP                     --/                       \-> value head (training only)
    memory map (optional) --> MapEncoder (2D conv)          --/   (added in Run 2: see memory_map.py)

No screen pixels: the rays already say "wall ahead, door on the left", and training is far cheaper.
A sequence passed to forward() without `done` must belong to one episode (memory is not reset inside it).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from memory_map import CHANNELS, SCALES, MemoryMap  # noqa: F401
from features import (  # noqa: F401  (numpy-only part, also used by the game worker processes)
    ACTIONS,
    MAX_ENTITIES,
    MIRROR_ACTION,
    NUM_ACTIONS,
    NUM_ENTITY_TYPES,
    NUM_HIT_TYPES,
    NUM_RAYS,
    NUM_VITALS,
    VITALS_SCALE,
    mirror,
    preprocess,
)

# Names agent.py uses.
ACTION_TABLE = ACTIONS
IDLE = 0
CHECKPOINT_FORMAT = 1


def encode_observation(observation: dict, prev_action: int) -> dict[str, np.ndarray]:
    """One environment observation -> the network's inputs (numpy arrays). Ignores rgb and depth."""
    return preprocess(observation, prev_action)


# ---------------------------------------------------------------- encoders
class RayEncoder(nn.Module):
    """1D convolution over the 64 rays that WRAPS AROUND (ray 63 is next to ray 0).

    A convolution slides the same small pattern detector across all directions, so
    "doorway at the edge of a wall" is recognised the same way ahead, left or behind.
    """

    def __init__(self, out_dim: int = 256):
        super().__init__()
        c_in = 1 + NUM_HIT_TYPES  # distance + one-hot hit type
        self.conv = nn.Sequential(
            nn.Conv1d(c_in, 32, 5, padding=2, padding_mode="circular"),
            nn.ReLU(),
            nn.Conv1d(32, 64, 5, stride=2, padding=2, padding_mode="circular"),  # 64 -> 32
            nn.ReLU(),
            nn.Conv1d(64, 64, 3, stride=2, padding=1, padding_mode="circular"),  # 32 -> 16
            nn.ReLU(),
        )
        self.fc = nn.Linear(64 * 16, out_dim)

    def forward(self, dist: torch.Tensor, hit: torch.Tensor) -> torch.Tensor:
        x = torch.cat([dist.unsqueeze(1), F.one_hot(hit, NUM_HIT_TYPES).float().transpose(1, 2)], 1)
        return F.relu(self.fc(self.conv(x).flatten(1)))


class EntityEncoder(nn.Module):
    """Attention pooling over a variable number of visible things (0 enemies or 30, same code).

    Each entity becomes a vector; a learned "query" asks "what here matters most?" and
    attention returns a weighted summary. A learned 'nothing visible' token is always
    present so the summary is well defined when no entity is in view.
    """

    def __init__(self, out_dim: int = 128, d: int = 64):
        super().__init__()
        self.type_emb = nn.Embedding(NUM_ENTITY_TYPES, 16)
        self.embed = nn.Sequential(nn.Linear(5 + 16, d), nn.ReLU(), nn.Linear(d, d))
        self.null_token = nn.Parameter(torch.zeros(1, 1, d))
        self.query = nn.Parameter(torch.randn(1, 1, d) * 0.1)
        self.attn = nn.MultiheadAttention(d, num_heads=4, batch_first=True)
        self.fc = nn.Linear(d, out_dim)

    def forward(self, feat: torch.Tensor, etype: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b = feat.shape[0]
        e = self.embed(torch.cat([feat, self.type_emb(etype)], -1))  # (B, 32, d)
        e = torch.cat([self.null_token.expand(b, -1, -1), e], 1)  # (B, 33, d)
        # The null token is always valid (1), so attention never sees an all-ignored list (that gives NaN).
        pad = torch.cat([torch.ones(b, 1, device=mask.device), mask.float()], 1) < 0.5  # True = ignore
        out, _ = self.attn(self.query.expand(b, -1, -1), e, e, key_padding_mask=pad)
        return F.relu(self.fc(out.squeeze(1)))


class MapEncoder(nn.Module):
    """2D convolution over the agent's own memory map (two crops: near and far, 7 layers each).

    Sees things the rays cannot: "I already explored the left", "there was a locked door
    behind me", "I saw a key over there". See memory_map.py for how the map is drawn.
    """

    def __init__(self, out_dim: int = 256):
        super().__init__()
        c_in = len(SCALES) * CHANNELS
        self.conv = nn.Sequential(
            nn.Conv2d(c_in, 32, 3, stride=2, padding=1),  # 32 -> 16
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 16 -> 8
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),  # 8 -> 4
            nn.ReLU(),
        )
        self.fc = nn.Linear(64 * 4 * 4, out_dim)

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        x = m.float().flatten(1, 2) / 255.0  # (N, 2, 7, 32, 32) uint8 -> (N, 14, 32, 32) in 0..1
        return F.relu(self.fc(self.conv(x).flatten(1)))


# ---------------------------------------------------------------- full network
class DoomNet(nn.Module):
    """Encoders -> fuse -> GRU memory -> policy and value heads.

    The GRU carries a summary of everything seen so far in this map ("there was a locked
    door behind me"). It is reset to zeros before every map.

    priv_dim > 0 turns on the ASYMMETRIC critic: during training only, the value head also
    sees hidden ground-truth facts. The policy never sees them, so the final agent stays legal.
    """

    def __init__(self, hidden: int = 256, priv_dim: int = 0, use_map: bool = False):
        super().__init__()
        self.use_map = use_map
        if use_map:
            self.map_enc = MapEncoder(256)
            # Starts at ZERO, so a brain loaded from an older (map-less) run plays exactly as
            # before on day one and learns to use the map gradually.
            self.map_proj = nn.Linear(256, hidden)
            nn.init.zeros_(self.map_proj.weight)
            nn.init.zeros_(self.map_proj.bias)
        self.rays = RayEncoder(256)
        self.ents = EntityEncoder(128)
        self.vit = nn.Sequential(nn.Linear(NUM_VITALS + NUM_ACTIONS, 64), nn.ReLU())
        self.fuse = nn.Sequential(nn.Linear(256 + 128 + 64, hidden), nn.ReLU())
        self.gru = nn.GRU(hidden, hidden)
        self.policy = nn.Linear(hidden, NUM_ACTIONS)
        self.priv_dim = priv_dim
        self.value = nn.Sequential(nn.Linear(hidden + priv_dim, 256), nn.ReLU(), nn.Linear(256, 1))
        self.hidden = hidden
        # Small initial policy weights -> near-uniform action choice at the start (good exploration).
        nn.init.orthogonal_(self.policy.weight, 0.01)
        nn.init.zeros_(self.policy.bias)

    def initial_state(self, batch: int, device="cpu") -> torch.Tensor:
        return torch.zeros(1, batch, self.hidden, device=device)

    def encode(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Per-step features before memory. x tensors have shape (N, ...)."""
        prev = F.one_hot(x["prev_action"], NUM_ACTIONS).float()
        z = self.fuse[0](
            torch.cat(
                [
                    self.rays(x["ray_dist"], x["ray_hit"]),
                    self.ents(x["ent_feat"], x["ent_type"], x["ent_mask"]),
                    self.vit(torch.cat([x["vitals"], prev], -1)),
                ],
                -1,
            )
        )
        if self.use_map:
            z = z + self.map_proj(self.map_enc(x["map"]))
        return F.relu(z)

    def forward(self, x: dict[str, torch.Tensor], h: torch.Tensor, done: torch.Tensor | None = None):
        """x tensors: (T, B, ...). h: (1, B, H). done: (T, B), 1 where an episode started at that step.

        Returns logits (T, B, A), memory outputs (T, B, H), final h.
        """
        t, b = x["ray_dist"].shape[:2]
        flat = {k: v.reshape(t * b, *v.shape[2:]) for k, v in x.items()}
        z = self.encode(flat).reshape(t, b, -1)
        if done is None:
            out, h = self.gru(z, h)
        else:  # step through time so memory is wiped exactly where a new map began
            outs = []
            for i in range(t):
                h = h * (1.0 - done[i]).view(1, b, 1)
                o, h = self.gru(z[i : i + 1], h)
                outs.append(o)
            out = torch.cat(outs, 0)
        return self.policy(out), out, h

    def value_of(self, mem: torch.Tensor, priv: torch.Tensor | None = None) -> torch.Tensor:
        if self.priv_dim:
            mem = torch.cat([mem, priv], -1)
        return self.value(mem).squeeze(-1)


def to_torch(x: dict[str, np.ndarray], device) -> dict[str, torch.Tensor]:
    return {k: torch.as_tensor(v, device=device) for k, v in x.items()}


def save_checkpoint(model: DoomNet, path) -> None:
    """Save the policy in the format agent.py loads (the training-only value head is dropped)."""
    state = {k: v.detach().cpu() for k, v in model.state_dict().items() if not k.startswith("value.")}
    torch.save({"format": CHECKPOINT_FORMAT, "model": state}, path)


def has_map(state: dict) -> bool:
    """True if these weights come from a brain that reads the memory map."""
    return any(k.startswith("map_enc.") for k in state)


def load_model(path, device: torch.device | str) -> DoomNet:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"unknown checkpoint format in {path}")
    model = DoomNet(use_map=has_map(checkpoint["model"]))
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    missing = [k for k in missing if not k.startswith("value.")]
    if missing or unexpected:  # never play with a half-loaded (partly random) brain
        raise ValueError(f"{path} does not match DoomNet: missing {missing[:3]}, unexpected {unexpected[:3]}")
    return model.to(device).eval()
