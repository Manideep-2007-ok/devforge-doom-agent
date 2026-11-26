"""Evaluation entry point. The scorer instantiates DoomAgent from this file.

Contract: docs/CONTRACT.md §6. Every action comes from the trained network in doom_net.py.

If act() ever fails, it presses nothing for that step, counts the error and carries on, because an
exception would leave the player standing still for the rest of the map. Set DOOM_STRICT=1 while
testing so that errors stop the run instead of being hidden.
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

from doom_net import ACTION_TABLE, IDLE, encode_observation, load_model

SUBMISSION_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = SUBMISSION_DIR / "weights" / "model.pt"

# True: draw each action from the network's probabilities. False: always take the most likely one.
# Compare both on held-out maps before the final submission.
SAMPLE_ACTIONS = True
SEED = 0


class DoomAgent:
    def __init__(self, observation_space, action_space, device: str):
        torch.set_num_threads(1)
        self.device = torch.device(device)
        self.model = load_model(WEIGHTS_PATH, self.device)
        self.generator = torch.Generator()  # CPU, so choices repeat across devices
        self.strict = os.environ.get("DOOM_STRICT") == "1"
        self.errors = 0
        self.reset()

    def reset(self) -> None:
        """Called before every map: forget everything from the previous one."""
        self.hidden = self.model.initial_state(1, self.device)
        self.prev_action = IDLE
        self.generator.manual_seed(SEED)

    def act(self, observation: dict) -> np.ndarray:
        try:
            return self._choose(observation)
        except Exception:
            if self.strict:
                raise
            self.errors += 1
            if self.errors == 1:  # print once, to keep the logs short
                traceback.print_exc(file=sys.stderr)
            self.prev_action = IDLE
            return ACTION_TABLE[IDLE].copy()

    @torch.inference_mode()
    def _choose(self, observation: dict) -> np.ndarray:
        encoded = encode_observation(observation, self.prev_action)
        inputs = {
            key: torch.from_numpy(value)[None, None].to(self.device)  # (time=1, batch=1, ...)
            for key, value in encoded.items()
        }
        logits, _, hidden = self.model(inputs, self.hidden)
        probabilities = logits[0, 0].softmax(dim=-1).cpu()
        if SAMPLE_ACTIONS:
            choice = int(torch.multinomial(probabilities, 1, generator=self.generator))
        else:
            if not torch.isfinite(probabilities).all():
                raise FloatingPointError("network output is not a number")
            choice = int(probabilities.argmax())
        self.hidden = hidden
        self.prev_action = choice
        return ACTION_TABLE[choice].copy()
