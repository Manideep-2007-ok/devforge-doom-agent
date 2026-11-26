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

from doom_net import ACTION_TABLE, IDLE, MemoryMap, encode_observation, load_model, small_rgb

SUBMISSION_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = SUBMISSION_DIR / "weights" / "model.pt"

# True: draw each action from the network's probabilities. False: always take the most likely one.
# Compare both on held-out maps before the final submission.
SAMPLE_ACTIONS = True
SEED = 0


class DoomAgent:
    def __init__(self, observation_space, action_space, device: str):
        torch.set_num_threads(1)
        self.observation_space = observation_space
        self.device = torch.device(device)
        self.model = load_model(WEIGHTS_PATH, self.device)
        self.generator = torch.Generator()  # CPU, so choices repeat across devices
        self.strict = os.environ.get("DOOM_STRICT") == "1"
        self.errors = 0
        self._warm_up()

    def _warm_up(self) -> None:
        """Make one throwaway decision during start-up (limit 60 s).

        The first decision loads extra code (about 1.5 s, once over 10 s on a laptop), and the
        first act() of a map is limited to 10 s. Paying that cost here keeps act() fast.
        """
        self.reset()  # the network needs its memory and previous action to exist
        try:
            self._choose(self.observation_space.sample())  # made-up input of the right shape
        except Exception:
            if self.strict:
                raise
            traceback.print_exc(file=sys.stderr)
        self.reset()  # forget the made-up input

    def reset(self) -> None:
        """Called before every map: forget everything from the previous one."""
        self.hidden = self.model.initial_state(1, self.device)
        self.prev_action = IDLE
        self.generator.manual_seed(SEED)
        self.memory = MemoryMap() if self.model.use_map else None  # the agent's own map, blank for a new map
        self.first_step = True

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
        if self.memory is not None:  # draw this observation into the map, then let the network read it
            last = None if self.first_step else ACTION_TABLE[self.prev_action]
            encoded["map"] = (self.memory.observe(observation, last) * 255).astype(np.uint8)
        if self.model.use_eye:  # half-size camera image
            encoded["eye"] = small_rgb(observation["rgb"])
        self.first_step = False
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
