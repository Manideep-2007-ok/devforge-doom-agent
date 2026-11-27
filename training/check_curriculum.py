"""Smoke-test authored maps and the PPO reward tracker on a ViZDoom host."""

import numpy as np

from doom_env import make_env
from doom_env.spaces import BUTTONS
from training.build_curriculum import LAYOUTS, MAP_DIR


STRAIGHT_ROUTES = {
    "train/01_straight": (0, 0, 0),
    "train/03_door": (0, 1, 0),
    "train/04_red_key": (1, 0, 1),
}


def check_map(name, make_tracker):
    env = make_env(MAP_DIR / f"{name}.wad", "MAP01", episode_ticks=1000, render_buffers=False)
    try:
        observation, _ = env.reset(seed=0)
        if not env.observation_space.contains(observation):
            raise RuntimeError(f"{name} returned an invalid observation")
        tracker = make_tracker()
        state = env.privileged_state()
        tracker.reset(state)
        direction = np.zeros(env.action_space.shape, dtype=np.int8)

        expected = STRAIGHT_ROUTES.get(name)
        if expected is None:
            env.step(direction)
            tracker(env.privileged_state())
            print(f"{name}: loaded and stepped")
            return

        direction[BUTTONS.index("MOVE_FORWARD")] = 1
        use = np.zeros_like(direction)
        use[BUTTONS.index("USE")] = 1
        last_x = state["x"]
        # These straight test maps face east. Pulse USE only after waiting at an obstacle.
        for step in range(1, 251):
            stalled = abs(state["x"] - last_x) < 0.5
            last_x = state["x"]
            _, _, terminated, truncated, _ = env.step(use if stalled and step % 20 == 0 else direction)
            state = env.privileged_state()
            tracker(state)
            if terminated or truncated:
                break

        stats = tracker.stats
        observed = stats["keys"], stats["doors"], stats["locked"]
        if not state["exit_reached"] or stats["exit"] != 1 or observed != expected:
            raise RuntimeError(f"{name} failed progression check: {stats}")
        print(f"{name}: {step} steps, exit, keys/doors/locked {observed}, reward {stats['ret']:.3f}")
    finally:
        env.close()


def main():
    from training.rewards import RewardTracker

    for name in LAYOUTS:
        check_map(name, RewardTracker)


if __name__ == "__main__":
    main()
