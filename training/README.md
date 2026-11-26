# Training

A recurrent PPO trainer written for this project (no Stable-Baselines). It trains `DoomNet` on the
same observations the agent receives during evaluation.

## Files

| File | Purpose |
| --- | --- |
| `training/ppo.py` | Trainer: parallel game workers, advantage estimates, PPO updates, logging, checkpoints |
| `training/rewards.py` | Training reward and the extra facts given to the critic |
| `training/export.py` | Copies a checkpoint's policy into `submission/weights/model.pt` |
| `training/bench_speed.py` | Game steps per second and memory per copy (Linux only) |
| `submission/doom_net.py` | Network, shared with the agent |
| `submission/features.py` | Input preparation, action list and left-right mirroring, shared with the agent |

## Setup

Python 3.13 and [uv](https://docs.astral.sh/uv/), from the repository root:

```bash
uv sync --group train
```

The `train` group adds TensorBoard.

## Check that the pipeline runs

Do this before any long run. It uses the small maps shipped for the tests and works without a GPU.

```bash
uv run --group train python training/ppo.py --name smoke --maps "tests/fixtures/*.wad" \
    --total-steps 20000 --workers 2 --envs-per-worker 4 --device cpu
```

These maps only show that nothing crashes. In one test, a barely trained network already reached the
exit in most episodes on them, so their scores say little about skill.

## Train

```bash
uv run --group train python training/ppo.py --name run1 --maps "path/to/maps/*.wad"
```

The default maps are `maps/two_rooms.wad` only, so pass `--maps` for real training. It takes
comma-separated patterns, each `path.wad` (map `MAP01`) or `path.wad:MAP03`.

| Setting | Default | Meaning |
| --- | --- | --- |
| `--total-steps` | 50e6 | Decisions to train on, counted across resumes |
| `--workers` | 10 | CPU processes running games |
| `--envs-per-worker` | 8 | Games per process |
| `--device` | cuda | Falls back to the CPU if no GPU is found |
| `--rollout` | 128 | Steps per game between updates; a multiple of `--seq-len` |
| `--seq-len` | 32 | Steps the memory is trained across |
| `--epochs`, `--minibatches` | 4, 4 | Passes over each batch, and parts per pass |
| `--lr`, `--lr-end` | 3e-4, 1e-4 | Learning rate, falling steadily |
| `--gamma`, `--gae-lambda` | 0.995, 0.95 | Weight of future reward |
| `--clip` | 0.2 | Limit on how far one update moves the policy |
| `--ent-coef`, `--ent-end` | 0.01, 0.001 | Reward for trying varied actions, falling steadily |
| `--episode-ticks` | 2100 | Episode length limit |
| `--episodes-per-map` | 8 | Average episodes before a game switches map |
| `--mirror-prob` | 0.5 | Share of episodes played left-right mirrored |
| `--asym-critic` | true | Let the critic see hidden facts (training only) |

## Watch a run

The console prints a summary every few updates: reward, kills, map cells visited, doors opened and
exit rate. For graphs:

```bash
uv run --group train tensorboard --logdir runs
```

## Checkpoints

Saved in `runs/<name>/ckpt/` every 10 minutes (`--save-every-min`) and when training stops
(Ctrl+C is safe). **Only the newest 5 are kept**, so copy any good one elsewhere. `runs/` is not
committed to Git. Settings of the latest start are written to `runs/<name>/config.json`.

To continue a run, pass the checkpoint and repeat your other settings, which are not restored:

```bash
uv run --group train python training/ppo.py --name run1 --maps "path/to/maps/*.wad" \
    --resume runs/run1/ckpt/<file>.pt
```

## Put a model into the submission

```bash
uv run --group train python training/export.py runs/run1/ckpt/<file>.pt
uv run python validate.py
```

`export.py` **overwrites** `submission/weights/model.pt`, then checks that the agent loads it and
acts. Commit the new file (Git LFS) only after the validator passes. Load only checkpoints you made
yourself: the script unpickles the whole file.

## How it works

**Network.** A convolution over the 64 rays that wraps around, attention over the visible entities,
and a small layer for health, ammo, keys, speed and the previous action. These feed a GRU memory,
which picks one of 13 button combinations. Screen pixels are not used.

**Loop.** Collect `rollout` steps from every game, estimate how much better each action did than the
critic expected (GAE), then update with clipped PPO on `seq-len` step sequences, so the memory is
trained too. Training stops if the loss stops being a number.

**Critic.** With `--asym-critic`, the value estimate also receives six facts from
`env.privileged_state()` and the reward tracker: time used, health, kills, keys, doors and cells
visited. The policy never receives them, and `export.py` removes the critic from the saved model.

## Reward

Computed from `env.privileged_state()`, in training only. Values are in `rewards.py`.

| Event | Reward |
| --- | --- |
| Reach the exit | +10, plus up to +2 for 100 health left |
| Pick up a key (once each) | +3 |
| Open a locked door (once each) | +3 |
| Open a normal door (once each) | +0.5 |
| Enter a new 64×64 cell | +0.05 |
| Kill an enemy | +0.5 |
| Lose health | −0.01 per point |
| Die | −3 |
| Run out of time | −1 |
| Every step | −0.001 |

## Not in the repository yet

A map generator or other training map set, held-out evaluation maps, a scoring script and a
staged curriculum.
