# Training

A recurrent PPO trainer written for this project (no Stable-Baselines). It trains `DoomNet` on the
same observations the agent receives during evaluation.

## Files

| File | Purpose |
| --- | --- |
| `training/ppo.py` | Trainer: parallel game workers, advantage estimates, PPO updates, logging, checkpoints |
| `training/rewards.py` | Training reward and the extra facts given to the critic |
| `training/mapgen.py` | Random training and test maps in tiers, each checked to be solvable |
| `training/exit_finder.py` | Reads the exit's position from a map file (training reward only) |
| `training/score.py` | Mini-scorer: plays a brain on test maps and gives one comparable number |
| `training/final_pick.py` | Chooses the brain to submit, using four test sets |
| `training/count_shots.py` | Counts shots fired with no enemy in front |
| `training/watch.py` | Watch a brain play, with its senses and its own memory map drawn |
| `training/check_memory_map.py` | Measures how far the memory map's self-tracking drifts |
| `training/export.py` | Copies a checkpoint's policy into `submission/weights/model.pt` |
| `training/bench_speed.py` | Game steps per second and memory per copy (Linux only) |
| `submission/doom_net.py` | Network, shared with the agent |
| `submission/features.py` | Input preparation, action list and left-right mirroring, shared with the agent |
| `submission/memory_map.py` | The agent's own top-down map, shared with the agent |

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

## Make maps

```bash
uv run python training/mapgen.py --out maps/gen/train   --per-tier 150 --seed 1
uv run python training/mapgen.py --out maps/gen/heldout --per-tier 5   --seed 999 --check
```

Tiers: 1 combat, 2 multi-room, 3 key + locked door, 4 two keys and two locked doors, 5 "key school"
(key near the start, locked door right after it; add it with `--tiers 1,2,3,4,5`). Each map is
checked before saving: the exit is reachable, each locked door really blocks it, and each key comes
before its door. Each map also gets a `.json` with its tier and time limit (2,100–4,200 ticks).
`maps/gen/` is not committed: the same seeds make the same maps. Never train on the held-out maps.

## Train

```bash
uv run --group train python training/ppo.py --name run1 --maps "maps/gen/train/*/*.wad"
```

The default maps are `maps/two_rooms.wad` only, so pass `--maps` for real training. It takes
comma-separated patterns, each `path.wad` (map `MAP01`) or `path.wad:MAP03`, and `@file` reads a
long list from a file.

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
| `--episode-ticks` | 2100 | Episode length limit for maps without a `.json` |
| `--episodes-per-map` | 8 | Average episodes before a game switches map |
| `--mirror-prob` | 0.5 | Share of episodes played left-right mirrored |
| `--asym-critic` | true | Let the critic see hidden facts (training only) |
| `--memory-map` | false | Give the network its own memory map (`true` to turn on) |
| `--eye` | false | Give the network the 80×60 camera image (`true` to turn on) |
| `--plr`, `--plr-uniform` | false, 0.5 | Pick maps it wins about half the time more often; this share of picks stays random |
| `--init-from` | — | Start a new run from another run's brain (new parts start at zero) |

Run 2 was started from the Run 1 brain with `--init-from`, `--memory-map true` and `--eye true`.

## Watch a run

The console prints a summary every few updates: reward, kills, map cells visited, doors opened and
exit rate. For graphs:

```bash
uv run --group train tensorboard --logdir runs
```

## Checkpoints

Saved in `runs/<name>/ckpt/` every 10 minutes (`--save-every-min`) and when training stops
(Ctrl+C is safe). Every save is kept (about 11 MB each), because an early one can turn out to be the
best. `runs/` is not committed to Git. Settings of the latest start are written to `runs/<name>/config.json`.

To continue a run, pass the checkpoint and repeat your other settings, which are not restored:

```bash
uv run --group train python training/ppo.py --name run1 --maps "path/to/maps/*.wad" \
    --resume runs/run1/ckpt/<file>.pt
```

## Test a brain

```bash
uv run python training/score.py runs/run1/ckpt/<file>.pt --seeds 3   # one brain on the held-out maps
uv run python training/score.py runs/run1/ckpt --best                  # every save; copies the best to best.pt
uv run python training/watch.py --ckpt runs/run1/ckpt/<file>.pt        # watch it play
```

The score copies the official categories with our own weights: exit 100 (+10 × health left,
+10 × time left), key 20, locked door 20, door 5, new 128×128 area 0.5, kill 3, damage −0.05 per
point, early death −20 × time left. Each map's points are multiplied by its tier. `--outside` tests
on the Freedoom levels that ship with ViZDoom instead.

## Put a model into the submission

```bash
uv run --group train python training/export.py runs/run1/ckpt/<file>.pt
uv run python validate.py
```

`export.py` **overwrites** `submission/weights/model.pt`. It refuses a brain with missing or extra
parts, then checks that the agent loads it, acts, and has exactly the trained weights. Commit the new file (Git LFS) only after the validator passes. Load only checkpoints you made
yourself: the script unpickles the whole file.

## How it works

**Network.** A convolution over the 64 rays that wraps around, attention over the visible entities,
and a small layer for health, ammo, keys, speed and the previous action. With `--memory-map`, a 2D
convolution also reads the agent's own map; with `--eye`, a small 2D convolution reads the 80×60
camera image. These feed a GRU memory, which picks one of 13 button combinations.

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
| Reach the exit | +10, plus up to +2 for health left and up to +2 for time left |
| Pick up a key (once each) | +3 |
| Open a locked door (once each) | +3 |
| Open a normal door (once each) | +0.5 |
| Enter a new 64×64 cell | +0.05 |
| Enter a cell for the 3rd+ time | −0.02 |
| Kill an enemy | +0.5 |
| Lose health | −0.01 per point, up to 2× when health is low |
| Gain health | +0.01 per point |
| Enemy within 128 units | −0.02 per step |
| Fireball or rocket within 160 units | −0.05 per step |
| Fire with no enemy ahead | −0.02 |
| Get closer to a visible key | +0.001 per unit (potential-based) |
| Get closer to the exit, within 512 units | +0.004 per unit (potential-based; exit read from the map file) |
| Die | −3, minus up to 2 more for dying early |
| Run out of time | −1 |
| Every step | −0.001 |

"Potential-based" means the reward is the change in closeness, so walking back and forth adds up to zero.
