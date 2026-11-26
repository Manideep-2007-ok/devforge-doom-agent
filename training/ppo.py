"""Recurrent PPO trainer for DoomNet. Written from scratch (no Stable-Baselines) so judges can read it.

The loop, repeated until the step budget is spent:

  1. COLLECT  - 10 CPU worker processes each run several game copies. The GPU network decides for
                all copies at once (one batched forward pass per step). We record 128 steps per copy.
  2. ADVANTAGE- for each step, "was this better or worse than the critic expected?" (GAE).
  3. UPDATE   - 4 passes over the batch, in 32-step sequences so the GRU memory is trained too.
                PPO clips each update so the policy only moves a small, safe step.
  4. LOG/SAVE - TensorBoard every update, checkpoint every 10 minutes.

    .venv/bin/python training/ppo.py --name smoke --maps "tests/fixtures/*.wad" --total-steps 2e6
"""

from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submission"))
sys.path.insert(0, str(ROOT / "training"))

from features import ACTIONS, MIRROR_ACTION, mirror, preprocess, small_rgb  # noqa: E402  (numpy only: workers never load torch)
from rewards import PRIV_DIM, RewardTracker, RewardWeights  # noqa: E402

ATTACK_BUTTON = 6  # column of ATTACK in each ACTIONS row (doom_env.spaces.BUTTONS order)


# ============================================================== config
@dataclass
class Config:
    name: str = "run"
    maps: str = "maps/two_rooms.wad"  # comma-separated globs, each "path.wad" or "path.wad:MAP03"
    total_steps: float = 50e6
    workers: int = 10
    envs_per_worker: int = 8  # 80 game copies: measured ~3.2k steps/s on the RTX 3050 laptop
    rollout: int = 128  # steps per copy per batch
    seq_len: int = 32  # GRU unroll length in training
    epochs: int = 4
    minibatches: int = 4
    lr: float = 3e-4
    lr_end: float = 1e-4
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    ent_end: float = 0.001
    max_grad_norm: float = 0.5
    episode_ticks: int = 2100
    episodes_per_map: int = 8  # loading a map costs ~170 ms vs ~1 ms to restart the same one
    mirror_prob: float = 0.5
    asym_critic: bool = True
    overlap: bool = True  # two game groups take turns: GPU decides for one while the other plays
    seed: int = 0
    resume: str = ""
    init_from: str = ""  # start a NEW run from another run's brain (e.g. Run 2 from Run 1); fresh optimizer
    memory_map: bool = False  # give the brain its own memory map (memory_map.py); Run 2 on
    eye: bool = False  # let the brain see the half-size camera image (to spot the exit); Run 2 on
    save_every_min: float = 10.0
    device: str = "cuda"


# ============================================================== maps
def expand_maps(spec: str) -> list[tuple[str, str]]:
    out = []
    for part in spec.split(","):
        part = part.strip()
        path, _, mapname = part.partition(":")
        for p in sorted(glob.glob(path)):
            out.append((p, mapname or "MAP01"))
    if not out:
        raise SystemExit(f"no maps matched {spec!r}")
    return out


def map_ticks(wad: str, default: int) -> int:
    """Tick cap for a map: from its <name>.json (written by mapgen.py) if present, else the default."""
    meta = Path(wad).with_suffix(".json")
    if meta.exists():
        try:
            return int(json.loads(meta.read_text()).get("ticks", default))
        except (ValueError, OSError):
            pass
    return default


# ============================================================== workers (CPU game copies)
def _strip(x: dict) -> dict:
    x = dict(x)
    x.pop("prev_action")
    return x


class _Slot:
    """One game copy: picks a random map each episode, shapes reward, mirrors half the episodes."""

    def __init__(self, maps, cfg: Config, rng: random.Random, weights: RewardWeights):
        self.maps, self.cfg, self.rng = maps, cfg, rng
        self.env = None
        self.key = None
        self.left_on_map = 0
        self.tracker = RewardTracker(weights)
        self.memory = None
        if cfg.memory_map:
            from memory_map import MemoryMap

            self.memory = MemoryMap()

    def new_episode(self):
        from doom_env import make_env

        if self.left_on_map <= 0:
            key = self.rng.choice(self.maps)
            # random count so the 40 copies don't all switch maps at the same moment
            self.left_on_map = self.rng.randint(max(1, self.cfg.episodes_per_map // 2), self.cfg.episodes_per_map * 3 // 2)
        else:
            key = self.key
        self.left_on_map -= 1
        if key != self.key:
            if self.env is not None:
                self.env.close()
            self.env = make_env(key[0], key[1], episode_ticks=map_ticks(key[0], self.cfg.episode_ticks), render_buffers=self.cfg.eye)
            self.key = key
        obs, _ = self.env.reset(seed=self.rng.randrange(1 << 30))
        self.last_obs = obs
        self.mirrored = self.rng.random() < self.cfg.mirror_prob
        ps = self.env.privileged_state()
        self.tracker.reset(ps)
        if self.memory is not None:
            self.memory.reset()
        return self._obs(obs, None), self.tracker.critic_features(ps)

    def _obs(self, obs, action_row):
        x = preprocess(obs, 0)
        if self.memory is not None:  # drawn from the REAL (unmirrored) view; mirror() flips it with the rest
            x["map"] = (self.memory.observe(obs, action_row) * 255).astype(np.uint8)
        if self.cfg.eye:
            x["eye"] = small_rgb(obs["rgb"])
        return _strip(mirror(x) if self.mirrored else x)

    def step(self, a: int):
        real = MIRROR_ACTION[a] if self.mirrored else a
        r = self.tracker.shot(self.last_obs, bool(ACTIONS[real][ATTACK_BUTTON]))
        obs, _, term, trunc, _ = self.env.step(ACTIONS[real])
        self.last_obs = obs
        ps = self.env.privileged_state()
        r += self.tracker(ps)
        done = term or trunc
        stats = None
        if done:
            stats = dict(self.tracker.stats, **{f"r_{k}": v for k, v in self.tracker.parts.items()})
            stats["map"] = os.path.basename(self.key[0]) + ":" + self.key[1]
            x, priv = self.new_episode()
        else:
            x, priv = self._obs(obs, ACTIONS[real]), self.tracker.critic_features(ps)
        return x, priv, r, done, stats


def worker_main(remote, maps, cfg: Config, seed: int, weights: RewardWeights):
    import signal

    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ctrl+C goes to the trainer, which shuts workers down cleanly
    os.environ["OMP_NUM_THREADS"] = "1"
    rng = random.Random(seed)
    slots = [_Slot(maps, cfg, rng, weights) for _ in range(cfg.envs_per_worker)]
    try:
        while True:
            cmd, data = remote.recv()
            # Results are packed into a few NumPy arrays per worker: far cheaper to send than many small dicts.
            if cmd == "reset":
                res = [s.new_episode() for s in slots]
                remote.send((stack([r[0] for r in res]), np.stack([r[1] for r in res])))
            elif cmd == "step":
                res = [s.step(int(a)) for s, a in zip(slots, data)]
                x, priv, r, d, st = zip(*res)
                remote.send((stack(list(x)), np.stack(priv), np.array(r, np.float32), np.array(d, np.float32),
                             [z for z in st if z is not None]))
            elif cmd == "close":
                break
    finally:
        for s in slots:
            if s.env is not None:
                s.env.close()


def stack(dicts: list[dict]) -> dict[str, np.ndarray]:
    return {k: np.stack([d[k] for d in dicts]) for k in dicts[0]}


def concat(dicts: list[dict]) -> dict[str, np.ndarray]:
    return {k: np.concatenate([d[k] for d in dicts]) for k in dicts[0]}


class VecEnv:
    def __init__(self, maps, cfg: Config, weights: RewardWeights):
        ctx = mp.get_context("spawn")
        self.remotes, self.procs = [], []
        for i in range(cfg.workers):
            a, b = ctx.Pipe()
            p = ctx.Process(target=worker_main, args=(b, maps, cfg, cfg.seed * 1000 + i, weights), daemon=True)
            p.start()
            self.remotes.append(a)
            self.procs.append(p)
        self.k = cfg.envs_per_worker
        self.n = cfg.workers * self.k
        self.set_groups(1)

    def set_groups(self, g: int):
        """Split the workers into g contiguous groups; group i owns env columns self.slices[i]."""
        w = len(self.remotes)
        cuts = [round(i * w / g) for i in range(g + 1)]
        self.groups = [list(range(cuts[i], cuts[i + 1])) for i in range(g)]
        self.slices = [slice(cuts[i] * self.k, cuts[i + 1] * self.k) for i in range(g)]

    def step_async(self, g: int, actions: np.ndarray):
        for j, i in enumerate(self.groups[g]):
            self.remotes[i].send(("step", actions[j * self.k : (j + 1) * self.k]))

    def step_wait(self, g: int):
        res = [self.remotes[i].recv() for i in self.groups[g]]
        o, p, rew, done, st = zip(*res)
        return concat(list(o)), np.concatenate(p), np.concatenate(rew), np.concatenate(done), [x for s in st for x in s]

    def reset(self):
        for r in self.remotes:
            r.send(("reset", None))
        res = [r.recv() for r in self.remotes]
        return concat([o for o, _ in res]), np.concatenate([p for _, p in res])

    def step(self, actions: np.ndarray):
        for i, r in enumerate(self.remotes):
            r.send(("step", actions[i * self.k : (i + 1) * self.k]))
        res = [r.recv() for r in self.remotes]
        o, p, rew, done, st = zip(*res)
        return concat(list(o)), np.concatenate(p), np.concatenate(rew), np.concatenate(done), [x for s in st for x in s]

    def close(self):
        for r in self.remotes:
            try:
                r.send(("close", None))
            except (BrokenPipeError, EOFError):
                pass
        for p in self.procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()


# ============================================================== helpers
class RunningStd:
    """Tracks the spread of discounted returns so rewards can be scaled to a stable size."""

    def __init__(self):
        self.mean, self.var, self.count = 0.0, 1.0, 1e-4

    def update(self, x: np.ndarray):
        bm, bv, bc = x.mean(), x.var(), x.size
        d = bm - self.mean
        tot = self.count + bc
        self.mean += d * bc / tot
        self.var = (self.var * self.count + bv * bc + d * d * self.count * bc / tot) / tot
        self.count = tot

    @property
    def std(self):
        return float(np.sqrt(self.var) + 1e-8)


def save_checkpoint(path: Path, net, opt, steps, ret_std, cfg):
    import torch

    torch.save(
        {"net": net.state_dict(), "opt": opt.state_dict(), "steps": steps,
         "ret_std": (ret_std.mean, ret_std.var, ret_std.count), "cfg": asdict(cfg)},
        path,
    )


# ============================================================== training loop
def train(cfg: Config):
    import torch
    import torch.nn.functional as F
    from torch.utils.tensorboard import SummaryWriter

    from doom_net import DoomNet, to_torch

    torch.set_num_threads(2)  # leave the other cores to the game workers
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    dev = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    run_dir = ROOT / "runs" / cfg.name
    (run_dir / "ckpt").mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    tb = SummaryWriter(str(run_dir / "tb"))

    maps = expand_maps(cfg.maps)
    print(f"[{cfg.name}] {len(maps)} maps, {cfg.workers}x{cfg.envs_per_worker} game copies, device {dev}", flush=True)

    net = DoomNet(priv_dim=PRIV_DIM if cfg.asym_critic else 0, use_map=cfg.memory_map, use_eye=cfg.eye).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, eps=1e-5)
    ret_std = RunningStd()
    steps = 0
    if cfg.resume:
        ck = torch.load(cfg.resume, map_location=dev, weights_only=False)
        net.load_state_dict(ck["net"])
        opt.load_state_dict(ck["opt"])
        steps = ck["steps"]
        ret_std.mean, ret_std.var, ret_std.count = ck["ret_std"]
        print(f"resumed from {cfg.resume} at {steps:,} steps", flush=True)
    elif cfg.init_from:
        ck = torch.load(cfg.init_from, map_location=dev, weights_only=False)
        missing, unexpected = net.load_state_dict(ck["net"], strict=False)
        if unexpected or any(not k.startswith(("map_enc.", "map_proj.", "eye_enc.", "eye_proj.")) for k in missing):
            raise SystemExit(f"--init-from does not fit: missing {missing[:4]} unexpected {unexpected[:4]}")
        ret_std.mean, ret_std.var, ret_std.count = ck["ret_std"]
        print(f"started from the brain in {cfg.init_from} (new parts: {len(missing)} weight sets)", flush=True)
    print(f"parameters: {sum(p.numel() for p in net.parameters()):,}", flush=True)

    venv = VecEnv(maps, cfg, RewardWeights())
    venv.set_groups(2 if cfg.overlap and cfg.workers >= 2 else 1)
    N, T = venv.n, cfg.rollout
    obs, priv = venv.reset()
    prev_a = np.zeros(N, np.int64)
    h = net.initial_state(N, dev)
    first = np.ones(N, np.float32)  # 1 = this obs is the first of an episode
    disc_ret = np.zeros(N, np.float32)

    keys = list(obs.keys()) + ["prev_action"]
    t_start = time.time()
    last_save = time.time()
    upd = 0
    recent: list[dict] = []

    try:
        while steps < cfg.total_steps:
            frac = min(steps / cfg.total_steps, 1.0)
            lr = cfg.lr + (cfg.lr_end - cfg.lr) * frac
            ent_coef = cfg.ent_coef + (cfg.ent_end - cfg.ent_coef) * frac
            for g in opt.param_groups:
                g["lr"] = lr

            # ---------------- 1. COLLECT
            # Games are split into G groups. While one group's games play their move on the CPU, the GPU
            # already picks the next move for the other group, so neither side sits waiting.
            # Same policy, same data: only the order of the work changes.
            G = len(venv.groups)
            buf = {k: np.zeros((T,) + obs[k].shape, obs[k].dtype) for k in obs}
            buf["prev_action"] = np.zeros((T, N), np.int64)
            b_act = np.zeros((T, N), np.int64)
            b_logp = np.zeros((T, N), np.float32)
            b_val = np.zeros((T, N), np.float32)
            b_rew = np.zeros((T, N), np.float32)
            b_first = np.zeros((T, N), np.float32)
            b_priv = np.zeros((T, N, priv.shape[1]), np.float32)
            b_h = torch.zeros(T, N, net.hidden, device=dev)
            t_of = [0] * G
            t0 = time.time()
            net.eval()
            env_s = 0.0

            def decide(g):
                """Pick actions for group g at its current step and send them to its games."""
                nonlocal h
                t, sl = t_of[g], venv.slices[g]
                x = {k: obs[k][sl] for k in obs}
                x["prev_action"] = prev_a[sl]
                for k in keys:
                    buf[k][t, sl] = x[k]
                b_first[t, sl] = first[sl]
                b_priv[t, sl] = priv[sl]
                with torch.no_grad():
                    xt = {k: v.unsqueeze(0) for k, v in to_torch(x, dev).items()}
                    ft = torch.as_tensor(first[sl], device=dev).unsqueeze(0)
                    hg = h[:, sl] * (1 - ft).unsqueeze(-1)
                    b_h[t, sl] = hg[0]  # memory at the START of this step
                    logits, mem, hg = net(xt, hg, ft)
                    h[:, sl] = hg
                    dist = torch.distributions.Categorical(logits=logits[0])
                    a = dist.sample()
                    v = net.value_of(mem[0], torch.as_tensor(priv[sl], device=dev) if cfg.asym_critic else None)
                a_np = a.cpu().numpy()
                b_act[t, sl] = a_np
                b_logp[t, sl] = dist.log_prob(a).cpu().numpy()
                b_val[t, sl] = v.cpu().numpy()
                venv.step_async(g, a_np)

            def collect(g):
                """Receive group g's results for its current step."""
                t, sl = t_of[g], venv.slices[g]
                o, p, rew, done, stats = venv.step_wait(g)
                for k in obs:
                    obs[k][sl] = o[k]
                priv[sl] = p
                disc_ret[sl] = disc_ret[sl] * cfg.gamma + rew
                ret_std.update(disc_ret[sl])
                disc_ret[sl] *= 1 - done
                b_rew[t, sl] = rew
                recent.extend(stats)
                prev_a[sl] = np.where(done > 0, 0, b_act[t, sl])
                first[sl] = done
                t_of[g] += 1

            for g in range(G):
                decide(g)
            for t in range(T):
                for g in range(G):
                    te = time.time()
                    collect(g)
                    env_s += time.time() - te
                    if t < T - 1:
                        decide(g)
            steps += N * T
            collect_s = time.time() - t0

            with torch.no_grad():
                x = dict(obs, prev_action=prev_a)
                xt = {k: v.unsqueeze(0) for k, v in to_torch(x, dev).items()}
                ft = torch.as_tensor(first, device=dev).unsqueeze(0)
                _, mem, _ = net(xt, h, ft)
                last_v = net.value_of(mem[0], torch.as_tensor(priv, device=dev) if cfg.asym_critic else None).cpu().numpy()

            # ---------------- 2. ADVANTAGE (GAE)
            rew = np.stack(b_rew) / ret_std.std
            val = np.stack(b_val)
            fst = np.stack(b_first)
            adv = np.zeros_like(rew)
            gae = np.zeros(N, np.float32)
            for t in reversed(range(T)):
                nxt_v = last_v if t == T - 1 else val[t + 1]
                nonterm = 1.0 - (first if t == T - 1 else fst[t + 1])  # next obs starts a new episode?
                delta = rew[t] + cfg.gamma * nxt_v * nonterm - val[t]
                gae = delta + cfg.gamma * cfg.gae_lambda * nonterm * gae
                adv[t] = gae
            ret = adv + val

            # ---------------- 3. UPDATE
            t1 = time.time()
            net.train()
            L = cfg.seq_len
            S = T // L  # sequences per copy

            def seqs(a):  # (T, N, ...) -> (L, S*N, ...)
                a = np.asarray(a)
                a = a.reshape(S, L, N, *a.shape[2:]).swapaxes(0, 1)
                return a.reshape(L, S * N, *a.shape[3:])

            data = {k: torch.as_tensor(seqs(np.stack(buf[k])), device=dev) for k in keys}
            d_act = torch.as_tensor(seqs(np.stack(b_act)), device=dev)
            d_logp = torch.as_tensor(seqs(np.stack(b_logp)), device=dev)
            d_val = torch.as_tensor(seqs(val), device=dev)
            d_adv = torch.as_tensor(seqs(adv), device=dev)
            d_ret = torch.as_tensor(seqs(ret), device=dev)
            d_first = torch.as_tensor(seqs(fst), device=dev)
            d_priv = torch.as_tensor(seqs(np.stack(b_priv)), device=dev)
            h_all = b_h  # (T, N, H)
            d_h0 = h_all.reshape(S, L, N, -1)[:, 0].reshape(1, S * N, -1)  # memory at each sequence start

            n_seq = S * N
            mb = max(n_seq // cfg.minibatches, 1)
            log = {"pg": 0.0, "vf": 0.0, "ent": 0.0, "kl": 0.0, "clipfrac": 0.0}
            n_mb = 0
            for _ in range(cfg.epochs):
                perm = torch.randperm(n_seq, device=dev)
                for i in range(0, n_seq, mb):
                    idx = perm[i : i + mb]
                    xb = {k: v[:, idx] for k, v in data.items()}
                    logits, mem, _ = net(xb, d_h0[:, idx].contiguous(), d_first[:, idx])
                    dist = torch.distributions.Categorical(logits=logits)
                    logp = dist.log_prob(d_act[:, idx])
                    ent = dist.entropy().mean()
                    ratio = torch.exp(logp - d_logp[:, idx])
                    a_b = d_adv[:, idx]
                    a_b = (a_b - a_b.mean()) / (a_b.std() + 1e-8)
                    pg = -torch.min(ratio * a_b, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * a_b).mean()
                    v = net.value_of(mem, d_priv[:, idx] if cfg.asym_critic else None)
                    v_clip = d_val[:, idx] + (v - d_val[:, idx]).clamp(-cfg.clip, cfg.clip)
                    vf = 0.5 * torch.max((v - d_ret[:, idx]) ** 2, (v_clip - d_ret[:, idx]) ** 2).mean()
                    loss = pg + cfg.vf_coef * vf - ent_coef * ent
                    if not torch.isfinite(loss):
                        raise FloatingPointError("NaN/Inf in loss - stopping so we don't train on garbage")
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
                    opt.step()
                    with torch.no_grad():
                        log["pg"] += pg.item()
                        log["vf"] += vf.item()
                        log["ent"] += ent.item()
                        log["kl"] += ((ratio - 1) - torch.log(ratio)).mean().item()
                        log["clipfrac"] += ((ratio - 1).abs() > cfg.clip).float().mean().item()
                    n_mb += 1
            update_s = time.time() - t1
            upd += 1

            # ---------------- 4. LOG / SAVE
            sps = N * T / (collect_s + update_s)
            for k, v in log.items():
                tb.add_scalar(f"loss/{k}", v / n_mb, steps)
            tb.add_scalar("perf/steps_per_s", sps, steps)
            tb.add_scalar("perf/collect_s", collect_s, steps)
            tb.add_scalar("perf/env_wait_s", env_s, steps)
            tb.add_scalar("perf/update_s", update_s, steps)
            tb.add_scalar("train/lr", lr, steps)
            tb.add_scalar("train/ent_coef", ent_coef, steps)
            if recent:
                for k in recent[0]:
                    if k != "map":
                        tb.add_scalar(f"episode/{k}", float(np.mean([s[k] for s in recent])), steps)
            if upd % 5 == 0 or upd == 1:
                ep = (f"eps {len(recent):3d} ret {np.mean([s['ret'] for s in recent]):6.2f} "
                      f"kills {np.mean([s['kills'] for s in recent]):.2f} cells {np.mean([s['cells'] for s in recent]):5.1f} "
                      f"doors {np.mean([s['doors'] + s['locked'] for s in recent]):.2f} exit {np.mean([s['exit'] for s in recent]):.2f}"
                      if recent else "no finished episodes yet")
                print(f"[{cfg.name}] {steps / 1e6:6.2f}M steps | {sps:5.0f} sps (env {env_s:.1f}s/collect {collect_s:.1f}s/update {update_s:.1f}s) | ent {log['ent'] / n_mb:.2f} "
                      f"kl {log['kl'] / n_mb:.4f} | {ep} | {(time.time() - t_start) / 60:.1f} min", flush=True)
            recent = []

            if time.time() - last_save > cfg.save_every_min * 60:
                p = run_dir / "ckpt" / f"ckpt_{steps // 1000:08d}k.pt"
                save_checkpoint(p, net, opt, steps, ret_std, cfg)
                # every save is kept (about 11 MB each): an early save can turn out to be the best one
                last_save = time.time()
    finally:
        save_checkpoint(run_dir / "ckpt" / f"ckpt_{steps // 1000:08d}k.pt", net, opt, steps, ret_std, cfg)
        venv.close()
        tb.close()
        print(f"[{cfg.name}] stopped at {steps:,} steps", flush=True)


def main():
    import signal

    def _stop(signum, frame):  # `kill`/timeout -> same clean path as Ctrl+C: save checkpoint, close workers
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _stop)
    ap = argparse.ArgumentParser()
    for f, v in asdict(Config()).items():
        flag = "--" + f.replace("_", "-")
        if isinstance(v, bool):
            ap.add_argument(flag, type=lambda s: s.lower() in ("1", "true", "yes"), default=v)
        else:
            ap.add_argument(flag, type=type(v), default=v)
    cfg = Config(**vars(ap.parse_args()))
    try:
        train(cfg)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
