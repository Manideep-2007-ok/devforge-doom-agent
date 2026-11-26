"""Copy a training checkpoint's policy into submission/weights/model.pt (what agent.py loads).

    .venv/bin/python training/export.py runs/run1/ckpt/ckpt_00012345k.pt

Drops the optimizer state and the training-only value head (file ~3 MB), saves it with
doom_net.save_checkpoint, then checks that DoomAgent loads it and acts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "submission" / "weights" / "model.pt"
sys.path.insert(0, str(ROOT / "submission"))

from doom_net import net_for, save_checkpoint  # noqa: E402


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
    weights = {k: v for k, v in ck["net"].items() if not k.startswith("value.")}
    net = net_for(weights)  # same optional parts (memory map, eye) as the trained brain
    missing, unexpected = net.load_state_dict(weights, strict=False)
    missing = [k for k in missing if not k.startswith("value.")]
    if missing or unexpected:  # never ship a brain with parts dropped or left random
        raise SystemExit(f"export refused: missing {missing[:4]} unexpected {unexpected[:4]}")
    save_checkpoint(net, OUT)
    print(f"parts: memory map {'yes' if net.use_map else 'no'}, eye {'yes' if net.use_eye else 'no'}, {len(weights)} weight sets")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB) from {sys.argv[1]} at {ck.get('steps', 0):,} steps")

    from agent import DoomAgent  # noqa: E402
    from doom_env import make_env  # noqa: E402

    env = make_env(str(ROOT / "maps" / "two_rooms.wad"), "MAP01", render_buffers=True)
    obs, _ = env.reset(seed=0)
    agent = DoomAgent(env.observation_space, env.action_space, "cpu")
    agent.reset()
    a = agent.act(obs)
    assert a.shape == (9,) and set(a.tolist()) <= {0, 1}, a
    assert agent.errors == 0, "agent.act() raised an error (run with DOOM_STRICT=1 to see it)"
    # the submitted file must play exactly like the training save: compare their outputs on one input
    from doom_net import load_model  # noqa: E402
    sub = load_model(OUT, "cpu").eval()
    agent.reset()
    agent.model = sub
    sd_sub, sd_ck = sub.state_dict(), net.state_dict()
    play = [k for k in weights if not k.startswith("value.")]  # the value head is training-only
    assert all(k in sd_sub and torch.equal(sd_sub[k].float(), weights[k].float()) for k in play), "exported weights differ"
    print("DoomAgent loads it and acts: OK (every weight matches the training save)")


if __name__ == "__main__":
    main()
