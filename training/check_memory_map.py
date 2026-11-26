"""Check the memory map's self-tracking against the game's true position (training-only check).

Plays the best Run 1 brain on held-out maps and compares where the agent THINKS it is (memory_map.py,
from its own buttons and speed) with where it really is (privileged state, never given to the agent).

    .venv/bin/python training/check_memory_map.py [checkpoint]
"""
import sys, glob, time, numpy as np, torch
from pathlib import Path
R=str(Path(__file__).resolve().parents[1])
sys.path[:0]=[R+'/env',R+'/submission',R+'/training']
torch.set_num_threads(1)
from doom_env import make_env
from agent import DoomAgent
from doom_net import DoomNet
from features import ACTIONS
from memory_map import MemoryMap
st=torch.load(sys.argv[1] if len(sys.argv) > 1 else R+'/runs/run1/ckpt/best.pt',map_location='cpu',weights_only=False)
w=st['net'] if 'net' in st else st['model']
maps=sorted(glob.glob(R+'/maps/gen/heldout/*/*.wad'))[::2]
tot_t=0;n=0
for wad in maps:
    e=make_env(wad,'MAP01',episode_ticks=3000,render_buffers=False); o,_=e.reset(seed=1)
    ag=DoomAgent(e.observation_space,e.action_space,'cpu'); net=DoomNet(); net.load_state_dict({k:v for k,v in w.items() if not k.startswith('value.')},strict=False); ag.model=net.eval(); ag.reset()
    ps=e.privileged_state(); x0,y0,a0=ps['x'],ps['y'],np.deg2rad(ps['angle'])
    mm=MemoryMap(); mm.observe(o); errs={}
    for i in range(1,701):
        act=ag.act(o); o,_,t,tr,_=e.step(act)
        t0=time.perf_counter(); mm.observe(o, act); tot_t+=time.perf_counter()-t0; n+=1
        if t or tr: break
        ps=e.privileged_state()
        dx,dy=ps['x']-x0,ps['y']-y0; c,s=np.cos(-a0),np.sin(-a0)
        tx,ty=dx*c-dy*s, dx*s+dy*c
        ta=((np.deg2rad(ps['angle'])-a0+np.pi)%(2*np.pi))-np.pi
        if i in (100,300,700): errs[i]=(round(float(np.hypot(mm.x-tx,mm.y-ty))),round(float(np.rad2deg(((mm.heading-ta+np.pi)%(2*np.pi))-np.pi)),1), round(float(np.hypot(tx,ty))))
    print(wad.split('/')[-2], wad.split('/')[-1][:12], 'steps', i, 'err(pos units, heading deg, dist travelled-from-start)', errs)
    e.close()
print('ms per step', round(1000*tot_t/n,3))
