"""Scratch: does a coarse a-posteriori LES show K-theory killing the wake?"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
# repo reorg: make sibling domain folders importable
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("general_two_clocks", "atmosphere", "glaciers", "ocean"):
    _p = os.path.join(_REPO_ROOT, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)
del _d, _p

from subglacial.flow import SubglacialFlow, SubglacialConfig
 
SPIN = 9000
WIN = 3000
EVERY = 50
 
 
def run_case(n, sgs, bs=0.6, seed=1):
    cfg = SubglacialConfig(n=n, sgs=sgs, backscatter=bs, f_amp=1.5, k_f=10.0,
                           f_band=2.0, seed=seed)
    f = SubglacialFlow(cfg)
    f.run(SPIN, ramp=2500)
    su = np.zeros((n, n)); sv = np.zeros((n, n))
    suu = np.zeros((n, n)); svv = np.zeros((n, n)); st = np.zeros((n, n))
    cnt = 0
    for s in range(WIN):
        f.step(cfg.U0)
        if s % EVERY == 0:
            su += f.u; sv += f.v; suu += f.u**2; svv += f.v**2; st += f.theta
            cnt += 1
    mu, mv = su/cnt, sv/cnt
    var = (suu/cnt - mu**2) + (svv/cnt - mv**2)
    band = f.wake_band()
    tke = 0.5*float(np.mean(var[band]))
    heat = float(np.mean((st/cnt)[band]))
    ke = f.kinetic_energy()
    return tke, heat, ke
 
 
cases = [
    ("DNS truth (n=128)", 128, "none"),
    ("coarse none (n=64)", 64, "none"),
    ("coarse Smagorinsky (n=64)", 64, "smagorinsky"),
    ("coarse backscatter (n=64)", 64, "backscatter"),
]
print(f"{'case':30s} {'wakeTKE':>12s} {'heat_wake':>12s} {'KE':>12s}")
for name, n, sgs in cases:
    tke, heat, ke = run_case(n, sgs)
    print(f"{name:30s} {tke:12.4e} {heat:12.4e} {ke:12.4e}")
