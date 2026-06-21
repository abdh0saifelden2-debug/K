import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, time
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
# repo reorg: make sibling domain folders importable
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ("general_two_clocks", "atmosphere", "glaciers", "ocean"):
    _p = os.path.join(_REPO_ROOT, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)
del _d, _p

from subglacial.flow import SubglacialFlow, SubglacialConfig
 
cfg = SubglacialConfig(n=128, sgs="none", f_amp=1.5, k_f=10.0, f_band=2.0, seed=1)
f = SubglacialFlow(cfg)
t0=time.time()
keh=[]
for s in range(9000):
    Ut = cfg.U0*min(1.0,(s+1)/2500)
    f.step(Ut)
    if s%50==0: keh.append(f.kinetic_energy())
    if s%1500==0: print(f"  step {s:5d} umax={np.abs(f.u).max():.3f} KE={f.kinetic_energy():.4e}")
print("walltime", time.time()-t0)
keh=np.array(keh[-40:])
print("KE last-window mean=%.4e std=%.2e (std/mean=%.3f)"%(keh.mean(),keh.std(),keh.std()/keh.mean()))
mf, band = f.melt_flux()
print("melt_flux", mf, "heat_in_wake", f.heat_in_wake())
 
X,Y=f.X,f.Y; chi=f.chi
om=np.ma.masked_where(chi>0.5, f.vorticity())
th=np.ma.masked_where(chi>0.5, f.theta)
sp_=np.ma.masked_where(chi>0.5, np.sqrt(f.u**2+f.v**2))
pr=np.ma.masked_where(chi>0.5, f.pressure())
fig,ax=plt.subplots(4,1,figsize=(9,12))
sd=np.nanstd(om)
ax[0].pcolormesh(X,Y,om,cmap="RdBu_r",shading="auto",vmin=-3*sd,vmax=3*sd); ax[0].set_title("vorticity")
ax[1].pcolormesh(X,Y,sp_,cmap="viridis",shading="auto"); ax[1].set_title("speed")
ax[2].pcolormesh(X,Y,th,cmap="inferno",shading="auto"); ax[2].set_title("heat")
ax[3].pcolormesh(X,Y,pr,cmap="RdBu_r",shading="auto"); ax[3].set_title("pressure")
for a in ax:
    a.contour(X,Y,chi,[0.5],colors="k",linewidths=0.8); a.set_aspect("equal"); a.set_ylim(0,3.0)
fig.tight_layout(); fig.savefig("_validate_sub.png",dpi=110)
print("saved _validate_sub.png")
