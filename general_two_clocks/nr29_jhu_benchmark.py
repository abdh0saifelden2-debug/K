r"""NR29 external benchmark — the backscatter-fraction law on JHU DNS data.

The gap this closes
-------------------
NR29 (`new_relationships6.py`, P1 §5c) derived and verified in-repo that the
subgrid energy-flux backscatter *volume fraction* obeys the zero-parameter
Gaussian flux law

    P(Pi < 0) = Phi(-<Pi> / sigma(Pi)),

with K-theory (Pi >= 0 pointwise) its mu/sigma -> infinity limit.  The in-repo
verification used this repo's own forced HIT snapshots — a referee can object
that the law was tuned to the house solver.  This module tests the same closed
form on the **community-standard truth**: the Johns Hopkins Turbulence Database
`isotropic1024coarse` DNS (Re_lambda ~ 433; Li et al. 2008, J. Turb. 9, N31;
Perlman et al. 2007), fetched through the public demo token of the JHTDB/
Giverny service.

Method (honest scope)
---------------------
* Fetch the velocity field on a **full-domain periodic lattice** (n^3 points at
  native stride 1024/n over [0, 2pi)^3) — a strided sample of the whole box is
  periodic by construction, so spectral filtering is exact on the lattice;
  striding aliases sub-lattice scales (noted, not hidden: the law under test
  relates moments of the *filtered lattice field* to its own sign fraction, so
  aliasing changes the field, not the validity of the comparison).
* Sharp spectral filter at k_c; SGS stress tau_ij = (u_i u_j)^~ - u~_i u~_j;
  resolved strain S~_ij; flux Pi = -tau_ij S~_ij (standard definitions, as in
  the in-repo NR29).
* Compare measured P(Pi<0) against Phi(-mean/std) at several k_c and (if the
  service allows) a second timepoint; report the forward-cascade sign <Pi> > 0
  (the NR31 3-D premise) as a side check.

Data access: demo token (rate-limited); the fetched lattice is cached to
``figures/nr29_jhu_cache_*.npz`` (not committed — regenerate via this script;
the derived JSON/PNG artifacts are committed).  Unit tests are offline
(synthetic fields); the network fetch happens only under ``--fetch``/main.

Run:
    python general_two_clocks/nr29_jhu_benchmark.py --n 64 --kc 8 12 16
      -> figures/nr29_jhu_benchmark.{json,png}
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
DEMO_TOKEN = "edu.jhu.pha.turbulence.testing-201406"
DATASET = "isotropic1024coarse"
N_NATIVE = 1024


# ------------------------------------------------------------- core physics --
def sharp_filter(uhat, kc, k2):
    out = uhat.copy()
    out[k2 > kc * kc] = 0.0
    return out


def pi_field(u, kc):
    """Subgrid energy flux Pi = -tau_ij S~_ij on a periodic lattice.

    u: (3, n, n, n) real velocity. Returns Pi (n,n,n)."""
    n = u.shape[1]
    k1 = np.fft.fftfreq(n, d=1.0 / n)
    kx, ky, kz = np.meshgrid(k1, k1, k1, indexing="ij")
    k2 = kx * kx + ky * ky + kz * kz
    kvec = (kx, ky, kz)

    uhat = [np.fft.fftn(u[i]) for i in range(3)]
    uf_hat = [sharp_filter(h, kc, k2) for h in uhat]
    uf = [np.real(np.fft.ifftn(h)) for h in uf_hat]

    tau = np.empty((3, 3) + u.shape[1:])
    for i in range(3):
        for j in range(i, 3):
            prod_hat = np.fft.fftn(u[i] * u[j])
            prod_f = np.real(np.fft.ifftn(sharp_filter(prod_hat, kc, k2)))
            tau[i, j] = prod_f - uf[i] * uf[j]
            tau[j, i] = tau[i, j]

    pi = np.zeros(u.shape[1:])
    for i in range(3):
        for j in range(3):
            Sij = 0.5 * np.real(np.fft.ifftn(
                1j * (kvec[j] * uf_hat[i] + kvec[i] * uf_hat[j])))
            pi += -tau[i, j] * Sij
    return pi


def nr29_check(pi):
    """Measured backscatter fraction vs the Gaussian flux law Phi(-mu/sigma)."""
    from scipy.stats import norm
    mu, sd = float(np.mean(pi)), float(np.std(pi))
    f_meas = float(np.mean(pi < 0))
    f_law = float(norm.cdf(-mu / sd)) if sd > 0 else float("nan")
    return {"mean_pi": mu, "std_pi": sd, "backscatter_fraction": f_meas,
            "nr29_closed_form": f_law, "abs_err": abs(f_meas - f_law),
            "forward_cascade": bool(mu > 0)}


# ---------------------------------------------------------------- data fetch --
def fetch_lattice(n=64, t=0.0, token=DEMO_TOKEN, cache_dir=FIGDIR,
                  batch=4096, verbose=True):
    """Full-domain periodic n^3 velocity lattice from JHTDB (cached)."""
    cache = os.path.join(cache_dir, f"nr29_jhu_cache_n{n}_t{t:g}.npz")
    if os.path.exists(cache):
        return np.load(cache)["u"], cache
    from givernylocal.turbulence_dataset import turb_dataset
    from givernylocal.turbulence_toolkit import getData
    ds = turb_dataset(dataset_title=DATASET, output_path="/tmp/jhu",
                      auth_token=token)
    ax = (2.0 * np.pi / N_NATIVE) * (np.arange(n) * (N_NATIVE // n))
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]).astype(np.float64)
    out = np.empty((pts.shape[0], 3), dtype=np.float64)
    for i0 in range(0, pts.shape[0], batch):
        sl = slice(i0, min(i0 + batch, pts.shape[0]))
        res = getData(ds, "velocity", t, "none", "lag4", "field", pts[sl],
                      verbose=False)
        arr = np.asarray(res[0] if isinstance(res, (list, tuple)) else res,
                         dtype=np.float64)
        out[sl] = arr.reshape(-1, 3)
        if verbose and (i0 // batch) % 16 == 0:
            print(f"  fetched {sl.stop}/{pts.shape[0]}", flush=True)
    u = out.T.reshape(3, n, n, n)
    os.makedirs(cache_dir, exist_ok=True)
    np.savez_compressed(cache, u=u)
    return u, cache


# ----------------------------------------------------------------------- run --
def run(n=64, kcs=(8, 12, 16), times=(0.0,), token=DEMO_TOKEN):
    rows = []
    for t in times:
        u, cache = fetch_lattice(n=n, t=t, token=token)
        for kc in kcs:
            pi = pi_field(u, kc)
            r = nr29_check(pi)
            r.update({"t": t, "kc": int(kc), "n_lattice": n,
                      "n_samples": int(pi.size)})
            rows.append(r)
            print(f"t={t} kc={kc}: f_meas={r['backscatter_fraction']:.4f} "
                  f"law={r['nr29_closed_form']:.4f} err={r['abs_err']:.4f} "
                  f"<Pi>{'>' if r['forward_cascade'] else '<='}0", flush=True)
    out = {
        "dataset": f"JHTDB {DATASET} (Re_lambda~433; Li et al. 2008)",
        "access": "public demo token, full-domain strided periodic lattice "
                  f"(n={n}, stride={N_NATIVE // n}; aliasing noted in module "
                  "docstring)",
        "law": "P(Pi<0) = Phi(-<Pi>/sigma(Pi))  [NR29, zero parameters]",
        "rows": rows,
        "max_abs_err": float(max(r["abs_err"] for r in rows)),
        "all_forward_cascade": bool(all(r["forward_cascade"] for r in rows)),
        "in_repo_comparator": "new_relationships6.py (NR29) tracks <=0.005 on "
                              "house HIT; this is the external check",
    }
    return out


def make_figure(out, path_png):                              # pragma: no cover
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = out["rows"]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for t in sorted({r["t"] for r in rows}):
        rr = [r for r in rows if r["t"] == t]
        ax.plot([r["kc"] for r in rr], [r["backscatter_fraction"] for r in rr],
                "o-", label=f"measured, t={t}")
        ax.plot([r["kc"] for r in rr], [r["nr29_closed_form"] for r in rr],
                "s--", label=f"NR29 law, t={t}")
    ax.set_xlabel("filter wavenumber k_c")
    ax.set_ylabel("backscatter volume fraction P(Pi<0)")
    ax.set_title("NR29 on JHTDB isotropic1024coarse\n"
                 f"max |err| = {out['max_abs_err']:.4f}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path_png, dpi=140)
    plt.close(fig)


def main():                                                  # pragma: no cover
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--kc", type=int, nargs="+", default=[8, 12, 16])
    ap.add_argument("--times", type=float, nargs="+", default=[0.0])
    ap.add_argument("--json-out", default=os.path.join(
        FIGDIR, "nr29_jhu_benchmark.json"))
    ap.add_argument("--png-out", default=os.path.join(
        FIGDIR, "nr29_jhu_benchmark.png"))
    a = ap.parse_args()
    out = run(n=a.n, kcs=tuple(a.kc), times=tuple(a.times))
    os.makedirs(os.path.dirname(a.json_out), exist_ok=True)
    with open(a.json_out, "w") as fh:
        json.dump(out, fh, indent=1)
    make_figure(out, a.png_out)
    print("wrote", a.json_out, "and", a.png_out)


if __name__ == "__main__":
    main()
