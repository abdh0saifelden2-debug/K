#!/usr/bin/env python
r"""Full-family GPU (CuPy) backend verification sweep.

Runs every backend-agnostic NumPy/CuPy solver in the repo (the ``scallop_*`` /
gate-probe family, ``flow3d``, the ``candidate4`` hydraulic anchor) in a fresh
subprocess and checks that the GPU path is clean. A whole class of bugs only
appears on a real CUDA device -- most notably ``np.asarray(<cupy array>)`` which
raises ``TypeError: Implicit conversion to a NumPy array is not allowed`` in
CuPy >= 10 -- and cannot be caught by the CPU ``pytest`` suite. This is the
committable form of the GPU verification workflow (verified 13/13 PASS on
Tesla P100-PCIE-16GB, cupy 14.0.1).

A case PASSES iff its subprocess returns 0 AND ``Implicit conversion`` does not
appear in stderr (a device->host conversion would either crash or print it).

Usage (on a Kaggle P100 notebook, after cloning the repo)::

    python glaciers/gpu_family_sweep.py --gpu --json /kaggle/working/gpu_family_sweep.json

Without ``--gpu`` the same cases run on the CPU backend (a harness smoke test;
``--smoke`` shrinks the grids so the CPU pass is quick).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _cli_cases(gpu, smoke):
    g = ["--gpu"] if gpu else []
    py = sys.executable
    if smoke:
        probe = ["--nx", "48", "--ny", "48", "--spinup", "40", "--measure", "20"]
        s3d = ["--n", "24", "--spinup", "40", "--snaps", "1", "--snap-every", "20",
               "--amps", "0.10", "--out", "/tmp/s3d.json"]
        gate = ["--n", "16", "--spinup", "30", "--snaps", "1", "--snap-every", "15"]
    else:
        probe = ["--nx", "96", "--ny", "96", "--spinup", "200", "--measure", "80"]
        s3d = ["--n", "48", "--spinup", "150", "--snaps", "2", "--snap-every", "60",
               "--amps", "0.10", "--out", "/tmp/s3d.json"]
        gate = ["--n", "24", "--spinup", "80", "--snaps", "2", "--snap-every", "30"]
    return [
        ("scallop_probe",            [py, "glaciers/scallop_probe.py", *g, *probe]),
        ("scallop_doublediff",       [py, "ocean/scallop_doublediff.py", *g, "--fast"]),
        ("scallop_channel_feedback", [py, "glaciers/scallop_channel_feedback.py", *g, "--fast"]),
        ("scallop_amplitude_band",   [py, "glaciers/scallop_amplitude_band.py", *g, "--fast"]),
        ("scallop_amplitude_closure",[py, "glaciers/scallop_amplitude_closure.py", *g, "--fast"]),
        ("scallop_g1_populations",   [py, "glaciers/scallop_g1_populations.py", *g, "--fast"]),
        ("scallop_shedding_det",     [py, "glaciers/scallop_shedding_deterministic.py", *g, "--fast"]),
        ("scallop_sweep",            [py, "glaciers/scallop_sweep.py", *g, "--fast"]),
        ("scallop3d_probe",          [py, "glaciers/scallop3d_probe.py", *g, *s3d]),
        ("subglacial.slip_gate",     [py, "-m", "subglacial.slip_gate", *g, *gate]),
        ("subglacial.wall_flux_gate",[py, "-m", "subglacial.wall_flux_gate", *g, *gate]),
    ]


def _special_cases(gpu, smoke):
    xp = "cupy" if gpu else "numpy"
    py = sys.executable
    if smoke:
        c4_cfg = "nx=64,ny=32,A=4.0,f_amp=0.5,Ri=0.5"
        c4_run = "spinup=25,measure=25"
        bat = "nx=48,ny=48,spinup=40,measure=20"
    else:
        c4_cfg = "nx=128,ny=64,A=4.0,f_amp=0.5,Ri=0.5"
        c4_run = "spinup=50,measure=50"
        bat = "nx=96,ny=96,spinup=200,measure=80"
    c4 = (f"import sys; sys.path.insert(0,'glaciers'); import {xp} as xp; "
          "import subglacial.candidate4_hydraulic_switch as m; "
          f"cfg=m.HydraulicConfig({c4_cfg}); "
          f"g=m.run_case(cfg,{c4_run},xp=xp); "
          "print('candidate4', g['H1_mean'], g['melt_mean'], g['umax'])")
    battery = (f"import sys; sys.path.insert(0,'glaciers'); import {xp} as xp; "
               "import scallop_battery as m; "
               f"m.main(xp=xp, {bat})")
    return [
        ("candidate4_anchor", [py, "-c", c4]),
        ("scallop_battery",   [py, "-c", battery]),
    ]


def run(gpu=False, smoke=False, per_case_timeout=1800):
    env = dict(os.environ)
    extra = os.pathsep.join([os.path.join(REPO, "glaciers"),
                             os.path.join(REPO, "general_two_clocks"), REPO])
    env["PYTHONPATH"] = extra + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["MPLBACKEND"] = "Agg"
    cases = _cli_cases(gpu, smoke) + _special_cases(gpu, smoke)
    results = []
    for name, cmd in cases:
        t0 = time.time()
        try:
            p = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True,
                               text=True, timeout=per_case_timeout)
            rc, err, out = p.returncode, p.stderr, p.stdout
            timed_out = False
        except subprocess.TimeoutExpired as e:
            rc, err, out, timed_out = -1, (e.stderr or ""), (e.stdout or ""), True
        dt = time.time() - t0
        implicit = "Implicit conversion" in (err or "")
        ok = (rc == 0) and not implicit and not timed_out
        results.append({
            "case": name, "pass": ok, "returncode": rc, "elapsed_s": round(dt, 1),
            "timed_out": timed_out, "implicit_conversion": implicit,
            "stderr_tail": (err or "")[-400:].strip(),
            "stdout_tail": (out or "")[-200:].strip(),
        })
        flag = "PASS" if ok else "FAIL"
        print(f"[{flag}] {name:<28} rc={rc} t={dt:6.1f}s"
              + ("  IMPLICIT-CONVERSION" if implicit else "")
              + ("  TIMEOUT" if timed_out else ""))
        sys.stdout.flush()
    n_pass = sum(r["pass"] for r in results)
    return {"backend": "cupy" if gpu else "numpy", "smoke": smoke,
            "n_pass": n_pass, "n_total": len(results), "cases": results}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true", help="use the CuPy backend (CUDA device)")
    ap.add_argument("--smoke", action="store_true", help="tiny grids for a fast CPU harness test")
    ap.add_argument("--timeout", type=int, default=1800, help="per-case timeout [s]")
    ap.add_argument("--json", default=None, help="write the summary JSON here")
    a = ap.parse_args()

    dev = "(CPU/numpy)"
    if a.gpu:
        try:
            import cupy as cp
            dev = f"cupy {cp.__version__} on {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}"
        except Exception as e:  # pragma: no cover
            dev = f"(cupy unavailable: {e})"
    print(f"=== GPU family sweep — backend={'cupy' if a.gpu else 'numpy'} {dev} smoke={a.smoke} ===")

    summary = run(gpu=a.gpu, smoke=a.smoke, per_case_timeout=a.timeout)
    summary["device"] = dev
    print(f"\nSWEEP: {summary['n_pass']}/{summary['n_total']} PASS  (backend={summary['backend']})")
    if a.json:
        os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
        with open(a.json, "w") as fh:
            json.dump(summary, fh, indent=2)
        print(f"json -> {os.path.abspath(a.json)}")
    return 0 if summary["n_pass"] == summary["n_total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
