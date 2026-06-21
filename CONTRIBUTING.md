# Contributing & Reviewing Guide

## For Reviewers

### Quick orientation

1. **Start with the README** — the [Roadmap](#roadmap) table shows all 10 parts
   and links to each report.
2. **The key documents:**
   - [`general_two_clocks/REPORT_THEORY.md`](general_two_clocks/REPORT_THEORY.md) — the mathematical derivation
     (Mori–Zwanzig → K-theory as mutilation → projected-FDT repair)
   - [`general_two_clocks/REPORT_CLOSURE.md`](general_two_clocks/REPORT_CLOSURE.md) — the benchmark that decides it
     (Part 8b results + physical implications)
3. **To reproduce the headline result:**
   ```bash
   pip install -r requirements.txt
   python general_two_clocks/run_closure.py --out-dir general_two_clocks/figures --report general_two_clocks/REPORT_CLOSURE.md
   ```
   This takes ~30s on a modern machine (256² DNS, no GPU needed).

### What to look for

- **Mathematical claims** live in `general_two_clocks/REPORT_THEORY.md`. The derivation is standard
  (Mori–Zwanzig / optimal prediction) — the novelty is the identification of
  K-theory as the specific mutilation and the two-clocks framing.
- **Numerical claims** live in the `run_*.py` scripts. Each generates its own
  figures and report — you can verify any number by re-running.
- **Scope claims** are explicit throughout: every part states what it does and
  does not prove. Watch for the "Scope" paragraphs.

### Code structure

Each part follows the same pattern:
```
library/module.py   ← reusable classes/functions (solver, loader, analysis)
run_partname.py     ← CLI script that uses the library, generates figures + report
REPORT_NAME.md      ← auto-generated report with computed numbers
figures/NN_name.png ← generated plots
```

### Running tests

```bash
pytest -v
```

All 308 tests use synthetic signals (and the bundled BEDMAP1 / Otis data) — no
data download needed.

---

## For Contributors

### Adding a new Part

1. Create a library module under the appropriate package
2. Create a `run_*.py` script following the existing CLI pattern
3. Add entries to the README roadmap table and layout tree
4. Add unit tests under the owning domain folder’s `tests/` (e.g. `glaciers/tests/`)
5. Update `requirements.txt` if new dependencies are needed

### Code style

- Pure Python + NumPy/SciPy/Matplotlib (no frameworks)
- Pseudo-spectral methods throughout (FFT-based)
- All figures are numbered sequentially (01–30+)
- Report benchmark sections are regenerated from code; hand-written analysis
  (marked with `<!-- BEGIN HAND-WRITTEN CONTENT (preserved across regeneration) -->`) is preserved across runs

### Commit conventions

- One commit per logical change
- Commit messages reference the Part number where relevant
- Bug fixes clearly state what was wrong and what was fixed
