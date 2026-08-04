# Cover letter — Physical Review Fluids (Paper 1)

> Copy-paste ready. Paste into the "Cover Letter" box on the APS submission server,
> or upload as a separate PDF. Fill the optional suggested/excluded-referee line if you wish.

---

To the Editors, *Physical Review Fluids*

Dear Editors,

I am pleased to submit the enclosed manuscript, **"What single-eddy-diffusivity closure discards: a Mori–Zwanzig audit, a scalar-independent memory time, and a divergence-cleaning design window,"** for consideration as a **Regular Article** in *Physical Review Fluids*.

Eddy-diffusivity ("K-theory") closures remain the workhorse for unresolved turbulent transport, yet the precise content of what a single scalar diffusivity discards is rarely stated term by term. This manuscript contributes three things, building on the established Mori–Zwanzig (MZ) view of model reduction:

1. **An operator-by-operator audit.** The single-diffusivity approximation is mapped onto four simultaneous, separable simplifications of the generalized Langevin equation — a delta-in-time memory kernel, a wavenumber-frozen spectral eddy viscosity, a deleted fluctuation–dissipation-tied orthogonal force, and the sign restriction that forbids backscatter — each tied to a specific GLE term and tested in an a-priori, frozen-field benchmark (256² periodic box, sharp filter at k_c = 32). The structural result is *predicted, not fitted*: the Leray-projected FDT force stays divergence-free to RMS(∇·m) ≈ 1.7×10⁻¹⁴ with the correct sign of energy backscatter, whereas a spectrum-matched surrogate reproduces the force spectrum yet breaks incompressibility and randomizes the transfer direction, and Smagorinsky is purely dissipative.

2. **A new, falsifiable measurement — a scalar-independent memory time.** The closure's memory time is the same for heat and salt to within measurement error across a 100× molecular-Lewis contrast, even though the scalar Nusselt numbers track the Lewis number strongly — direct evidence that the memory belongs to the *flow*, as the MZ orthogonal dynamics require. The implied clock-mismatch correction cuts a transient K-theory scalar-solver error by ≈15× and vanishes identically in steady turbulence.

3. **A divergence-cleaning design window.** When exact projection is replaced by hyperbolic (Dedner) divergence cleaning, the constraint residual collapses only inside a finite window 2 ≲ γτ ≲ 12 and *re-grows* beyond it: pushing the cleaning rate up to clean fast locally stalls the global low-wavenumber modes through the slow telegrapher root — an over-cleaning failure mode absent from the original cleaning analysis.

I have taken care to bound the claims honestly. The scope is stated explicitly (a-priori, 2-D, periodic; the clean FDT/projection commutation is special to the spectral box and does not survive solid walls), and a time-integrated a-posteriori test (§5b) reports the genuinely null result that in pure 2-D no eddy-viscosity closure beats no-model on the resolved spectrum. The contribution is the audit, the scalar-independent memory time, and the cleaning window — not a universal closure.

This work fits the scope of *Physical Review Fluids*: it concerns the fundamental structure of turbulence closures and yields concrete, falsifiable measurements rather than an ad hoc model. All data and code that reproduce the figures are publicly available (repository link in the manuscript). The manuscript is single-authored and original, and it is not under consideration at any other journal.

*(Optional — suggested referees: ____; referees to exclude: ____.)*

Thank you for considering this work.

Sincerely,
Abdelrahman Saifelden
Alexandria Higher Institute of Engineering and Technology (AIET), Alexandria, Egypt
Abdulrahman.Saifelden22011411@aiet.edu.eg
