#!/usr/bin/env python3
"""Content + build for the Paper 1-4 talk decks (v2). Style/engine in deck_kit.py.
Run:  python build_slides.py   ->  paperN_*.pptx + paperN_SCRIPT.md
"""
from __future__ import annotations
import os
from pptx.dml.color import RGBColor
from deck_kit import Deck, fig, OUT

AUTHOR = "Presenter: [your name] · [affiliation] · [ORCID]"


# =========================================================================== #
def paper1():
    d = Deck(RGBColor(0x1F, 0x3B, 0x73), "Paper 1 · MZ closure repair", "Paper 1")
    d.title(
        "Single-eddy-diffusivity closure is the Markovian collapse of a memory kernel",
        "Turbulence models close the small scales with a simple down-gradient rule AND that's "
        "convenient, BUT the rule secretly erases the flow's memory, THEREFORE we show it is the "
        "memoryless limit of an exact memory kernel — and put the memory back.", AUTHOR,
        "Open with the punchline: the most common turbulence closure isn't a law, it's the zero-memory "
        "limit of something exact — and that's both why it fails and how to fix it. Keep it warm; "
        "promise one clean idea, one decisive test.")
    d.analogy(
        "Think of an echo-cancelling app that erases a room's reverb",
        ["A down-gradient closure is like an app that smooths away a room's echo in real time.",
         "It works beautifully — until the echo IS the signal you cared about.",
         "Turbulence has a memory (an echo in time). The standard closure quietly deletes it."],
        "Picture an audio 'de-reverb' slider: handy for a phone call, fatal for measuring the room. "
        "The question of this talk is: what exactly does the closure delete, and can we put it back? "
        "Pause after the analogy and let it land before the formal version.",
        big="The closure deletes the flow's memory. This talk is about that deleted memory.")
    d.assert_slide(
        "The standard closure replaces an exact memory integral with a single instant",
        "Here's the one equation of the talk, in words. The true unresolved force depends on the whole "
        "history of the flow through a Mori-Zwanzig memory kernel. Down-gradient K-theory keeps only "
        "the present instant — it collapses that kernel to a spike. State it, don't dwell.",
        cards=[("∫ K(t−s)·(s) ds", "exact: a memory integral over history"),
               ("K(0)·(t)", "K-theory: only the present instant"),
               ("memory → 0", "the approximation that's usually hidden")],
        support="K-theory is the Markovian-delta projection of an exact, non-local, causal response.")
    d.assert_slide(
        "How wrong it is, is set by one number — a memory (Deborah) number",
        "The error isn't mysterious; it's governed by a single dimensionless ratio: memory time over "
        "the resolved timescale. When that ratio is tiny, K-theory is exact; when it's order one, the "
        "closure is wrong by a predictable amount. This converts a vague worry into a number you can check.",
        cards=[("De = τ_c / τ_event", "memory time ÷ resolved time"),
               ("error ∝ De^(p+1)", "a Chapman–Enskog / Knudsen ladder"),
               ("De → 0", "the only limit where K-theory is exact")])
    d.assert_slide(
        "The repair keeps the memory and restores the backscatter the closure threw away",
        "The fix is not a fudge factor. We keep the kernel (a non-local-in-time eddy viscosity) and "
        "restore its reactive part through a projected fluctuation-dissipation relation — which brings "
        "back backscatter, energy flowing back up to large scales. Net eddy viscosity can even go negative.",
        image=fig("validation", "reports", "h3_cmn_reduced_model.png"),
        caption="Kernel-based reduced model (CMN) tracking the full trajectory.",
        cards=[("non-local ∫K", "memory restored"),
               ("Re Z < 0", "backscatter restored (projected FDT)"),
               ("∫K can be < 0", "net negative eddy viscosity → growth")])
    d.assert_slide(
        "The decisive test: the repaired kernel reproduces the exact subgrid force to round-off",
        "This is the slide to slow down on. It's an a-priori test against the exact filtered force — not "
        "a tuned fit. The projection is exact to 1e-16, the kernel equals the eliminated channel's Green's "
        "function to 1e-18, and the reduced model tracks the full trajectory to 1e-8. The memory is "
        "necessary to get the interior peak. This is a structural identity, not a calibration.",
        cards=[("5×10⁻¹⁶", "MZ projection exact to"),
               ("7×10⁻¹⁸", "kernel = Green's function to"),
               ("9×10⁻⁸", "full trajectory reproduced to")],
        status=("VERIFIED", "a-priori against the exact force; certified to round-off, not fit"))
    d.assert_slide(
        "The memory time is a real, measurable property of the flow — not of the tracer",
        "One last point that makes it useful: the memory clock is the same regardless of which scalar you "
        "advect, and it inverts for a mean transport speed. So it's a physical handle, not a bookkeeping "
        "device — something you could measure.",
        cards=[("one τ_c", "sets kernel cutoff AND coupling roll-off"),
               ("scalar-independent", "a property of the flow"),
               ("inverts for V̄", "a measurable handle")])
    d.scope(
        ["MZ structure of the kernel (round-off)", "Error ∝ De^(p+1) ladder",
         "Backscatter via projected FDT", "1-D + 2-D advection / plume proxy"],
        ["Down-gradient K-theory as a universal law", "(1+CV²)-type local enhancement"],
        ["Full 3-D turbulent ice-flow solver test", "In-situ field calibration of τ_c"],
        "Be honest and quick: verified a-priori and in proxies, not yet in a full 3-D solver; that's the "
        "next step, not a hidden caveat.")
    d.assert_slide(
        "The takeaway: closure failure is a memory problem, and memory is repairable",
        "Land the plane. One sentence: when a down-gradient model fails, it's because the memory number "
        "isn't small — and we can put the memory (and its backscatter) back, verified. Point them to the "
        "public code and the companion relationships NR1, NR13, NR15, NR23.",
        cards=[("memory number", "tells you WHEN it fails"),
               ("projected MZ kernel", "tells you HOW to fix it"),
               ("public + reproducible", "github.com/abdh0saifelden2-debug/K")])
    return d.save("paper1_closure_theory.pptx")


# =========================================================================== #
def paper2():
    d = Deck(RGBColor(0x0F, 0x6E, 0x6E), "Paper 2 · Melt ceiling", "Paper 2")
    d.title(
        "A boundary-condition-robust ceiling on turbulent basal melt",
        "We worry turbulence speeds up melt under grounded ice AND it sounds plausible, BUT when we test "
        "it the melt is capped by a thin conductive skin, THEREFORE turbulence does not raise the net "
        "melt — there is a ceiling.", AUTHOR,
        "Hook with the stake: basal melt sets how fast ice can ungroundi, and a popular assumption is that "
        "turbulence cranks it up. We tested that assumption and it failed — in a useful way.")
    d.analogy(
        "Think of a wool blanket, not the wind outside it",
        ["How fast heat reaches the ice is set by a thin conductive layer right at the interface —",
         "like a wool blanket. You can blow on a blanket as hard as you like;",
         "the blanket, not the wind, sets how fast heat gets through."],
        "Everyone reaches for 'more turbulence = more mixing = more melt'. But if a conductive skin is the "
        "bottleneck, stirring the outside barely matters. Hold the blanket image; we'll now show the data "
        "agrees with the blanket, not the wind.",
        big="If conduction is the bottleneck, stirring the outside barely moves the melt.")
    d.assert_slide(
        "Turning the turbulence off entirely barely changes the melt",
        "This is the cleanest version of the result. We sweep the solver across closures and boundary "
        "conditions, then switch the flow off completely — pure conduction. The melt is reproduced to "
        "about one percent. If momentum transport drove melt, that couldn't happen.",
        image=fig("validation", "reports", "g6_local_flux_law.png"),
        caption="Local flux law across the swept control and all closures.",
        cards=[("flow off ⇒ ~1%", "pure conduction reproduces the melt"),
               ("ceiling", "set by the conductive sublayer"),
               ("not momentum", "transport doesn't enhance net melt")])
    d.assert_slide(
        "Three plausible turbulent enhancement mechanisms each net to ≈ zero",
        "We didn't just assert it — we tried the three ways turbulence could plausibly boost melt. "
        "Averaged over the bed, each one cancels: net melt is area times local flux, and the average is "
        "bounded. The popular (1+CV²) variance enhancement is actually falsified here.",
        image=fig("figures", "34_subglacial_heatflux.png"),
        caption="Subglacial heat-flux field; area-averaged enhancement ≈ 0.",
        cards=[("3 mechanisms", "all net to ≈ no gain"),
               ("(1+CV²)", "variance enhancement falsified"),
               ("area × local flux", "mean stays bounded")])
    d.assert_slide(
        "Scallops are a real LOCAL signal — but the basin mean stays under the ceiling",
        "Be fair to the other side: there IS a positive signal. Scalloped beds grow channels locally, "
        "plus thirty-three percent in area, and it's calibration-free — the latent-heat constant cancels. "
        "But locally positive and mean-bounded are not the same thing; the average still respects the ceiling.",
        image=fig("figures", "49_scallop_channel_feedback.png"),
        caption="Scallop → channel feedback geometry.",
        cards=[("+33% area", "local channel growth (calibration-free)"),
               ("ρL cancels", "no free latent-heat knob"),
               ("mean bounded", "still under the ceiling")])
    d.assert_slide(
        "The ceiling survives the modelling choices a skeptic would attack — same answer",
        "This is why it's trustworthy. No-slip versus slip walls reproduce the trajectory bit-for-bit; the "
        "result isn't an artifact of one wall model or one closure. Robustness is the headline.",
        cards=[("no-slip = slip", "trajectories match bit-for-bit"),
               ("across closures", "not one-model luck"),
               ("BC-robust", "the ceiling holds")],
        status=("VERIFIED", "robust across boundary conditions and closures"),
        caption="Boundary-condition robustness (slip gate / wall-flux).")
    d.assert_slide(
        "The deliverable is a regime map: where conduction pins melt vs where channels matter",
        "End on something usable. We hand modellers a map: most of the grounded cavity sits in the "
        "conduction-pinned regime, with channelization mattering only in a corner. That's a drop-in "
        "closure for basal-melt parameterizations.",
        image=fig("validation", "reports", "type_iii_regime.png"),
        caption="Regime map in the effective-pressure / forcing plane.")
    d.scope(
        ["Conductive-sublayer ceiling (BC-robust)", "Flow-off reproduces melt (~1%)",
         "Scallop +33% local (calibration-free)", "Regime-map deliverable"],
        ["Momentum-transport melt enhancement", "(1+CV²) variance truncation"],
        ["Network concentration gain g∈[0.1,0.9]", "Validation at a real cavity"],
        "Keep it crisp: the falsified item (turbulent enhancement) is the point of the paper, not an "
        "embarrassment. The one open knob is the network gain.")
    d.assert_slide(
        "The takeaway: don't budget for turbulent melt enhancement — budget for the ceiling",
        "Land it: in grounded subglacial cavities, the safe modelling assumption is a conduction-set "
        "ceiling, not turbulent enhancement. Connects to NR5. Point to the public code.",
        cards=[("ceiling, not boost", "the modelling takeaway"),
               ("conduction-set", "the mechanism"),
               ("reproducible", "github.com/abdh0saifelden2-debug/K")])
    return d.save("paper2_subglacial_melt_ceiling.pptx")


# =========================================================================== #
def paper3():
    d = Deck(RGBColor(0x5B, 0x2C, 0x83), "Paper 3 · Scallop parity break", "Paper 3")
    d.title(
        "Scallop migration is a parity break that down-gradient closures cannot reproduce",
        "Scallops on the bed drift downstream AND we'd like our model to capture it, BUT a symmetric "
        "down-gradient model literally cannot point downstream, THEREFORE migration is a clean, measurable "
        "test the standard closure fails.", AUTHOR,
        "Hook with a paradox: a model can fit the shape of a bedform perfectly and still get the physics "
        "wrong — because it can't say which way the form moves. That 'which way' is the whole game.")
    d.analogy(
        "Think of a clock with no difference between tick and tock",
        ["A down-gradient closure is mirror-symmetric: it has no built-in sense of 'downstream'.",
         "It's like a clock whose tick and tock are identical — it can't tell you which way time runs.",
         "But scallops DO run one way. So the closure is missing something structural, not just tuned wrong."],
        "Symmetry sounds harmless, even elegant. The point of this talk is that this particular symmetry "
        "is a defect: it forces a real, observed direction to be exactly zero. Let the clock image sit "
        "before the formal statement.",
        big="A symmetric closure forces the migration speed to be exactly zero — by construction.")
    d.assert_slide(
        "Migration is the out-of-phase part of the bed's response — the imaginary part",
        "Here's the mechanism in one line. Drive the bed with a topographic wave and read the flux "
        "response. The in-phase part grows or decays the bedform; the out-of-phase (quadrature) part moves "
        "it. Migration speed is literally the imaginary part of a complex response.",
        cards=[("in-phase (Re)", "grows / decays the scallop"),
               ("out-of-phase (Im)", "MOVES the scallop — migration"),
               ("phase ψ", "the single clean observable")],
        support="Harmonic decomposition of the wall flux: Re(s) vs Im(s).")
    d.assert_slide(
        "A symmetric closure has zero imaginary part — so it cannot migrate, ever",
        "This is the keystone. Because down-gradient closures are parity-symmetric, their response has no "
        "imaginary part: Im(s) = 0. That's not a small error you can tune away — it's a structural "
        "impossibility. The closure predicts a migration speed of exactly zero.",
        cards=[("K-theory: Im(s) = 0", "no migration possible"),
               ("structural", "not a tuning gap"),
               ("prediction", "migration speed = 0 exactly")])
    d.assert_slide(
        "In the resolved solver, the scallop genuinely migrates — the parity break is measured",
        "And here's the evidence: the full solver produces a non-zero imaginary part and real downstream "
        "migration on the same bed where the closure gives zero. We package it as an index I = |tan ψ|/2π. "
        "The model and the closure disagree, and the solver is right.",
        image=fig("figures", "49_scallop_channel_feedback.png"),
        caption="Scallop geometry used in the harmonic probe.",
        cards=[("Im(s) ≠ 0", "measured in the solver"),
               ("I = |tan ψ|/2π", "the migration index"),
               ("closure = 0", "contradicted by the solver")],
        status=("VERIFIED", "migration measured in the resolved solver"))
    d.assert_slide(
        "The phase test is constant-free and needs no temperature difference",
        "Why this is a good test: the phase observable needs no absolute calibration and no ΔT. You can "
        "read the parity break from bedform geometry and the flow alone — so it's checkable in the field, "
        "not just in a solver. Note in passing: the K² curvature and Mullins-Sekerka |k| ansätze are falsified.",
        cards=[("constant-free", "no absolute calibration"),
               ("ΔT-free", "no temperature difference needed"),
               ("field-testable", "from geometry + flow alone")],
        status=("FALSIFIED", "the K² curvature and Mullins–Sekerka |k| ansätze (wavelength test)"))
    d.scope(
        ["Migration = Im(s) parity break (solver)", "Closure forced to Im(s)=0",
         "Robust across amplitude & drive", "Constant-free index I=|tan ψ|/2π"],
        ["K² curvature ansatz", "Mullins–Sekerka |k| ansatz"],
        ["One open gate (bounded)", "Field measurement on a real reach"],
        "Be plain: the falsified ansätze are competitors we ruled out; the open item is a single bounded gate.")
    d.assert_slide(
        "The takeaway: migration is a clean, measurable failure of down-gradient closures",
        "Land it: if you want one experiment to tell whether a down-gradient closure is missing the "
        "reactive physics, measure bedform migration — it must be zero in the closure and isn't in reality. "
        "Connects to Paper 1's reactive part and NR2/NR4/NR9.",
        cards=[("one clean test", "bedform migration / phase ψ"),
               ("closure must say 0", "reality says otherwise"),
               ("reproducible", "github.com/abdh0saifelden2-debug/K")])
    return d.save("paper3_scallop_parity_break.pptx")


# =========================================================================== #
def paper4():
    d = Deck(RGBColor(0x14, 0x5A, 0x32), "Paper 4 · Hydrology forecasts", "Paper 4")
    d.title(
        "Falsifiable forecasts for subglacial hydrology",
        "Subglacial water drives the ice AND we have predictions, BUT most aren't checkable on real data, "
        "THEREFORE we make three and hold each to verify-or-falsify on open Antarctic data — including the "
        "one we lost.", AUTHOR,
        "Open with the stance, not the results: the field is full of plausible stories about water under "
        "the ice; we made three concrete predictions and put each on trial against open data. Promise "
        "honesty — one is verified twice, one is falsified, one is an honest null.")
    d.analogy(
        "Three bets on the ice — and we show our hand on every one",
        ["We make three predictions about water under the ice, and we test each on open data.",
         "One we verify — twice, on two independent datasets. One we falsify outright.",
         "One comes back an honest null — and we can say exactly why."],
        "Most talks show you only the bets they won. The discipline here is that we report all three the "
        "same way. That's what makes them forecasts and not stories.",
        big="A forecast you can't lose isn't a forecast. Here are three we could lose.")
    d.assert_slide(
        "Ocean water should intrude grounded ice exactly where ocean pressure beats the water pressure",
        "Set up bet one simply. Define a dimensionless number, RTN, that compares the ocean's pressure at "
        "the bed to the subglacial water pressure. When it exceeds one, the ocean can push in. It's a door "
        "that only opens when the outside pressure wins. The prediction is directional: this should cluster "
        "at grounding lines.",
        cards=[("RTN = (p_ocean − p_atm) / p_w", "ocean head vs subglacial water pressure"),
               ("RTN > 1", "ocean can intrude grounded ice"),
               ("directional", "should concentrate at grounding lines")],
        support="A 'door that opens only when the outside pressure wins.'")
    d.assert_slide(
        "On real Bedmap2 geometry, the intrusion signal piles up at the grounding line",
        "Bet one, first dataset. On real Antarctic geometry the RTN>1 cells sit a median six kilometres "
        "from the grounding line; everything else is two hundred-plus kilometres away. The signal decays "
        "monotonically inland. Show the map; let them see the concentration.",
        image=fig("validation", "reports", "rtn_bedmap2.png"),
        caption="RTN>1 concentration vs distance-to-grounding-line (Bedmap2).",
        cards=[("6 km", "median dist. to GL (RTN>1)"),
               ("222 km", "median for all other cells"),
               ("monotonic", "decays smoothly inland")])
    d.assert_slide(
        "The signal survives an independent dataset — so it isn't an artifact of one map",
        "This is the slide that earns trust. We rerun on BedMachine v4 — a completely independent thickness "
        "inversion — and get the same decay and the same fractions. The two thickness maps agree at r equals "
        "0.97, and the loader tests pass. One dataset is a claim; two is a result.",
        image=fig("validation", "reports", "rtn_bedmachine.png"),
        caption="Independent BedMachine v4 cross-check (decimated to 2 km).",
        cards=[("2.5 / 1.4 / 0.8%", "RTN>1 fraction, φ=0.8/0.9/0.95"),
               ("r = 0.970", "BedMachine vs Bedmap2 thickness"),
               ("10/10", "loader tests pass")],
        status=("VERIFIED", "reproduces on an independent thickness inversion"))
    d.assert_slide(
        "Our first guess for surge timing — ice thermal diffusion — is wrong by ~80,000×",
        "Bet two, and we lose it cleanly. The natural hypothesis is that ice-stream surges lag the water "
        "forcing by the time heat diffuses through the ice. On 131 real lakes that time is 150,000 years; "
        "observed lags are months. It's off by ten thousands. A thick ice slab simply can't feel a fast "
        "pulse — and saying so out loud is the point.",
        cards=[("τ_ice ≈ 151,000 yr", "thermal-diffusion lag on 131 lakes"),
               ("0.02–2 yr", "observed post-drainage lags"),
               ("~8×10⁴× too slow", "a clean, diagnostic falsification")],
        status=("FALSIFIED", "thermal sliding-lag kernel, on real geometry"))
    d.assert_slide(
        "The memory isn't in the ice — it's in the plumbing, and it's an exact memory kernel",
        "The falsification is productive: it tells us where the memory lives. The cavity-to-channel "
        "hydraulic system has a linear response that is an exact Mori-Zwanzig memory kernel — the same "
        "structure certified in Paper 1, here to one part in 10^18. So we replaced a wrong clock with a "
        "right one.",
        image=fig("validation", "reports", "h3_cmn_reduced_model.png"),
        caption="Reduced cavity↔channel memory model vs full trajectory.",
        cards=[("hydrology, not ice", "where the memory lives"),
               ("exact MZ kernel", "kernel = Green's function"),
               ("7×10⁻¹⁸", "certified to")],
        status=("VERIFIED", "same MZ structure certified in Paper 1"))
    d.assert_slide(
        "Bet three is an honest null on open data — and we can say exactly why",
        "Bet three: test the lag directly on satellite data. We get no significant signal — peak half a "
        "sigma, zero of five events. But this isn't a quiet failure: the cause is temporal aliasing, the "
        "real lag is sub-annual and the data is quarterly. The fix is named: sub-annual GPS or fast-outlet "
        "velocity. An honest null, not a swept-under-the-rug one.",
        image=fig("validation", "reports", "lake_lag_itslive.png"),
        caption="Matched lake-drainage → velocity lag (CryoSat-2 + ITS_LIVE).",
        cards=[("+0.56σ, 0/5", "no significant lag detected"),
               ("temporal aliasing", "sub-annual lag vs quarterly data"),
               ("fix: sub-annual GPS", "the route to a verdict")],
        status=("OPEN", "unverifiable here for a stated, fixable reason — not a refutation"))
    d.assert_slide(
        "We can read the sliding law off the surface — three independent ways",
        "Now the constructive payoff. There's a closed-form effective-pressure 'master curve' for sliding, "
        "verified to 1e-4, that you can probe three independent ways — drainage steps, ocean-thermal "
        "gating, and ocean tides — and invert for the flotation threshold to a few percent. It even gives "
        "an ungrounding early-warning.",
        image=fig("validation", "reports", "sn_master_curve.png"),
        caption="s_N(N) master curve and inversion.",
        cards=[("|s_N| = m/(1−(N_c/N)^m)", "closed-form, verified 1.4×10⁻⁴"),
               ("3 probes", "drainage / ocean-thermal / tides"),
               ("N_c to a few %", "+ ungrounding early-warning")])
    d.assert_slide(
        "And the tidal forcing it relies on is measured, not assumed",
        "Close the loop on a real number. The tidal probe needs a forcing amplitude — and we measure it "
        "from the CATS2008 tide model across the BedMachine grounding zone, not assume it. The tidal "
        "pressure swing is 0.3% of the ice overburden, growing toward the grounding line. Eleven hundred "
        "valid cells.",
        image=fig("validation", "reports", "tidal_forcing_gz.png"),
        caption="Tidal pressure swing vs distance-to-grounding-line (CATS2008 × BedMachine).",
        cards=[("0.30%", "median tidal swing / overburden"),
               ("0.21% → 0.45%", "grows toward the grounding line"),
               ("n = 1108", "valid grounding-zone cells")],
        status=("VERIFIED", "the forcing magnitude is measured (CATS2008 × BedMachine)"))
    d.scope(
        ["RTN GL concentration (Bedmap2 + BedMachine)", "MZ kernel structure (round-off)",
         "s_N(N) master curve (1.4×10⁻⁴)", "Tidal forcing measured (CATS2008)"],
        ["Thermal (τ_ice) sliding-lag kernel", "Surge memory living in ice thermics"],
        ["Matched lag (honest null; aliasing)", "Ro thinning vs hydraulic (needs DInSAR)",
         "Absolute s_N calibration (per-event)"],
        "Show all three outcomes on one slide — verified, falsified, open — because that even-handedness "
        "is the contribution. Don't hide the red column; point at it.")
    d.assert_slide(
        "The takeaway: real forecasts you can lose — and a method that pays off when you do",
        "Land the whole talk: we made three falsifiable forecasts on open data, won one twice, lost one "
        "cleanly, and turned an honest null into a concrete next measurement — plus a field-readable "
        "sliding law and an ungrounding early-warning. Everything is public and reproducible.",
        cards=[("verify / falsify / null", "all reported the same way"),
               ("productive falsification", "wrong clock → right clock"),
               ("public + reproducible", "github.com/abdh0saifelden2-debug/K")])
    return d.save("paper4_subglacial_hydrology_forecasts.pptx")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for fn in (paper1, paper2, paper3, paper4):
        pptx, md = fn()
        print("wrote", os.path.basename(pptx), "+", os.path.basename(md))
