# Speaker script — Paper 1

Write-to-speak narration, one block per slide (also embedded in the deck's notes pane). Say it in your own words; let the slide carry the visual.

## Slide 1 — Single-eddy-diffusivity closure is the Markovian collapse of a memory kernel

Open with the punchline: the most common turbulence closure isn't a law, it's the zero-memory limit of something exact — and that's both why it fails and how to fix it. Keep it warm; promise one clean idea, one decisive test.

## Slide 2 — Think of an echo-cancelling app that erases a room's reverb

Picture an audio 'de-reverb' slider: handy for a phone call, fatal for measuring the room. The question of this talk is: what exactly does the closure delete, and can we put it back? Pause after the analogy and let it land before the formal version.

## Slide 3 — The standard closure replaces an exact memory integral with a single instant

Here's the one equation of the talk, in words. The true unresolved force depends on the whole history of the flow through a Mori-Zwanzig memory kernel. Down-gradient K-theory keeps only the present instant — it collapses that kernel to a spike. State it, don't dwell.

## Slide 4 — How wrong it is, is set by one number — a memory (Deborah) number

The error isn't mysterious; it's governed by a single dimensionless ratio: memory time over the resolved timescale. When that ratio is tiny, K-theory is exact; when it's order one, the closure is wrong by a predictable amount. This converts a vague worry into a number you can check.

## Slide 5 — The repair keeps the memory and restores the backscatter the closure threw away

The fix is not a fudge factor. We keep the kernel (a non-local-in-time eddy viscosity) and restore its reactive part through a projected fluctuation-dissipation relation — which brings back backscatter, energy flowing back up to large scales. Net eddy viscosity can even go negative.

## Slide 6 — The decisive test: the repaired kernel reproduces the exact subgrid force to round-off

This is the slide to slow down on. It's an a-priori test against the exact filtered force — not a tuned fit. The projection is exact to 1e-16, the kernel equals the eliminated channel's Green's function to 1e-18, and the reduced model tracks the full trajectory to 1e-8. The memory is necessary to get the interior peak. This is a structural identity, not a calibration.

## Slide 7 — The memory time is a real, measurable property of the flow — not of the tracer

One last point that makes it useful: the memory clock is the same regardless of which scalar you advect, and it inverts for a mean transport speed. So it's a physical handle, not a bookkeeping device — something you could measure.

## Slide 8 — Scope

Be honest and quick: verified a-priori and in proxies, not yet in a full 3-D solver; that's the next step, not a hidden caveat.

## Slide 9 — The takeaway: closure failure is a memory problem, and memory is repairable

Land the plane. One sentence: when a down-gradient model fails, it's because the memory number isn't small — and we can put the memory (and its backscatter) back, verified. Point them to the public code and the companion relationships NR1, NR13, NR15, NR23.

