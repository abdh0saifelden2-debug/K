# Speaker script — Paper 4

Write-to-speak narration, one block per slide (also embedded in the deck's notes pane). Say it in your own words; let the slide carry the visual.

## Slide 1 — Falsifiable forecasts for subglacial hydrology

Open with the stance, not the results: the field is full of plausible stories about water under the ice; we made three concrete predictions and put each on trial against open data. Promise honesty — one is verified twice, one is falsified, one is an honest null.

## Slide 2 — Three bets on the ice — and we show our hand on every one

Most talks show you only the bets they won. The discipline here is that we report all three the same way. That's what makes them forecasts and not stories.

## Slide 3 — Ocean water should intrude grounded ice exactly where ocean pressure beats the water pressure

Set up bet one simply. Define a dimensionless number, RTN, that compares the ocean's pressure at the bed to the subglacial water pressure. When it exceeds one, the ocean can push in. It's a door that only opens when the outside pressure wins. The prediction is directional: this should cluster at grounding lines.

## Slide 4 — On real Bedmap2 geometry, the intrusion signal piles up at the grounding line

Bet one, first dataset. On real Antarctic geometry the RTN>1 cells sit a median six kilometres from the grounding line; everything else is two hundred-plus kilometres away. The signal decays monotonically inland. Show the map; let them see the concentration.

## Slide 5 — The signal survives an independent dataset — so it isn't an artifact of one map

This is the slide that earns trust. We rerun on BedMachine v4 — a completely independent thickness inversion — and get the same decay and the same fractions. The two thickness maps agree at r equals 0.97, and the loader tests pass. One dataset is a claim; two is a result.

## Slide 6 — Our first guess for surge timing — ice thermal diffusion — is wrong by ~80,000×

Bet two, and we lose it cleanly. The natural hypothesis is that ice-stream surges lag the water forcing by the time heat diffuses through the ice. On 131 real lakes that time is 150,000 years; observed lags are months. It's off by ten thousands. A thick ice slab simply can't feel a fast pulse — and saying so out loud is the point.

## Slide 7 — The memory isn't in the ice — it's in the plumbing, and it's an exact memory kernel

The falsification is productive: it tells us where the memory lives. The cavity-to-channel hydraulic system has a linear response that is an exact Mori-Zwanzig memory kernel — the same structure certified in Paper 1, here to one part in 10^18. So we replaced a wrong clock with a right one.

## Slide 8 — Bet three is an honest null on open data — and we can say exactly why

Bet three: test the lag directly on satellite data. We get no significant signal — peak half a sigma, zero of five events. But this isn't a quiet failure: the cause is temporal aliasing, the real lag is sub-annual and the data is quarterly. The fix is named: sub-annual GPS or fast-outlet velocity. An honest null, not a swept-under-the-rug one.

## Slide 9 — We can read the sliding law off the surface — three independent ways

Now the constructive payoff. There's a closed-form effective-pressure 'master curve' for sliding, verified to 1e-4, that you can probe three independent ways — drainage steps, ocean-thermal gating, and ocean tides — and invert for the flotation threshold to a few percent. It even gives an ungrounding early-warning.

## Slide 10 — And the tidal forcing it relies on is measured, not assumed

Close the loop on a real number. The tidal probe needs a forcing amplitude — and we measure it from the CATS2008 tide model across the BedMachine grounding zone, not assume it. The tidal pressure swing is 0.3% of the ice overburden, growing toward the grounding line. Eleven hundred valid cells.

## Slide 11 — Scope

Show all three outcomes on one slide — verified, falsified, open — because that even-handedness is the contribution. Don't hide the red column; point at it.

## Slide 12 — The takeaway: real forecasts you can lose — and a method that pays off when you do

Land the whole talk: we made three falsifiable forecasts on open data, won one twice, lost one cleanly, and turned an honest null into a concrete next measurement — plus a field-readable sliding law and an ungrounding early-warning. Everything is public and reproducible.

