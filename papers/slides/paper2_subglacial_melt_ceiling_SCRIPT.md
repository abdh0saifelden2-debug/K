# Speaker script — Paper 2

Write-to-speak narration, one block per slide (also embedded in the deck's notes pane). Say it in your own words; let the slide carry the visual.

## Slide 1 — A boundary-condition-robust ceiling on turbulent basal melt

Hook with the stake: basal melt sets how fast ice can ungroundi, and a popular assumption is that turbulence cranks it up. We tested that assumption and it failed — in a useful way.

## Slide 2 — Think of a wool blanket, not the wind outside it

Everyone reaches for 'more turbulence = more mixing = more melt'. But if a conductive skin is the bottleneck, stirring the outside barely matters. Hold the blanket image; we'll now show the data agrees with the blanket, not the wind.

## Slide 3 — Turning the turbulence off entirely barely changes the melt

This is the cleanest version of the result. We sweep the solver across closures and boundary conditions, then switch the flow off completely — pure conduction. The melt is reproduced to about one percent. If momentum transport drove melt, that couldn't happen.

## Slide 4 — Three plausible turbulent enhancement mechanisms each net to ≈ zero

We didn't just assert it — we tried the three ways turbulence could plausibly boost melt. Averaged over the bed, each one cancels: net melt is area times local flux, and the average is bounded. The popular (1+CV²) variance enhancement is actually falsified here.

## Slide 5 — Scallops are a real LOCAL signal — but the basin mean stays under the ceiling

Be fair to the other side: there IS a positive signal. Scalloped beds grow channels locally, plus thirty-three percent in area, and it's calibration-free — the latent-heat constant cancels. But locally positive and mean-bounded are not the same thing; the average still respects the ceiling.

## Slide 6 — The ceiling survives the modelling choices a skeptic would attack — same answer

This is why it's trustworthy. No-slip versus slip walls reproduce the trajectory bit-for-bit; the result isn't an artifact of one wall model or one closure. Robustness is the headline.

## Slide 7 — The deliverable is a regime map: where conduction pins melt vs where channels matter

End on something usable. We hand modellers a map: most of the grounded cavity sits in the conduction-pinned regime, with channelization mattering only in a corner. That's a drop-in closure for basal-melt parameterizations.

## Slide 8 — Scope

Keep it crisp: the falsified item (turbulent enhancement) is the point of the paper, not an embarrassment. The one open knob is the network gain.

## Slide 9 — The takeaway: don't budget for turbulent melt enhancement — budget for the ceiling

Land it: in grounded subglacial cavities, the safe modelling assumption is a conduction-set ceiling, not turbulent enhancement. Connects to NR5. Point to the public code.

