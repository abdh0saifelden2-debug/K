# Data citation & acknowledgement statements

Required citation and acknowledgement language for every **external data product** used
anywhere in this repository's papers. Pasting the relevant block into a manuscript's
*Data availability* and *Acknowledgements* sections satisfies (a) each data provider's
own use policy and (b) The Cryosphere / Copernicus submission requirements.

Sources for the policies and the exact "cite-as" strings are the providers' own landing
pages (NEON citing guidelines; USAP-DC dataset pages; NSIDC BedMachine v4 user guide),
retrieved 2026-06-18.

---

## 0. Which paper needs which statement

| Paper | Target venue | External provider data ingested | Provider statement needed? |
|---|---|---|---|
| **P1** closure theory | PRFluids / JFM | none — DNS, own compute | **No** (code/repo only) |
| **P2** subglacial melt ceiling | The Cryosphere / J. Glaciol. | none — DNS + *literature* citation of Bushuk et al. 2019 | **No** (code/repo only) |
| **P3** scallop parity break *(being submitted now)* | **The Cryosphere** | **none** — solver output + *literature* citation of Bushuk et al. 2019 | **No — public repo link only** |
| **P4a** hydraulic memory kernel | The Cryosphere / J. Glaciol. | USAP-DC 601439; USAP-DC 601470; BedMachine (NSIDC-0756); Bedmap2 | **Yes** → §2, §3, §4 |
| **P4b** sliding-law field test | The Cryosphere / J. Glaciol. | CATS2008 (USAP-DC 601235); BedMachine (NSIDC-0756); ITS\_LIVE; CryoSat-2 dates | **Yes** → §2, §3, §4 |
| Two-clocks **observational** analysis (`general_two_clocks/`) | not yet drafted as a TeX paper | NEON DP4.00200 / DP1.00004 / DP1.00003 | **Yes** → §1 (when written up) |

> **Key point for the current submission:** P3 uses **no** NEON, USAP-DC, or NSIDC data.
> A literature citation (Bushuk et al. 2019, *JFM*) is **not** a data deposit and needs no
> data-use statement. P3's only requirement is its public-repository availability line,
> which it already has. **Do not add a NEON/USAP-DC statement to P3.**

---

## 1. NEON — National Ecological Observatory Network  *(two-clocks observational work only)*

License: CC0 1.0 (public domain). Attribution is still required by scholarly norm; the
"author" of the data is **NEON itself**, not individuals.

**Acknowledgement (verbatim, required):**
> The National Ecological Observatory Network is a program sponsored by the U.S. National
> Science Foundation (NSF) and operated under cooperative agreement by Battelle. This
> material is based in part upon work supported by the NSF through the NEON Program.

**Data citations (one per product; insert the RELEASE DOI from each product page):**
> National Ecological Observatory Network (NEON). (2024). Bundled data products – eddy
> covariance (DP4.00200.001) [Data set]. NEON. https://doi.org/<release-DOI>. Site WREF.
> Accessed <date>.
>
> National Ecological Observatory Network (NEON). (2024). Barometric pressure
> (DP1.00004.001) [Data set]. NEON. https://doi.org/<release-DOI>. Site WREF. Accessed <date>.
>
> National Ecological Observatory Network (NEON). (2024). Triple aspirated air temperature
> (DP1.00003.001) [Data set]. NEON. https://doi.org/<release-DOI>. Site WREF. Accessed <date>.

Notes: each product detail page on `data.neonscience.org` carries a copy-paste citation
with the release DOI; `neonUtilities` emits BibTeX. If **provisional** (non-released) data
were used, NEON requires archiving the exact version used and citing that archived copy.
Our analysis used the public NEON API (no auth) for WREF 2020 — fill the accessed date and
the release DOI for each product from its page before submitting any paper that uses NEON.

---

## 2. USAP-DC — U.S. Antarctic Program Data Center  *(P4a, P4b)*

USAP-DC terms: acknowledge **both** the original data contributors (via their publications
and the Data DOI) **and** USAP-DC (www.usap-dc.org). Each dataset has a Data DOI of the
form `10.15784/<id>`.

**601439 — Antarctic Active Subglacial Lake Inventory from ICESat Altimetry** (lake
drainage volumes used in P4a §5 forcing). License CC BY 4.0.
> Smith, B., Fricker, H., Joughin, I., & Tulaczyk, S. (2012). Antarctic Active Subglacial
> Lake Inventory from ICESat Altimetry. U.S. Antarctic Program (USAP) Data Center.
> https://doi.org/10.15784/601439.

Underlying peer-reviewed paper (cite as well):
> Smith, B. E., Fricker, H. A., Joughin, I. R., & Tulaczyk, S. (2009). An inventory of
> active subglacial lakes in Antarctica detected by ICESat (2003–2008). *Journal of
> Glaciology*, 55(192), 573–595. https://doi.org/10.3189/002214309789470879.

**601470 — Antarctic Ice Thickness, Slipperiness, and Subglacial Lake Locations**
(131-lake statistics; independent thickness cross-check in P4a §3). License CC BY 4.0.
> Stubblefield, A., Arthern, R., Kingslake, J., & Siegfried, M. (2021). Antarctic Ice
> Thickness, Slipperiness, and Subglacial Lake Locations. U.S. Antarctic Program (USAP)
> Data Center. https://doi.org/10.15784/601470.

Underlying peer-reviewed papers (cite as well):
> Arthern, R. J., Hindmarsh, R. C. A., & Williams, C. R. (2015). Flow speed within the
> Antarctic ice sheet and its controls inferred from satellite observations. *J. Geophys.
> Res. Earth Surf.*, 120, 1171–1188. https://doi.org/10.1002/2014JF003239.
>
> Siegfried, M. R., & Fricker, H. A. (2018). Thirteen years of subglacial lake activity in
> Antarctica from multi-mission satellite altimetry. *Annals of Glaciology*, 59(76pt1),
> 42–55. https://doi.org/10.1017/aog.2017.36.

**601235 — CATS2008: Circum-Antarctic Tidal Simulation version 2008** (tidal forcing in
P4b). License **CC BY-NC 4.0** (noncommercial — academic use only).
> Howard, S. L., Erofeeva, S., & Padman, L. (2019). CATS2008: Circum-Antarctic Tidal
> Simulation version 2008. U.S. Antarctic Program (USAP) Data Center.
> https://doi.org/10.15784/601235.

Underlying peer-reviewed papers (cite as well):
> Padman, L., Fricker, H. A., Coleman, R., Howard, S., & Erofeeva, S. (2002). A new tidal
> model for the Antarctic ice shelves and seas. *Annals of Glaciology*, 34, 247–254.
> https://doi.org/10.3189/172756402781817752.
>
> Padman, L., Erofeeva, S., & Fricker, H. A. (2008). Improving Antarctic tide models by
> assimilation of ICESat laser altimetry over ice shelves. *Geophys. Res. Lett.*, 35,
> L22504. https://doi.org/10.1029/2008GL035592.

**USAP-DC acknowledgement line (required):**
> Data were obtained from the U.S. Antarctic Program Data Center (USAP-DC,
> https://www.usap-dc.org), funded by the U.S. National Science Foundation.

---

## 3. NSIDC / NASA  *(P4a, P4b)*

**BedMachine Antarctica, Version 4** (NSIDC-0756). NSIDC requires the data citation **and**
requests acknowledgement of the peer-reviewed paper.
> Morlighem, M. (2025). MEaSUREs BedMachine Antarctica (NSIDC-0756, Version 4) [Data set].
> Boulder, Colorado USA. NASA National Snow and Ice Data Center Distributed Active Archive
> Center. https://doi.org/10.5067/POJQI54A45HX. [Date Accessed].

Peer-reviewed paper (cite as well):
> Morlighem, M., Rignot, E., Binder, T., Blankenship, D. D., Drews, R., Eagles, G., et al.
> (2020). Deep glacial troughs and stabilizing ridges unveiled beneath the margins of the
> Antarctic ice sheet. *Nature Geoscience*, 13, 132–137.
> https://doi.org/10.1038/s41561-019-0510-8.

**ITS\_LIVE — Regional Glacier and Ice Sheet Surface Velocities** (NASA MEaSUREs, NSIDC).
License CC0. Cite the version actually downloaded:
> Gardner, A., Fahnestock, M., Greene, C. A., Kennedy, J. H., Liukis, M., Lopez, L., &
> Scambos, T. (2025). MEaSUREs ITS\_LIVE Regional Glacier and Ice Sheet Surface Velocities,
> Version 2 (NSIDC-0776) [Data set]. Boulder, Colorado USA. NASA NSIDC DAAC.
> https://doi.org/10.5067/JQ6337239C96. [Date Accessed].
> *(Version 1 DOI, if that was the one used: https://doi.org/10.5067/6II6VW8LLWJ7.)*

Peer-reviewed paper (cite as well):
> Gardner, A. S., Moholdt, G., Scambos, T., Fahnestock, M., Ligtenberg, S., van den Broeke,
> M., & Nilsson, J. (2018). Increased West Antarctic and unchanged East Antarctic ice
> discharge over the last 7 years. *The Cryosphere*, 12(2), 521–547.
> https://doi.org/10.5194/tc-12-521-2018.

---

## 4. Published datasets cited as literature (no provider deposit statement)

- **Bedmap2** — Fretwell, P., et al. (2013). Bedmap2: improved ice bed, surface and
  thickness datasets for Antarctica. *The Cryosphere*, 7, 375–393.
  https://doi.org/10.5194/tc-7-375-2013.
- **Bushuk et al. (2019)** — Ice scallops: a laboratory investigation of the ice–water
  interface. *Journal of Fluid Mechanics*, 873, 942–976. https://doi.org/10.1017/jfm.2019.398.
  *(Used by P2 and P3 as a literature/consistency citation only; the raw h(x,t) arrays are
  not openly deposited — see P3 §6. No data-deposit statement applies.)*

---

## ✓ Attribution reconciled (601470 vs BedMachine)

Earlier P4a / omnibus-P4 text said *"independent BedMachine thickness (USAP-DC 601470),"*
which conflated two distinct sources. Verified against the USAP-DC dataset landing page
(https://doi.org/10.15784/601470), the truth is:
- **USAP-DC 601470** (Stubblefield, Arthern, Kingslake & Siegfried 2021) distributes
  **Arthern et al. (2015)** ice-thickness + sliding maps and **Siegfried & Fricker (2018)**
  lake stats — **not** BedMachine. This is the *independent thickness* used in the
  thermal-kernel robustness check (P4a §3.3 / omnibus §4.3; 601470-vs-Bedmap2 `r=0.970`).
- **BedMachine v4** is a *separate* NSIDC product (NSIDC-0756, Morlighem 2025,
  DOI 10.5067/POJQI54A45HX). It is used **only** for the independent continental RTN
  cross-check (P4a §6.2 / omnibus §3.3; `run_rtn_bedmachine.py`).

**Resolution applied (option a):** the manuscripts now cite 601470 / Stubblefield + Arthern
(2015) for the thermal-kernel robustness and drop the word "BedMachine" there, while
keeping BedMachine v4 / Morlighem only where it is genuinely used (the continental screen).
The `run_usapdc_lakes.py` docstring has been corrected to match; the dated reproduction-log
entries retain the original "BedMachine H" wording as a historical record. The previously
co-reported `r=0.970` is the 601470(Arthern)-vs-Bedmap2 thickness correlation and is stated
only there; it is no longer attached to the BedMachine-v4 screen.
