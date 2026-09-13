# Interface design

Source of the ClaimSight interface design, authored as a multi-artboard canvas.

| File | Screen |
|---|---|
| `Upload.dc.html` | Photo drop zone |
| `Analyzing.dc.html` | Analysis in progress |
| `Main.dc.html` | Claim report — the core screen |
| `PhotoDetail.dc.html` | Single photo with detection overlay |
| `DirectionConsole.dc.html`, `DirectionWorkbench.dc.html` | Rejected direction sketches, kept for the record |
| `canvas.json` | Artboard layout, pages and annotations |
| `claimsight-interface.html` | The compiled canvas — open it in a browser |

## Design system

Cool neutrals computed in oklch (hue 265, chroma ≤ 0.011); two accents sharing
lightness and chroma (L=0.58, C=0.155) at opposite hues — `#3979D4` for actions
and covered states, `#C54D47` for missing photos and severe damage.
Type is Manrope for interface copy and IBM Plex Mono for machine output.

That last split is a rule, not a decoration: **monospace means a value produced
by a model, sans-serif means text written by a human.** An adjuster can tell at
a glance what the system is asserting from what the interface is saying.

Layout follows three HIG principles — deference (soft elevation and tinted
fills rather than hard rules), clarity (wide headings with negative tracking),
depth (continuous corner radii, one dark surface for the photo viewer only).
