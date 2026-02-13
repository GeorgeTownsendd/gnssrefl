# Arc QC Filters

After `extract_arcs` produces arcs (see [Extracting Satellite Arcs](extract_arcs.md)),
both `gnssir` and `phase` apply QC filters to each arc individually before
accepting results. This page documents every filter and its default threshold.

## Shared filters (gnssir + phase)

| Filter | Rejects when | Json key | Default |
|--------|-------------|----------|---------|
| ediff | Arc doesn't cover nearly the full [e1, e2] elevation range: requires emin ≤ e1 + ediff and emax ≥ e2 - ediff. With the default ediff=2 and e1=5, e2=25, an arc needs emin ≤ 7° and emax ≥ 23°. | `ediff` | 2 deg |
| tooclose | LSP peak RH is within 0.10 m of `minH` or `maxH`, or LSP returned no result | `minH`, `maxH` | 0.5, 8 m |
| No noise data | Noise region `[NReg[0], NReg[1]]` doesn't overlap with `[minH, maxH]` | `NReg` | [0.5, 8] m |
| Amplitude | `peak_amplitude <= reqAmp` | `reqAmp` | 5.0 |
| Peak-to-noise | `peak_amplitude / mean_noise <= PkNoise` | `PkNoise` | 2.8 |
| Arc duration | `arc_duration >= delTmax` | `delTmax` | 75 min |

All defaults are set by `gnssir_input` and stored in the station json file.


## Phase-only filters

These run before the shared filters in `phase_tracks()`:

| Filter | Rejects when |
|--------|-------------|
| No apriori track | Satellite and azimuth don't match any predefined track from the apriori file |
| L2C/L5 capability | Satellite doesn't transmit the requested frequency on the date being processed |

