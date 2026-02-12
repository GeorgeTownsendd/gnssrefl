# Arc QC Filters

After `extract_arcs` produces arcs (see [Extracting Satellite Arcs](extract_arcs.md)),
both `gnssir` and `phase` apply QC filters to each arc individually before
accepting results. This page documents every filter and its default threshold.

## Shared filters (gnssir + phase)

| Filter | Rejects when | Json key | Default |
|--------|-------------|----------|---------|
| ediff: start | `min_elevation - e1 > ediff` | `ediff` | 2 deg |
| ediff: end | `e2 - max_elevation > ediff` | `ediff` | 2 deg |
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

