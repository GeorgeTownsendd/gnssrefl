# Regenerating Golden Files

Golden files are the expected output that regression tests compare against.
Regenerate them when output changes are **intentional**.

## Tests

- `test_gnssir_output_unchanged`: gnssir results for day 011 match golden output
- `test_gnssir_midnite_output_unchanged`: gnssir -midnite results for day 011 match golden output
- `test_gnssir_arcs_unchanged`: individual arc files for day 011 match golden output
- `test_gnssir_midnite_arcs_unchanged`: individual arc files with -midnite for day 011 match golden output
- `test_phase_output_unchanged`: phase estimates for day 011 match golden output

## Prerequisites

- gnssrefl installed (`pip install .`) with CLI commands available on `$PATH`
- `$REFL_CODE` environment variable set (see install docs)
- If regenerating the apriori file, a full year of snr data from mchl/mchl00aus

## 1. Golden files

Run non-midnite first, copy, then midnite, copy (commands overwrite output files).

```bash
# gnssir day 011
gnssir mchl 2025 11 -savearcs T
cp $REFL_CODE/2025/results/mchl/011.txt test/data/expected/gnssir/2025/
cp $REFL_CODE/2025/arcs/mchl/011/*.txt test/data/expected/arcs/2025/011/

# phase day 011
phase mchl 2025 11
cp $REFL_CODE/2025/phase/mchl/011.txt test/data/expected/phase/2025/

# gnssir -midnite day 011
gnssir mchl 2025 11 -midnite T -savearcs T
cp $REFL_CODE/2025/results/mchl/011.txt test/data/expected/gnssir_midnite/2025/
cp $REFL_CODE/2025/arcs/mchl/011/*.txt test/data/expected/arcs_midnite/2025/011/
```

## 2. Supplementary files

Only needed if station config or input data changes.

```bash
# SNR files — days 010-012 needed for midnite buffer on day 011
cp $REFL_CODE/2025/snr/mchl/mchl010*.25.snr66* test/data/refl_code/2025/snr/mchl/
cp $REFL_CODE/2025/snr/mchl/mchl011*.25.snr66* test/data/refl_code/2025/snr/mchl/
cp $REFL_CODE/2025/snr/mchl/mchl012*.25.snr66* test/data/refl_code/2025/snr/mchl/

# Station JSON config — created by gnssir_input mchl
cp $REFL_CODE/input/mchl/mchl.json test/data/refl_code/input/mchl/

# Apriori RH file — create with vwc_input mchl after running gnssir mchl 2025 1 -doy_end 365
cp $REFL_CODE/input/mchl/mchl_phaseRH_L2.txt test/data/refl_code/input/mchl/
```
