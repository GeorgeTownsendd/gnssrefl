#!/usr/bin/env python
"""
Regenerate golden output files for regression tests.

Run this script manually when output changes are INTENTIONAL:
    python test/generate_golden_output.py

It runs gnssir and phase against the fixture data and overwrites
test/data/expected/ with the new baseline output.
"""
import os
import shutil
from pathlib import Path
from unittest.mock import patch

SCRIPT_DIR = Path(__file__).parent
FIXTURE_DIR = SCRIPT_DIR / "data" / "refl_code"
EXPECTED_DIR = SCRIPT_DIR / "data" / "expected"


def setup_refl_code(tmp_path):
    """Copy fixture data into a temporary REFL_CODE tree."""
    # Copy SNR files
    snr_dst = tmp_path / "2025" / "snr" / "mchl"
    snr_dst.mkdir(parents=True)
    snr_src = FIXTURE_DIR / "2025" / "snr" / "mchl"
    for f in snr_src.glob("*.snr66.gz"):
        shutil.copy2(f, snr_dst / f.name)

    # Copy JSON config and apriori RH
    input_dst = tmp_path / "input" / "mchl"
    input_dst.mkdir(parents=True)
    input_src = FIXTURE_DIR / "input" / "mchl"
    shutil.copy2(input_src / "mchl.json", input_dst / "mchl.json")
    shutil.copy2(input_src / "mchl_phaseRH_L2.txt", input_dst / "mchl_phaseRH_L2.txt")

    # Create required output directories
    for subdir in ["Files", "logs", "phase"]:
        (tmp_path / subdir).mkdir(parents=True, exist_ok=True)
    (tmp_path / "2025" / "results" / "mchl").mkdir(parents=True)
    (tmp_path / "2025" / "phase" / "mchl").mkdir(parents=True)


def main():
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="gnssrefl_golden_"))
    print(f"Working in {tmp}")
    setup_refl_code(tmp)

    with patch.dict(os.environ, {
        "REFL_CODE": str(tmp),
        "ORBITS": ".",
        "EXE": ".",
    }):
        # Import after patching env so modules pick up the right REFL_CODE
        from gnssrefl.gnssir_cl import gnssir
        from gnssrefl.quickPhase import quickphase

        # 1. gnssir day 011
        print("Running gnssir day 011 ...")
        gnssir("mchl", 2025, 11, screenstats=False, plt=False)
        result = tmp / "2025" / "results" / "mchl" / "011.txt"
        dst = EXPECTED_DIR / "gnssir" / "2025" / "011.txt"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(result, dst)
        print(f"  -> {dst}")

        # 2. phase day 011 — needs gnssir results from step 1
        print("Running phase day 011 ...")
        quickphase("mchl", 2025, 11, screenstats=False, plt=False)
        phase_result = tmp / "2025" / "phase" / "mchl" / "011.txt"
        dst = EXPECTED_DIR / "phase" / "2025" / "011.txt"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(phase_result, dst)
        print(f"  -> {dst}")

        # 3. gnssir day 011 with savearcs
        # Remove previous result so we get a fresh run
        result.unlink(missing_ok=True)
        print("Running gnssir day 011 with savearcs ...")
        gnssir("mchl", 2025, 11, savearcs=True, screenstats=False, plt=False)
        arcs_src = tmp / "2025" / "arcs" / "mchl" / "011"
        arcs_dst = EXPECTED_DIR / "arcs" / "2025" / "011"
        if arcs_dst.exists():
            shutil.rmtree(arcs_dst)
        arcs_dst.mkdir(parents=True)
        for f in arcs_src.glob("*.txt"):
            shutil.copy2(f, arcs_dst / f.name)
        print(f"  -> {arcs_dst}/ ({len(list(arcs_dst.glob('*.txt')))} files)")

    # Clean up
    shutil.rmtree(tmp)
    print("Done. Golden output files updated.")


if __name__ == "__main__":
    main()
