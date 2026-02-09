"""Golden file regression tests for gnssir, phase, and arc output.

These tests run gnssir and phase against committed fixture data and compare
output to known-good baselines in test/data/expected/. Any unintended change
to numeric output will cause a failure with a track-level diff showing which
satellite arcs were added, removed, or changed.

To regenerate golden files after an INTENTIONAL change:
    python test/generate_golden_output.py
"""
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

EXPECTED_DIR = Path(__file__).parent / "data" / "expected"

# gnssir column layout (17 columns)
GNSSIR_COLS = [
    "year", "doy", "RH", "sat", "UTCtime", "Azim", "Amp", "eminO", "emaxO",
    "NumbOf", "freq", "rise", "EdotF", "PkNoise", "DelT", "MJD", "refr",
]

# phase column layout (16 columns)
PHASE_COLS = [
    "Year", "DOY", "Hour", "Phase", "Nv", "Azimuth", "Sat", "Ampl",
    "emin", "emax", "DelT", "aprioriRH", "freq", "estRH", "pk2noise", "LSPAmp",
]


def _load_output(path):
    """Load a gnssir/phase output file as a numpy array, skipping % headers."""
    return np.loadtxt(path, comments="%")


def _key_gnssir(row):
    """Track key for gnssir: (sat, rise/set)."""
    return (int(row[3]), int(row[11]))


def _key_phase(row):
    """Track key for phase: (sat, rounded azimuth)."""
    return (int(row[6]), int(round(row[5])))


def _match_rows(actual, expected, key_fn, time_fn, max_dt=0.5):
    """Match actual rows to expected rows by track key + nearest absolute time.

    Parameters
    ----------
    max_dt : float
        Maximum time separation (in the units of time_fn, i.e. days) to
        consider two rows the same arc.  Default 0.5 days — same arc is
        within minutes, different-day arcs are ~1.0 apart.

    Returns (matched, missing, new) where:
      matched: list of (actual_row, expected_row) pairs
      missing: list of expected rows with no actual match
      new:     list of actual rows with no expected match
    """
    # Index expected rows by key
    exp_by_key = defaultdict(list)
    for i, row in enumerate(expected):
        exp_by_key[key_fn(row)].append((i, row))

    used_exp = set()
    matched = []
    new_rows = []

    for act_row in actual:
        key = key_fn(act_row)
        candidates = exp_by_key.get(key, [])
        # Filter out already-matched candidates
        available = [(i, r) for i, r in candidates if i not in used_exp]
        if not available:
            new_rows.append(act_row)
            continue
        # Pick nearest by absolute time, reject if too far apart
        act_time = time_fn(act_row)
        best_idx, best_row = min(
            available, key=lambda pair: abs(time_fn(pair[1]) - act_time)
        )
        if abs(time_fn(best_row) - act_time) > max_dt:
            new_rows.append(act_row)
            continue
        used_exp.add(best_idx)
        matched.append((act_row, best_row))

    # Remaining expected rows are missing
    missing = [row for i, row in enumerate(expected) if i not in used_exp]

    return matched, missing, new_rows


def _fmt_gnssir_row(row, prefix=""):
    """Format a gnssir row for the diff summary."""
    sat = int(row[3])
    rise = "rise" if int(row[11]) == 1 else "set"
    return (
        f"{prefix}sat {sat:>2} {rise:<4}  UTC={row[4]:.2f}  doy={int(row[1])}  "
        f"RH={row[2]:.3f}  az={row[5]:.1f}"
    )


def _fmt_phase_row(row, prefix=""):
    """Format a phase row for the diff summary."""
    sat = int(row[6])
    return (
        f"{prefix}sat {sat:>2}  hour={row[2]:.2f}  doy={int(row[1])}  "
        f"Phase={row[3]:.3f}  estRH={row[13]:.3f}  az={row[5]:.1f}"
    )


# gnssir columns that are already shown in the track identifier (sat, rise, UTCtime)
_GNSSIR_SKIP = {3, 4, 11}
# phase columns already shown in the track identifier (Sat, Hour, Azimuth)
_PHASE_SKIP = {2, 5, 6}


def _fmt_gnssir_changed(act, exp):
    """Format a changed gnssir row showing field-level diffs."""
    sat = int(exp[3])
    rise = "rise" if int(exp[11]) == 1 else "set"
    avg_time = (act[4] + exp[4]) / 2
    parts = [f"sat {sat:>2} {rise:<4}  UTC~{avg_time:.2f}"]
    labels = GNSSIR_COLS
    for i in range(len(act)):
        if i in _GNSSIR_SKIP:
            continue
        if not np.isclose(act[i], exp[i], rtol=1e-6):
            parts.append(f"{labels[i]}: {exp[i]:.4g} -> {act[i]:.4g}")
    return "  " + "  ".join(parts)


def _fmt_phase_changed(act, exp):
    """Format a changed phase row showing field-level diffs."""
    sat = int(exp[6])
    avg_time = (act[2] + exp[2]) / 2
    parts = [f"sat {sat:>2}  hour~{avg_time:.2f}"]
    labels = PHASE_COLS
    for i in range(len(act)):
        if i in _PHASE_SKIP:
            continue
        if not np.isclose(act[i], exp[i], rtol=1e-6):
            parts.append(f"{labels[i]}: {exp[i]:.4g} -> {act[i]:.4g}")
    return "  " + "  ".join(parts)


def _compare_tracks(actual, expected, fmt):
    """Compare actual vs expected output with track-aware matching.

    Parameters
    ----------
    actual : np.ndarray — rows from the test run
    expected : np.ndarray — rows from the golden file
    fmt : str — "gnssir" or "phase"
    """
    if fmt == "gnssir":
        key_fn = _key_gnssir
        time_fn = lambda row: row[15]              # MJD (absolute time)
        fmt_row = _fmt_gnssir_row
        fmt_changed = _fmt_gnssir_changed
    else:
        key_fn = _key_phase
        time_fn = lambda row: row[1] + row[2] / 24  # doy + hour/24
        fmt_row = _fmt_phase_row
        fmt_changed = _fmt_phase_changed

    matched, missing, new = _match_rows(actual, expected, key_fn, time_fn)

    # Check matched pairs for value changes
    changed = []
    for act_row, exp_row in matched:
        if not np.allclose(act_row, exp_row, rtol=1e-6):
            changed.append((act_row, exp_row))

    n_diffs = len(missing) + len(new) + len(changed)
    if n_diffs == 0:
        return

    lines = [f"Output changed -- {n_diffs} track difference{'s' if n_diffs != 1 else ''}:"]

    if missing:
        lines.append("")
        lines.append("MISSING (expected but not in actual):")
        for row in missing:
            lines.append(fmt_row(row, prefix="  "))

    if new:
        lines.append("")
        lines.append("NEW (in actual but not expected):")
        for row in new:
            lines.append(fmt_row(row, prefix="  "))

    if changed:
        lines.append("")
        lines.append("CHANGED:")
        for act_row, exp_row in changed:
            lines.append(fmt_changed(act_row, exp_row))

    pytest.fail("\n".join(lines))


def test_gnssir_output_unchanged(refl_code_with_mchl):
    from gnssrefl.gnssir_cl import gnssir

    tmp = refl_code_with_mchl
    gnssir("mchl", 2025, 11, screenstats=False, plt=False)

    actual = _load_output(tmp / "2025" / "results" / "mchl" / "011.txt")
    expected = _load_output(EXPECTED_DIR / "gnssir" / "2025" / "011.txt")
    _compare_tracks(actual, expected, "gnssir")


def test_phase_output_unchanged(refl_code_with_mchl):
    from gnssrefl.gnssir_cl import gnssir
    from gnssrefl.quickPhase import quickphase

    tmp = refl_code_with_mchl
    # phase needs gnssir results first
    gnssir("mchl", 2025, 11, screenstats=False, plt=False)
    quickphase("mchl", 2025, 11, screenstats=False, plt=False)

    actual = _load_output(tmp / "2025" / "phase" / "mchl" / "011.txt")
    expected = _load_output(EXPECTED_DIR / "phase" / "2025" / "011.txt")
    _compare_tracks(actual, expected, "phase")


def test_gnssir_arcs_unchanged(refl_code_with_mchl):
    from gnssrefl.gnssir_cl import gnssir

    tmp = refl_code_with_mchl
    gnssir("mchl", 2025, 11, savearcs=True, screenstats=False, plt=False)

    actual_dir = tmp / "2025" / "arcs" / "mchl" / "011"
    expected_dir = EXPECTED_DIR / "arcs" / "2025" / "011"

    actual_files = {f.name for f in actual_dir.glob("*.txt")}
    expected_files = {f.name for f in expected_dir.glob("*.txt")}

    missing = sorted(expected_files - actual_files)
    new = sorted(actual_files - expected_files)
    common = sorted(actual_files & expected_files)

    changed = []
    for name in common:
        act = np.loadtxt(actual_dir / name, comments="%")
        exp = np.loadtxt(expected_dir / name, comments="%")
        if act.shape != exp.shape or not np.allclose(act, exp, rtol=1e-6):
            # Identify which columns diverged
            col_names = ["elev", "dSNR", "sec"]
            bad_cols = []
            if act.shape == exp.shape:
                for ci, cn in enumerate(col_names):
                    if not np.allclose(act[:, ci], exp[:, ci], rtol=1e-6):
                        bad_cols.append(cn)
            else:
                bad_cols.append(f"shape {exp.shape}->{act.shape}")
            changed.append((name, bad_cols))

    n_diffs = len(missing) + len(new) + len(changed)
    if n_diffs == 0:
        return

    lines = [f"Arc output changed -- {n_diffs} arc difference{'s' if n_diffs != 1 else ''}:"]

    if missing:
        lines.append("")
        lines.append("MISSING arcs (expected but not produced):")
        for name in missing:
            lines.append(f"  {name}")

    if new:
        lines.append("")
        lines.append("NEW arcs (produced but not expected):")
        for name in new:
            lines.append(f"  {name}")

    if changed:
        lines.append("")
        lines.append("CHANGED arcs:")
        for name, bad_cols in changed:
            lines.append(f"  {name}  diverged in: {', '.join(bad_cols)}")

    pytest.fail("\n".join(lines))
