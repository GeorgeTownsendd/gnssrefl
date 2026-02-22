"""
extract_arcs.py - Standalone module for extracting satellite arcs from SNR data.

This module provides an API for detecting and extracting satellite arcs from Signal-to-Noise Ratio (SNR) data files.

A "pass" represents data from a continuous satellite pass (horizon to horizon) across the sky.

A pass will contain two "arcs", which are determined by splitting the pass at the maximum elevation.
"""

import glob
import os
import shutil
import subprocess

import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union

from gnssrefl.read_snr_files import read_snr
from gnssrefl.utils import circular_mean_deg, circular_distance_deg
from gnssrefl.utils import FileManagement

# Constants
GAP_TIME_LIMIT = 600  # seconds (10 minutes)
MIN_ARC_POINTS = 20
RESULT_COLUMNS = [
    'year', 'doy', 'RH', 'sat', 'UTCtime', 'Azim', 'Amp',
    'eminO', 'emaxO', 'NumbOf', 'freq', 'rise', 'EdotF',
    'PkNoise', 'DelT', 'MJD', 'refr',
]
PHASE_COLUMNS = [
    'year', 'doy', 'Hour', 'Phase', 'Nv', 'Azimuth', 'Sat', 'Ampl',
    'emin', 'emax', 'DelT', 'aprioriRH', 'freq', 'estRH', 'pk2noise', 'LSPAmp',
]

VWC_TRACK_COLUMNS = [
    'Year', 'DOY', 'Hour', 'MJD', 'AzMinEle',
    'PhaseOrig', 'AmpLSPOrig', 'AmpLSOrig', 'DeltaRHOrig',
    'AmpLSPSmooth', 'AmpLSSmooth', 'DeltaRHSmooth',
    'PhaseVegCorr', 'SlopeCorr', 'SlopeFinal',
    'PhaseCorrected', 'VWC',
]

# (snr file column, constellation offset) -> user-facing freq code
# Constellation offset derived from sat number: GPS=0, GLONASS=100, Galileo=200, Beidou=300
COL_CONST_TO_FREQ = {
    (6, 200): 206, (6, 300): 306,
    (7, 0): 1, (7, 100): 101, (7, 200): 201, (7, 300): 301,
    (8, 0): 20, (8, 100): 102, (8, 300): 302,
    (9, 0): 5, (9, 200): 205, (9, 300): 305,
    (10, 200): 207, (10, 300): 307,
    (11, 200): 208, (11, 300): 308,
}
SNR_COLUMNS = [6, 7, 8, 9, 10, 11]



def _freq_for_column_and_sat(column, sat, l2c_sats=None, l5_sats=None):
    """Map (SNR column, sat number) to user-facing freq code. Returns None if invalid."""
    offset = (sat // 100) * 100
    # GPS special cases: L2C vs legacy L2, and L5 transmit check
    if offset == 0:
        if column == 8:
            return 20 if (l2c_sats is None or sat in l2c_sats) else 2
        if column == 9 and l5_sats is not None and sat not in l5_sats:
            return None
    return COL_CONST_TO_FREQ.get((column, offset))


def _load_result_file(path):
    """Load a gnssir result file into a 2-D numpy array.

    Parameters
    ----------
    path : str
        Path to the result file

    Returns
    -------
    np.ndarray
        2-D array with shape (N, 17+)
    """
    data = np.loadtxt(path, comments='%')
    if data.size == 0:
        return None
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


def attach_gnssir_processing_results(
    arcs: List[Tuple[Dict, Dict]],
    results: Union[str, np.ndarray],
    time_tolerance: float = 0.17,
) -> List[Tuple[Dict, Dict]]:
    """Attach gnssir processing results to extracted arcs.

    For each arc, finds the matching row in the gnssir result file based on
    satellite number, frequency, rise/set direction, and UTC time proximity.
    The matched result is stored in ``metadata['gnssir_processing_results']``
    as a dict, or ``None`` if no match is found.

    Parameters
    ----------
    arcs : list of (metadata, data) tuples
        Output from ``extract_arcs()`` or related functions.
    results : str or np.ndarray
        Either a path to a gnssir result file, or a 2-D numpy array already
        loaded from one (N rows x 17+ columns, see ``RESULT_COLUMNS``).
    time_tolerance : float
        Maximum allowed difference in hours between the arc timestamp and the
        result UTCtime for a match. Default 0.17 (~10 minutes).

    Returns
    -------
    list of (metadata, data) tuples
        Same arcs with ``metadata['gnssir_processing_results']`` added.
    """
    if isinstance(results, str):
        results = _load_result_file(results)

    if results is None:
        for metadata, _data in arcs:
            metadata['gnssir_processing_results'] = None
        return arcs

    # Pre-index columns
    COL_SAT = RESULT_COLUMNS.index('sat')       # 3
    COL_UTC = RESULT_COLUMNS.index('UTCtime')    # 4
    COL_FREQ = RESULT_COLUMNS.index('freq')      # 10
    COL_RISE = RESULT_COLUMNS.index('rise')      # 11

    # Build lookup: (sat, freq, rise) -> list of (row_index, utctime)
    lookup: Dict[Tuple[int, int, int], List[Tuple[int, float]]] = {}
    for i in range(results.shape[0]):
        key = (int(results[i, COL_SAT]),
               int(results[i, COL_FREQ]),
               int(results[i, COL_RISE]))
        lookup.setdefault(key, []).append((i, results[i, COL_UTC]))

    # Fields to extract from result row
    result_fields = {
        'RH': (RESULT_COLUMNS.index('RH'), float),
        'Amp': (RESULT_COLUMNS.index('Amp'), float),
        'PkNoise': (RESULT_COLUMNS.index('PkNoise'), float),
        'MJD': (RESULT_COLUMNS.index('MJD'), float),
        'UTCtime': (COL_UTC, float),
        'Azim': (RESULT_COLUMNS.index('Azim'), float),
        'eminO': (RESULT_COLUMNS.index('eminO'), float),
        'emaxO': (RESULT_COLUMNS.index('emaxO'), float),
        'NumbOf': (RESULT_COLUMNS.index('NumbOf'), int),
        'DelT': (RESULT_COLUMNS.index('DelT'), float),
        'EdotF': (RESULT_COLUMNS.index('EdotF'), float),
        'refr': (RESULT_COLUMNS.index('refr'), int),
        'rise': (COL_RISE, int),
    }

    for metadata, data in arcs:
        arc_rise = 1 if metadata['arc_type'] == 'rising' else -1
        key = (metadata['sat'], metadata['freq'], arc_rise)
        candidates = lookup.get(key, [])

        arc_utc = metadata['arc_timestamp']  # hours
        best_idx = None
        best_dt = time_tolerance

        for row_idx, utctime in candidates:
            dt = abs(utctime - arc_utc)
            if dt < best_dt:
                best_dt = dt
                best_idx = row_idx

        if best_idx is not None:
            row = results[best_idx]
            metadata['gnssir_processing_results'] = {
                name: typ(row[col]) for name, (col, typ) in result_fields.items()
            }
        else:
            metadata['gnssir_processing_results'] = None

    return arcs


def attach_phase_processing_results(
    arcs: List[Tuple[Dict, Dict]],
    results: Union[str, np.ndarray],
    time_tolerance: float = 0.17,
) -> List[Tuple[Dict, Dict]]:
    """Attach phase processing results to extracted arcs.

    For each arc, finds the matching row in the phase result file based on
    satellite number, frequency, and UTC time proximity.
    The matched result is stored in ``metadata['phase_processing_results']``
    as a dict, or ``None`` if no match is found.

    Parameters
    ----------
    arcs : list of (metadata, data) tuples
        Output from ``extract_arcs()`` or related functions.
    results : str or np.ndarray
        Either a path to a phase result file, or a 2-D numpy array already
        loaded from one (N rows x 16 columns, see ``PHASE_COLUMNS``).
    time_tolerance : float
        Maximum allowed difference in hours between the arc timestamp and the
        phase Hour for a match. Default 0.17 (~10 minutes).

    Returns
    -------
    list of (metadata, data) tuples
        Same arcs with ``metadata['phase_processing_results']`` added.
    """
    if isinstance(results, str):
        results = _load_result_file(results)

    if results is None:
        for metadata, _data in arcs:
            metadata['phase_processing_results'] = None
        return arcs

    # Pre-index columns
    COL_SAT = PHASE_COLUMNS.index('Sat')        # 6
    COL_HOUR = PHASE_COLUMNS.index('Hour')       # 2
    COL_FREQ = PHASE_COLUMNS.index('freq')       # 12

    # Build lookup: (sat, freq) -> list of (row_index, hour)
    lookup: Dict[Tuple[int, int], List[Tuple[int, float]]] = {}
    for i in range(results.shape[0]):
        key = (int(results[i, COL_SAT]), int(results[i, COL_FREQ]))
        lookup.setdefault(key, []).append((i, results[i, COL_HOUR]))

    # Fields to extract from phase row
    phase_fields = {
        'Phase': (PHASE_COLUMNS.index('Phase'), float),
        'Nv': (PHASE_COLUMNS.index('Nv'), int),
        'Azimuth': (PHASE_COLUMNS.index('Azimuth'), float),
        'Ampl': (PHASE_COLUMNS.index('Ampl'), float),
        'emin': (PHASE_COLUMNS.index('emin'), float),
        'emax': (PHASE_COLUMNS.index('emax'), float),
        'DelT': (PHASE_COLUMNS.index('DelT'), float),
        'aprioriRH': (PHASE_COLUMNS.index('aprioriRH'), float),
        'estRH': (PHASE_COLUMNS.index('estRH'), float),
        'pk2noise': (PHASE_COLUMNS.index('pk2noise'), float),
        'LSPAmp': (PHASE_COLUMNS.index('LSPAmp'), float),
    }

    for metadata, data in arcs:
        key = (metadata['sat'], metadata['freq'])
        candidates = lookup.get(key, [])

        arc_utc = metadata['arc_timestamp']  # hours
        best_idx = None
        best_dt = time_tolerance

        for row_idx, hour in candidates:
            dt = abs(hour - arc_utc)
            if dt < best_dt:
                best_dt = dt
                best_idx = row_idx

        if best_idx is not None:
            row = results[best_idx]
            metadata['phase_processing_results'] = {
                name: typ(row[col]) for name, (col, typ) in phase_fields.items()
            }
        else:
            metadata['phase_processing_results'] = None

    return arcs


def attach_vwc_track_results(
    arcs: List[Tuple[Dict, Dict]],
    station: str,
    year: int,
    doy: int,
    extension: str = '',
    az_tolerance: float = 5.0,
    time_tolerance: float = 0.25,
) -> List[Tuple[Dict, Dict]]:
    """Attach VWC track results from ``vwc -save_tracks T -vegetation_model 2``.

    Matches track file rows to arcs by sat, freq suffix, azimuth, and hour.
    Sets ``metadata['vwc_track_results']`` to a dict keyed by
    :data:`VWC_TRACK_COLUMNS`, or ``None``.
    """
    import glob
    import re
    from gnssrefl.phase_functions import get_temporal_suffix

    def _set_all_none():
        for metadata, _data in arcs:
            metadata['vwc_track_results'] = None
        return arcs

    refl_code = os.environ.get('REFL_CODE', '')
    if not refl_code:
        return _set_all_none()

    subdir = os.path.join(station, extension) if extension else station
    track_dir = os.path.join(refl_code, 'Files', subdir, 'individual_tracks')
    if not os.path.isdir(track_dir):
        return _set_all_none()

    # Parse track filenames: az{NNN}_sat{NN}_{station}_{year}{suffix}.txt
    filename_re = re.compile(r'az(\d{3})_sat(\d{2})_')
    sat_lookup: Dict[int, List[Tuple[int, str]]] = {}
    for fpath in glob.glob(os.path.join(track_dir, 'az*_sat*_*.txt')):
        m = filename_re.match(os.path.basename(fpath))
        if m:
            sat_lookup.setdefault(int(m.group(2)), []).append((int(m.group(1)), fpath))
    if not sat_lookup:
        return _set_all_none()

    vwc_fields = {name: (i, float) for i, name in enumerate(VWC_TRACK_COLUMNS)}
    for col in ('Year', 'DOY', 'MJD'):
        vwc_fields[col] = (VWC_TRACK_COLUMNS.index(col), int)

    for metadata, _data in arcs:
        metadata['vwc_track_results'] = None
        candidates = sat_lookup.get(metadata['sat'])
        if not candidates:
            continue

        try:
            freq_suffix = get_temporal_suffix(metadata['freq'])
        except Exception:
            continue
        candidates = [(az, fp) for az, fp in candidates if freq_suffix in os.path.basename(fp)]
        if not candidates:
            continue

        # Closest azimuth within tolerance
        az_dists = np.array([circular_distance_deg(metadata['az_min_ele'], az) for az, _ in candidates])
        best_i = int(np.argmin(az_dists))
        if az_dists[best_i] > az_tolerance:
            continue

        track_data = _load_result_file(candidates[best_i][1])
        if track_data is None:
            continue
        day_mask = (track_data[:, 0].astype(int) == year) & (track_data[:, 1].astype(int) == doy)
        day_rows = track_data[day_mask]
        if len(day_rows) == 0:
            continue
        raw_diffs = np.abs(day_rows[:, 2] - metadata['arc_timestamp'])
        hour_diffs = np.minimum(raw_diffs, 24 - raw_diffs)
        best_row_i = int(np.argmin(hour_diffs))
        if hour_diffs[best_row_i] > time_tolerance:
            continue
        row = day_rows[best_row_i]
        metadata['vwc_track_results'] = {
            name: typ(row[col]) for name, (col, typ) in vwc_fields.items()
        }

    return arcs


def _get_arc_filename(sdir, sat, freq, az_min_ele, arc_timestamp):
    """Build deterministic arc filename from metadata.

    Parameters
    ----------
    sdir : str
        Output directory (must end with '/')
    sat : int
        Satellite number
    freq : int
        Frequency code
    az_min_ele : float
        Azimuth at minimum elevation (degrees)
    arc_timestamp : float
        Arc midpoint in hours UTC

    Returns
    -------
    str
        Full file path, or '' if freq mapping fails
    """
    csat = f'{sat:03d}'

    if freq < 100:
        constell = 'G'
        fout = freq
    elif freq < 200:
        constell = 'R'
        fout = freq - 100
    elif freq < 300:
        constell = 'E'
        fout = freq - 200
    else:
        constell = 'C'
        fout = freq - 300

    cf = '_L2_' if freq == 20 else f'_L{fout}_'
    cf += constell

    cazim = f'az{round(az_min_ele):03d}'
    hh = int(arc_timestamp) % 24
    mm = int((arc_timestamp % 1) * 60)
    ctime = f'{hh:02d}{mm:02d}z'

    return f'{sdir}sat{csat}{cf}{cazim}_{ctime}.txt'


def _write_arc_file(fname, data, meta, station, year, doy, savearcs_format='txt'):
    """Write a single arc to disk.

    Parameters
    ----------
    fname : str
        Output file path (for txt; .pickle replaces .txt extension)
    data : dict
        Arc data with keys 'ele', 'snr', 'seconds'
    meta : dict
        Arc metadata
    station : str
        Station name
    year : int
        Year
    doy : int
        Day of year
    savearcs_format : str
        'txt' or 'pickle'
    """
    import gnssrefl.gps as g

    eangles = data['ele']
    dsnr = data['snr']
    sec = data['seconds']

    if savearcs_format == 'txt':
        headerline = ' elev-angle (deg), dSNR (volts/volts), sec of day'
        xyz = np.vstack((eangles, dsnr, sec)).T
        fmt = '%12.7f  %12.7f  %10.0f'
        np.savetxt(fname, xyz, fmt=fmt, delimiter=' ', newline='\n',
                   comments='%', header=headerline)
    else:
        import pickle
        d = g.doy2ymd(year, doy)
        month, day = d.month, d.day
        MJD = g.getMJD(year, month, day, meta['arc_timestamp'])
        docstring = ('arrays are eangles (degrees), dsnrData is SNR '
                     'with/DC removed, and sec (seconds of the day),\n')
        pname = fname[:-4] + '.pickle'
        with open(pname, 'wb') as f:
            pickle.dump([station, eangles, dsnr, sec, meta['sat'],
                         meta['freq'], meta['az_min_ele'], year, doy,
                         meta['arc_timestamp'], MJD, docstring], f)


def setup_arcs_directory(station, year, doy, extension='', nooverwrite=False):
    """Create arcs directory, optionally clearing old contents.

    Parameters
    ----------
    station : str
        Station name
    year : int
        Year
    doy : int
        Day of year
    extension : str
        Strategy extension
    nooverwrite : bool
        If True, do not clear existing arcs. Default False.

    Returns
    -------
    str
        Path to arcs directory (with trailing '/')
    """
    fm = FileManagement(station, "arcs_directory", year=year, doy=doy,
                        extension=extension)
    sdir = str(fm.get_directory_path(ensure_directory=True)) + '/'
    if not nooverwrite:
        for f in glob.glob(sdir + '*.txt') + glob.glob(sdir + '*.pickle'):
            os.remove(f)
        failqc = sdir + 'failQC/'
        if os.path.isdir(failqc):
            for f in glob.glob(failqc + '*.txt') + glob.glob(failqc + '*.pickle'):
                os.remove(f)
    os.makedirs(sdir + 'failQC/', exist_ok=True)
    return sdir


def save_arc(meta, data, sdir, station, year, doy, savearcs_format='txt'):
    """Save a single arc file to sdir.

    Parameters
    ----------
    meta : dict
        Arc metadata (must contain sat, freq, az_min_ele, arc_timestamp,
        num_pts, delT)
    data : dict
        Arc data (must contain ele, snr, seconds)
    sdir : str
        Output directory (with trailing '/')
    station : str
        Station name
    year : int
        Year
    doy : int
        Day of year
    savearcs_format : str
        'txt' or 'pickle'
    """
    if meta['num_pts'] <= 0 or meta['delT'] == 0:
        return
    fname = _get_arc_filename(sdir, meta['sat'], meta['freq'],
                              meta['az_min_ele'], meta['arc_timestamp'])
    if not fname:
        return
    _write_arc_file(fname, data, meta, station, year, doy, savearcs_format)


def move_arc_to_failqc(meta, station, year, doy, extension=''):
    """Move a saved arc file from arcs/ to arcs/failQC/.

    Resolves the arcs directory path internally from station metadata.

    Parameters
    ----------
    meta : dict
        Arc metadata (must contain sat, freq, az_min_ele, arc_timestamp)
    station : str
        Station name
    year : int
        Year
    doy : int
        Day of year
    extension : str
        Strategy extension
    """
    fm = FileManagement(station, "arcs_directory", year=year, doy=doy,
                        extension=extension)
    sdir = str(fm.get_directory_path()) + '/'
    fname = _get_arc_filename(sdir, meta['sat'], meta['freq'],
                              meta['az_min_ele'], meta['arc_timestamp'])
    if not fname or not os.path.isfile(fname):
        return
    dest = _get_arc_filename(sdir + 'failQC/', meta['sat'], meta['freq'],
                             meta['az_min_ele'], meta['arc_timestamp'])
    shutil.move(fname, dest)


def apply_refraction(snr_array, lsp, year, doy):
    """Apply refraction correction to SNR elevation angles.

    Parameters
    ----------
    snr_array : np.ndarray
        SNR data array with elevation angles in column 1.
    lsp : dict
        Station analysis parameters containing refraction model config.
    year : int
        Full year.
    doy : int
        Day of year.

    Returns
    -------
    np.ndarray
        Copy of *snr_array* with corrected elevations; rows where the
        correction is invalid are removed.
    """
    from gnssrefl.refraction import correct_elevations
    corrected, valid_mask = correct_elevations(snr_array[:, 1], lsp, year, doy)
    snr_array = snr_array.copy()
    snr_array[:, 1] = corrected
    return snr_array[valid_mask]


def extract_arcs_from_station(
    station: str,
    year: int,
    doy: int,
    freq: Optional[Union[int, List[int]]] = None,
    snr_type: int = 66,
    buffer_hours: float = 2,
    attach_results: bool = False,
    extension: str = '',
    lsp: Optional[Dict[str, Any]] = None,
    gzip: bool = False,
    **kwargs,
) -> List[Tuple[Dict[str, Any], Dict[str, np.ndarray]]]:
    """
    Extract satellite arcs for a station/year/day.

    Resolves the SNR file path (handling .gz/.xz decompression), loads it,
    and extracts arcs in one call.

    Parameters
    ----------
    station : str
        Station name (4 characters, e.g. 'mchl')
    year : int
        Full year (e.g. 2025)
    doy : int
        Day of year (1-366)
    freq : int, list of int, or None
        Frequency code(s). A single int (e.g. ``1``), a list (e.g.
        ``[1, 2, 5]``), or ``None`` (default) to auto-detect all
        frequencies that have data in the file.
    snr_type : int
        SNR file type (66, 77, 88, etc.). Default: 66
    buffer_hours : float
        Hours of data to include from adjacent days for midnight-crossing arcs.
        Default: 2
    attach_results : bool
        If True, look up the gnssir result file, phase result file, and VWC
        individual track files, attaching processing results to each arc's
        metadata via ``attach_gnssir_processing_results()``,
        ``attach_phase_processing_results()``, and
        ``attach_vwc_track_results()``. Arcs with no matching result get
        ``gnssir_processing_results = None``,
        ``phase_processing_results = None``, and/or
        ``vwc_track_results = None``. Default: False
    extension : str
        Subdirectory under ``$REFL_CODE/<year>/results/<station>/`` where
        result files are stored (e.g. ``'gnssir'``). Only used when
        *attach_results* is True. Default: ``''`` (no subdirectory)
    lsp : dict, optional
        Station analysis parameters (from json). When provided and
        ``lsp['refraction']`` is True, refraction correction is applied
        to elevation angles before arc extraction. Default: None
    gzip : bool
        If True, gzip-compress the SNR file after reading. Default: False
    **kwargs
        Additional keyword arguments passed to ``extract_arcs()``
        (e1, e2, azlist, sat_list, etc.)

    Returns
    -------
    list of (metadata, data) tuples
        See ``extract_arcs()`` for format details.

    Raises
    ------
    FileNotFoundError
        If the SNR file does not exist and cannot be decompressed.
    """
    import gnssrefl.gps as g
    obsfile, _, snr_exists = g.define_and_xz_snr(station, year, doy, snr_type)
    if not snr_exists:
        raise FileNotFoundError(
            f"SNR file not found for station={station}, year={year}, "
            f"doy={doy}, snr_type={snr_type}: {obsfile}"
        )

    # Load SNR data (same as extract_arcs_from_file)
    screenstats = kwargs.get('screenstats', False)
    allGood, snr_array, _, _ = read_snr(
        obsfile, buffer_hours=buffer_hours, screenstats=screenstats,
    )
    if not allGood:
        raise RuntimeError(f"read_snr failed for: {obsfile}")

    if gzip and os.path.isfile(obsfile):
        subprocess.call(['gzip', '-f', obsfile])

    # Apply refraction correction
    if lsp is not None and lsp.get('refraction', False):
        snr_array = apply_refraction(snr_array, lsp, year, doy)

    arcs = extract_arcs(snr_array, freq=freq, year=year, doy=doy, **kwargs)

    # Save individual arc files to disk
    if lsp is not None and lsp.get('savearcs', False):
        nooverwrite = lsp.get('nooverwrite', False)
        savearcs_format = lsp.get('savearcs_format', 'txt')
        sdir = setup_arcs_directory(station, year, doy, extension, nooverwrite)
        print('Writing individual arcs to', sdir)
        for meta, data in arcs:
            save_arc(meta, data, sdir, station, year, doy, savearcs_format)

    if attach_results and extension:
        result_dir = FileManagement(station, 'gnssir_result', year, doy, extension=extension).get_file_path(ensure_directory=False).parent
        phase_dir = FileManagement(station, 'phase_file', year, doy, extension=extension).get_file_path(ensure_directory=False).parent
        if not result_dir.is_dir() and not phase_dir.is_dir():
            raise FileNotFoundError(
                f"Extension directory '{extension}' not found under "
                f"{result_dir.parent} or {phase_dir.parent}"
            )

    if attach_results:
        result_path = FileManagement(station, 'gnssir_result', year, doy, extension=extension).get_file_path(ensure_directory=False)
        if result_path.is_file():
            attach_gnssir_processing_results(arcs, str(result_path))
        else:
            for metadata, _data in arcs:
                metadata['gnssir_processing_results'] = None

        phase_path = FileManagement(station, 'phase_file', year, doy, extension=extension).get_file_path(ensure_directory=False)
        if phase_path.is_file():
            attach_phase_processing_results(arcs, str(phase_path))
        else:
            for metadata, _data in arcs:
                metadata['phase_processing_results'] = None

        attach_vwc_track_results(arcs, station, year, doy, extension)

    return arcs


def extract_arcs_from_file(
    obsfile: str,
    freq: Optional[Union[int, List[int]]] = None,
    buffer_hours: float = 2,
    gzip: bool = False,
    **kwargs,
) -> List[Tuple[Dict[str, Any], Dict[str, np.ndarray]]]:
    """
    Extract satellite arcs from an SNR file.

    Loads the file with ``read_snr()`` and extracts arcs in one call.

    Parameters
    ----------
    obsfile : str
        Path to the SNR observation file.
    freq : int, list of int, or None
        Frequency code(s). A single int (e.g. ``1``), a list (e.g.
        ``[1, 2, 5]``), or ``None`` (default) to auto-detect all
        frequencies that have data in the file.
    buffer_hours : float
        Hours of data to include from adjacent days for midnight-crossing arcs.
        Default: 2
    gzip : bool
        If True, gzip-compress the SNR file after reading. Default: False
    **kwargs
        Additional keyword arguments passed to ``extract_arcs()``
        (e1, e2, azlist, sat_list, etc.)

    Returns
    -------
    list of (metadata, data) tuples
        See ``extract_arcs()`` for format details.

    Raises
    ------
    FileNotFoundError
        If *obsfile* does not exist.
    RuntimeError
        If ``read_snr()`` fails to load the file.
    """
    if not os.path.isfile(obsfile):
        raise FileNotFoundError(f"SNR file not found: {obsfile}")

    screenstats = kwargs.get('screenstats', False)
    allGood, snr_array, _, _ = read_snr(
        obsfile, buffer_hours=buffer_hours, screenstats=screenstats,
    )
    if not allGood:
        raise RuntimeError(f"read_snr failed for: {obsfile}")

    if gzip and os.path.isfile(obsfile):
        subprocess.call(['gzip', '-f', obsfile])

    return extract_arcs(snr_array, freq=freq, **kwargs)


def extract_arcs(
    snr_array: np.ndarray,
    freq: Optional[Union[int, List[int]]] = None,
    e1: float = 5.0,
    e2: float = 25.0,
    ellist: Optional[List[float]] = None,
    azlist: Optional[List[float]] = None,
    sat_list: Optional[List[int]] = None,
    min_pts: int = MIN_ARC_POINTS,
    polyV: int = 4,
    pele: Optional[List[float]] = None,
    dbhz: bool = False,
    screenstats: bool = False,
    detrend: bool = True,
    split_arcs: bool = True,
    filter_to_day: bool = True,
    year: Optional[int] = None,
    doy: Optional[int] = None,
    dec: int = 1,
) -> List[Tuple[Dict[str, Any], Dict[str, np.ndarray]]]:
    """
    Extract satellite arcs from SNR data array.

    Parameters
    ----------
    snr_array : np.ndarray
        2D array with columns: [sat, ele, azi, seconds, edot, snr1, snr2, ...]
        This is the output of loading an SNR file with np.loadtxt()
    freq : int, list of int, or None
        Frequency code(s). A single int (e.g. ``1``), a list (e.g.
        ``[1, 2, 5]``), or ``None`` (default) to auto-detect all
        frequencies that have data in the file.
    e1 : float
        Minimum elevation angle (degrees) for analysis. Default: 5.0
    e2 : float
        Maximum elevation angle (degrees) for analysis. Default: 25.0
    ellist : list of floats, optional
        Multiple elevation angle ranges as pairs, e.g., [5, 10, 7, 12] means
        ranges (5-10 deg) and (7-12 deg). When provided and non-empty, this
        overrides e1/e2. Each pair is processed independently. Default: None
    azlist : list of floats, optional
        Azimuth regions as pairs, e.g., [0, 90, 180, 270] means 0-90 and 180-270.
        Default: [0, 360] (all azimuths)
    sat_list : list of int, optional
        Specific satellites to process. Default: all satellites in data
    min_pts : int
        Minimum points required per arc. Default: 20
    polyV : int
        Polynomial order for DC removal. Default: 4
    dbhz : bool
        If True, keep SNR in dB-Hz; if False, convert to linear units. Default: False
    screenstats : bool
        If True, print debug information. Default: False
    detrend : bool
        If True (default), remove DC component via polynomial fit.
        If False, return SNR converted to linear units only (no detrending).
    split_arcs : bool
        If True (default), split data into separate arcs by time gaps and elevation
        direction changes. If False, return all data for each satellite as a single
        arc without splitting or validation. Useful for phase processing.
    filter_to_day : bool
        If True (default), only return arcs whose midpoint (arc_timestamp) falls
        within the principal day (0-24 hours). This prevents double-counting arcs
        when processing consecutive days with buffer_hours. Arc data may still
        extend beyond day boundaries. If False, return all arcs regardless of
        their midpoint time.
    year : int, optional
        Full year (e.g. 2025). Used with *doy* to determine which GPS satellites
        transmit L2C and L5 signals. If not provided, all GPS satellites on
        column 8 are assumed L2C.
    doy : int, optional
        Day of year (1-366). See *year*.
    dec : int
        Decimation factor. Keep only observations whose seconds-of-day is
        divisible by *dec*. Default: 1 (no decimation).

    Returns
    -------
    list of (metadata, data) tuples
        Each arc is represented as:
        - metadata: dict with keys: sat, freq, arc_num, arc_type, ele_start, ele_end,
          az_min_ele, az_avg, time_start, time_end, time_avg, num_pts, delT, edot_factor, cf
        - data: dict with keys: ele, azi, snr, seconds, edot (all np.ndarray)
    """
    if azlist is None:
        azlist = [0, 360]
    if pele is None:
        pele = [e1, e2]

    # Apply decimation before any other filtering
    if dec != 1:
        keep = np.remainder(snr_array[:, 3], dec) == 0
        snr_array = snr_array[keep]

    # Pre-filter SNR data to pele range (replicates gnssir's elevation mask)
    pele_mask = (snr_array[:, 1] >= pele[0]) & (snr_array[:, 1] <= pele[1])
    snr_array = snr_array[pele_mask]

    ncols = snr_array.shape[1]

    # Build column list and per-column freq/constellation info
    freq_list = [freq] if isinstance(freq, int) else (list(freq) if freq is not None else None)
    if freq_list is not None:
        column_list = [_get_snr_column(f) for f in freq_list]
        # Explicit freqs: record freq and target constellation per column
        explicit = {_get_snr_column(f): (f, (f // 100) * 100) for f in freq_list}
    else:
        column_list = [c for c in SNR_COLUMNS if c <= ncols]
        explicit = None

    # L2C/L5 satellite sets for GPS freq assignment
    l2c_sats = l5_sats = None
    if year is not None and doy is not None:
        from gnssrefl.gps import l2c_l5_list
        l2c_arr, l5_arr = l2c_l5_list(year, doy)
        l2c_sats = set(int(s) for s in l2c_arr)
        l5_sats = set(int(s) for s in l5_arr)

    all_arcs = []

    for column in column_list:
        # Convert to 0-based index
        icol = column - 1

        # Check if column exists
        if column > ncols:
            if screenstats:
                print(f"Warning: SNR file has {ncols} columns, need column {column}")
            continue

        # Extract columns
        sats = snr_array[:, 0].astype(int)
        ele_all = snr_array[:, 1]
        azi_all = snr_array[:, 2]
        seconds_all = snr_array[:, 3]
        edot_all = snr_array[:, 4] if ncols > 4 else np.zeros_like(seconds_all)
        snr_all = snr_array[:, icol]

        # Get unique satellites
        if sat_list is None:
            unique_sats = np.unique(sats)
        else:
            unique_sats = np.array(sat_list)

        # Parse elevation list
        elev_pairs = _parse_elevation_list(e1, e2, ellist)
        if screenstats and len(elev_pairs) > 1:
            print(f'Using {len(elev_pairs)} elevation angle ranges: {elev_pairs}')

        for sat in unique_sats:
            # Determine freq code for this column + satellite
            if explicit is not None:
                exp_freq, exp_const = explicit[column]
                if (sat // 100) * 100 != exp_const:
                    continue  # wrong constellation for this freq
                sat_freq = exp_freq
            else:
                sat_freq = _freq_for_column_and_sat(column, sat, l2c_sats, l5_sats)
                if sat_freq is None:
                    continue

            sat_mask = sats == sat

            if np.sum(sat_mask) < min_pts:
                continue

            sat_ele = ele_all[sat_mask]
            sat_azi = azi_all[sat_mask]
            sat_seconds = seconds_all[sat_mask]
            sat_edot = edot_all[sat_mask]
            sat_snr = snr_all[sat_mask]

            for pair_e1, pair_e2 in elev_pairs:
                if split_arcs:
                    # Detect arc boundaries on full satellite data
                    # (arc detection validates against e1/e2 but uses all elevation data)
                    arc_boundaries = _detect_arc_boundaries(
                        sat_ele, sat_azi, sat_seconds,
                        pair_e1, pair_e2, sat,
                        min_pts=min_pts,
                    )
                else:
                    # No splitting - treat all satellite data as one arc
                    # This returns ALL data without validation, used in phase processing
                    arc_boundaries = [(0, len(sat_ele), sat, 1)]

                for sind, eind, sat_num, arc_num in arc_boundaries:
                    # Extract arc data (full elevation range)
                    arc_ele = sat_ele[sind:eind].copy()
                    arc_azi = sat_azi[sind:eind].copy()
                    arc_seconds = sat_seconds[sind:eind].copy()
                    arc_edot = sat_edot[sind:eind].copy()
                    arc_snr = sat_snr[sind:eind].copy()

                    # Remove zero/invalid SNR values from the arc
                    # Use > 1 to filter zeros in both dB-Hz and linear
                    nonzero_mask = arc_snr > 1
                    if np.sum(nonzero_mask) < min_pts:
                        if screenstats:
                            print(f"No useful data on frequency {sat_freq} / sat {sat}: all zeros")
                        continue

                    arc_ele = arc_ele[nonzero_mask]
                    arc_azi = arc_azi[nonzero_mask]
                    arc_seconds = arc_seconds[nonzero_mask]
                    arc_edot = arc_edot[nonzero_mask]
                    arc_snr = arc_snr[nonzero_mask]

                    # Check minimum points for polynomial fit
                    reqN = 20
                    if len(arc_ele) <= reqN:
                        continue

                    # Process SNR: either detrend or just convert to linear units
                    if detrend:
                        arc_snr_processed = _remove_dc_component(arc_ele, arc_snr, polyV, dbhz, pele)
                    else:
                        # Just convert to linear units, no detrending
                        if dbhz:
                            arc_snr_processed = arc_snr.copy()
                        else:
                            arc_snr_processed = np.power(10, (arc_snr / 20))

                    # Apply e1/e2 filter (after DC removal) - skip if not splitting arcs
                    if split_arcs:
                        e_mask = (arc_ele > pair_e1) & (arc_ele <= pair_e2)
                        Nvv = np.sum(e_mask)

                        if Nvv < 15:
                            continue

                        # Check azimuth compliance using circular mean of filtered arc
                        filtered_azi = arc_azi[e_mask]
                        arc_azim = circular_mean_deg(filtered_azi)

                        if not _check_azimuth_compliance(arc_azim, azlist):
                            if screenstats:
                                print(f"Azimuth {arc_azim:.2f} not in requested region")
                            continue

                        # Apply e1/e2 filter to all arrays
                        final_ele = arc_ele[e_mask]
                        final_azi = arc_azi[e_mask]
                        final_seconds = arc_seconds[e_mask]
                        final_edot = arc_edot[e_mask]
                        final_snr = arc_snr_processed[e_mask]
                    else:
                        # No filtering - return all data for this satellite
                        # Phase will do its own filtering by azimuth and elevation
                        final_ele = arc_ele
                        final_azi = arc_azi
                        final_seconds = arc_seconds
                        final_edot = arc_edot
                        final_snr = arc_snr_processed

                    # Compute metadata using filtered data
                    metadata = _compute_arc_metadata(
                        final_ele, final_azi, final_seconds,
                        sat, sat_freq, arc_num,
                    )

                    # Create data dictionary
                    data = {
                        'ele': final_ele,
                        'azi': final_azi,
                        'snr': final_snr,
                        'seconds': final_seconds,
                        'edot': final_edot,
                    }

                    all_arcs.append((metadata, data))

    # Optionally remove any arcs where the mean time is not 0 <= h < 24 (when using buffer_hours > 0)
    if filter_to_day:
        all_arcs = [
            (meta, data) for meta, data in all_arcs
            if 0 <= meta['arc_timestamp'] < 24
        ]

    return all_arcs

def _parse_elevation_list(
    e1: float,
    e2: float,
    ellist: Optional[List[float]],
) -> List[Tuple[float, float]]:
    """
    Parse elevation angle parameters into list of (e1, e2) pairs.

    Parameters
    ----------
    e1 : float
        Default minimum elevation angle
    e2 : float
        Default maximum elevation angle
    ellist : list of float or None
        Elevation pairs as flat list [e1a, e2a, e1b, e2b, ...]

    Returns
    -------
    list of (float, float)
        List of (min_elev, max_elev) tuples to process

    Raises
    ------
    ValueError
        If ellist has odd length (incomplete pair)
    """
    if ellist is None or len(ellist) == 0:
        return [(e1, e2)]

    if len(ellist) % 2 != 0:
        raise ValueError(
            f"ellist must contain pairs of elevation angles, "
            f"got {len(ellist)} values (odd number)"
        )

    return [(ellist[i], ellist[i + 1]) for i in range(0, len(ellist), 2)]


def _get_snr_column(freq: int) -> int:
    """
    Map frequency code to SNR column index

    SNR file format:
        1: sat, 2: ele, 3: azi, 4: seconds, 5: edot
        6: S6, 7: S1 (L1), 8: S2 (L2/L2C), 9: S5 (L5)
        10: S7 (E5b), 11: S8 (E5a+b)

    Docs: https://gnssrefl.readthedocs.io/en/latest/pages/file_structure.html#the-snr-data-format

    Parameters
    ----------
    freq : int
        Frequency code (1, 2, 5, 20, 101, 102, 201, 205, 206, 207, 208, etc.)

    Returns
    -------
    int
        Column number (1-based) for the SNR data

    Raises
    ------
    ValueError
        If frequency code is not recognized
    """
    # L1 frequencies -> S1 column (7)
    if freq in [1, 101, 201, 301]:
        return 7
    # L2/L2C frequencies -> S2 column (8)
    elif freq in [2, 20, 102, 302]:
        return 8
    # L5 frequencies -> S5 column (9)
    elif freq in [5, 205, 305]:
        return 9
    # E6/B3 frequencies -> S6 column (6)
    elif freq in [206, 306]:
        return 6
    # E5b/B2 frequencies -> S7 column (10)
    elif freq in [207, 307]:
        return 10
    # E5a+b frequencies -> S8 column (11)
    elif freq in [208, 308]:
        return 11
    else:
        raise ValueError(f"Unrecognized frequency code: {freq}")


def _check_azimuth_compliance(init_azim: float, azlist: List[float]) -> bool:
    """
    Check if azimuth is within allowed regions.

    Parameters
    ----------
    init_azim : float
        Azimuth angle (degrees) at the lowest elevation point of the arc
    azlist : list of float
        Azimuth regions as pairs [az1_start, az1_end, az2_start, az2_end, ...]
        e.g., [0, 90, 180, 270] means 0-90 and 180-270 degrees

    Returns
    -------
    bool
        True if azimuth is within any of the allowed regions
    """
    N = int(len(azlist) / 2)
    for a in range(N):
        azim1 = azlist[2 * a]
        azim2 = azlist[2 * a + 1]
        if (init_azim >= azim1) and (init_azim <= azim2):
            return True
    return False


def _detect_arc_boundaries(
    ele: np.ndarray,
    azm: np.ndarray,
    seconds: np.ndarray,
    e1: float,
    e2: float,
    sat: int,
    min_pts: int = MIN_ARC_POINTS,
    gap_time: float = GAP_TIME_LIMIT,
) -> List[Tuple[int, int, int, int]]:
    """
    Detect arc boundaries based on time gaps and elevation direction changes.

    This is refactored from new_rise_set_again() in gnssir_v2.py.

    Parameters
    ----------
    ele : np.ndarray
        Elevation angles (degrees)
    azm : np.ndarray
        Azimuth angles (degrees)
    seconds : np.ndarray
        Seconds of day
    e1 : float
        Minimum elevation angle (degrees) for analysis
    e2 : float
        Maximum elevation angle (degrees) for analysis
    sat : int
        Satellite number
    min_pts : int
        Minimum points required per arc
    gap_time : float
        Time gap threshold for splitting arcs (seconds)

    Returns
    -------
    list of tuples
        Each tuple is (start_idx, end_idx, sat, arc_num) for valid arcs
    """
    if len(ele) < min_pts:
        return []

    # Find breakpoints
    ddate = np.ediff1d(seconds)
    delv = np.ediff1d(ele)

    # Initialize with the last index
    bkpt = np.array([len(ddate)])

    # Add time gap breakpoints
    bkpt = np.append(bkpt, np.where(ddate > gap_time)[0])

    # Add elevation direction change breakpoints
    bkpt = np.append(bkpt, np.where(np.diff(np.sign(delv)))[0])

    # Remove duplicates and sort
    bkpt = np.unique(bkpt)
    bkpt = np.sort(bkpt)

    valid_arcs = []
    iarc = 0

    for ii in range(len(bkpt)):
        if ii == 0:
            sind = 0
        else:
            sind = bkpt[ii - 1] + 1
        eind = bkpt[ii] + 1

        # Extract arc data
        arc_ele = ele[sind:eind]

        if len(arc_ele) == 0:
            continue

        # Check minimum point count
        if (eind - sind) < min_pts:
            continue

        iarc += 1
        valid_arcs.append((sind, eind, sat, iarc))

    return valid_arcs


def _remove_dc_component(
    ele: np.ndarray,
    snr: np.ndarray,
    polyV: int,
    dbhz: bool,
    pele: Optional[List[float]] = None,
) -> np.ndarray:
    """
    Remove direct signal component via polynomial fit.

    Parameters
    ----------
    ele : np.ndarray
        Elevation angles (degrees)
    snr : np.ndarray
        Raw SNR values
    polyV : int
        Polynomial order for DC removal
    dbhz : bool
        If True, keep SNR in dB-Hz; if False, convert to linear units first
    pele : list of float, optional
        Elevation angle range [min, max] for polynomial fit.
        If provided, the polynomial is fit on data within this range
        but evaluated (and removed) over the full arc.

    Returns
    -------
    np.ndarray
        Detrended SNR data
    """
    data = snr.copy()

    # Convert to linear units if needed
    if not dbhz:
        data = np.power(10, (data / 20))

    # Fit polynomial on pele range if provided, otherwise full arc
    if pele is not None:
        pele_mask = (ele >= pele[0]) & (ele <= pele[1])
        if np.sum(pele_mask) > polyV + 1:
            model = np.polyfit(ele[pele_mask], data[pele_mask], polyV)
        else:
            model = np.polyfit(ele, data, polyV)
    else:
        model = np.polyfit(ele, data, polyV)
    fit = np.polyval(model, ele)
    data = data - fit

    return data


def _compute_arc_metadata(
    ele: np.ndarray,
    azi: np.ndarray,
    seconds: np.ndarray,
    sat: int,
    freq: int,
    arc_num: int,
) -> Dict[str, Any]:
    """
    Compute metadata for an arc including edot factor.

    Parameters
    ----------
    ele : np.ndarray
        Elevation angles (degrees)
    azi : np.ndarray
        Azimuth angles (degrees)
    seconds : np.ndarray
        Seconds of day
    sat : int
        Satellite number
    freq : int
        Frequency code
    arc_num : int
        Arc index number

    Returns
    -------
    dict
        Metadata dictionary
    """
    # Determine arc type from elevation trend
    if len(ele) >= 2:
        arc_type = 'rising' if ele[-1] > ele[0] else 'setting'
    else:
        arc_type = 'unknown'

    # Get index of minimum elevation angle
    ie = np.argmin(ele)
    init_azim = azi[ie]

    # Compute edot factor (from window_new lines 975-987)
    # edot in radians/sec
    model = np.polyfit(seconds, ele * np.pi / 180, 1)
    avgEdot_fit = model[0]

    # Average tan(elev)
    cunit = np.mean(np.tan(np.pi * ele / 180))

    # edot factor: tan(e)/edot in units of 1/(radians/hour)
    edot_factor = cunit / (avgEdot_fit * 3600) if avgEdot_fit != 0 else 0.0

    # Scale factor (wavelength/2)
    import gnssrefl.gps as g
    cf = g.arc_scaleF(freq, sat)

    return {
        'sat': sat,
        'freq': freq,
        'arc_num': arc_num,
        'arc_type': arc_type,
        'ele_start': float(np.min(ele)),
        'ele_end': float(np.max(ele)),
        'az_min_ele': float(init_azim),
        'az_avg': float(circular_mean_deg(azi)),
        'time_start': float(np.min(seconds)),
        'time_end': float(np.max(seconds)),
        'arc_timestamp': float(np.mean(seconds) / 3600),  # hours UTC
        'num_pts': len(ele),
        'delT': float((np.max(seconds) - np.min(seconds)) / 60),  # minutes
        'edot_factor': float(edot_factor),
        'cf': float(cf),
    }
