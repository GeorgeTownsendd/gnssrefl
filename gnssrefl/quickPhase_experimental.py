import argparse
import numpy as np
import sys
import subprocess
from scipy import optimize
from functools import partial
from pathlib import Path
import os
from importlib.metadata import version

import gnssrefl.gps as g
import gnssrefl.phase_functions as qp
import gnssrefl.read_snr_files as read_snr
from gnssrefl.utils import str2bool, FileManagement, FileTypes


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("station", help="station", type=str)
    parser.add_argument("year", help="year", type=int)
    parser.add_argument("doy", help="doy", type=int)
    parser.add_argument("-snr", default=None, help="snr file ending", type=int)
    parser.add_argument("-fr", default=None, help="frequency (e.g. 1, 20, or 'all'). If not set, reads from json.", type=str)
    parser.add_argument("-doy_end", "-doy_end", default=None, type=int, help="doy end")
    parser.add_argument("-year_end", "-year_end", default=None, type=int, help="year end")
    parser.add_argument("-extension", default='', help="analysis extension for json file", type=str)
    parser.add_argument("-e1", default=None, type=float),
    parser.add_argument("-e2", default=None, type=float),
    parser.add_argument("-plt", default=None, type=str, help="plots come to the screen - which you do not want!")
    parser.add_argument("-screenstats", default=None, type=str, help="stats come to the screen")
    parser.add_argument("-gzip", default=None, type=str, help="gzip SNR files after use, default is True" )

    args = parser.parse_args().__dict__

    # convert all expected boolean inputs from strings to booleans
    boolean_args = ['plt', 'screenstats', 'gzip']
    args = str2bool(args, boolean_args)

    # only return a dictionary of arguments that were added from the user - all other defaults will be set in code below
    return {key: value for key, value in args.items() if value is not None}


def quickphase(station: str, year: int, doy: int, year_end: int = None, doy_end: int = None, snr: int = 66,
        fr: str = None, e1: float = 5, e2: float = 30, plt: bool = False, screenstats: bool = False, gzip: bool = True, extension: str = ''):
    """
    quickphase computes phase, which are subquently used in vwc. The command line call is phase_experimental
    (experimental version with alternative phase estimation methods).
    
    Examples
    --------
    phase_experimental p038 2021 4
        analyzes data for year 2021 and day of year 4

    phase_experimental p038 2021 1 -doy_end 365 
        analyzes data for the whole year

    Parameters
    ----------
    station: str
        4 character ID of the station.
    year: int
        full Year to evaluate.
    doy: int
        day of year to evaluate.
    year_end: int, optional
        year to end analysis. Using this option will create a range from year-year_end.
        Default is None.
    doy_end: int, optional
        Day of year to end analysis. Using this option will create a range of doy-doy_end.
        If also using year_end, then this will be the day to end analysis in the year_end requested.
        Default is None.
    snr : int, optional
        SNR format. This tells the code what elevation angles are in the SNR file
        value options:

            66 (default) : data with elevation angles less than 30 degrees

            99 : data with elevation angles between 5 and 30 degrees

            88 : data with all elevation angles 

            50 : data with elevation angles less than 10 degrees

    fr : str, optional
        GNSS frequency. Currently only supports L2C. Default is 20 (l2c)
    e1 : float, optional
        Elevation angle lower limit in degrees for the LSP. default is 5
    e2: float, optional
        Elevation angle upper limit in degrees for the LSP. default is 30
    plt: bool, optional
        Whether to plot results. Default is False
    screenstats: bool, optional
        Whether to print stats to the screen. Default is False
    gzip : bool, optional
        gzip the SNR file after use.  Default is True

    Returns
    -------
    Saves a file for each day in the doy-doy_end range: $REFL_CODE/<year>/phase/<station>/<doy>.txt

    columns in files:
        year doy hour phase nv azimuth sat ampl emin emax delT aprioriRH freq estRH pk2noise LSPAmp

    """

    print("WARNING: This is the EXPERIMENTAL phase estimation command.")
    print("         Code is untested and likely to produce terrible results.")
    print("         DO NOT use for production analysis or published results.")
    print("         For research and algorithm development only.")
    
    compute_lsp = True # used to be an optional input
    if len(station) != 4:
        print('Station name must be four characters long. Exiting.')
        sys.exit()

    # Use the helper function to get the list of frequencies.
    fr_list = qp.get_vwc_frequency(station, extension, fr)

    # Check that an apriori file exists for each requested frequency.
    for f in fr_list:
        ex = qp.apriori_file_exist(station, f)
        if not ex:
            print(f'No apriori RH file exists for frequency {f}. Please run vwc_input.')
            sys.exit()

    # in case you want to analyze multiple days of data
    if not doy_end:
        doy_end = doy

    exitS = g.check_inputs(station, year, doy, snr)

    if exitS:
        sys.exit()

    g.result_directories(station, year, '')

    # this should really be read from the json
    pele = [5, 30]  # polynomial fit limits  for direct signal

    # TODO maybe instead of specific doy we can do only year and pick up all those files just like the other parts?
    if year_end:
        year_range = np.arange(year, year_end+1)
        for y in np.arange(year, year_end+1):
            # If first year in multi-year range, then start on doy requested and finish year.
            if y == year_range[0]:
                date_range = np.arange(doy, 366)
            # If last year in multi-year range, then start on doy 1 finish on doy_end.
            elif y == year_range[-1]:
                date_range = np.arange(1, doy_end+1)
            # If year within multi-year range then do whole year start to finish.
            else:
                date_range = np.arange(1, 366)

            for d in date_range:
                print('Analyzing year/day of year ' + str(y) + '/' + str(d))
                phase_tracks_experimental(station, y, d, snr, fr_list, e1, e2, pele, plt, screenstats, compute_lsp, gzip, extension)
    else:
        for d in np.arange(doy, doy_end + 1):
            phase_tracks_experimental(station, year, d, snr, fr_list, e1, e2, pele, plt, screenstats, compute_lsp, gzip, extension)


def main():
    args = parse_arguments()
    quickphase(**args)


def create_phase_header(station):
    """
    Generate header for phase output files with version information.
    
    Parameters
    ----------
    station : str
        4-character station identifier
        
    Returns
    -------
    str
        Multi-line header string for phase output files
    """
    versionNumber = 'v' + str(version('gnssrefl')) + '-EXPERIMENTAL'
    tem = ' station ' + station + ' https://github.com/kristinemlarson/gnssrefl ' + versionNumber + '\n'
    line2 = ' EXPERIMENTAL PHASE PROCESSING - Code is untested and likely to produce terrible results \n'
    line3 = ' Year DOY Hour   Phase   Nv  Azimuth  Sat  Ampl emin emax  DelT aprioriRH  freq estRH  pk2noise LSPAmp\n'
    line4 = ' (1)  (2)  (3)    (4)   (5)    (6)    (7)  (8)  (9)  (10)  (11)   (12)     (13)  (14)    (15)    (16)'
    all = tem + line2 + line3 + line4
    return all


def interferometric_model(x, a, b, rh_apriori, freq):
    """
    Sine wave model for GNSS-IR interferometric pattern fitting.
    
    Implements the theoretical relationship between elevation angle and SNR
    oscillations caused by multipath interference from ground reflections.
    
    Parameters
    ----------
    x : numpy.ndarray
        Sine of elevation angles (dimensionless)
    a : float
        Amplitude parameter to be estimated
    b : float
        Phase parameter to be estimated (radians)
    rh_apriori : float
        A priori reflector height (meters)
    freq : int
        GNSS frequency identifier (1=L1, 20=L2C)
    
    Returns
    -------
    numpy.ndarray
        Modeled SNR oscillations
    """
    if (freq == 20) or (freq == 2):
        wavelength = g.constants.wL2
    elif freq == 5:
        wavelength = g.constants.wL5
    else:
        wavelength = g.constants.wL1
    
    freq_least_squares = 2*np.pi*2*rh_apriori/wavelength
    return a * np.sin(freq_least_squares * x + b)


def phase_tracks_experimental(station, year, doy, snr_type, fr_list, e1, e2, pele, plot, screenstats, compute_lsp, gzip, extension=''):
    """
    Extract interferometric phase from GNSS SNR data for soil moisture estimation.
    
    Processes SNR observations along predefined satellite ground tracks to estimate
    phase delays caused by changes in surface dielectric properties. Uses least
    squares fitting of theoretical interferometric patterns.
    
    Parameters
    ----------
    station : str
        4-character station identifier
    year : int
        Year of observations
    doy : int
        Day of year
    snr_type : int
        SNR file format identifier
    fr_list : list of int
        GNSS frequencies to process
    e1, e2 : float
        Elevation angle limits (degrees)
    pele : list of float
        Polynomial removal elevation limits (degrees) 
    plot : bool
        Generate diagnostic plots
    screenstats : bool
        Print processing statistics
    compute_lsp : bool
        Compute Lomb-Scargle periodogram
    gzip : bool
        Compress SNR files after processing
    extension : str
        Output subdirectory name
    """
    
    min_amp = 3
    poly_v = 4
    min_num_pts = 20

    # get the SNR filename
    obsfile, obsfilecmp, snrexist = g.define_and_xz_snr(station, year, doy, snr_type)

    # noise region - hardwired for normal sites ~ 2-3 meters tall
    noise_region = [0.5, 8]

    l2c_list, l5_sat = g.l2c_l5_list(year, doy)

    if not snrexist:
        print('No SNR file on this day.')
        pass
    else:
        header = create_phase_header(station)
        output_path = FileManagement(station, FileTypes.phase_file, year, doy).get_file_path()
        if extension:
            output_path = output_path.parent / extension / output_path.name
            output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Saving experimental phase file to: {output_path}")
        with open(output_path, 'w') as my_file:
            np.savetxt(my_file, [], header=header, comments='%')
            # read the SNR file into memory
            sat, ele, azi, t, edot, s1, s2, s5, s6, s7, s8, snr_exists = read_snr.read_one_snr(obsfile, 1)

            for freq in fr_list:
                # read apriori reflector height results
                apriori_results = qp.read_apriori_rh(station, freq)

                print('Analyzing Frequency ', freq, ' Year ', year, ' Day of Year ', doy)

                rows, columns = np.shape(apriori_results)

                for i in range(0, rows):
                    compute_lsp = True
                    azim = apriori_results[i, 3]
                    sat_number = apriori_results[i, 2]
                    az1 = apriori_results[i, 5]
                    az2 = apriori_results[i, 6]
                    rh_apriori = apriori_results[i, 1]

                    x, y, nv, cf, utctime, avg_azim, avg_edot, edot2, del_t = g.window_data(s1, s2, s5, s6, s7, s8, sat, ele, azi,
                                                                                        t, edot, freq, az1, az2, e1, e2,
                                                                                        sat_number, poly_v, pele, screenstats)
                    if (freq == 20) and (sat_number not in l2c_list):
                        if screenstats: 
                            print('Asked for L2C but this is not L2C transmitting on this day: ', int(sat_number))
                        compute_lsp = False
                    elif (freq == 5) and (sat_number not in l5_sat):
                        if screenstats:
                            print('Asked for L5 but this is not L5 transmitting on this day: ', int(sat_number))
                        compute_lsp = False

                    if screenstats:
                        print(f'Track {i:2.0f} Sat {sat_number:3.0f} Azimuth {azim:5.1f} RH {rh_apriori:6.2f} {nv:5.0f}')

                    if compute_lsp and (nv > min_num_pts):
                        min_height = 0.5
                        max_height = 8
                        desired_p = 0.01

                        max_f, max_amp, emin_obs, emax_obs, rise_set, px, pz = g.strip_compute(x, y, cf, max_height,
                                                                                           desired_p, poly_v, min_height)

                        nij = pz[(px > noise_region[0]) & (px < noise_region[1])]
                        noise = 0
                        if len(nij) > 0:
                            noise = np.mean(nij)
                            obs_pk2noise = max_amp/noise

                            if screenstats:
                                print(f'LSP RH {max_f:7.3f} m {obs_pk2noise:6.1f} Amp {max_amp:6.1f} {min_amp:6.1f}')
                        else:
                            max_amp = 0

                        # Phase estimation via least squares fitting
                        if (nv > min_num_pts) and (max_amp > min_amp):
                            minmax = np.max(x) - np.min(x)
                            if (minmax > 22) and (del_t < 120):
                                # Transform to sine space for interferometric model
                                x_data = np.sin(np.deg2rad(x))
                                y_data = y
                                
                                # Fit theoretical interferometric model
                                test_function_apriori = partial(interferometric_model, rh_apriori=rh_apriori, freq=freq)
                                params, params_covariance = optimize.curve_fit(test_function_apriori, x_data, y_data, p0=[2, 2])

                                # Convert phase to degrees and apply constraints
                                phase = params[1]*180/np.pi
                                min_el = min(x)
                                max_el = max(x)
                                amp = np.absolute(params[0])
                                raw_amp = params[0]
                                
                                # Normalize phase to 0-360 degree range
                                phase = phase % 360

                                # Correct for negative amplitude
                                if raw_amp < 0:
                                    phase = (phase + 180) % 360

                                result = [[year, doy, utctime, phase, nv, avg_azim, sat_number, amp, min_el, max_el, del_t, rh_apriori, freq, max_f, obs_pk2noise, max_amp]]
                                np.savetxt(my_file, result, fmt="%4.0f %3.0f %6.2f %8.3f %5.0f %6.1f %3.0f %5.2f %5.2f %5.2f %6.2f %5.3f %2.0f %6.3f %6.2f %6.2f", comments="%")

        # gzip SNR file if requested
        if gzip:
            subprocess.call(['gzip', obsfile])


if __name__ == "__main__":
    main()
