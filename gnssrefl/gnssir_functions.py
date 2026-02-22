import datetime
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.interpolate
import scipy.signal
import subprocess
import sys
import time
import warnings

from importlib.metadata import version

import gnssrefl.gps as g
import gnssrefl.retrieve_rh as r
from gnssrefl.utils import FileManagement, FileTypes, rewrite_azel

def gnssir_guts_v2(station,year,doy, snr_type, extension,lsp, debug):
    """

    Computes lomb scargle periodograms for a given station, year, day of year etc.

    Arcs are determined differently than in the first version of the code, which
    was quadrant based. This identifies arcs and applies azimuth constraints after the fact.

    2023-aug-02 trying to fix the issue with azimuth print out being different than
    azimuth at lowest elevation angle

    if screenstats is True, it prints to a log file now, directory $REFL_CODE/logs/ssss


    Parameters
    ----------
    station : str
        4 character station name
    year : int
        full year
    doy : int
        day of year
    snr_type : int
        snr file type
    extension : str
        optional subdirectory to save results

    lsp : dictionary
        e1 : float
            min elev angle, deg
        e2 : float
            max elev angle, deg
        freqs: list of int
            frequencies to use 
        minH : float
            min reflector height, m
        maxH : float
            max reflector height, m 
        NReg : list of floats
            noise region for RH peak2noise , meters
        azval2 : list of floats
            new pairs of azimuth regions, i.e. [0 180 270 360]
        delTmax : float
            max allowed arc length in minutes
        pele: list of floats 
            min and max elev angle in DC removal
        PkNoise : float
            peak to noise value for QC
        ediff : float
            elev angle difference for arc length, QC
        reqAmp : list of floats
            list of required periodogram amplitude for QC for each frequency
        ellist: list of floats
            added 23jun16, allow multiple elevation angle regions
        apriori_rh : float
            a priori reflector height, used in NITE, meters
        savearcs : bool
            if true, elevation angle and detrended SNR data are saved for each arc
            default is False
        savearcs_format : str
            if arcs are to be saved, will they be txt or pickle format
        midnite : bool
            whether midnite arcs are alloweed 
        dbhz : bool
            whether db-hz (True) or volts/volts (False) are used for SNR data
        
    debug : bool
        debugging value to help track down bugs

    """

    #   make sure environment variables exist.  set to current directory if not
    g.check_environ_variables()

    # make sure REFL_CODE/Files/station directory exists ... 
    g.checkFiles(station, '')
    midnite = lsp['midnite']

    if 'azlist' in lsp.keys():
        azlist = lsp['azlist']
        if len(azlist) > 0:
            print('Using an augmented azimuth angle list', azlist)
    else:
        azlist = [];
        #print('no augmented elevation angle list')

    if 'ellist' in lsp.keys():
        ellist = lsp['ellist']
        if len(ellist) > 0:
            print('Using an augmented elevation angle list', ellist)
    else:
        ellist = [];
        #print('no augmented elevation angle list')

    # this must have been experimental and it does not seem to be used ...
    variable_azel = False
    if (len(ellist) >0) & (len(azlist) & 0) :
        if len(ellist) == len(azlist) :
            print('You are using a beta version of the code that sets')
            print('variable elevation angle limits for different azimuth regions.')
            print('Be careful! Especially for tall sites.')
            variable_azel = True
        
    if (len(ellist) > 0) and midnite:
        print('Testing midnite option on multiple elevation angle bins')
        if False:
            print('You have invoked multiple elevation angle bins and the midnite crossing option.')
            print('This has not been implemented yet.  Please submit a PR if you speak python or ')
            print('an Issue if your project needs this.')
            midnite = False

    # this is also checked in the command line - but for people calling the code ...
    if ((lsp['maxH'] - lsp['minH']) < 5):
        print('Requested reflector heights (', lsp['minH'], ',', lsp['maxH'], ') are too close together. Exiting.')
        print('They must be at least 5 meters apart - and preferably further than that.')
        return

    e1=lsp['e1']; e2=lsp['e2']; minH = lsp['minH']; maxH = lsp['maxH']
    ediff = lsp['ediff']; NReg = lsp['NReg']  
    PkNoise = lsp['PkNoise']; prec = lsp['desiredP']; delTmax = lsp['delTmax']
    azvalues = rewrite_azel(lsp.get('azval2'))
    if not azvalues:
        print('This module requires azval2 to be set in gnssir_input. This record is not present in your json.')
        sys.exit()

    pele = lsp['pele'] ; pfitV = lsp['polyV']

    freqs = lsp['freqs'] ; reqAmp = lsp['reqAmp']

    ok = g.is_it_legal(freqs)
    if not ok:
        print('There is something wrong. Fix your json list of frequencies. Exiting')
        sys.exit()

    plot_screen = lsp['plt_screen'] 
    onesat = lsp['onesat']; screenstats = lsp['screenstats']
    # testing this out - turned out not to be useful/needed
    #new_direct_signal = False

    gzip = lsp['gzip']
    if 'dec' in lsp.keys():
        dec = int(lsp['dec'])
    else:
        dec = 1 # so Jupyter notebooks do not need to be rewritten

    # no need to print to screen if default
    if (dec != 1):
        print('Using decimation value: ', dec)

    ann = g.make_nav_dirs(year) # make sure directories are there for orbits

    fname = FileManagement(station, 'gnssir_result', year, doy, extension=extension).get_file_path()
    resultExist = fname.is_file()
    if screenstats:
        logid, logfilename = open_gnssir_logfile(station,year,doy,extension)
    else:
        logid = None
        logfilename = None

    if (lsp['nooverwrite'] ) & (resultExist ):
        print('>>>>> The result file already exists for this day and you have selected the do not overwrite option')
        return

    print('LSP Results will be written to:', fname)
    irefr = lsp.get('refr_model', 1) if lsp.get('refraction', False) else 0

    buffer_hours = 2 if midnite else 0
    if midnite:
        print('Midnite option enabled: loading +/- 2 hours from adjacent days')

    from gnssrefl.extract_arcs import extract_arcs_from_station
    try:
        arcs = extract_arcs_from_station(
            station, year, doy, freq=freqs, snr_type=snr_type,
            buffer_hours=buffer_hours, dec=dec,
            e1=e1, e2=e2, ellist=ellist, azlist=azvalues,
            polyV=lsp['polyV'], pele=pele, dbhz=lsp['dbhz'],
            gzip=gzip, lsp=lsp,
            sat_list=lsp['onesat'],
        )
    except FileNotFoundError as e:
        print(str(e))
        return

    r.retrieve_rh(station,year,doy,extension,lsp,arcs,screenstats,irefr,logid,logfilename,lsp['dbhz'])

def local_update_plot(x,y,px,pz,ax1, ax2,failure):
    """
    updates optional result plot for SNR data and Lomb Scargle periodograms

    Parameters
    ----------
    x : numpy array
        elevation angle (deg)
    y : numpy array
        SNR (volt/volt)
    px : numpy array
        reflector height (m)
    pz : numpy array
        spectral amplitude (volt/volt)
    ax1 : matplotlib figure control
        top plot
    ax2 : matplotlib figure control
        bottom plot
    failure : boolean
        whether periodogram fails QC 

    """
    if failure:
        ax1.plot(x,y,color='gray',linewidth=0.5)
        ax2.plot(px,pz,color='gray',linewidth=0.5)
    else:
        ax1.plot(x,y)
        ax2.plot(px,pz)


def plot2screen(station, f,ax1,ax2,pltname):
    """
    Add axis information and Send the plot to the screen.
    https://www.semicolonworld.com/question/57658/matplotlib-adding-an-axes-using-the-same-arguments-as-a-previous-axes

    Parameters
    ----------
    station : string
        4 character station ID

    """
    ax2.set_xlabel('Reflector Height (m)'); 
    #ax2.set_title('SNR periodogram')
    ax2.set_ylabel('volts/volts')
    ax1.set_ylabel('volts/volts')
    ax1.set_xlabel('Elevation Angles (deg)')
    ax1.grid(True, linestyle='-')
    ax2.grid(True, linestyle='-')
    ax1.set_title(station + ' SNR Data/' + g.ftitle(f) + ' Frequency')
    plt.show()

    return True


def open_gnssir_logfile(station,year,doy,extension):
    """
    opens a logfile when asking for screen output

    Parameters
    ----------
    station : str
        4 ch station name
    year : int
        full year
    doy : int
        day of year
    extension : str
        analysis extension name (for storage of results)
        if not set you should send empty string

    Returns
    -------
    fileid : ?
        I don't know the proper name of this - but what comes out
        when you open a file so you can keep writing to it

    """
    xdir = os.environ['REFL_CODE']
    if len(extension) == 0:
        logdir = xdir + '/logs/' + station + '/' + str(year) + '/'
    else:
        logdir = xdir + '/logs/' + station + '/' + extension + '/' + str(year) + '/'

    if not os.path.isdir(logdir):
        subprocess.call(['mkdir', '-p',logdir])
    fout = 0
    cdoy = '{:03d}'.format(doy)
#   extra file with rejected arcs

    filename = logdir + cdoy + '_gnssir.txt' 
    fileid = open(filename,'w+')
    v = str(g.version('gnssrefl'))
    fileid.write('gnssrefl version {0:s} \n'.format(v))

    return fileid, filename

def retrieve_Hdates(a):
    """
    Retrieves character strings of dates and attempts to QC
    them.  

    Parameters
    ----------
    a : list of str
        online input to gnssir_input for Hdates 

    Returns
    -------
    Hdate : list of str
        full dates (2024-10-11 15:12) of Hortho values

    """
    NV = len(a)

    Hdates = []
    if  NV % 2 != 0:
        print(a)
        print('Your Hdates have an uneven number of entries. There ')
        print('needs to be one date and one HH:MM for each Hortho entry')
        sys.exit()

    for i in range(0,int(NV/2)):
        index = i*2 + 1
        #print(i, index, a[index])
        if len(a[index]) != 5:
            print(a[index], ' is an invalid time. It must be exactly five characters long including the :')
            sys.exit()
        else:
            H= a[index-1] + ' ' + a[index]
            Hdates.append(H)
            #o=datetime.datetime.fromisoformat(H)
            #ts = datetime.datetime.utctimetuple(o)
            #year = ts.tm_year ; mm  = ts.tm_mon ; dd =  ts.tm_mday
            #hh = ts.tm_hour ; minutes = ts.tm_min ; sec = 0
            #print(year, mm, dd, hh, minutes)

    return Hdates

def convert_Hdates_mjd(Hdates,remove_hhmm):
    """
    takes a list of dates in format yyyy-mm-dd hh:mm and turns them into a list of mjd

    Parameters
    ----------
    Hdates : list of str
         date strings in the format yyyy-mm-dd hh:mm

    remove_hhmm : bool
         whether you want to ignore hh:mm

    Returns
    -------
    mjd_Hortho : list of floats
        modified julian dates of character string dates

    """
    mjd_Hortho = []
    print(Hdates)
    for i in range(0,len(Hdates)):
        # convert to mjd
        if remove_hhmm : 
            m  = g.datestring_mjd(Hdates[i][0:10])
        else:
            m  = g.datestring_mjd(Hdates[i])
        mjd_Hortho.append(m)

    return mjd_Hortho 
