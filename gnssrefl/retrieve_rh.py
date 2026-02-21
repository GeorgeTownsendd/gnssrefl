import matplotlib.pyplot as plt
import numpy as np
import os
import subprocess
import sys

import gnssrefl.gnssir_v2 as guts
import gnssrefl.gps as g
from gnssrefl.utils import FileManagement, format_qc_summary

def retrieve_rh(station,year,doy,extension, lsp, arcs, screenstats, irefr,logid,logfilename,dbhz):
    """
    new worker code that estimates LSP from GNSS SNR data.
    it will now live here and be called by gnssir_v2.py

    Parameters
    ----------
    station : str
        name of station
    year : int
        calendar year
    doy : int
        day of year
    extension : str
        strategy extension
    lsp : dict
        inputs to LSP analysis
    arcs : list of (metadata, data) tuples
        pre-extracted satellite arcs from extract_arcs_from_station
    screenstats : bool
        whether you want stats to the screen
    irefr: int
        which refrction model is used
    logid : file ID
        opened in earlier function
    logfilename : str
        name of the log file ...
    dbhz : bool
        keep dbhz units  (or not)

    """
    fundy = False
    if station == 'bof3':
        fundy = True

    xdir = os.environ['REFL_CODE']
    docstring = 'arrays are eangles (degrees), dsnrData is SNR with/DC removed, and sec (seconds of the day),\n'

    # Use FileManagement for arcs directory with extension support
    # Only create directory if savearcs is enabled
    test_savearcs = lsp.get('savearcs', False)
    fm = FileManagement(station, "arcs_directory", year=year, doy=doy, extension=extension)
    sdir = str(fm.get_directory_path(ensure_directory=test_savearcs)) + '/'

    all_lsp = [] # variable to save the results so you can sort them

    d = g.doy2ymd(year,doy); month = d.month; day = d.day

    e1=lsp['e1']; e2=lsp['e2']; minH = lsp['minH']; maxH = lsp['maxH']
    ediff = lsp['ediff']; NReg = lsp['NReg']
    plot_screen = lsp['plt_screen']
    PkNoise = lsp['PkNoise']; prec = lsp['desiredP']; delTmax = lsp['delTmax']
    freqs = lsp['freqs'] ; reqAmp = lsp['reqAmp']
    reqAmp_dict = {f: reqAmp[i] for i, f in enumerate(freqs)}

    # this must have been something i was doing privately
    if 'savearcs' in lsp:
        test_savearcs = lsp['savearcs']
    else:
        test_savearcs = False

    # default will be plain txt
    if 'savearcs_format' in lsp:
        savearcs_format = lsp['savearcs_format']
    else:
        savearcs_format = 'txt'

    # Group pre-extracted arcs by frequency
    arcs_by_freq = {}
    for meta, data in arcs:
        arcs_by_freq.setdefault(meta['freq'], []).append((meta, data))

    if True: # so we don't have to reindent everything ...
        total_arcs = 0
        qc_lines = []
#       the main loop a given list of frequencies
        for f in freqs:
            freq_arcs = arcs_by_freq.get(f, [])
            found_results = False
            if plot_screen:
                # no idea if this will work
                fig, (ax1, ax2) = plt.subplots(2, 1,figsize=(10,7))
            rj = 0
            gj = 0
            if screenstats:
                logid.write('=================================================================================\n')
                logid.write('Looking at {0:4s} {1:4.0f} {2:3.0f} frequency {3:3.0f} ReqAmp {4:7.2f} \n'.format(station, year, doy,f,reqAmp_dict[f]))
                logid.write('=================================================================================\n')

            # Process each arc
            n_total = len(freq_arcs)
            n_filter_ediff = 0; n_filter_tooclose = 0; n_filter_amp = 0
            n_filter_pk2noise = 0; n_filter_delT = 0
            for a, (meta, data) in enumerate(freq_arcs):
                # ediff QC: check arc elevation coverage
                if (meta['ele_start'] - e1) > ediff:
                    n_filter_ediff += 1
                    continue
                if (meta['ele_end'] - e2) < -ediff:
                    n_filter_ediff += 1
                    continue
                found_results = True

                # Map extract_arcs output to expected variables
                satNu = meta['sat']
                x = data['ele']
                y = data['snr']
                secxonds = data['seconds']
                cf = meta['cf']
                az_min_ele = meta['az_min_ele']
                Edot2 = meta['edot_factor']
                delT = meta['delT']
                meanTime = meta['arc_timestamp']
                Nvv = meta['num_pts']
                Nv = Nvv
                UTCtime = meanTime

                # LSP computation
                MJD = g.getMJD(year,month,day, meanTime)
                maxF, maxAmp, eminObs, emaxObs,riseSet,px,pz = g.strip_compute(x,y,cf,maxH,prec,minH)

                tooclose = False
                if (maxF == 0) & (maxAmp == 0):
                    tooclose = True
                    Noise = 1
                    iAzim = 0
                else:
                    nij = pz[(px > NReg[0]) & (px < NReg[1])]

                Noise = 1
                if len(nij) > 0:
                    Noise = np.mean(nij)

                iAzim = int(az_min_ele)

                if abs(maxF - minH) < 0.10:  # peak too close to min value
                    tooclose = True

                if abs(maxF - maxH) < 0.10:  # peak too close to max value
                    tooclose = True

                if (not tooclose) & (delT < delTmax) & (maxAmp > reqAmp_dict[f]) & (maxAmp/Noise > PkNoise):
                    # QC passed - save arc
                    if test_savearcs and (Nv > 0):
                        newffile = guts.arc_name(sdir,satNu,f,a,az_min_ele)
                        if (len(newffile) > 0) and (delT != 0):
                            file_info = [station,satNu,f,az_min_ele,year,month,day,doy,meanTime,docstring]
                            guts.write_out_arcs(newffile,x,y,secxonds,file_info,savearcs_format)

                    xyear,xmonth,xday,xhr,xmin,xsec,xdoy = g.simpleTime(MJD)
                    betterUTC = xhr + xmin/60 + xsec/3600
                    if lsp['mmdd']:
                        onelsp = [xyear,xdoy,maxF,satNu,betterUTC,az_min_ele,maxAmp,eminObs,emaxObs,Nv,f,riseSet,Edot2,maxAmp/Noise,delT,MJD,irefr,xmonth,xday,xhr,xmin,xsec]
                    else:
                        onelsp = [xyear,xdoy,maxF,satNu,betterUTC,az_min_ele,maxAmp,eminObs,emaxObs,Nv,f,riseSet,Edot2,maxAmp/Noise,delT,MJD,irefr]

                    gj += 1
                    all_lsp.append(onelsp)

                    if screenstats:
                        T = ' ' + g.nicerTime(betterUTC)
                        logid.write('SUCCESS Azimuth {0:3.0f} Sat {1:3.0f} RH {2:7.3f} m PkNoise {3:4.1f} Amp {4:4.1f} Fr{5:3.0f} UTC {6:6s} DT {7:3.0f} \n'.format(iAzim,satNu,maxF,maxAmp/Noise,maxAmp,f,T,round(delT)))
                    if plot_screen:
                        failed = False
                        guts.local_update_plot(x,y,px,pz,ax1,ax2,failed)
                else:
                    # QC failed - count which filter caught it first
                    if tooclose:
                        n_filter_tooclose += 1
                    elif maxAmp <= reqAmp_dict[f]:
                        n_filter_amp += 1
                    elif maxAmp/Noise <= PkNoise:
                        n_filter_pk2noise += 1
                    elif delT >= delTmax:
                        n_filter_delT += 1
                    if test_savearcs and (Nv > 0):
                        newffile = guts.arc_name(sdir+'failQC/',satNu,f,a,az_min_ele)
                        if (len(newffile) > 0) and (delT != 0):
                            file_info = [station,satNu,f,az_min_ele,year,month,day,doy,meanTime,docstring]
                            guts.write_out_arcs(newffile,x,y,secxonds,file_info,savearcs_format)
                    rj += 1
                    if screenstats:
                        logid.write('FAILED QC for Azimuth {0:.1f} Satellite {1:2.0f} UTC {2:5.2f} RH {3:5.2f} \n'.format(iAzim,satNu,UTCtime,maxF))
                        g.write_QC_fails(delT,lsp['delTmax'],eminObs,emaxObs,e1,e2,ediff,maxAmp,Noise,PkNoise,reqAmp_dict[f],tooclose,logid)
                    if plot_screen:
                        failed = True
                        guts.local_update_plot(x,y,px,pz,ax1,ax2,failed)

            qc_filters = [
                ('ediff', n_filter_ediff), ('tooclose', n_filter_tooclose),
                ('amp', n_filter_amp), ('pk2noise', n_filter_pk2noise),
                ('delT', n_filter_delT),
            ]
            qc_line = format_qc_summary(f, n_total, qc_filters, gj)
            qc_lines.append(qc_line)
            if screenstats:
                logid.write('=================================================================================\n')
                logid.write('     Frequency  {0:3.0f}   good arcs: {1:3.0f}  rejected arcs: {2:3.0f} \n'.format( f, gj, rj))
                logid.write(qc_line + '\n')
                logid.write('=================================================================================\n')
            total_arcs = gj + total_arcs
            if found_results and plot_screen:
                print('data found for this frequency: ',f)
                ax1.set_xlabel('Elevation Angles (deg)')
                ax1.grid(True, linestyle='-'); ax2.grid(True, linestyle='-')
                ax1.set_title(station + ' Raw Data/Periodogram for ' + g.ftitle(f) + ' Frequency')
                ax2.set_xlabel('Reflector Height (m)');
                if dbhz:
                    ax2.set_ylabel('db-Hz') ; 
                    ax1.set_ylabel('db-Hz')
                else:
                    ax2.set_ylabel('volts/volts') ; 
                    ax1.set_ylabel('volts/volts')

                plotname = f'{xdir}/Files/{station}/gnssir_freq{f:03d}.png'
                print(plotname)
                g.save_plot(plotname)
                plt.show()
            else:
                if plot_screen: 
                    print('no data found for this frequency: ',f)

        # try moving this
        if found_results and plot_screen:
            guts.plot2screen(station, f, ax1, ax2,lsp['pltname']) 

        if screenstats:
            logid.close()
            print('Screen stat information printed to: ', logfilename)

        # look like someone asked me to sort the LSP results ... 
        # convert to numpy array
        allL = np.asarray(all_lsp)
        longer_line = lsp['mmdd']
        #print('writing out longer line ', longer_line)
        if len(allL) > 0:
            head = g.lsp_header(station,longer_line=longer_line) # header
        # sort the results for felipe
            ii = np.argsort(allL[:,15])
            allL = allL[ii,:]

            if longer_line:
                f = '%4.0f %3.0f %6.3f %3.0f %6.3f %6.2f %6.2f %6.2f %6.2f %4.0f  %3.0f  %2.0f %8.5f %6.2f %7.2f %12.6f %2.0f %2.0f %2.0f %2.0f %2.0f %2.0f '
            else:
                f = '%4.0f %3.0f %6.3f %3.0f %6.3f %6.2f %6.2f %6.2f %6.2f %4.0f  %3.0f  %2.0f %8.5f %6.2f %7.2f %12.6f %2.0f'

        # this is really just overwriting what I had before. However, This will be sorted.
            testfile,fe = g.LSPresult_name(station,year,doy,extension)
            print('Writing sorted LSP results to : ', testfile, '\n')
            np.savetxt(testfile, allL, fmt=f, delimiter=' ', newline='\n',header=head, comments='%')
        else:
            print('No good retrievals found so no LSP file should be created ')
            lspname,orgexist = g.LSPresult_name(station,year,doy,extension)
            print(lspname,orgexist)
            if orgexist:
                subprocess.call(['rm', '-f',lspname])

        if qc_lines:
            print('\n'.join(qc_lines))
