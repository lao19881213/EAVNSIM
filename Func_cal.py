import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

# 0. UI parameters
# 1)stations
teleSet = {'Ef','Mc','On','Tr','Jb1','Jb2','Cm','Wb','W1','Nt','Sh','Tm65','Ur','Mh','Ys','Sr','Ar','Hh','My','Km','Sv','Zc','Bd','Wz','Ka','Ny','ALMA','Pv','Ro70','Ro34','Pb','Ku','Ky','Kt','At','Mp','Pa','Ho','Cd','Ap','Go','Gb','Y1','Y27','Sc','Hn','Nl','Fd','La','Kp','Pt','Ov','Br','Mk'}
teleDict = {x:0 for x in teleSet}
# 2)observed band
obevBandDict ={'P - 92cm':'92cm',
               'P - 49cm':'49cm',
               'UHF - 30cm':'30cm',
               'L - 21cm':'21cm',
               'L - 18cm': '18cm',
               'S - 13cm':'13cm',
               'C - 6cm':'6cm',
               'C - 5cm':'5cm',
               'X - 3.6cm':'3.6cm',
               'U - 2cm':'2cm',
               'K - 1.3cm':'1.3cm',
               'Ka - 9mm':'9mm',
               'Q - 7mm':'7mm',
               'W - 3mm':'3mm'}
obevBandList = list(obevBandDict.keys())
# 3)data rate
datarateList = [2048, 1024, 512, 256, 128, 64, 32, 16, 8]
# 4)channel number
channelNumberDist = {'8192 ch':'8192', '4096 ch':'4096', '2048 ch':'2048', '1024 ch':'1024', '512 ch': '512',
                     '256 ch':'256', '128 ch':'128', '64 ch':'64', '32 ch':'32', '16 ch':'16'}
channelNumberList = list(channelNumberDist.keys())
# 5)integration time
integrateTimeDist = {'16 s':'16', '8 s':'8', '4 s':'4', '2 s':'2', '1 s':'1', '1/2 s':'0.5', '1/4 s':'0.25',
                     '1/8 s':'0.125', '60 ms':'0.060', '30 ms':'0.030', '15 ms':'0.015'}
integrateTimeList = list(integrateTimeDist.keys())
# 6)polarizations number
polarNumberDist = {'4 pols':'4', '2 pols':'2', '1 pol':'1'}
polarNumberList = list(polarNumberDist.keys())
# 7)subband number
subBandNumberDist = {'16 sb':'16', '8 sb':'8', '4 sb':'4', '2 sb':'2', '1 sb':'1'}
subBandNumberList = list(subBandNumberDist.keys())
# 8)subband bandwidth
subBandBandwidthDist = {'128 MHz':'128', '64 MHz':'64', '32 MHz':'32', '16 MHz':'16', '8 MHz':'8',
                        '4 MHz':'4', '2 MHz':'2', '1 MHz':'1', '0.5 MHz':'0.5'}
subBandBandwidthList = list(subBandBandwidthDist.keys())
# 9)baseline length
baselineLenDist = {'12000 km (EVN+VLBA)':'12000.0', '10000 km (Full EVN)':'10000.0', '9000 km (VLBA)':'9000.0',
                   '5000 km':'5000.0', '2500 km (Western EVN)':'2500.0', '1000 km':'1000.0'}
baselineLenList = list(baselineLenDist.keys())

# 1. main window
def center_window(root, width, height):
    screenwidth = root.winfo_screenwidth()
    screenheight = root.winfo_screenheight()
    size = '%dx%d+%d+%d' % (width, height, (screenwidth - width)/2, (screenheight - height)/2-50)
    print(size)
    root.geometry(size)

window = tk.Tk()
window.resizable(False, False)
window.title("EVN Calculator")
center_window(window, 600, 500)

# 2. framework #flat, groove, raised, ridge, solid, or sunken
tk.Label(window, text='   Configuration   ', relief='ridge', bg='lightblue', font=('Arial',18)).pack(side='top',anchor='w',padx=5)
frm_config = tk.Frame(window, border=5)
frm_config.pack(side='top',anchor='w')
tk.Label(window, text='   Calculation   ', relief='ridge', bg='lightblue', font=('Arial',18)).pack(side='top',anchor='w',padx=5)
frm_run = tk.Frame(window)
frm_run.pack(side='top',anchor='w')

frm_result = tk.Frame(window, border=1)
frm_result.pack(side='top',anchor='w')

# 3. config frame
config_left_frm = tk.Frame(frm_config,border=1)
config_left_frm.grid(row=0, column=0, padx=5)

iRow, iColumn, showNum = 0, 0, 0
for teleStation in teleDict:
    teleDict[teleStation] = tk.IntVar()
    tk.Checkbutton(config_left_frm, text=teleStation, variable=teleDict[teleStation],
                           onvalue=1, offvalue=0).grid(row=iRow, column=iColumn, ipadx=2, pady=1)
    showNum += 1
    iRow = showNum // 9
    iColumn = showNum % 9

config_right_frm = tk.Frame(frm_config,border=3)
config_right_frm.grid(row=1, column=0)

# waveband and datarate
waveband = tk.StringVar(window,'L - 18cm')
tk.Label(config_right_frm,text='Observe Band:').grid(row=0,column=0)
ttk.Combobox(config_right_frm,textvariable=waveband,values=obevBandList,width='12').grid(row=0,column=1)

datarate = tk.StringVar(window,datarateList[1])
tk.Label(config_right_frm,text='Data Rate:').grid(row=0,column=2)
ttk.Combobox(config_right_frm,textvariable=datarate,values=datarateList,width='18').grid(row=0,column=3)

# spectrum channel number and integration time
specChNum = tk.StringVar(window,'16 ch')
tk.Label(config_right_frm,text='Channel Num:').grid(row=1,column=0)
ttk.Combobox(config_right_frm,textvariable=specChNum,values=channelNumberList,width='12').grid(row=1,column=1)

integrateTime = tk.StringVar(window, '2 s')
tk.Label(config_right_frm,text='Integrate Time:').grid(row=1,column=2)
ttk.Combobox(config_right_frm,textvariable=integrateTime,values=integrateTimeList,width='18').grid(row=1,column=3)

# Number of polarizations, bandwidth of a subband [MHz]
polarNum = tk.StringVar(window,'2 pols')
tk.Label(config_right_frm,text='Polarization Num:').grid(row=2,column=0)
ttk.Combobox(config_right_frm,textvariable=polarNum,values=polarNumberList,width='12').grid(row=2,column=1)

subBandBandwidth = tk.StringVar(window,'16 MHz')
tk.Label(config_right_frm,text='Subband BW:').grid(row=2,column=2)
ttk.Combobox(config_right_frm,textvariable=subBandBandwidth,values=subBandBandwidthList,width='18').grid(row=2,column=3)

# subbands per polarizations and baseline
subBandNum = tk.StringVar(window,'8 sb')
tk.Label(config_right_frm,text='Subband Num:').grid(row=3,column=0)
ttk.Combobox(config_right_frm,textvariable=subBandNum,values=subBandNumberList,width='12').grid(row=3,column=1)


baselineLen = tk.StringVar(window,'10000 km (Full EVN)')
tk.Label(config_right_frm,text='Baseline Len:').grid(row=3,column=2)
ttk.Combobox(config_right_frm,textvariable=baselineLen,values=baselineLenList,width='18').grid(row=3,column=3)

# On-source integration time [min]
def limitInputSize(*args):
    value = onSourceTime.get()
    if len(value) > 4: onSourceTime.set(value[:5])
def test(content):
    return content.isdigit()
onSourceTime = tk.StringVar(window,'150')
onSourceTime.trace('w', limitInputSize)

testCMD= window.register(test)
tk.Label(config_right_frm,text='On-Source Time [min]:').grid(row=4,column=1)
tk.Entry(config_right_frm, bg="#282B2B", fg="white", width=12, textvariable=onSourceTime,
         validate="key",validatecommand=(testCMD,'%P')).grid(row=4,column=2)

# 4. Run frame
# 4.1 parameter definition and value-obtain
selectedStationArray = []
missTelescopeArray = []
stationNum = 0
SEFD = {}
wavelength = 0.0
dRate, tObs = 0.0, 0.0
stationEffectNum, specNum, subNum, polNum = 0, 0, 0, 0
baseLen, subBW, tInt = 0.0, 0.0, 0.0
trueDatarate = 0.0
parNum, crossNum = 0, 0


def obtain_input():
    # SEFD and wavelength
    global SEFD, wavelength
    SEFD = {}
    wavelength = 0.0
    temp_wb = obevBandDict[waveband.get()]
    if temp_wb == '92cm':
        SEFD = _band_92cm
        wavelength = 92.0
    elif temp_wb == '49cm':
        SEFD = _band_49cm
        wavelength = 49.0
    elif temp_wb == '30cm':
        SEFD = _band_UFH
        wavelength = 30.0
    elif temp_wb == '21cm':
        SEFD = _band_21cm
        wavelength = 21.0
    elif temp_wb == '18cm':
        SEFD = _band_18cm
        wavelength = 18.0
    elif temp_wb == '13cm':
        SEFD = _band_13cm
        wavelength = 13.0
    elif temp_wb == '6cm':
        SEFD = _band_6cm
        wavelength = 6.0
    elif temp_wb == '5cm':
        SEFD = _band_5cm
        wavelength = 5.0
    elif temp_wb == '3.6cm':
        SEFD = _band_4cm
        wavelength = 3.6
    elif temp_wb == '2cm':
        SEFD = _band_2cm
        wavelength = 2.0
    elif temp_wb == '1.3cm':
        SEFD = _band_13mm
        wavelength = 1.3
    elif temp_wb == '9mm':
        SEFD = _band_9mm
        wavelength = 0.9
    elif temp_wb == '7mm':
        SEFD = _band_7mm
        wavelength = 0.7
    else: # 3mm
        SEFD = _band_3mm
        wavelength = 0.3

    # station #
    global selectedStationArray, stationNum, missTelescopeArray
    selectedStationArray = []
    missTelescopeArray = []
    for teleStation in teleDict:
        if teleDict[teleStation].get() != 0:
            selectedStationArray.append(teleStation)
            if SEFD[teleStation] == -1:
                missTelescopeArray.append(teleStation)
    stationNum = len(selectedStationArray)

    # drate, Tobs
    global dRate, tObs
    dRate = float(datarate.get())
    tObs = float(onSourceTime.get()) * 60

    # stationEffectNum, specNum, subNum, polNum
    global specNum, subNum, polNum
    specNum = int(channelNumberDist[specChNum.get()])
    subNum = int(subBandNumberDist[subBandNum.get()])
    polNum = int(polarNumberDist[polarNum.get()])
    # baseLen, subBW, tInt
    global baseLen, subBW, tInt
    baseLen = float(baselineLenDist[baselineLen.get()])
    subBW = float(subBandBandwidthDist[subBandBandwidth.get()])
    tInt = float(integrateTimeDist[integrateTime.get()])

    # stationEffectNum
    global stationEffectNum
    if stationNum % 4 == 0:
        stationEffectNum = int(stationNum/4) * 4.0
    else:
        stationEffectNum = int(1 + stationNum/4) * 4.0

    # parNum, crossNum
    global parNum, crossNum
    if polNum == 1:
        parNum, crossNum = 1, 1
    else:
        parNum, crossNum = 2, 1
    if polNum == 4:
        crossNum = 2

    # $Nsb*$Nparpol*$BWsb*4;
    global trueDatarate
    trueDatarate = subNum * parNum * subBW * 4

# selectedStationArray, missTelescopeArray, stationNum, SEFD, wavelength
# dRate, tObs, stationEffectNum, specNum, subNum, polNum,baseLen, subBW, tInt
def calculate():
    m,sum1, sum2 = 2, 0, 0
    for tel1 in selectedStationArray:
        for tel2 in selectedStationArray:
            if tel1 != tel2:
                t1 = SEFD[tel1]
                t2 = SEFD[tel2]
                sum1 += (t1 * t2) ** (1-m)
                sum2 += (t1 * t2) ** (-m/2)

    sum1 *= 0.5
    sum2 *= 0.5
    mean_sefd = sum1 ** (1/2) / sum2
    return 1000 * 1.43 * mean_sefd / ((dRate *1000000.0/2.0 * tObs) ** (1/2))


def noise_calculation():
    unit = ' mJy'
    err_msg = ''
    if len(missTelescopeArray) > 0:
        err_msg = 'There are no receivers in this band (or SEFD is not ' \
                  'yet available) at following stations: ' + str(missTelescopeArray)
        return 'N/A', err_msg
    elif tObs <= 0:
        err_msg = 'Please specify a reasonable observation time'
        return 'N/A', err_msg
    else:
        if stationNum == 1:
            noise = SEFD[selectedStationArray[0]]
            unit = ' Jy'
        else:
            noise = calculate()
            if noise > 100:
                unit = " Jy"
                noise /= 1000.0
            elif noise < 0.1:
                unit = " uJy"
                noise *= 1000.0
            if stationNum == 2:
                unit += '(1 sigma)'
            else:
                unit += '/beam'
                if dRate != trueDatarate:
                    err_msg = "Warning: the total data rate " + str(dRate) \
                              + "Mbps does not math the subBand Bandwidth setting"
    return '{:.3f}'.format(noise) + unit, err_msg


def fov_bw_calculation():
    fov_bw = 49500.0 * specNum / (baseLen * subBW)
    unit = ' arcsec'
    if fov_bw >= 60.0:
        fov_bw /= 60.0
        unit = ' arcmin'
    err_msg = 'We assuming {} km for the maximum baseline, '.format(baseLen)
    return '{:.3f}'.format(fov_bw) + unit, err_msg


def fov_tm_calculation():
    fov_tm = 18560.0 * wavelength / (baseLen * tInt)
    unit = ' arcsec'
    if fov_tm >= 60.0:
        fov_tm /= 60.0
        unit = ' arcmin'
    err_msg = 'and the smearing values are calculated for 10% loss in the response of a point source.'
    return '{:.3f}'.format(fov_tm) + unit, err_msg


def capacity_calculation():
    unit = ' GB'
    err_msg = ''

    if parNum * subNum > 16:
        err_msg = "Warning: The number of subbands*polarizations exceeds 16. This has to be correlated in multiple passes.Decrease the number of subbands or polarizations to see the results for a single pass."
        return "N/A", err_msg
    elif stationNum > 16:
        err_msg = "Warning: More than 16 stations. This has to be correlated in multiple passes.Decrease the number of subbands or polarizations to see the results for a single pass."
        return "N/A", err_msg
    else:
        corr_usage_1 = (stationNum * stationNum * parNum * crossNum * subNum * specNum) / (131072.0)
        cor_cap = 1.75 * corr_usage_1 * (tObs/3600.0) / tInt
        if cor_cap < 1.0:
            cor_cap *= 1000.0
            unit = ' MB'
        return '{:.3f}'.format(cor_cap) + unit, err_msg


def run_calculation():
    # selectedStationArray, missTelescopeArray, stationNum, SEFD, wavelength
    # dRate, tObs, stationEffectNum, specNum, subNum, polNum,baseLen, subBW, tInt
    obtain_input()
    print(stationNum, stationEffectNum, wavelength, specNum, polNum, subNum)
    print(dRate, tInt, subBW, baseLen, tObs)

    # reset result
    tmFOV.set("")
    bwFOV.set("")
    fitsFile.set("")
    thermalNoise.set("")

    # format cleaning
    noiseOutWin.config(bg='lightblue')
    bwFovOutWin.config(bg='lightblue')
    tmFovOutWin.config(bg='lightblue')
    fitsCapOutWin.config(bg='lightblue')

    # start calculation
    if stationNum > 0:
        noise_set, err_noise = noise_calculation()
        thermalNoise.set(noise_set)
        if noise_set == 'N/A':
            noiseOutWin.config(bg='red')

        if stationNum < 3:
            bwFOV.set('N/A')
            bwFovOutWin.config(bg='red')
            tmFOV.set('N/A')
            tmFovOutWin.config(bg='red')
            fitsFile.set('N/A')
            fitsCapOutWin.config(bg='red')
            messagebox.showinfo(title="Warning",
                                message=err_noise + "\n\nNote: Please select a station array (N>2) if you wanna see the 'smearing' information")
        else:
            fov_bw_set, err_fov_bw = fov_bw_calculation()
            bwFOV.set(fov_bw_set)
            fov_tm_set, err_fov_tm = fov_tm_calculation()
            tmFOV.set(fov_tm_set)
            fits_cap_set, err_fits_cap = capacity_calculation()
            fitsFile.set(fits_cap_set)
            if fits_cap_set == 'N/A':
                fitsCapOutWin.config(bg='red')
            if err_noise != "":
                if err_fits_cap != "":
                    messagebox.showinfo(title="Warning", message=err_noise + '\n' + err_fits_cap)
                else:
                    messagebox.showinfo(title="Warning", message=err_noise)
    else:
        messagebox.showinfo(title="Warning", message="Warning:Please select the observation stations!")

btn_run = tk.Button(frm_run, text='RUN', width=10, height=2, command=run_calculation)
btn_run.pack(side='left', padx=100)


def reset_all():
    # reset all UI parameters
    for teleStation in teleDict:
        teleDict[teleStation].set(0)
    waveband.set('L - 18cm')
    datarate.set(datarateList[1])
    specChNum.set('16 ch')
    integrateTime.set('2 s')
    baselineLen.set('10000 km (Full EVN)')
    polarNum.set('2 pols')
    subBandNum.set('8 sb')
    subBandBandwidth.set('16 MHz')
    onSourceTime.set('150')
    # reset calculation parameters
    obtain_input()
    # reset result
    tmFOV.set("")
    bwFOV.set("")
    fitsFile.set("")
    thermalNoise.set("")
    # format cleaning
    noiseOutWin.config(bg='lightblue')
    bwFovOutWin.config(bg='lightblue')
    tmFovOutWin.config(bg='lightblue')
    fitsCapOutWin.config(bg='lightblue')


btn_reset = tk.Button(frm_run, text='RESET', width=10, height=2, command=reset_all)
btn_reset.pack(side='left',padx=30)

# 5. Result frame
frm_result_top = tk.Frame(frm_result)
frm_result_top.pack(side='top', anchor='w')

thermalNoise = tk.StringVar(window,"")
tk.Label(frm_result_top, text="Thermal Noise:").grid(row=0, column=0)
noiseOutWin = tk.Label(frm_result_top, textvariable=thermalNoise, width=20, bg='lightblue', fg='white')
noiseOutWin.grid(row=0, column=1,pady=5)

fitsFile = tk.StringVar(window,"")
tk.Label(frm_result_top, text="FITS File Size:").grid(row=1, column=0)
fitsCapOutWin = tk.Label(frm_result_top, textvariable=fitsFile, width=20, bg='lightblue', fg='white')
fitsCapOutWin.grid(row=1, column=1)

bwFOV = tk.StringVar(window,"")
tk.Label(frm_result_top, text="Bandwidth Smearing:").grid(row=0, column=2)
bwFovOutWin = tk.Label(frm_result_top, textvariable=bwFOV, width=20, bg='lightblue', fg='white')
bwFovOutWin.grid(row=0, column=3)

tmFOV = tk.StringVar(window,"")
tk.Label(frm_result_top, text="Time Smearing:").grid(row=1, column=2)
tmFovOutWin = tk.Label(frm_result_top, textvariable=tmFOV, width=20, bg='lightblue', fg='white')
tmFovOutWin.grid(row=1, column=3)


# debugInfo = tk.StringVar()
# debugView = tk.Label(window, textvariable=debugInfo, bg='red', font=('Arial',12),width=600,height=2)
# debugView.pack(side='top', anchor='w')


# show window
label = tk.Label(window,text='Copyright \N{COPYRIGHT SIGN} 2017, Zsolt Paragi and SHAO',bd=1,relief='sunken', anchor='e')
label.pack(side='bottom', fill='x')

# pre-kown infomation
_band_92cm = {'Jb1': 132, 'Jb2': -1, 'Cm': -1, 'Wb': 150, 'W1': 2100, 'Ef': 600, 'Mc': -1, 'Nt': -1, 'On': -1, 'Sh': -1, 'Tm65': -1, 'Ur': 3020, 'Tr': -1, 'Mh': -1, 'Ys': -1, 'Sr': 76, 'Ar': 12, 'Wz': -1, 'Hh': -1, 'My': -1, 'Km': -1, 'Sv': -1, 'Zc': -1, 'Bd': -1, 'Ro70': -1, 'Ka': -1, 'Ny': -1, 'ALMA': -1, 'Y1': 3900, 'Y27': 167, 'Ro34': -1, 'Go': -1, 'Gb': 35, 'Sc': 2742, 'Hn': 2742, 'Nl': 2742, 'Fd': 2742, 'La': 2742, 'Kp': 2742, 'Pt': 2742, 'Ov': 2742, 'Br': 2742, 'Mk': 2742, 'Pv': -1, 'Pb': -1, 'At': -1, 'Mp': -1, 'Pa': -1, 'Ho': -1, 'Cd': -1, 'Ap': -1, 'Ku': -1, 'Ky': -1, 'Kt': -1}
_band_49cm = {'Jb1': 83, 'Jb2': -1, 'Cm': -1, 'Wb': -1, 'W1': 1260, 'Ef': 600, 'Mc': -1, 'Nt': -1, 'On': -1, 'Sh': -1, 'Tm65': -1, 'Ur': -1, 'Tr': -1, 'Mh': -1, 'Ys': -1, 'Sr': -1, 'Ar': -1, 'Wz': -1, 'Hh': -1, 'My': -1, 'Km': -1, 'Sv': -1, 'Zc': -1, 'Bd': -1, 'Ro70': -1, 'Ka': -1, 'Ny': -1, 'ALMA': -1, 'Y1': -1, 'Y27': -1, 'Ro34': -1, 'Go': -1, 'Gb': 24, 'Sc': 2744, 'Hn': 2744, 'Nl': 2744, 'Fd': 2744, 'La': 2744, 'Kp': 2744, 'Pt': 2744, 'Ov': 2744, 'Br': 2744, 'Mk': 2744, 'Pv': -1, 'Pb': -1, 'At': -1, 'Mp': -1, 'Pa': -1, 'Ho': -1, 'Cd': -1, 'Ap': -1, 'Ku': -1, 'Ky': -1, 'Kt': -1}
_band_UFH = {'Jb1': 100, 'Jb2': -1, 'Cm': -1, 'Wb': 120, 'W1': 1680, 'Ef': 65, 'Mc': -1, 'Nt': -1, 'On': 900, 'Sh': -1, 'Tm65': -1, 'Ur': 2400, 'Tr': 2000, 'Mh': -1, 'Ys': -1, 'Sr': -1, 'Ar': 3, 'Wz': -1, 'Hh': -1, 'My': -1, 'Km': -1, 'Sv': -1, 'Zc': -1, 'Bd': -1, 'Ro70': -1, 'Ka': -1, 'Ny': -1, 'ALMA': -1, 'Y1': -1, 'Y27': -1, 'Ro34': -1, 'Go': -1, 'Gb': 13, 'Sc': -1, 'Hn': -1, 'Nl': -1, 'Fd': -1, 'La': -1, 'Kp': -1, 'Pt': -1, 'Ov': -1, 'Br': -1, 'Mk': -1, 'Pv': -1, 'Pb': -1, 'At': -1, 'Mp': -1, 'Pa': -1, 'Ho': -1, 'Cd': -1, 'Ap': -1, 'Ku': -1, 'Ky': -1, 'Kt': -1}
_band_21cm = {'Jb1': 65, 'Jb2': 350, 'Cm': 220, 'Wb': 40, 'W1': 420, 'Ef': 20, 'Mc': 700, 'Nt': 820, 'On': 320, 'Sh': -1, 'Tm65': 39, 'Ur': 350, 'Tr': 300, 'Mh': -1, 'Ys': -1, 'Sr': 67, 'Ar': 3.5, 'Wz': -1, 'Hh': -1, 'My': -1, 'Km': -1, 'Sv': 360, 'Zc': 300, 'Bd': 330, 'Ro70': -1, 'Ka': -1, 'Ny': -1, 'ALMA': -1, 'Y1': 420, 'Y27': 17.9, 'Ro34': -1, 'Go': -1, 'Gb': 10, 'Sc': 289, 'Hn': 289, 'Nl': 289, 'Fd': 289, 'La': 289, 'Kp': 289, 'Pt': 289, 'Ov': 289, 'Br': 289, 'Mk': 289, 'Pv': -1, 'Pb': -1, 'At': 68, 'Mp': 240, 'Pa': 40, 'Ho': 470, 'Cd': 1000, 'Ap': 6000, 'Ku': -1, 'Ky': -1, 'Kt': -1}
_band_18cm = {'Jb1': 65, 'Jb2': 320, 'Cm': 212, 'Wb': 40, 'W1': 420, 'Ef': 19, 'Mc': 700, 'Nt': 784, 'On': 320, 'Sh': 670, 'Tm65': 39, 'Ur': 270, 'Tr': 300, 'Mh': -1, 'Ys': -1, 'Sr': 67, 'Ar': 3, 'Wz': -1, 'Hh': 450, 'My': -1, 'Km': -1, 'Sv': 360, 'Zc': 300, 'Bd': 330, 'Ro70': 35, 'Ka': -1, 'Ny': -1, 'ALMA': -1, 'Y1': 420, 'Y27': 17.9, 'Ro34': -1, 'Go': 49, 'Gb': 10, 'Sc': 314, 'Hn': 314, 'Nl': 314, 'Fd': 314, 'La': 314, 'Kp': 314, 'Pt': 314, 'Ov': 314, 'Br': 314, 'Mk': 314, 'Pv': -1, 'Pb': -1, 'At': 68, 'Mp': 240, 'Pa': 40, 'Ho': 470, 'Cd': 1000, 'Ap': 6000, 'Ku': -1, 'Ky': -1, 'Kt': -1}
_band_13cm = {'Jb1': -1, 'Jb2': -1, 'Cm': -1, 'Wb': 60, 'W1': 840, 'Ef': 300, 'Mc': 400, 'Nt': 770, 'On': 1110, 'Sh': 800, 'Tm65': 46, 'Ur': 680, 'Tr': -1, 'Mh': 4500, 'Ys': -1, 'Sr': -1, 'Ar': 3, 'Wz': 1250, 'Hh': 380, 'My': -1, 'Km': 350, 'Sv': 330, 'Zc': 330, 'Bd': 330, 'Ro70': 20, 'Ka': 240, 'Ny': 850, 'ALMA': -1, 'Y1': 370, 'Y27': 15.8, 'Ro34': 150, 'Go': -1, 'Gb': 12, 'Sc': 347, 'Hn': 347, 'Nl': 347, 'Fd': 347, 'La': 347, 'Kp': 347, 'Pt': 347, 'Ov': 347, 'Br': 347, 'Mk': 347, 'Pv': -1, 'Pb': -1, 'At': 106, 'Mp': 530, 'Pa': 30, 'Ho': 650, 'Cd': 400, 'Ap': -1, 'Ku': -1, 'Ky': -1, 'Kt': -1}
_band_6cm = {'Jb1': 80, 'Jb2': 320, 'Cm': 136, 'Wb': 120, 'W1': 840, 'Ef': 20, 'Mc': 170, 'Nt': 260, 'On': 600, 'Sh': 720, 'Tm65': 26, 'Ur': 200, 'Tr': 220, 'Mh': -1, 'Ys': 160, 'Sr': -1, 'Ar': 5, 'Wz': -1, 'Hh': 795, 'My': -1, 'Km': -1, 'Sv': 250, 'Zc': 400, 'Bd': 200, 'Ro70': -1, 'Ka': -1, 'Ny': -1, 'ALMA': -1, 'Y1': 310, 'Y27': 13.2, 'Ro34': -1, 'Go': -1, 'Gb': 13, 'Sc': 210, 'Hn': 210, 'Nl': 210, 'Fd': 210, 'La': 210, 'Kp': 210, 'Pt': 210, 'Ov': 210, 'Br': 210, 'Mk': 210, 'Pv': -1, 'Pb': -1, 'At': 70, 'Mp': 350, 'Pa': 110, 'Ho': 640, 'Cd': 450, 'Ap': -1, 'Ku': -1, 'Ky': -1, 'Kt': -1}
_band_5cm = {'Jb1': -1, 'Jb2': 300, 'Cm': 410, 'Wb': -1, 'W1': 1600, 'Ef': 25, 'Mc': 840, 'Nt': 1100, 'On': 1500, 'Sh': 1500, 'Tm65': 26, 'Ur': -1, 'Tr': 400, 'Mh': -1, 'Ys': 160, 'Sr': 50, 'Ar': 5, 'Wz': -1, 'Hh': 680, 'My': -1, 'Km': -1, 'Sv': -1, 'Zc': -1, 'Bd': -1, 'Ro70': -1, 'Ka': -1, 'Ny': -1, 'ALMA': -1, 'Y1': 310, 'Y27': 13.2, 'Ro34': -1, 'Go': -1, 'Gb': 13, 'Sc': 278, 'Hn': 278, 'Nl': 278, 'Fd': 278, 'La': 278, 'Kp': 278, 'Pt': 278, 'Ov': 278, 'Br': 278, 'Mk': 278, 'Pv': -1, 'Pb': -1, 'At': 70, 'Mp': 350, 'Pa': 110, 'Ho': 640, 'Cd': 450, 'Ap': -1, 'Ku': -1, 'Ky': -1, 'Kt': -1}
_band_4cm = {'Jb1': -1, 'Jb2': -1, 'Cm': -1, 'Wb': 120, 'W1': 1680, 'Ef': 20, 'Mc': 320, 'Nt': 770, 'On': 1000, 'Sh': 800, 'Tm65': 48, 'Ur': 480, 'Tr': -1, 'Mh': 3200, 'Ys': 210, 'Sr': -1, 'Ar': 6, 'Wz': 750, 'Hh': 940, 'My': -1, 'Km': 480, 'Sv': 200, 'Zc': 200, 'Bd': 200, 'Ro70': 18, 'Ka': 300, 'Ny': 1255, 'ALMA': -1, 'Y1': 250, 'Y27': 10.7, 'Ro34': 106, 'Go': -1, 'Gb': 15, 'Sc': 327, 'Hn': 327, 'Nl': 327, 'Fd': 327, 'La': 327, 'Kp': 327, 'Pt': 327, 'Ov': 327, 'Br': 327, 'Mk': 327, 'Pv': -1, 'Pb': -1, 'At': 86, 'Mp': 430, 'Pa': 43, 'Ho': 560, 'Cd': 600, 'Ap': 3500, 'Ku': 1000, 'Ky': 1000, 'Kt': 1000}
_band_2cm = {'Jb1': -1, 'Jb2': -1, 'Cm': -1, 'Wb': -1, 'W1': -1, 'Ef': 45, 'Mc': -1, 'Nt': -1, 'On': -1, 'Sh': -1, 'Tm65': -1, 'Ur': -1, 'Tr': -1, 'Mh': -1, 'Ys': -1, 'Sr': -1, 'Ar': -1, 'Wz': -1, 'Hh': -1, 'My': -1, 'Km': -1, 'Sv': -1, 'Zc': -1, 'Bd': -1, 'Ro70': -1, 'Ka': -1, 'Ny': -1, 'ALMA': -1, 'Y1': 350, 'Y27': 15.0, 'Ro34': -1, 'Go': -1, 'Gb': 20, 'Sc': 543, 'Hn': 543, 'Nl': 543, 'Fd': 543, 'La': 543, 'Kp': 543, 'Pt': 543, 'Ov': 543, 'Br': 543, 'Mk': 543, 'Pv': -1, 'Pb': -1, 'At': -1, 'Mp': -1, 'Pa': -1, 'Ho': -1, 'Cd': -1, 'Ap': -1, 'Ku': 1000, 'Ky': 1000, 'Kt': 1000}
_band_13mm = {'Jb1': -1, 'Jb2': 910, 'Cm': 720, 'Wb': -1, 'W1': -1, 'Ef': 90, 'Mc': 700, 'Nt': 800, 'On': 1380, 'Sh': -1, 'Tm65': -1, 'Ur': 2950, 'Tr': 500, 'Mh': 2608, 'Ys': 295, 'Sr': 138, 'Ar': -1, 'Wz': -1, 'Hh': 3000, 'My': -1, 'Km': -1, 'Sv': 710, 'Zc': 710, 'Bd': 710, 'Ro70': 83, 'Ka': -1, 'Ny': -1, 'ALMA': -1, 'Y1': 560, 'Y27': 23.9, 'Ro34': -1, 'Go': 65, 'Gb': 30, 'Sc': 640, 'Hn': 640, 'Nl': 640, 'Fd': 640, 'La': 640, 'Kp': 640, 'Pt': 640, 'Ov': 640, 'Br': 640, 'Mk': 640, 'Pv': -1, 'Pb': -1, 'At': 106, 'Mp': 675, 'Pa': 810, 'Ho': 1800, 'Cd': 2500, 'Ap': -1, 'Ku': 1288, 'Ky': 1288, 'Kt': 1288}
_band_9mm = {'Jb1': -1, 'Jb2': -1, 'Cm': -1, 'Wb': -1, 'W1': -1, 'Ef': -1, 'Mc': -1, 'Nt': -1, 'On': -1, 'Sh': -1, 'Tm65': -1, 'Ur': -1, 'Tr': -1, 'Mh': -1, 'Ys': -1, 'Sr': -1, 'Ar': -1, 'Wz': -1, 'Hh': -1, 'My': -1, 'Km': -1, 'Sv': -1, 'Zc': -1, 'Bd': -1, 'Ro70': -1, 'Ka': -1, 'Ny': -1, 'ALMA': -1, 'Y1': 710, 'Y27': 30.3, 'Ro34': -1, 'Go': -1, 'Gb': -1, 'Sc': -1, 'Hn': -1, 'Nl': -1, 'Fd': -1, 'La': -1, 'Kp': -1, 'Pt': -1, 'Ov': -1, 'Br': -1, 'Mk': -1, 'Pv': -1, 'Pb': -1, 'At': -1, 'Mp': -1, 'Pa': -1, 'Ho': -1, 'Cd': -1, 'Ap': -1, 'Ku': -1, 'Ky': -1, 'Kt': -1}
_band_7mm = {'Jb1': -1, 'Jb2': -1, 'Cm': -1, 'Wb': -1, 'W1': -1, 'Ef': 200, 'Mc': -1, 'Nt': 900, 'On': 1310, 'Sh': -1, 'Tm65': -1, 'Ur': -1, 'Tr': -1, 'Mh': 4500, 'Ys': -1, 'Sr': 98, 'Ar': -1, 'Wz': -1, 'Hh': -1, 'My': -1, 'Km': -1, 'Sv': -1, 'Zc': -1, 'Bd': -1, 'Ro70': -1, 'Ka': -1, 'Ny': -1, 'ALMA': -1, 'Y1': 1260, 'Y27': 53.8, 'Ro34': -1, 'Go': -1, 'Gb': 60, 'Sc': 1181, 'Hn': 1181, 'Nl': 1181, 'Fd': 1181, 'La': 1181, 'Kp': 1181, 'Pt': 1181, 'Ov': 1181, 'Br': 1181, 'Mk': 1181, 'Pv': -1, 'Pb': -1, 'At': 180, 'Mp': 900, 'Pa': -1, 'Ho': -1, 'Cd': -1, 'Ap': -1, 'Ku': 1919, 'Ky': 1919, 'Kt': 1919}
_band_3mm = {'Jb1': -1, 'Jb2': -1, 'Cm': -1, 'Wb': -1, 'W1': -1, 'Ef': 930, 'Mc': -1, 'Nt': -1, 'On': 5100, 'Sh': -1, 'Tm65': -1, 'Ur': -1, 'Tr': -1, 'Mh': 17650, 'Ys': 2540, 'Sr': 367, 'Ar': -1, 'Wz': -1, 'Hh': -1, 'My': -1, 'Km': -1, 'Sv': -1, 'Zc': -1, 'Bd': -1, 'Ro70': -1, 'Ka': -1, 'Ny': -1, 'ALMA': 70, 'Y1': -1, 'Y27': -1, 'Ro34': -1, 'Go': -1, 'Gb': 140, 'Sc': -1, 'Hn': -1, 'Nl': 4236, 'Fd': 4236, 'La': 4236, 'Kp': 4236, 'Pt': 4236, 'Ov': 4236, 'Br': 4236, 'Mk': 4236, 'Pv': 640, 'Pb': 450, 'At': 1440, 'Mp': 3750, 'Pa': -1, 'Ho': -1, 'Cd': -1, 'Ap': -1, 'Ku': 3000, 'Ky': 3000, 'Kt': 3000}


window.mainloop()
