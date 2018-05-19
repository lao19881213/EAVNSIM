"""
@functions: source model, dirty map, clean img, radplot
@author: Zhen ZHAO
@date: May 16, 2018
"""
import os
import time
import sys
import matplotlib as mpl
import matplotlib.image as plimg
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.ndimage.interpolation as spndint
import numpy as np
import load_conf as lc
import trans_time as tt
import Func_uv as fuv


# 1. source model
def test_source_model():
    # basic setting
    # n_H = 200
    n_pix = 512
    source_model = 'Faceon-Galaxy.model'
    # open the model file
    model_dir = os.path.join(os.getcwd(), 'SOURCE_MODELS')
    model_file = os.path.join(model_dir, source_model)
    plt.figure(num=1)
    plot_model(n_pix, model_file)
    plt.show()


def plot_model(Npix, model_file):

    model_prepared = prepare_model(Npix, model_file)
    if len(model_prepared) <= 1:
        return False
    else:
        Np4 = Npix // 4
        gamma = 0.5
        modelim = model_prepared[1]

        modelfft = model_prepared[2]
        Xaxmax = model_prepared[3]
        temp = modelim[Np4:(Npix - Np4), Np4:(Npix - Np4)]
        modelPlotPlot = plt.imshow(np.power(temp, gamma), picker=True,
                                   interpolation='nearest', vmin=0.0, vmax=np.max(modelim) ** gamma, cmap=cm.jet)

        # modflux = modelim[Nphf, Nphf]
        # fmtM = r'%.2e Jy/pixel' "\n"  r'$\Delta\alpha = $ % 4.2f / $\Delta\delta = $ % 4.2f'
        # plt.text(0.05, 0.87, fmtM % (modflux, 0.0, 0.0),
        #                      bbox=dict(facecolor='white', alpha=0.7))
        plt.setp(modelPlotPlot, extent=(Xaxmax / 2., -Xaxmax / 2., -Xaxmax / 2., Xaxmax / 2.))
        plt.xlabel('Dec offset (as)')
        plt.ylabel('Dec offset (as)')
        plt.title('MODEL IMAGE')
        return True, modelim, modelfft, Xaxmax


def prepare_model(Npix, model_file):
    model_reading = read_model(str(model_file))
    if len(model_reading) <= 1:
        print("wrong model settings")
        return False
    else:
        # read in the data
        models = model_reading[1]
        imsize = model_reading[2]
        Xaxmax = model_reading[3]
        imfiles = model_reading[4]
        Nphf = Npix // 2
        modelim = np.zeros((Npix, Npix), dtype=np.float32)

        pixsize = float(imsize) / Npix
        xx = np.linspace(-imsize / 2., imsize / 2., Npix)
        yy = np.ones(Npix, dtype=np.float32)
        distmat = np.zeros((Npix, Npix), dtype=np.float32)

        for model in models:
            xsh = -model[1]
            ysh = -model[2]
            xpix = np.rint(xsh / pixsize).astype(np.int32)
            ypix = np.rint(ysh / pixsize).astype(np.int32)
            centy = np.roll(xx, ypix)
            centx = np.roll(xx, xpix)
            distmat[:] = np.outer(centy ** 2., yy) + np.outer(yy, centx ** 2.)
            if model[0] == 'D':
                mask = np.logical_or(distmat <= model[4] ** 2., distmat == np.min(distmat))
                modelim[mask] += float(model[3]) / np.sum(mask)
            elif model[0] == 'G':
                gauss = np.exp(-distmat / (2. * model[4] ** 2.))
                modelim[:] += float(model[3]) * gauss / np.sum(gauss)
            elif model[0] == 'P':
                if np.abs(xpix + Nphf) < Npix and np.abs(ypix + Nphf) < Npix:
                    yint = ypix + Nphf
                    xint = xpix + Nphf
                    modelim[yint, xint] += float(model[3])

        for imfile in imfiles:
            if not os.path.exists(imfile[0]):
                imfile[0] = os.path.join(os.path.join(os.getcwd(), 'PICTURES'), imfile[0])
                if not os.path.exists(imfile[0]):
                    print('File %s does NOT exist. Cannot read the model!' % imfile[0])
                    return

            Np4 = Npix // 4
            img = plimg.imread(imfile[0]).astype(np.float32)
            dims = np.shape(img)
            d3 = min(2, dims[2])
            d1 = float(max(dims))
            avimg = np.average(img[:, :, :d3], axis=2)
            avimg -= np.min(avimg)
            avimg *= imfile[1] / np.max(avimg)
            if d1 == Nphf:
                pass
            else:
                zoomimg = spndint.zoom(avimg, float(Nphf) / d1)
                zdims = np.shape(zoomimg)
                zd0 = min(zdims[0], Nphf)
                zd1 = min(zdims[1], Nphf)
                sh0 = (Nphf - zdims[0]) // 2
                sh1 = (Nphf - zdims[1]) // 2
                # print(sh0, Np4, zd0, sh1, zd1)
                modelim[sh0 + Np4:sh0 + Np4 + zd0, sh1 + Np4:sh1 + Np4 + zd1] += zoomimg[:zd0, :zd1]

        modelim[modelim < 0.0] = 0.0
        modelfft = np.fft.fft2(np.fft.fftshift(modelim))
        # print(modelim)
        return True, modelim, modelfft, Xaxmax


def read_model(model_file):

    img_size = 4.
    img_file = []

    if len(model_file) == 0:
        models = [['G',0.,0.4,1.0,0.1],['D',0.,0.,2.,0.5],['P',-0.4,-0.5,0.1]]
        Xaxmax = img_size/2.
        return True, models, img_size, Xaxmax, img_file

    if len(model_file) > 0:
        if not os.path.exists(model_file):
            print("\n\nModel file %s does not exist!\n\n" % model_file)
            return False
        else:
            fix_size = False
            temp_model = []
            temp_img_files = []
            temp_img_size = img_size
            Xmax = 0.0
            fi = open(model_file)
            for li, l in enumerate(fi.readlines()):
                comm = l.find('#')
                if comm >= 0:
                    l = l[:comm]
                it = l.split()
                if len(it) > 0:
                    if it[0] == 'IMAGE':
                        temp_img_files.append([str(it[1]), float(it[2])])
                    elif it[0] in ['G', 'D', 'P']:
                        temp_model.append([it[0]] + list(map(float, it[1:])))
                        if temp_model[-1][0] != 'P':
                            temp_model[-1][4] = np.abs(temp_model[-1][4])
                            Xmax = np.max([np.abs(temp_model[-1][1]) + temp_model[-1][4],
                                           np.abs(temp_model[-1][2]) + temp_model[-1][4], Xmax])
                    elif it[0] == 'IMSIZE':
                        temp_img_size = 2. * float(it[1])
                        fix_size = True
                    else:
                        print("\n\nWRONG SYNTAX IN LINE %i:\n\n %s...\n\n" % (li + 1, l[:max(10, len(l))]))
            if len(temp_model) + len(temp_img_files) == 0:
                print("\n\nThere should be at least 1 model component!\n\n")
            if not fix_size:
                temp_img_size = Xmax * 1.1
            temp_XaxMax = temp_img_size / 2

            return True, temp_model, temp_img_size, temp_XaxMax, temp_img_files

    return False


# 2. dirty beam
def test_dirty_beam():
    start_time = tt.time_2_mjd(lc.StartTimeGlobalYear, lc.StartTimeGlobalMonth,
                               lc.StartTimeGlobalDay, lc.StartTimeGlobalHour,
                               lc.StartTimeGlobalMinute, lc.StartTimeGlobalSecond, 0)
    stop_time = tt.time_2_mjd(lc.StopTimeGlobalYear, lc.StopTimeGlobalMonth,
                              lc.StopTimeGlobalDay, lc.StopTimeGlobalHour,
                              lc.StopTimeGlobalMinute, lc.StopTimeGlobalSecond, 0)
    time_step = tt.time_2_day(lc.TimeStepGlobalDay, lc.TimeStepGlobalHour, lc.TimeStepGlobalMinute,
                              lc.TimeStepGlobalSecond)

    dict_u, dict_v, dict_bl_sta1, dict_bl_sta2, dict_bl_duration \
        = fuv.func_uv(start_time, stop_time, time_step, lc.pos_mat_src[0], lc.pos_mat_sat, lc.pos_mat_vlbi,
                      lc.pos_mat_telemetry, lc.obs_freq, lc.baseline_type, lc.unit_flag, lc.cutoff_mode, lc.precession_mode)
    temp_u = []
    for each in dict_u.values():
        if each is not None:
            temp_u.extend(each)
    temp_v = []
    for each in dict_v.values():
        if each is not None:
            temp_v.extend(each)
    n_pix = 512
    plt.figure(num=2)
    max_u = max(max(temp_u), max(temp_v))
    plot_dirty_beam(n_pix, temp_u, temp_v, max_u)
    plt.show()


def plot_dirty_beam(Npix, u, v, max_u):
    Np4 = Npix // 4
    beam_prepared = prepare_beam(Npix, u, v, max_u)
    mask = beam_prepared[0]
    beam = beam_prepared[1]
    beamScale = beam_prepared[2]

    # beamPlotPlot = plt.imshow(beam[Np4:Npix-Np4, Np4:Npix-Np4], picker=True, interpolation='nearest')
    beamPlotPlot = plt.imshow(beam[Np4:Npix - Np4, Np4:Npix - Np4], picker=True)

    plt.setp(beamPlotPlot)
    plt.xlabel('RA offset (as)')
    plt.ylabel('Dec offset (as)')
    plt.title('DIRTY BEAM')

    return mask, beamScale


def prepare_beam(Npix, u, v, max_u):
    mask = np.zeros((Npix, Npix), dtype=np.float32)
    beam = np.zeros((Npix, Npix), dtype=np.float32)

    # 1. griding uv
    ctr = Npix // 2 + 1
    scale_uv = Npix / 2 / max_u * 0.95 * 0.5
    for index in range(0, len(u)):
        mask[int(ctr + round(u[index] * scale_uv)), int(ctr + round(v[index] * scale_uv))] += 1
    mask = np.transpose(mask)

    # 2. robust sampling
    # robust = 0.0
    # Nbas = len(u)
    # nH = 200 # time_duration // time_step
    # robfac = (5. * 10. ** (-robust)) ** 2. * (2. * Nbas * nH) / np.sum( mask** 2.)
    # robustsamp = np.zeros((Npix, Npix), dtype=np.float32)
    # robustsamp[:] = mask / (1. + robfac * mask)

    # 3. beam
    beam[:] = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(mask))).real
    Nphf = Npix//2
    beamScale = np.max(beam[Nphf:Nphf + 1, Nphf:Nphf + 1])
    beam[:] /= beamScale

    return mask, beam, beamScale


# 3. test all
def test_src_beam_map():
    n_pix = 512
    # 1. source model
    source_model = 'Faceon-Galaxy.model'
    # open the model file
    model_dir = os.path.join(os.getcwd(), 'SOURCE_MODELS')
    model_file = os.path.join(model_dir, source_model)
    # plot beam
    plt.figure(num=1)
    model_plotted = plot_model(n_pix, model_file)
    model_fft = model_plotted[2]

    # 2. beam
    start_time = tt.time_2_mjd(lc.StartTimeGlobalYear, lc.StartTimeGlobalMonth,
                               lc.StartTimeGlobalDay, lc.StartTimeGlobalHour,
                               lc.StartTimeGlobalMinute, lc.StartTimeGlobalSecond, 0)
    stop_time = tt.time_2_mjd(lc.StopTimeGlobalYear, lc.StopTimeGlobalMonth,
                              lc.StopTimeGlobalDay, lc.StopTimeGlobalHour,
                              lc.StopTimeGlobalMinute, lc.StopTimeGlobalSecond, 0)
    time_step = tt.time_2_day(lc.TimeStepGlobalDay, lc.TimeStepGlobalHour, lc.TimeStepGlobalMinute,
                              lc.TimeStepGlobalSecond)

    dict_u, dict_v, dict_bl_sta1, dict_bl_sta2, dict_bl_duration \
        = fuv.func_uv(start_time, stop_time, time_step, lc.pos_mat_src[0], lc.pos_mat_sat, lc.pos_mat_vlbi,
                      lc.pos_mat_telemetry, lc.obs_freq, lc.baseline_type, lc.unit_flag, lc.cutoff_mode,
                      lc.precession_mode)
    temp_u = []
    for each in dict_u.values():
        if each is not None:
            temp_u.extend(each)
    temp_v = []
    for each in dict_v.values():
        if each is not None:
            temp_v.extend(each)
    # plot beam
    plt.figure(num=2)
    max_u = max(max(temp_u), max(temp_v))
    beam_plotted = plot_dirty_beam(n_pix, temp_u, temp_v, max_u)
    mask = beam_plotted[0]
    beam_scale = beam_plotted[1]
    # 3. dirty map
    dirty_map = np.zeros((n_pix, n_pix), dtype=np.float32)
    dirty_map[:] = np.fft.fftshift(np.fft.ifft2(model_fft * np.fft.ifftshift(mask))).real / beam_scale
    Np4 = n_pix // 4
    plt.figure(num=3)
    dirty_plot = plt.imshow(dirty_map[Np4:n_pix - Np4, Np4:n_pix - Np4], picker=True)
    plt.setp(dirty_plot)
    plt.xlabel('RA offset (as)')
    plt.ylabel('Dec offset (as)')
    plt.title('DIRTY IMAGE')

    plt.show()


if __name__ == "__main__":
    # test_source_model()
    # test_dirty_beam()
    test_src_beam_map()
