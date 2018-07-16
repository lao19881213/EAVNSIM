"""
@functions: source model, dirty map, clean img, radplot
@author: Zhen ZHAO
@date: May 16, 2018
"""
import os
import matplotlib as mpl
import matplotlib.image as plimg
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.ndimage.interpolation as spndint
import scipy.optimize as spfit
import numpy as np
import load_conf as lc
import utility as ut
import Func_uv as fuv
from Func_uv import FuncUv


class FuncImg(object):
    def __init__(self, model_name, n_pix, coverage_u, coverage_v, max_uv,
                 set_clean_window, clean_gain, clean_threshold, clean_niter):
        self.n_pix = n_pix
        self.n_phf = self.n_pix // 2
        # 1. source model
        # 1.1 get source model file directory
        self.source_model = model_name
        model_dir = os.path.join(os.getcwd(), 'SOURCE_MODELS')
        self.model_file = os.path.join(model_dir, self.source_model)
        # 1.2 source model result
        self.img_size = 4.
        self.img_file = []
        self.models = []
        self.x_max = 0

        self.model_img = []
        self.model_fft = []

        # 2. dirty beam
        # 2.1 parameter settings
        self.u = coverage_u
        self.v = coverage_v
        self.max_u = max_uv
        # 2.2 dirty beam result
        self.dirty_beam = []
        self.mask = []
        self.beam_scale = 0

        # 3. dirty map
        self.dirty_map = np.zeros((self.n_pix, self.n_pix), dtype=np.float32)

        # 4. cleaner
        # 4.1 settings
        self.clean_window = set_clean_window
        self.clean_gain = clean_gain
        self.clean_thresh = clean_threshold
        self.clean_niter = clean_niter
        # 4.2 clean results
        self.clean_img = []
        self.res_img = []

        # to avoid multiple runing
        self.is_model_obtained = False
        self.is_beam_obtained = False
        self.is_map_obtained = False

    # 1.source model
    def _read_model(self):
        """
        :return: models, img_size, Xaxmax, img_file
        """

        if len(self.model_file) == 0:
            self.models = [['G', 0., 0.4, 1.0, 0.1], ['D', 0., 0., 2., 0.5], ['P', -0.4, -0.5, 0.1]]
            self.x_max = self.img_size / 2.
            return True

        if len(self.model_file) > 0:
            if not os.path.exists(self.model_file):
                print("\n\nModel file %s does not exist!\n\n" % self.model_file)
                return False
            else:
                fix_size = False
                temp_model = []
                temp_img_files = []
                temp_img_size = self.img_size
                Xmax = 0.0
                fi = open(self.model_file)
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

                self.models = temp_model
                self.imsize = temp_img_size
                self.imfiles = temp_img_files
                if not fix_size:
                    self.imsize = Xmax * 1.1
                self.x_max = self.imsize / 2
                fi.close()

                return True

        return False

    def _prepare_model(self):
        """
        :return: modelim, modelfft
        """
        if self._read_model():
            # create temp variable
            models = self.models
            imsize = self.imsize
            imfiles = self.imfiles
            Npix = self.n_pix
            Nphf = self.n_phf

            pixsize = float(imsize) / Npix
            xx = np.linspace(-imsize / 2., imsize / 2., Npix)
            yy = np.ones(Npix, dtype=np.float32)
            distmat = np.zeros((Npix, Npix), dtype=np.float32)
            modelim = np.zeros((Npix, Npix), dtype=np.float32)

            # read model
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

            # read image file
            for imfile in imfiles:
                if not os.path.exists(imfile[0]):
                    imfile[0] = os.path.join(os.path.join(os.getcwd(), 'PICTURES'), imfile[0])
                    if not os.path.exists(imfile[0]):
                        print('File %s does NOT exist. Cannot read the model!' % imfile[0])
                        return False

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

            # obtain modelim, modelfft
            modelim[modelim < 0.0] = 0.0
            self.model_img = modelim
            self.model_fft = np.fft.fft2(np.fft.fftshift(modelim))
            return True
        else:
            print("wrong model settings")
            return False

    def get_result_src_model_with_update(self):
        """
        :return: model_img, max_range
        """
        if self._prepare_model():
            self.is_model_obtained = True
            Npix = self.n_pix
            Np4 = Npix // 4
            show_modelim = self.model_img[Np4:(Npix - Np4), Np4:(Npix - Np4)]
            return show_modelim, self.x_max
        else:
            return None, None

    def update_result_src_model(self):
        if self._prepare_model():
            self.is_model_obtained = True
        else:
            self.is_model_obtained = False

    def get_result_src_model(self):
        if self.is_model_obtained:
            Npix = self.n_pix
            Np4 = Npix // 4
            show_modelim = self.model_img[Np4:(Npix - Np4), Np4:(Npix - Np4)]
            return show_modelim, self.x_max
        else:
            return [], 0.0

    # 2.dirty beam
    def _prepare_beam(self):
        mask = np.zeros((self.n_pix, self.n_pix), dtype=np.float32)
        beam = np.zeros((self.n_pix, self.n_pix), dtype=np.float32)

        # 1. griding uv
        ctr = self.n_pix // 2 + 1
        scale_uv = self.n_pix / 2 / self.max_u * 0.95 * 0.5
        for index in range(0, len(self.u)):
            mask[int(ctr + round(self.u[index] * scale_uv)), int(ctr + round(self.v[index] * scale_uv))] += 1
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
        beam_scale = np.max(beam[self.n_phf:self.n_phf + 1, self.n_phf:self.n_phf + 1])
        beam[:] /= beam_scale

        # return
        self.dirty_beam = beam
        self.mask = mask
        self.beam_scale = beam_scale

    def get_result_dirty_beam_with_update(self):
        self._prepare_beam()
        self.is_beam_obtained = True
        Npix = self.n_pix
        Np4 = Npix // 4
        show_beam = self.dirty_beam[Np4:Npix - Np4, Np4:Npix - Np4]
        return show_beam

    # for multiprocessing purpose (separate updating and getter)
    def update_result_dirty_beam(self):
        self._prepare_beam()
        self.is_beam_obtained = True

    def get_result_dirty_beam(self):
        if self.is_beam_obtained:
            Npix = self.n_pix
            Np4 = Npix // 4
            show_beam = self.dirty_beam[Np4:Npix - Np4, Np4:Npix - Np4]
            return show_beam
        else:
            return []

    # 3.dirty map
    def _prepare_map(self):
        if not self.is_model_obtained:
            self._prepare_model()
        if not self.is_beam_obtained:
            self._prepare_beam()

        self.dirty_map[:] = np.fft.fftshift(
            np.fft.ifft2(self.model_fft * np.fft.ifftshift(self.mask))).real / self.beam_scale

    def get_result_dirty_map_with_update(self):
        self._prepare_map()
        self.is_map_obtained = True
        Np4 = self.n_pix // 4
        show_dirty = self.dirty_map[Np4:self.n_pix - Np4, Np4:self.n_pix - Np4]
        return show_dirty

    # for multiprocessing purpose (separate updating and getter)
    def update_result_dirty_map(self):
        self._prepare_map()
        self.is_map_obtained = True

    def get_result_dirty_map(self):
        if self.is_map_obtained:
            Np4 = self.n_pix // 4
            show_dirty = self.dirty_map[Np4:self.n_pix - Np4, Np4:self.n_pix - Np4]
            return show_dirty
        else:
            return []

    # 4.cleaner
    def overlap_indices(self):
        pass

    def do_clean(self):
        # clean_img, res_img = do_clean(dirty_map, dirty_beam, True, 0.2, 0, 100)
        if not self.is_map_obtained:
            self._prepare_map()
        image_shape = self.dirty_map.shape
        clean_img = np.zeros(image_shape)
        res_img = np.array(self.dirty_map)
        # clean window
        window = []
        if self.clean_window is True:
            window = np.ones(image_shape, np.bool)
        # clean iterations
        for i in range(self.clean_niter):
            mx, my = np.unravel_index(np.fabs(res_img[window]).argmax(), res_img.shape)
            mval = res_img[mx, my] * self.clean_gain
            clean_img[mx, my] += mval
            a1o, a2o = overlap_indices(self.dirty_map, self.dirty_beam,
                                       mx - image_shape[0] / 2,
                                       my - image_shape[1] / 2)
            # print(a1o, a2o)
            res_img[a1o[0]:a1o[1], a1o[2]:a1o[3]] -= self.dirty_beam[a2o[0]:a2o[1], a2o[2]:a2o[3]] * mval
            if np.fabs(res_img).max() < self.clean_thresh:
                break
        # result
        # print("="*20, self.clean_niter, "="*20)
        self.clean_img = clean_img
        self.res_img = res_img

    def get_result_clean_map_with_update(self):
        self.do_clean()
        Np4 = self.n_pix // 4
        show_clean = self.clean_img[Np4:self.n_pix - Np4, Np4:self.n_pix - Np4]
        show_res = self.res_img[Np4:self.n_pix - Np4, Np4:self.n_pix - Np4]
        return show_clean + show_res, show_res

    # for multiprocessing purpose (separate updating and getter)
    def update_result_clean_map(self):
        self.do_clean()

    def get_result_clean_map(self):
        if self.is_map_obtained:
            Np4 = self.n_pix // 4
            show_clean = self.clean_img[Np4:self.n_pix - Np4, Np4:self.n_pix - Np4]
            show_res = self.res_img[Np4:self.n_pix - Np4, Np4:self.n_pix - Np4]
            return show_clean, show_res
        else:
            return [], []


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
    print(model_prepared)
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
        plt.xlabel('RA offset')
        plt.ylabel('DEC offset')
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
        models = [['G', 0., 0.4, 1.0, 0.1], ['D', 0., 0., 2., 0.5], ['P', -0.4, -0.5, 0.1]]
        Xaxmax = img_size / 2.
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
            fi.close()

            return True, temp_model, temp_img_size, temp_XaxMax, temp_img_files

    return False


# 2. dirty beam
def test_dirty_beam():
    start_time = ut.time_2_mjd(lc.StartTimeGlobalYear, lc.StartTimeGlobalMonth,
                               lc.StartTimeGlobalDay, lc.StartTimeGlobalHour,
                               lc.StartTimeGlobalMinute, lc.StartTimeGlobalSecond, 0)
    stop_time = ut.time_2_mjd(lc.StopTimeGlobalYear, lc.StopTimeGlobalMonth,
                              lc.StopTimeGlobalDay, lc.StopTimeGlobalHour,
                              lc.StopTimeGlobalMinute, lc.StopTimeGlobalSecond, 0)
    time_step = ut.time_2_day(lc.TimeStepGlobalDay, lc.TimeStepGlobalHour, lc.TimeStepGlobalMinute,
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
    plt.xlabel('RA offset')
    plt.ylabel('Dec offset')
    plt.title('DIRTY BEAM')

    return beam, mask, beamScale


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
    Nphf = Npix // 2
    beamScale = np.max(beam[Nphf:Nphf + 1, Nphf:Nphf + 1])
    beam[:] /= beamScale

    return mask, beam, beamScale


# 3. test all
def test_src_beam_map():
    n_pix = 512
    # 1. source model
    source_model = 'point.model'
    # open the model file
    model_dir = os.path.join(os.getcwd(), 'SOURCE_MODELS')
    model_file = os.path.join(model_dir, source_model)
    # plot model
    plt.figure(num=1)
    model_plotted = plot_model(n_pix, model_file)
    model_fft = model_plotted[2]

    # 2. beam
    start_time = ut.time_2_mjd(lc.StartTimeGlobalYear, lc.StartTimeGlobalMonth,
                               lc.StartTimeGlobalDay, lc.StartTimeGlobalHour,
                               lc.StartTimeGlobalMinute, lc.StartTimeGlobalSecond, 0)
    stop_time = ut.time_2_mjd(lc.StopTimeGlobalYear, lc.StopTimeGlobalMonth,
                              lc.StopTimeGlobalDay, lc.StopTimeGlobalHour,
                              lc.StopTimeGlobalMinute, lc.StopTimeGlobalSecond, 0)
    time_step = ut.time_2_day(lc.TimeStepGlobalDay, lc.TimeStepGlobalHour, lc.TimeStepGlobalMinute,
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
    dirty_beam = beam_plotted[0]
    mask = beam_plotted[1]
    beam_scale = beam_plotted[2]
    # 3. dirty map
    dirty_map = np.zeros((n_pix, n_pix), dtype=np.float32)
    dirty_map[:] = np.fft.fftshift(np.fft.ifft2(model_fft * np.fft.ifftshift(mask))).real / beam_scale
    Np4 = n_pix // 4
    plt.figure(num=3)
    dirty_plot = plt.imshow(dirty_map[Np4:n_pix - Np4, Np4:n_pix - Np4], picker=True)
    plt.setp(dirty_plot)
    plt.xlabel('RA offset')
    plt.ylabel('Dec offset ')
    plt.title('DIRTY IMAGE')

    # cleaner
    clean_img, res_img = do_clean(dirty_map, dirty_beam, True, 0.2, 0, 100)
    plt.figure(num=4)
    # 4.residual image
    residual_plot = plt.imshow(res_img[Np4:n_pix - Np4, Np4:n_pix - Np4], interpolation='nearest', picker=True)
    plt.setp(residual_plot)
    plt.xlabel('RA offset ')
    plt.ylabel('Dec offset')
    plt.title('RESIDUAL')

    # clean image
    plt.figure(num=5)
    clean_plot = plt.imshow(clean_img[Np4:n_pix - Np4, Np4:n_pix - Np4], picker=True)
    plt.setp(clean_plot)
    plt.xlabel('RA offset')
    plt.ylabel('Dec offset')
    plt.title('CLEAN IMAGE')
    plt.show()


# 4. cleaner
def do_clean(dirty, psf, window, gain, thresh, niter):
    clean_img = np.zeros(np.shape(dirty))
    res_img = np.copy(dirty)
    clean_beam = get_clean_beam(psf)
    if window is True:
        window = np.ones(dirty.shape, np.bool)
    for i in range(niter):
        mx, my = np.unravel_index(np.fabs(res_img[window]).argmax(), res_img.shape)
        mval = res_img[mx, my] * gain
        clean_img[mx, my] += mval
        a1o, a2o = overlap_indices(dirty, psf,
                                   mx - dirty.shape[0] / 2,
                                   my - dirty.shape[1] / 2)
        # print(a1o, a2o)
        res_img[a1o[0]:a1o[1], a1o[2]:a1o[3]] -= psf[a2o[0]:a2o[1], a2o[2]:a2o[3]] * mval
        if np.fabs(res_img).max() < thresh:
            break
    return clean_img, res_img


def get_clean_beam(beam):
    main_lobe = np.where(beam > 0.6)
    clean_beam = np.zeros(np.shape(beam))
    Npix = len(beam[0])
    print(Npix)

    if len(main_lobe[0]) < 5:
        print('ERROR!', 'The main lobe of the PSF is too narrow!\n CLEAN model will not be restored')
        clean_beam[:] = 0.0
        clean_beam[Npix / 2, Npix / 2] = 1.0
    else:
        dX = main_lobe[0] - Npix / 2
        dY = main_lobe[1] - Npix / 2
        #  if True:
        try:
            fit = spfit.leastsq(lambda x: np.exp(-(dX * dX * x[0] + dY * dY * x[1] + dX * dY * x[2])) - beam[main_lobe], [1., 1., 0.])
            ddX = np.outer(np.ones(Npix),
                           np.arange(-Npix / 2, Npix / 2).astype(np.float64))
            ddY = np.outer(np.arange(-Npix / 2, Npix / 2).astype(np.float64),
                           np.ones(Npix))

            clean_beam[:] = np.exp(-(ddY * ddY * fit[0][0] + ddX * ddX * fit[0][1] + ddY * ddX * fit[0][2]))

            del ddX, ddY
        except:
            print('ERROR!', 'Problems fitting the PSF main lobe!\n CLEAN model will not be restored')
            clean_beam[:] = 0.0
            clean_beam[Npix / 2, Npix / 2] = 1.0

    return clean_beam


def overlap_indices(a1, a2, shiftx, shifty):
    if shiftx >= 0:
        a1xbeg = shiftx
        a2xbeg = 0
        a1xend = a1.shape[0]
        a2xend = a1.shape[0] - shiftx
    else:
        a1xbeg = 0
        a2xbeg = -shiftx
        a1xend = a1.shape[0] + shiftx
        a2xend = a1.shape[0]

    if shifty >= 0:
        a1ybeg = shifty
        a2ybeg = 0
        a1yend = a1.shape[1]
        a2yend = a1.shape[1] - shifty
    else:
        a1ybeg = 0
        a2ybeg = -shifty
        a1yend = a1.shape[1] + shifty
        a2yend = a1.shape[1]

    return (int(a1xbeg), int(a1xend), int(a1ybeg), int(a1yend)), (int(a2xbeg), int(a2xend), int(a2ybeg), int(a2yend))


def test1():
    # for u, v
    start_time = ut.time_2_mjd(lc.StartTimeGlobalYear, lc.StartTimeGlobalMonth,
                               lc.StartTimeGlobalDay, lc.StartTimeGlobalHour,
                               lc.StartTimeGlobalMinute, lc.StartTimeGlobalSecond, 0)
    stop_time = ut.time_2_mjd(lc.StopTimeGlobalYear, lc.StopTimeGlobalMonth,
                              lc.StopTimeGlobalDay, lc.StopTimeGlobalHour,
                              lc.StopTimeGlobalMinute, lc.StopTimeGlobalSecond, 0)
    time_step = ut.time_2_day(lc.TimeStepGlobalDay, lc.TimeStepGlobalHour, lc.TimeStepGlobalMinute,
                              lc.TimeStepGlobalSecond)
    myFuncUV = FuncUv(start_time, stop_time, time_step, lc.pos_mat_src[0], lc.pos_mat_src, lc.pos_mat_sat,
                      lc.pos_mat_vlbi, lc.pos_mat_telemetry, lc.obs_freq, lc.baseline_type, lc.unit_flag,
                      lc.cutoff_mode, lc.precession_mode)
    coverage_u, coverage_v, max_uv = myFuncUV.get_result_single_uv_with_update()
    print("=" * 50)
    print(max_uv)
    print("=" * 50)
    # n_pix = 512
    # source_model = 'point.model'
    # clean_gain = 0.2
    # clean_threshold = 0
    # clean_niter = 100
    # color_map_name = 'hot' # 'jet', 'rainbow', 'Greys', 'hot', 'cool', 'nipy_spectral'

    # show image parameters
    gamma = 0.5
    set_clean_window = True
    # https://matplotlib.org/examples/color/colormaps_reference.html
    colormap = cm.get_cmap(lc.color_map_name)  # 'jet', 'rainbow', 'Greys', 'hot', 'cool', 'nipy_spectral'

    myFuncImg = FuncImg(lc.source_model, lc.n_pix, coverage_u, coverage_v, max_uv,
                        set_clean_window, lc.clean_gain, lc.clean_threshold, lc.clean_niter)

    # plot source model
    plt.figure(1)
    show_model, x_max = myFuncImg.get_result_src_model_with_update()

    norm = mpl.colors.Normalize(vmin=160, vmax=300)
    modelPlot = plt.imshow(np.power(show_model, gamma), picker=True,
                           interpolation='nearest', vmin=0.0, vmax=np.max(show_model) ** gamma,
                           cmap=colormap, norm=norm)
    plt.setp(modelPlot, extent=(x_max / 2., -x_max / 2., -x_max / 2., x_max / 2.))
    plt.colorbar(shrink=0.9, )  # orientation='horizontal'
    plt.xlabel('RA offset')
    plt.ylabel('DEC offset')
    plt.title('MODEL IMAGE')

    # plot dirty beam
    plt.figure(2)
    show_beam = myFuncImg.get_result_dirty_beam_with_update()
    beamPlot = plt.imshow(show_beam, picker=True, interpolation='nearest', cmap=colormap, norm=norm)
    plt.setp(beamPlot, extent=(x_max / 2., -x_max / 2., -x_max / 2., x_max / 2.))
    plt.colorbar(shrink=0.9)
    plt.xlabel('RA offset')
    plt.ylabel('Dec offset')
    plt.title('DIRTY BEAM')

    # plot dirty map
    plt.figure(3)
    show_map = myFuncImg.get_result_dirty_map_with_update()
    dirtyPlot = plt.imshow(show_map, picker=True, interpolation='nearest', cmap=colormap, norm=norm)
    plt.setp(dirtyPlot, extent=(x_max / 2., -x_max / 2., -x_max / 2., x_max / 2.))
    plt.colorbar(shrink=0.9)
    plt.xlabel('RA offset')
    plt.ylabel('Dec offset ')
    plt.title('DIRTY IMAGE')

    # plot cleaner
    clean_img, res_img = myFuncImg.get_result_clean_map_with_update()
    plt.figure(4)
    residual_plot = plt.imshow(res_img, interpolation='nearest', picker=True,
                               cmap=colormap, norm=norm)
    plt.colorbar(shrink=0.9)
    plt.setp(residual_plot)
    plt.xlabel('RA offset ')
    plt.ylabel('Dec offset')
    plt.title('RESIDUAL')

    plt.figure(5)
    clean_plot = plt.imshow(clean_img, picker=True, interpolation='nearest', cmap=colormap, norm=norm)
    # plt.setp(clean_plot)
    plt.colorbar(shrink=0.9)
    plt.xlabel('RA offset')
    plt.ylabel('Dec offset')
    plt.title('CLEAN IMAGE')

    # show
    plt.show()


def test():
    # for u, v
    start_time = ut.time_2_mjd(lc.StartTimeGlobalYear, lc.StartTimeGlobalMonth,
                               lc.StartTimeGlobalDay, lc.StartTimeGlobalHour,
                               lc.StartTimeGlobalMinute, lc.StartTimeGlobalSecond, 0)
    stop_time = ut.time_2_mjd(lc.StopTimeGlobalYear, lc.StopTimeGlobalMonth,
                              lc.StopTimeGlobalDay, lc.StopTimeGlobalHour,
                              lc.StopTimeGlobalMinute, lc.StopTimeGlobalSecond, 0)
    time_step = ut.time_2_day(lc.TimeStepGlobalDay, lc.TimeStepGlobalHour, lc.TimeStepGlobalMinute,
                              lc.TimeStepGlobalSecond)
    myFuncUV = FuncUv(start_time, stop_time, time_step, lc.pos_mat_src[0], lc.pos_mat_src, lc.pos_mat_sat,
                      lc.pos_mat_vlbi, lc.pos_mat_telemetry, lc.obs_freq, lc.baseline_type, lc.unit_flag,
                      lc.cutoff_mode, lc.precession_mode)
    coverage_u, coverage_v, max_uv = myFuncUV.get_result_single_uv_with_update()
    print("=" * 50)
    print(max_uv)
    print("=" * 50)
    # n_pix = 512
    # source_model = 'point.model'
    # clean_gain = 0.2
    # clean_threshold = 0
    # clean_niter = 100
    # color_map_name = 'hot' # 'jet', 'rainbow', 'Greys', 'hot', 'cool', 'nipy_spectral'

    # show image parameters
    gamma = 0.5
    set_clean_window = True
    # https://matplotlib.org/examples/color/colormaps_reference.html
    colormap = cm.get_cmap(lc.color_map_name)  # 'jet', 'rainbow', 'Greys', 'hot', 'cool', 'nipy_spectral'

    myFuncImg = FuncImg(lc.source_model, lc.n_pix, coverage_u, coverage_v, max_uv,
                        set_clean_window, lc.clean_gain, lc.clean_threshold, lc.clean_niter)

    figs = plt.figure(figsize=(3, 2))
    norm = mpl.colors.Normalize(vmin=160, vmax=300)

    # 1. plot source model
    img_model = figs.add_subplot(222, aspect='equal')
    show_model, x_max = myFuncImg.get_result_src_model_with_update()
    modelPlot = img_model.imshow(np.power(show_model, gamma), picker=True,
                                 interpolation='nearest', vmin=0.0, vmax=np.max(show_model) ** gamma,
                                 cmap=colormap, norm=norm)
    plt.setp(modelPlot, extent=(x_max / 2., -x_max / 2., -x_max / 2., x_max / 2.))
    img_model.set_xlabel('RA offset')
    img_model.set_ylabel('DEC offset')
    img_model.set_title('MODEL IMAGE')

    # plot dirty beam
    img_beam = figs.add_subplot(221, aspect='equal')
    show_beam = myFuncImg.get_result_dirty_beam_with_update()
    beamPlot = img_beam.imshow(show_beam, picker=True, interpolation='nearest', cmap=colormap, norm=norm)
    plt.setp(beamPlot, extent=(x_max / 2., -x_max / 2., -x_max / 2., x_max / 2.))
    img_beam.set_xlabel('RA offset')
    img_beam.set_ylabel('Dec offset')
    img_beam.set_title('DIRTY BEAM')

    # plot dirty map
    img_dirty = figs.add_subplot(223, aspect='equal')
    show_map = myFuncImg.get_result_dirty_map_with_update()
    dirtyPlot = img_dirty.imshow(show_map, picker=True, interpolation='nearest', cmap=colormap, norm=norm)
    plt.setp(dirtyPlot, extent=(x_max / 2., -x_max / 2., -x_max / 2., x_max / 2.))
    img_dirty.set_xlabel('RA offset')
    img_dirty.set_ylabel('Dec offset ')
    img_dirty.set_title('DIRTY IMAGE')

    # plot cleaner
    clean_img, res_img = myFuncImg.get_result_clean_map_with_update()
    img_clean = figs.add_subplot(224, aspect='equal')

    clean_plot = img_clean.imshow(clean_img, picker=True, interpolation='nearest', cmap=colormap, norm=norm)
    plt.setp(clean_plot)
    img_clean.set_xlabel('RA offset')
    img_clean.set_ylabel('Dec offset')
    img_clean.set_title('CLEAN IMAGE')

    # show
    # figs_obs.colorbar(ax, ticks=bounds, )
    figs.colorbar(clean_plot, shrink=0.9)
    # figs.show()
    plt.show()


if __name__ == "__main__":
    # test_source_model()
    # test_dirty_beam()
    # test_src_beam_map()
    test()
