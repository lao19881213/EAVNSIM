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


def test_source_model():
    # basic setting
    n_H = 200
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
        Nphf = Npix // 2
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

    if len(model_file)>0:
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
            models = temp_model
            img_file = temp_img_files
            if not fix_size:
                temp_img_size = Xmax * 1.1
            temp_XaxMax = temp_img_size / 2

            return True, temp_model, temp_img_size, temp_XaxMax, temp_img_files

    return False


def test_dirty_beam():
    pass


def test_dirty_map():
    pass


if __name__ == "__main__":
    test_source_model()
