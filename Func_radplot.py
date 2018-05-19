"""
@functions: visibility amplitude versus projected UV distance
@author: Zhen ZHAO
@date: May 19, 2018
"""
import pyfits as pf
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
import cmath
import os


# Data of the input file is instored in the following format:
# ['UU---SIN', 'VV---SIN', 'WW---SIN', 'DATE', '_DATE', 'BASELINE', 'INTTIM', 'GATEID', 'CORR-ID', 'DATA']
def rad_plot(file_name):
    obs_fits_dir = os.path.join(os.getcwd(), 'OBSERVE_DATA')
    fits_file = os.path.join(obs_fits_dir, str(file_name))
    if len(file_name) == 0:
        print("\n\nWrong input!!\n\n")
        return False
    else:
        if not os.path.exists(fits_file):
            print("\n\nModel file %s does not exist!\n\n" % fits_file)
            return False
        else:
            hdu_lst = pf.open(fits_file)
            PSCAL2 = hdu_lst[0].header['PSCAL2']
            data_in = hdu_lst[0].data
            uu = data_in['UU---SIN'] / PSCAL2 / 1e6
            temp_u = list(uu)
            temp_u.extend(list(-uu))
            plot_u = np.array(temp_u)
            vv = data_in['VV---SIN'] / PSCAL2 / 1e6
            temp_v = list(vv)
            temp_v.extend(list(-vv))
            plot_v = np.array(temp_v)

            DATA = data_in['DATA']
            vis_re = DATA[:, 0, 0, 0, 0, 0, 0]
            vis_im = DATA[:, 0, 0, 0, 0, 0, 1]
            vis = vis_re + vis_im * cmath.sqrt(-1)
            bl = np.sqrt(uu ** 2 + vv ** 2)

            return (plot_u, plot_v), (bl, vis)


def test_rad_plot():
    file_name = '0106+013_1.fits'
    data_uv, data_rad = rad_plot(file_name)
    plt.figure(num=1)
    plt.plot(data_uv[0], data_uv[1], 'ko', markersize=1)
    max_u = max(np.abs(data_uv[0]))
    max_v = max(np.abs(data_uv[1]))
    plt.xlim(-max_u, max_u)
    plt.ylim(-max_v, max_v)
    plt.title('UV PLOT')
    plt.xlabel("U (m)")
    plt.xlabel("V (m)")

    plt.figure(num=2)
    plt.plot(data_rad[0], abs(data_rad[1]), 'ko', markersize=2)
    plt.xlabel("UV Distance")
    plt.ylabel("Visibility Amplitude")
    plt.title('RAD PLOT')

    plt.show()


if __name__ == "__main__":
    test_rad_plot()
