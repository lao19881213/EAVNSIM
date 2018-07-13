"""
@functions: visibility amplitude versus projected UV distance
@author: Zhen ZHAO
@date: May 19, 2018
"""
import tkinter as tk
from tkinter import messagebox

import load_conf as lc
import astropy.io.fits as pf
import numpy as np
import cmath
import os
import matplotlib as mpl
mpl.use("TkAgg")
import matplotlib.pyplot as plt


# Data of the input file is instored in the following format:
# ['UU---SIN', 'VV---SIN', 'WW---SIN', 'DATE', '_DATE', 'BASELINE', 'INTTIM', 'GATEID', 'CORR-ID', 'DATA']
class FuncRadPlot(object):
    def __init__(self, file):
        self.file_name = file
        obs_fits_dir = os.path.join(os.getcwd(), 'OBSERVE_DATA')
        self.fits_file = os.path.join(obs_fits_dir, str(self.file_name))

        self.plot_u = None
        self.plot_v = None
        self.baseline = None
        self.vis = None

        self.data_state = self._extract_rad_plot_data()

    def get_data_state(self):
        if self.data_state == 0:
            return True
        return False

    def reset_file(self, file):
        self.__init__(file)

    def _extract_rad_plot_data(self):
        if len(self.file_name) == 0:
            # print("\n\nWrong input!!\n\n")
            # print("info", tk.messagebox.showinfo("About", "Wrong input!!"))
            # return False
            return 1
        else:
            if not os.path.exists(self.fits_file):
                # print("\n\nModel file %s does not exist!\n\n" % fits_file)
                # print("info", tk.messagebox.showinfo("About", "Model file %s does not exist!" % self.fits_file))
                # return False
                return 2
            else:
                hdu_lst = pf.open(self.fits_file)
                PSCAL2 = hdu_lst[0].header['PSCAL2']
                data_in = hdu_lst[0].data
                uu = data_in['UU---SIN'] / PSCAL2 / 1e6
                temp_u = list(uu)
                temp_u.extend(list(-uu))
                self.plot_u = np.array(temp_u)
                vv = data_in['VV---SIN'] / PSCAL2 / 1e6
                temp_v = list(vv)
                temp_v.extend(list(-vv))
                self.plot_v = np.array(temp_v)

                DATA = data_in['DATA']
                vis_re = DATA[:, 0, 0, 0, 0, 0, 0]
                vis_im = DATA[:, 0, 0, 0, 0, 0, 1]
                self.vis = vis_re + vis_im * cmath.sqrt(-1)
                self.baseline = np.sqrt(uu ** 2 + vv ** 2)

                return 0

    def get_result_uv_data(self):
        if self.data_state == 0:
            return self.plot_u, self.plot_v
        elif self.data_state == 1:
            print("info", tk.messagebox.showinfo("About", "Wrong input!!"))
        elif self.data_state == 2:
            print("info", tk.messagebox.showinfo("About", "Model file %s does not exist!" % self.fits_file))

    def get_result_rad_data(self):
        if self.data_state == 0:
            return self.baseline, self.vis
        elif self.data_state == 1:
            print("info", tk.messagebox.showinfo("About", "Wrong input!!"))
        elif self.data_state == 2:
            print("info", tk.messagebox.showinfo("About", "Model file %s does not exist!" % self.fits_file))

    def test_rad_plot(self):
        if self.data_state == 0:
            plt.figure(num=1)
            plt.plot(self.plot_u, self.plot_v, 'ko', markersize=1)
            max_u = max(np.abs(self.plot_u))
            max_v = max(np.abs(self.plot_v))
            plt.xlim(-max_u, max_u)
            plt.ylim(-max_v, max_v)
            plt.title('UV PLOT')
            plt.xlabel("U (m)")
            plt.xlabel("V (m)")

            plt.figure(num=2)
            plt.plot(self.baseline, abs(self.vis), 'ko', markersize=2)
            plt.xlabel("UV Distance")
            plt.ylabel("Visibility Amplitude")
            plt.title('RAD PLOT')

        plt.show()


if __name__ == "__main__":

    myFunc = FuncRadPlot(lc.rad_plot_file)
    myFunc.test_rad_plot()
    # print(myFunc.get_uv_data())
