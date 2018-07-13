"""
@functions: telescope visibility
@author: Zhen ZHAO
@date: April 26, 2018
"""
import load_conf as lc
import utility as ut
import model_effect as me
import model_satellite as ms
import model_obs_ability as mo
import numpy as np
import matplotlib.pyplot as plt


class FuncObs(object):
    def __init__(self, start_mjd, stop_mjd, t_step, pos_main_src, pos_vlbi, pos_sat, pos_telem,
                 baseline_type, cutoff_dict, procession):
        # 1. input parameter
        self.start_time_mjd = start_mjd
        self.stop_time_mjd = stop_mjd
        self.time_step = t_step
        self.pos_src = pos_main_src
        self.pos_mat_vlbi = pos_vlbi
        self.pos_mat_sat = pos_sat
        self.pos_mat_telem = pos_telem
        self.bl_type = baseline_type
        self.cutoff_dict = cutoff_dict
        self.cutoff_angle = cutoff_dict['CutAngle']
        self.procession_mode = procession

        # 2. Az-El result
        self.result_azimuth = []
        self.result_elevation = []
        self.result_hour = []

        # 3. sky survey
        self.result_pos_sun = []
        self.result_pos_moon = []
        self.num_array = None

    def _func_tv_az_el(self):

        if type(self.pos_src[1]) == str:
            self.pos_src[1] = ut.time_str_2_rad(self.pos_src[1])
        if type(self.pos_src[2]) == str:
            self.pos_src[2] = ut.angle_str_2_rad(self.pos_src[2])
        ra_src = self.pos_src[1]
        dec_src = self.pos_src[2]

        lst_az = []
        lst_el = []
        lst_hour = []

        for i in range(len(self.pos_mat_vlbi)):
            lst_az_1 = []
            lst_el_1 = []
            lst_hour_1 = []
            long_vlbi, lat_vlbi, height_vlbi = ut.itrf_2_geographic(self.pos_mat_vlbi[i][1], self.pos_mat_vlbi[i][2],
                                                                    self.pos_mat_vlbi[i][3])
            for itr_mjd in np.arange(self.start_time_mjd, self.stop_time_mjd, self.time_step):
                source_azimuth, source_elevation = ut.equatorial_2_horizontal(itr_mjd, ra_src, dec_src, long_vlbi,
                                                                              lat_vlbi)
                azimuth_deg = ut.rad_2_angle(source_azimuth)
                elevation_deg = ut.rad_2_angle(source_elevation)
                if elevation_deg < 0:
                    elevation_deg = 0
                h1 = (itr_mjd - self.start_time_mjd) * 24
                lst_az_1.append(azimuth_deg)
                lst_el_1.append(elevation_deg)
                lst_hour_1.append(h1)
            lst_az.append(lst_az_1)
            lst_el.append(lst_el_1)
            lst_hour.append(lst_hour_1)

        self.result_azimuth = lst_az
        self.result_elevation = lst_el
        self.result_hour = lst_hour

    def get_result_az_el_with_update(self):
        self._func_tv_az_el()
        return self.result_azimuth, self.result_elevation, self.result_hour

    # for multiprocessing purpose (separate updating and getter)
    def update_result_az_el(self):
        self._func_tv_az_el()

    def get_result_az_el(self):
        return self.result_azimuth, self.result_elevation, self.result_hour

    def _func_sky_survey(self):
        # 统计可见望远镜/基线个数的数组
        num_array = []

        # cutoff angle setting
        vb = np.ones((1, 360), dtype=float)
        vb *= self.cutoff_angle
        vb = vb[0].tolist()

        # calculate the position of sun and moon
        sun_ra, sun_dec = me.sun_ra_dec_cal(self.start_time_mjd, self.start_time_mjd, 1)
        moon_ra, moon_dec = me.moon_ra_dec_cal(self.start_time_mjd, self.start_time_mjd, 1)

        # survey the whole sky
        ra_list = np.arange(0.125, 24, 0.25)
        dec_list = np.arange(-88.75, 90, 2.5)
        for src_dec in dec_list:
            src_dec = ut.angle_2_rad(src_dec, 0, 0)
            for src_ra in ra_list:
                src_ra = ut.time_2_rad(src_ra, 0, 0)
                num1 = 0  # sta和vlbi能观测到source的计数
                # test vlbi station
                for i in self.pos_mat_vlbi:  # get vlbi station
                    longitude, latitude, height = ut.itrf_2_geographic(i[1], i[2], i[3])
                    visibility = mo.obs_judge_active_vlbi_station(src_ra, src_dec, self.start_time_mjd, longitude, latitude, vb)
                    if visibility:
                        num1 = num1 + 1
                # # test satellite
                # for j in self.pos_mat_sat:
                #     visibility = mo.obs_judge_active_satellite_with_kepler(self.start_time_mjd,
                #                                                            src_ra, src_dec,
                #                                                            self.pos_mat_sat[j], self.pos_mat_telem,
                #                                                            self.bl_type, self.cutoff_dict,
                #                                                            self.procession_mode)
                #
                #     if visibility:
                #         num1 = num1 + 1
                # add num
                num_array.append(num1)

        num_array = np.array(num_array)
        num_array.shape = len(dec_list), len(ra_list)

        # 图像是，[0, 96], [0,72], 需要转化坐标
        img_sun_ra = sun_ra[0] * (96 / 24)
        img_sun_dec = sun_dec[0] * 0.4 + 36

        img_moon_ra = moon_ra[0] * (96 / 24)
        img_moon_dec = moon_dec[0] * 0.4 + 36

        # self.result_pos_sun = [sun_ra, sun_dec]
        # self.result_pos_moon = [moon_ra, moon_dec]
        self.result_pos_sun = [img_sun_ra, img_sun_dec]
        self.result_pos_moon = [img_moon_ra, img_moon_dec]
        self.num_array = num_array

    def get_result_sky_survey_with_update(self):
        self._func_sky_survey()
        return self.result_pos_sun, self.result_pos_moon, self.num_array

    # for multiprocessing purpose (separate updating and getter)
    def update_result_sky_survey(self):
        self._func_sky_survey()

    def get_result_sky_survey(self):
        return self.result_pos_sun, self.result_pos_moon, self.num_array


def test():
    # load data from configurations
    start_time = ut.time_2_mjd(lc.StartTimeGlobalYear, lc.StartTimeGlobalMonth,
                               lc.StartTimeGlobalDay, lc.StartTimeGlobalHour,
                               lc.StartTimeGlobalMinute, lc.StartTimeGlobalSecond, 0)
    stop_time = ut.time_2_mjd(lc.StopTimeGlobalYear, lc.StopTimeGlobalMonth,
                              lc.StopTimeGlobalDay, lc.StopTimeGlobalHour,
                              lc.StopTimeGlobalMinute, lc.StopTimeGlobalSecond, 0)
    time_step = ut.time_2_day(lc.TimeStepGlobalDay, lc.TimeStepGlobalHour, lc.TimeStepGlobalMinute,
                              lc.TimeStepGlobalSecond)

    myFuncObs = FuncObs(start_time, stop_time, time_step, lc.pos_mat_src[0],
                        lc.pos_mat_vlbi, lc.pos_mat_sat, lc.pos_mat_telemetry,
                        lc.baseline_type, lc.cutoff_mode, lc.precession_mode)

    # az - el
    azimuth, elevation, hour_lst = myFuncObs.get_result_az_el_with_update()

    plt.figure(1)
    plt.subplot(2, 1, 1)
    for i in np.arange(0, len(azimuth)):
        az1 = azimuth[i]
        h1 = hour_lst[i]
        plt.plot(h1, az1, '.-', markersize=1)
    plt.xlabel("Time(h)")
    plt.ylabel("Azimuth($^\circ$)")
    plt.title("The azimuth of source in VLBI stations")

    plt.subplot(2, 1, 2)
    for i in np.arange(0, len(elevation)):
        el1 = elevation[i]
        h1 = hour_lst[i]
        plt.plot(h1, el1, '.-', markersize=1)
    plt.xlabel("Time(h)")
    plt.ylabel("Elevation($^\circ$)")
    plt.title("The elevation of source in VLBI stations")

    # sky survey
    pos_sun, pos_moon, num_array = myFuncObs.get_result_sky_survey_with_update()
    plt.figure(2)
    # draw survey
    array_max = np.max(num_array)
    bounds = np.arange(0, array_max + 1, 1)
    ax = plt.pcolor(num_array, edgecolors=(0.5, 0.5, 0.5), linewidths=1)
    plt.colorbar(ax, ticks=bounds, shrink=1)
    plt.yticks([0, 24, 36, 48, 72], [-90, -30, 0, 30, 90])
    plt.xticks([0, 16, 32, 48, 64, 80, 96], [0, 4, 8, 12, 16, 20, 24])
    plt.plot([48, 48], [0, 72], color='black', linewidth=0.8, linestyle='-.', alpha=0.4)
    plt.plot([0, 96], [36, 36], color='black', linewidth=0.8, linestyle='-.', alpha=0.4)
    plt.xlabel("RA(H)")
    plt.ylabel(r'Dec ($^\circ$)')
    plt.title("SKY SURVEY")
    # draw soon, moon
    plt.plot(pos_sun[0], pos_sun[1], color='red', marker='o', markerfacecolor=(1, 0, 0), alpha=1, markersize=20)
    plt.plot(pos_moon[0], pos_moon[1], color='blue', marker='o', markerfacecolor='w', alpha=1, markersize=10)

    plt.show()


if __name__ == "__main__":
    # plt.figure(num=1)
    # test_ae_el()
    # plt.figure(num=2)
    # test_sky_survey()
    # plt.show()
    test()
