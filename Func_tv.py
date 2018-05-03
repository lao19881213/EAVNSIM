"""
@functions: telescope visibility
@author: Zhen ZHAO
@date: April 26, 2018
"""
import load_conf as lc
import trans_time as tt
import trans_coordinate as tc
import trans_unit as tu
import model_effect as me
import model_satellite as ms
import model_obs_ability as mo
import numpy as np
import matplotlib.pyplot as plt


def func_tv_az_el(start_time_mjd, stop_time_mjd, time_step, pos_src, pos_mat_vlbi):
    # print("pos_src=",pos_src)
    if type(pos_src[1]) == str:
        pos_src[1] = tt.time_str_2_rad(pos_src[1])
    if type(pos_src[2]) == str:
        pos_src[2] = tu.angle_2_rad(pos_src[2])
    ra_src = pos_src[1]
    dec_src = pos_src[2]

    lst_az = []
    lst_el = []
    lst_hour = []
    for i in range(len(pos_mat_vlbi)):
        lst_az_1 = []
        lst_el_1 = []
        lst_hour_1 = []
        long_vlbi, lat_vlbi, height_vlbi = tc.itrf_2_geographic(pos_mat_vlbi[i][1], pos_mat_vlbi[i][2], pos_mat_vlbi[i][3])
        for itr_mjd in np.arange(start_time_mjd, stop_time_mjd, time_step):
            source_azimuth, source_elevation = tc.equatorial_2_horizontal(itr_mjd, ra_src, dec_src, long_vlbi, lat_vlbi)
            azimuth_deg = tu.rad_2_angle(source_azimuth)
            elevation_deg = tu.rad_2_angle(source_elevation)
            if elevation_deg < 0:
                elevation_deg = 0
            h1 = (itr_mjd - start_time_mjd) * 24
            lst_az_1.append(azimuth_deg)
            lst_el_1.append(elevation_deg)
            lst_hour_1.append(h1)
        lst_az.append(lst_az_1)
        lst_el.append(lst_el_1)
        lst_hour.append(lst_hour_1)
    return lst_az, lst_el, lst_hour


def test_ae_el():
    # load data from configurations
    start_time = tt.time_2_mjd(lc.StartTimeGlobalYear, lc.StartTimeGlobalMonth,
                               lc.StartTimeGlobalDay, lc.StartTimeGlobalHour,
                               lc.StartTimeGlobalMinute, lc.StartTimeGlobalSecond, 0)
    stop_time = tt.time_2_mjd(lc.StopTimeGlobalYear, lc.StopTimeGlobalMonth,
                              lc.StopTimeGlobalDay, lc.StopTimeGlobalHour,
                              lc.StopTimeGlobalMinute, lc.StopTimeGlobalSecond, 0)
    time_step = tt.time_2_day(lc.TimeStepGlobalDay, lc.TimeStepGlobalHour, lc.TimeStepGlobalMinute,
                              lc.TimeStepGlobalSecond)
    # invoke the AZ-EL calculation functions
    azimuth, elevation, hour_lst = func_tv_az_el(start_time, stop_time, time_step, lc.SouGllst[0], lc.VLBGllst)

    # plot it
    for i in np.arange(0, len(azimuth)):
        az1 = azimuth[i]
        h1 = hour_lst[i]
        plt.subplot(2, 1, 1)
        plt.plot(h1, az1)
        plt.xlabel("Time(h)")
        plt.ylabel("Azimuth($^\circ$)")
        plt.title("The azimuth of source in VLBI stations")
        # plt.title_MPL(1, "The azimuth of source in VLBI stations")

    for j in np.arange(0, len(elevation)):
        el1 = elevation[j]
        h1 = hour_lst[j]

        plt.subplot(2, 1, 2)
        plt.plot(h1, el1)
        plt.xlabel("Time(h)")
        plt.ylabel("Elevation($^\circ$)")
        plt.title("The elevation of source in VLBI stations")
    plt.show()


if __name__ == "__main__":
    # test az-el
    test_ae_el()
