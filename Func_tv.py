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
    azimuth, elevation, hour_lst = func_tv_az_el(start_time, stop_time, time_step, lc.pos_mat_src[0], lc.pos_mat_vlbi)

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
    # plt.show()


def func_sky_survey(start_mjd, pos_mat_vlbi, pos_mat_sat):
    # 统计个数的数组
    num_array = []
    # 水平角设置
    vb = np.ones((1, 360), dtype=int)
    vb *= 15
    vb = vb[0].tolist()
    # calculate the position of sun and moon
    sun_ra, sun_dec = me.sun_ra_dec_cal(start_mjd, start_mjd, 1)
    moon_ra, moon_dec = me.moon_ra_dec_cal(start_mjd, start_mjd, 1)

    # survey the whole sky
    ra_list = np.arange(0.125, 24, 0.25)
    dec_list = np.arange(-88.75, 90, 2.5)
    for src_dec in dec_list:
        src_dec = tu.angle_2_rad(src_dec, 0, 0)
        for src_ra in ra_list:
            src_ra = tt.time_2_rad(src_ra, 0, 0)
            num1 = 0  # sta和vlbi能观测到source的计数
            # test vlbi station
            for i in pos_mat_vlbi:  # get vlbi station
                longitude, latitude, height = tc.itrf_2_geographic(i[1], i[2], i[3])
                visibility = mo.obs_judge_active_vlbi_station(src_ra, src_dec, start_mjd, longitude, latitude, vb)
                if visibility:
                    num1 = num1 + 1
            # test satellite
            # for j in pos_mat_sat:
            #     SatX, SatY, SatZ, VSatX, VSatY, VSatZ = ms.kepler_2_cartesian(j[1], j[2], j[3], j[4], j[5], j[6])
            #     visibility = mo.obs_satellite_to_source(src_ra, SouDEC, start_mjd, SatX, SatY, SatZ)
            #     if visibility:
            #         num1 = num1 + 1
            num_array.append(num1)

    num_array = np.array(num_array)
    num_array.shape = len(dec_list), len(ra_list)

    return (sun_ra, sun_dec), (moon_ra, moon_dec), num_array  # 太阳赤经，赤纬，月球赤经赤纬，可见望远镜数


def test_sky_survey():
    start_time = tt.time_2_mjd(lc.StartTimeGlobalYear, lc.StartTimeGlobalMonth,
                               lc.StartTimeGlobalDay, lc.StartTimeGlobalHour,
                               lc.StartTimeGlobalMinute, lc.StartTimeGlobalSecond, 0)
    pos_sun, pos_moon, num_array = func_sky_survey(start_time, lc.pos_mat_vlbi, lc.pos_mat_sat)
    # 图像是，[0, 96], [0,72], 需要转化坐标
    print(pos_moon, pos_sun)
    img_sun_ra = pos_sun[0][0] * (96/24)
    img_moon_ra = pos_moon[0][0] * (96 / 24)
    img_sun_dec = pos_sun[1][0] * 0.4 + 36
    img_moon_dec = pos_moon[1][0] * 0.4 + 36
    # draw soon, moon
    plt.plot(img_sun_ra, img_sun_dec, color='red', marker='o', markerfacecolor=(1, 0, 0), alpha=1, markersize=20)
    plt.plot(img_moon_ra, img_moon_dec, color='blue', marker='o', markerfacecolor='w', alpha=1, markersize=10)
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
    # plt.show()


if __name__ == "__main__":
    plt.figure(num=1)
    test_ae_el()
    plt.figure(num=2)
    test_sky_survey()
    plt.show()
