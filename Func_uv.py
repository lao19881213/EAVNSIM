"""
@functions: basic uv coverage and sky coverage
@author: Zhen ZHAO
@date: May 2, 2018
"""

import load_conf as lc
import trans_time as tt
import trans_coordinate as tc
import trans_unit as tu
import model_effect as me
import model_satellite as ms
import model_obs_ability as mo
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def func_all_year_uv(start_mjd, time_step, pos_src, pos_mat_sat, pos_mat_vlbi, pos_mat_telemetry,
                     obs_freq, baseline_type, flag_unit, cutoff_mode, precession_mode):
    # result storing
    result_time_u = []
    result_time_v = []

    # generated 12 all year time, and calculate u,v
    date = tt.mjd_2_time(start_mjd)
    year = date[1]
    month = date[2]
    # print("year", year, "month", month)
    for _ in range(0, 12):
        # generate time
        if month > 13:
            year += 1
            month -= 12
            temp_start = tt.time_2_mjd(year, month, 1, 0, 0, 0, 0)
            temp_end = tt.time_2_mjd(year, month, 2, 0, 0, 0, 0)
        else:
            temp_start = tt.time_2_mjd(year, month, 1, 0, 0, 0, 0)
            temp_end = tt.time_2_mjd(year, month, 2, 0, 0, 0, 0)
        month += 1
        # print("temp_start", temp_start, "temp_end", temp_end)
        # invoke u,v
        dict_u, dict_v, dict_bl_sta1, dict_bl_sta2, dict_bl_duration \
            = func_uv(temp_start, temp_end, time_step, pos_src, pos_mat_sat, pos_mat_vlbi,
                      pos_mat_telemetry, obs_freq, baseline_type, flag_unit, cutoff_mode, precession_mode)
        # print("dict_u=", dict_u)
        # record u,v
        temp_u = []
        for each in dict_u.values():
            if each is not None:
                temp_u.extend(each)
        temp_v = []
        for each in dict_v.values():
            if each is not None:
                temp_v.extend(each)
        result_time_u.append(temp_u)
        result_time_v.append(temp_v)
    # calculate max {u,v}
    max_range = 1
    for k in range(0, len(result_time_u)):
        if len(result_time_u[k]) > 0 and len(result_time_v[k]) > 0:
            temp1 = np.max(np.abs(result_time_u[k]))
            temp2 = np.max(np.abs(result_time_v[k]))
            temp = max(temp1, temp2)
            if max_range < temp:
                max_range = temp
    # return
    return result_time_u, result_time_v, max_range * 1.3


def func_all_sky_uv(start_mjd, stop_mjd, time_step, pos_mat_sat, pos_mat_vlbi,
                    pos_mat_telemetry, obs_freq, baseline_type, flag_unit, cutoff_mode, precession_mode):
    # dict_mat_u = {"gg": None, "gs": None, "ss": None}
    # dict_mat_v = {"gg": None, "gs": None, "ss": None}
    result_mat_u = []
    result_mat_v = []
    for i in (2, 6, 10, 14, 18, 22):  # dra
        for j in (-60, -30, 0, 30, 60):  # dra
            ra = tt.time_2_rad(i, 0, 0)
            dec = tu.angle_2_rad(j, 0, 0)
            # print(ra, dec)
            pos_src = ['source-scan', ra, dec]
            dict_u, dict_v, dict_bl_sta1, dict_bl_sta2, dict_bl_duration \
                = func_uv(start_mjd, stop_mjd, time_step, pos_src, pos_mat_sat, pos_mat_vlbi,
                          pos_mat_telemetry, obs_freq, baseline_type, flag_unit, cutoff_mode, precession_mode)
            temp_u = []
            for each in dict_u.values():
                if each is not None:
                    temp_u.extend(each)
            temp_v = []
            for each in dict_v.values():
                if each is not None:
                    temp_v.extend(each)
            # if temp_v != [] and temp_u != []:
            #     result_mat_u.append(temp_u)
            #     result_mat_v.append(temp_v)
            result_mat_u.append(temp_u)
            result_mat_v.append(temp_v)

    max_range = 1
    for k in range(0, len(result_mat_u)):
        if len(result_mat_u[k]) > 0 and len(result_mat_v[k]) > 0:
            temp1 = np.max(np.abs(result_mat_u[k]))
            temp2 = np.max(np.abs(result_mat_v[k]))
            temp = max(temp1, temp2)
            if max_range < temp:
                max_range = temp

    return result_mat_u, result_mat_v, max_range


def func_uv(start_mjd, stop_mjd, time_step, pos_src, pos_mat_sat, pos_mat_vlbi, pos_mat_telemetry,
            obs_freq, baseline_type, flag_unit, cutoff_mode, precession_mode):
    # station info (lst_ground, lst_space)
    lst_ground = pos_mat_vlbi  # 将地面站看作是VLBI站
    lst_space = []
    for i in np.arange(len(pos_mat_sat)):
        if type(pos_mat_sat[i][7]) == str:
            # 将远地点和近地点数值转换成半长轴和离心率
            pos_mat_sat[i][1], pos_mat_sat[i][2] = ms.semi_axis_cal(pos_mat_sat[i][1], pos_mat_sat[i][2])
            pos_mat_sat[i][7] = tt.time_str_2_mjd(pos_mat_sat[i][7])
        # 卫星名称，半长轴，偏心率
        lst_space.append([pos_mat_sat[i][0], pos_mat_sat[i][1], pos_mat_sat[i][2]])

    # source info (src_ra, src_dec)
    if type(pos_src[1]) == str:
        src_ra = tt.time_str_2_rad(pos_src[1])
    else:
        src_ra = pos_src[1]

    if type(pos_src[2]) == str:
        src_dec = tu.angle_str_2_rad(pos_src[2])
    else:
        src_dec = pos_src[2]

    # observation info (obs_wlen, max_baseline, max_range)
    obs_wlen = tu.freq_2_wavelength(obs_freq)
    max_baseline = get_max_baseline(lst_ground, lst_space, baseline_type)
    if flag_unit == 0:
        max_range = max_baseline*1000 / obs_wlen
    else:
        max_range = max_baseline * 1000
    # results
    dict_u = {"gg": None, "gs": None, "ss": None}
    dict_v = {"gg": None, "gs": None, "ss": None}
    dict_bl_sta1 = {"gg": None, "gs": None, "ss": None}
    dict_bl_sta2 = {"gg": None, "gs": None, "ss": None}
    dict_bl_duration = {"gg": None, "gs": None, "ss": None}

    # according to the baseline type, calculate the corresponding uv coverage
    if (baseline_type & 1) != 0:  # ground to ground
        lst_u, lst_v, bl_sta1_name, bl_sta2_name, bl_duration \
            = func_uv_gg(start_mjd, stop_mjd, time_step, src_ra, src_dec,
                         pos_mat_vlbi, obs_wlen, cutoff_mode)
        dict_u["gg"] = lst_u
        dict_v["gg"] = lst_v
        dict_bl_sta1["gg"] = bl_sta1_name
        dict_bl_sta2["gg"] = bl_sta2_name
        dict_bl_duration["gg"] = bl_duration
    if (baseline_type & 2) != 0:  # ground to ground
        pass
    if (baseline_type & 4) != 0:  # ground to ground
        pass

    return dict_u, dict_v, dict_bl_sta1, dict_bl_sta2, dict_bl_duration


def func_uv_gg(start_mjd, stop_mjd, time_step, src_ra, src_dec, pos_mat_vlbi, obs_wlen, cutoff_mode):
    # define output
    lst_u = []
    lst_v = []
    lst_w = []
    baseline_sta1_name = []  # 一条地地基线对应的两个站名
    baseline_sta2_name = []
    baseline_duration = []  # 基线存在的时间

    # traverse all the time period
    for timestamp in np.arange(start_mjd, stop_mjd, time_step):
        active_station = mo.obs_all_active_vlbi(timestamp, src_ra, src_dec, pos_mat_vlbi, cutoff_mode)
        uv_matrix = tc.trans_matrix_uv_itrf(timestamp, src_ra, src_dec)
        # traverse all the combinations of ground stations
        for i in np.arange(len(pos_mat_vlbi)):
            for j in np.arange(i + 1, len(pos_mat_vlbi)):
                if active_station[2 * i + 1] is True and active_station[2 * j + 1] is True:
                    sta1_pos = pos_mat_vlbi[i][1:4]
                    sta2_pos = pos_mat_vlbi[j][1:4]
                    u, v, w = get_uv_coordination(uv_matrix, sta1_pos, sta2_pos, obs_wlen, 1)  # 单位为m
                    lst_u.extend([u, -u])
                    lst_v.extend([v, -v])
                    lst_w.extend([w, -w])
                    baseline_sta1_name.extend([pos_mat_vlbi[i][0]])
                    baseline_sta2_name.extend([pos_mat_vlbi[j][0]])
                    baseline_duration.extend([timestamp])

    # return the value
    return lst_u, lst_v, baseline_sta1_name, baseline_sta2_name, baseline_duration


def func_uv_gs(start_mjd, stop_mjd, time_step, src_ra, src_dec, pos_mat_sat, pos_mat_telemetry,
               pos_mat_vlbi, obs_freq, flag_unit, cutoff_mode, precession_mode):
    pass


def func_uv_ss(start_mjd, stop_mjd, time_step, src_ra, src_dec, pos_mat_sat, pos_mat_telemetry,
               obs_freq, flag_unit, cutoff_mode, precession_mode):
    pass


def get_uv_coordination(mat_uv, pos_sta1, pos_sta2, obs_wlen, flag_unit):
    d = np.array(pos_sta1) - np.array(pos_sta2)
    dtran = np.array([d])
    uvc = np.dot(mat_uv, dtran.T)
    if flag_unit == 0:
        return uvc[0][0] * 1000 / obs_wlen, uvc[1][0] * 1000 / obs_wlen, uvc[2][
            0] * 1000 / obs_wlen
    else:
        return uvc[0][0] * 1000, uvc[1][0] * 1000, uvc[2][0] * 1000


# 计算最大基线长度
# whichbaselines:0001 -> GroundToGround (1)
#               0010 -> GroundToSpace (2)
#               0100 -> SpaceToSpace(4)

#               0011 -> GroundToGround, GroundToSpace(3)
#               0110 -> GroundToSpace, SpaceToSpace,(6)
#               0111 -> GroundToGround,GroundToSpace,SpaceToSpace(7)
def get_max_baseline(lst_ground, lst_space, baseline_type):
    max_baseline = 0
    if (baseline_type & 1) != 0:
        for i in np.arange(len(lst_ground)):
            for j in np.arange(i + 1, len(lst_ground)):
                delta_x = lst_ground[i][1] - lst_ground[j][1]
                delta_y = lst_ground[i][2] - lst_ground[j][2]
                delta_z = lst_ground[i][3] - lst_ground[j][3]
                distance = delta_x ** 2 + delta_y ** 2 + delta_z ** 2
                baseline = np.sqrt(distance)
                if max_baseline < baseline:
                    max_baseline = baseline

    if (baseline_type & 2) != 0:
        for m in np.arange(len(lst_space)):
            baseline = lst_space[m][1] * (1 + lst_space[m][2])
            if baseline > max_baseline:
                max_baseline = baseline
        max_baseline = max_baseline + lc.earth_radius

    elif (baseline_type & 4) != 0:
        max_apogee = lc.earth_radius  # 卫星的最大远地点距离
        second_max_apogee = 0
        for m in np.arange(len(lst_space)):
            temp = lst_space[m][1] * (1 + lst_space[m][2])  # 半长轴 偏心率
            if temp > max_apogee:
                second_max_apogee = max_apogee
                max_apogee = temp
            elif temp > second_max_apogee:
                second_max_apogee = temp
        max_baseline = max_apogee + second_max_apogee

    return max_baseline


def test_single_uv():
    # load data from configurations
    start_time = tt.time_2_mjd(lc.StartTimeGlobalYear, lc.StartTimeGlobalMonth,
                               lc.StartTimeGlobalDay, lc.StartTimeGlobalHour,
                               lc.StartTimeGlobalMinute, lc.StartTimeGlobalSecond, 0)
    stop_time = tt.time_2_mjd(lc.StopTimeGlobalYear, lc.StopTimeGlobalMonth,
                              lc.StopTimeGlobalDay, lc.StopTimeGlobalHour,
                              lc.StopTimeGlobalMinute, lc.StopTimeGlobalSecond, 0)
    time_step = tt.time_2_day(lc.TimeStepGlobalDay, lc.TimeStepGlobalHour, lc.TimeStepGlobalMinute,
                              lc.TimeStepGlobalSecond)
    # invoke the calculation functions
    dict_u, dict_v, dict_bl_sta1, dict_bl_sta2, dict_bl_duration = \
        func_uv(start_time, stop_time, time_step, lc.pos_mat_src[0], lc.pos_mat_sat, lc.pos_mat_vlbi,
                lc.pos_mat_telemetry, lc.obs_freq, lc.baseline_type, lc.unit_flag, lc.cutoff_mode, lc.precession_mode)
    # plot it
    plt.figure(1)
    max_u = 0
    max_v = 0
    if (lc.baseline_type & 1) != 0:
        x = dict_u["gg"]
        y = dict_v["gg"]
        if x is not None and y is not None:
            x = np.array(x)
            y = np.array(y)
            plt.scatter(x, y, s=5, marker='.', color='b')
            max_u = np.max(list(np.abs(x)))
            max_v = np.max(list(np.abs(y)))

    if (lc.baseline_type & 2) != 0:
        x = dict_u["gs"]
        y = dict_v["gs"]
        if x is not None and y is not None:
            x = np.array(x)
            y = np.array(y)
            plt.scatter(x, y)
            max_u = np.max(list(np.abs(x)))
            max_v = np.max(list(np.abs(y)))

    if (lc.baseline_type & 4) != 0:
        x = dict_u["ss"]
        y = dict_v["ss"]
        if x is not None and y is not None:
            x = np.array(x)
            y = np.array(y)
            plt.scatter(x, y)
            max_u = np.max(list(np.abs(x)))
            max_v = np.max(list(np.abs(y)))

    # set the axis
    if max_u > 0 or max_v > 0:
        max_range = 1.3 * max(max_u, max_v)
        plt.xlim(-max_range, +max_range)
        plt.ylim(-max_range, +max_range)
    plt.title("UV Plot")
    if lc.unit_flag == 0:
        plt.xlabel("u$(\lambda)$")
        plt.ylabel("v$(\lambda)$")
    else:
        plt.xlabel("u$(m)$")
        plt.ylabel("v$(m)$")
    plt.grid()


def test_sky_uv():
    # load data from configurations
    start_time = tt.time_2_mjd(lc.StartTimeGlobalYear, lc.StartTimeGlobalMonth,
                               lc.StartTimeGlobalDay, lc.StartTimeGlobalHour,
                               lc.StartTimeGlobalMinute, lc.StartTimeGlobalSecond, 0)
    stop_time = tt.time_2_mjd(lc.StopTimeGlobalYear, lc.StopTimeGlobalMonth,
                              lc.StopTimeGlobalDay, lc.StopTimeGlobalHour,
                              lc.StopTimeGlobalMinute, lc.StopTimeGlobalSecond, 0)
    time_step = tt.time_2_day(lc.TimeStepGlobalDay, lc.TimeStepGlobalHour, lc.TimeStepGlobalMinute,
                              lc.TimeStepGlobalSecond)
    result_mat_u, result_mat_v, max_range = func_all_sky_uv(start_time, stop_time, time_step, lc.pos_mat_sat,
                                                            lc.pos_mat_vlbi, lc.pos_mat_telemetry, lc.obs_freq,
                                                            lc.baseline_type, lc.unit_flag, lc.cutoff_mode,
                                                            lc.precession_mode)
    # correctly obtained results
    if len(result_mat_u) == 0 or len(result_mat_v) == 0:
        return
    # print("u:", result_mat_u[0:10], "\nv:", result_mat_v[0:10])
    k = 0
    # print(len(result_mat_u))
    for i in (2, 6, 10, 14, 18, 22):
        for j in (-60, -30, 0, 30, 60):
            if len(result_mat_u[k]) > 0 and len(result_mat_v[k]) > 0:
                temp_u = np.array(result_mat_u[k]) / max_range
                temp_v = np.array(result_mat_v[k]) / max_range * 10
                temp_u += i
                temp_v += j
                plt.scatter(temp_u, temp_v, s=3, marker='.', color='b')
            k += 1
    # plot sun position
    sun_ra, sun_dec = me.sun_ra_dec_cal(start_time, stop_time, time_step)
    plt.plot(np.array(sun_ra), np.array(sun_dec), '.k', linewidth=2)
    plt.plot(sun_ra[0], sun_dec[0], 'or', alpha=0.5, markersize=20)
    # ticks
    plt.title("All Sky UV Plot")
    plt.xlabel(r"Ra($H$)")
    plt.ylabel(r'Dec ($^\circ$)')
    plt.xticks([0, 2, 6, 10, 14, 18, 22, 24])
    plt.yticks([-90, -60, -30, 0, 30, 60, 90])
    plt.xlim(0, 24)
    plt.ylim(-90, +90)
    plt.grid()


def test_time_uv():
    # load data from configurations
    start_time = tt.time_2_mjd(lc.StartTimeGlobalYear, lc.StartTimeGlobalMonth,
                               lc.StartTimeGlobalDay, lc.StartTimeGlobalHour,
                               lc.StartTimeGlobalMinute, lc.StartTimeGlobalSecond, 0)
    time_step = tt.time_2_day(lc.TimeStepGlobalDay, lc.TimeStepGlobalHour, lc.TimeStepGlobalMinute,
                              lc.TimeStepGlobalSecond)

    result_time_u, result_time_v, max_range = func_all_year_uv(start_time, time_step, lc.pos_mat_src[0], lc.pos_mat_sat,
                                                               lc.pos_mat_vlbi, lc.pos_mat_telemetry, lc.obs_freq,
                                                               lc.baseline_type, lc.unit_flag, lc.cutoff_mode,
                                                               lc.precession_mode)
    print(result_time_u, result_time_v, max_range)
    # correctly obtained results
    if len(result_time_u) == 0 or len(result_time_v) == 0:
        return
    # 横着有3个是6分, 纵轴有4个是8分, 所以取最小公倍数, 24
    k = 0
    for irow in (21, 15, 9, 3): # 24份对应的画点的位置
        for icol in (20, 12, 4):
            if len(result_time_u[k]) > 0 and len(result_time_v[k]) > 0:
                temp_u = result_time_u[k] / max_range * 4
                temp_v = result_time_v[k] / max_range * 3
                temp_u += icol
                temp_v += irow
                plt.scatter(temp_u, temp_v, s=3, marker='.', color='b')
            k += 1
    plt.title("All Year Round UV Plot")
    plt.xlim(0, 24)
    plt.ylim(0, 24)
    plt.xticks([4, 12, 20], [1, 2, 3])
    plt.yticks([3, 9, 15, 21], [4, 3, 2, 1])
    plt.xlabel(r"$i_{th}$\ month")
    plt.ylabel(r"Quarter")
    plt.grid()


if __name__ == "__main__":
    plt.figure(num=1)
    test_single_uv()
    plt.figure(num=2)
    test_sky_uv()
    plt.figure(num=3)
    test_time_uv()
    plt.show()
