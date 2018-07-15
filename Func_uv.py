"""
@functions: basic uv coverage and sky coverage
@author: Zhen ZHAO
@date: May 2, 2018
"""

import load_conf as lc
import utility as ut
import model_effect as me
import model_satellite as ms
import model_obs_ability as mo
import numpy as np
import matplotlib.pyplot as plt


class FuncUv(object):
    def __init__(self, start_t, stop_t, step_t, p_main_src, p_multi_src, p_sat, p_vlbi, p_tele,
                 freq, bl_type, f_unit, cutoff_type, precession_type):
        # 1. input parameter
        self.start_mjd = start_t
        self.stop_mjd = stop_t
        self.time_step = step_t

        self.pos_src = p_main_src
        self.pos_multi_src = p_multi_src

        self.pos_mat_sat = p_sat
        self.pos_mat_vlbi = p_vlbi
        self.pos_mat_telemetry = p_tele
        self.obs_freq = freq
        self.baseline_type = bl_type
        self.flag_unit = f_unit  # 0->lambda, 1->km
        self.cutoff_mode = cutoff_type
        self.precession_mode = precession_type

        # obtain/sparse all the srcs
        self.src_num = len(self.pos_multi_src)
        for i in range(self.src_num):
            tmp_ra = self.pos_multi_src[i][1]
            tmp_dec = self.pos_multi_src[i][2]
            # print(self.pos_multi_src[i][1])
            if type(tmp_ra) == str:
                self.pos_multi_src[i][1] = ut.time_str_2_rad(tmp_ra)
            if type(tmp_dec) == str:
                self.pos_multi_src[i][2] = ut.angle_str_2_rad(tmp_dec)

        # 2. store old/the very first result
        self.result_u = []
        self.result_v = []
        self.max_range_single_uv = 1

        # 3. all year uv result
        self.result_time_u = []
        self.result_time_v = []
        self.max_range_time_uv = 1

        # 4. all sky uv result
        self.result_sky_u = []
        self.result_sky_v = []
        self.max_range_sky_uv = 1

        # 5. multiple src results
        self.result_multi_src_name = []
        self.result_multi_src_u = []
        self.result_multi_src_v = []
        self.max_range_multi_src = 1

        self._ini_para()

    def _ini_para(self):
        # 2. functional variables
        # 2.1 station info (self.lst_ground, self.lst_space)
        self.lst_ground = self.pos_mat_vlbi  # 将地面站看作是VLBI站
        self.lst_space = []
        for i in np.arange(len(self.pos_mat_sat)):
            if type(self.pos_mat_sat[i][7]) == str:
                # 将远地点和近地点数值转换成半长轴和离心率
                self.pos_mat_sat[i][1], self.pos_mat_sat[i][2] = ms.semi_axis_cal(self.pos_mat_sat[i][1],
                                                                                  self.pos_mat_sat[i][2])
                self.pos_mat_sat[i][7] = ut.time_str_2_mjd(self.pos_mat_sat[i][7])
            # 卫星名称，半长轴，偏心率
            self.lst_space.append([self.pos_mat_sat[i][0], self.pos_mat_sat[i][1], self.pos_mat_sat[i][2]])

        # 2.2 source info (self.src_ra, self.src_dec)
        if type(self.pos_src[1]) == str:
            self.src_ra = ut.time_str_2_rad(self.pos_src[1])
            # print(self.pos_src[1], self.src_ra)
        else:
            self.src_ra = self.pos_src[1]

        if type(self.pos_src[2]) == str:
            self.src_dec = ut.angle_str_2_rad(self.pos_src[2])
            # print(self.pos_src[2], self.src_dec)
        else:
            self.src_dec = self.pos_src[2]

        # 2.3 observation info (obs_wlen, max_baseline, max_range)
        self.obs_wlen = ut.freq_2_wavelength(self.obs_freq)
        max_baseline = self._get_max_baseline()
        if self.flag_unit == 0:
            self.max_range_single_uv = max_baseline * 1000 / self.obs_wlen
        else:
            self.max_range_single_uv = max_baseline * 1000

        # 3. temp single uv result
        self.dict_u = {"gg": None, "gs": None, "ss": None}
        self.dict_v = {"gg": None, "gs": None, "ss": None}
        self.dict_bl_sta1 = {"gg": None, "gs": None, "ss": None}
        self.dict_bl_sta2 = {"gg": None, "gs": None, "ss": None}
        self.dict_bl_duration = {"gg": None, "gs": None, "ss": None}

        self.result_tmp_u = []
        self.result_tmp_v = []
        self.max_range_tmp = 1

    # 1. multiple srcs
    def get_result_multi_src_with_update(self):
        self._func_multi_source_uv()
        return self.result_multi_src_name, self.result_multi_src_u, self.result_multi_src_v, self.max_range_multi_src

    def _func_multi_source_uv(self):
        for i in range(self.src_num):
            tmp_name = self.pos_multi_src[i][0]
            tmp_ra = self.pos_multi_src[i][1]
            tmp_dec = self.pos_multi_src[i][2]

            tmp_src = self.pos_src
            temp_u, temp_v, temp_max = self._get_reset_source_info([tmp_name, tmp_ra, tmp_dec])
            self.pos_src = tmp_src

            self.result_multi_src_name.append(tmp_name)
            self.result_multi_src_u.append(temp_u)
            self.result_multi_src_v.append(temp_v)

            if self.max_range_multi_src < temp_max:
                self.max_range_multi_src = temp_max

    # for multiprocessing purpose (separate updating and getter)
    def update_result_multi_src(self):
        self._func_multi_source_uv()

    def get_result_multi_src(self):
        return self.result_multi_src_name, self.result_multi_src_u, self.result_multi_src_v, self.max_range_multi_src

    # 2. all sky uv
    def get_result_sky_uv_with_update(self):
        self._func_all_sky_uv()
        return self.result_sky_u, self.result_sky_v, self.max_range_sky_uv * 1.3

    def _func_all_sky_uv(self):
        for i in (2, 6, 10, 14, 18, 22):  # dra
            for j in (-60, -30, 0, 30, 60):  # dra
                ra = ut.time_2_rad(i, 0, 0)
                dec = ut.angle_2_rad(j, 0, 0)
                # print(ra, dec)
                pos_src = ['source-%d-%d' % (i, j), ra, dec]
                record_source = self.pos_src
                temp_u, temp_v, temp_max = self._get_reset_source_info(pos_src)
                self.pos_src = record_source
                self.result_sky_u.append(temp_u)
                self.result_sky_v.append(temp_v)
                # calculate max {u,v}
                if self.max_range_sky_uv < temp_max:
                    self.max_range_sky_uv = temp_max

    # for multiprocessing purpose (separate updating and getter)
    def update_result_sky_uv(self):
        self._func_all_sky_uv()

    def get_result_sky_uv(self):
        return self.result_sky_u, self.result_sky_v, self.max_range_sky_uv * 1.3

    # 3. all year round uv
    def get_result_year_uv_with_update(self):
        self._func_all_year_uv()
        # print(self.start_mjd, self.pos_src[0], self.max_range_time_uv, self.result_time_u[0])
        return self.result_time_u, self.result_time_v, self.max_range_time_uv * 1.3

    def _func_all_year_uv(self):
        # generated 12 all year time, and calculate u,v
        date = ut.mjd_2_time(self.start_mjd)
        year = date[1]
        month = date[2]

        for _ in range(0, 12):
            # generate time
            if month > 13:
                year += 1
                month -= 12
                temp_start = ut.time_2_mjd(year, month, 1, 0, 0, 0, 0)
                temp_end = ut.time_2_mjd(year, month, 2, 0, 0, 0, 0)
            else:
                temp_start = ut.time_2_mjd(year, month, 1, 0, 0, 0, 0)
                temp_end = ut.time_2_mjd(year, month, 2, 0, 0, 0, 0)
            month += 1

            temp_u, temp_v, temp_max = self._get_reset_time_info(temp_start, temp_end, self.time_step)

            self.result_time_u.append(temp_u)
            self.result_time_v.append(temp_v)
            # calculate max {u,v}
            if self.max_range_time_uv < temp_max:
                self.max_range_time_uv = temp_max

    # for multiprocessing purpose (separate updating and getter)
    def update_result_year_uv(self):
        self._func_all_year_uv()

    def get_result_year_uv(self):
        return self.result_time_u, self.result_time_v, self.max_range_time_uv * 1.3

    # 4. single uv function
    def get_result_single_uv_with_update(self):
        self._func_uv()
        self._parse_result_dict()
        self.result_u, self.result_v, self.max_range_single_uv = self._get_tmp_single_uv()
        return self.result_u, self.result_v, self.max_range_single_uv

    # for multiprocessing purpose (separate updating and getter)
    def update_result_single_uv(self):
        self._func_uv()
        self._parse_result_dict()
        self.result_u, self.result_v, self.max_range_single_uv = self._get_tmp_single_uv()

    def get_result_single_uv(self):
        return self.result_u, self.result_v, self.max_range_single_uv

    # other implementations
    def _get_reset_source_info(self, p_src):
        self.pos_src = p_src
        self._ini_para()

        return self._get_tmp_single_uv()

    def _get_reset_time_info(self, temp_start, temp_end, time_step):
        self.start_mjd = temp_start
        self.stop_mjd = temp_end
        self.time_step = time_step
        self._ini_para()

        return self._get_tmp_single_uv()

    def _get_tmp_single_uv(self):
        self._func_uv()
        self._parse_result_dict()
        return self.result_tmp_u, self.result_tmp_v, self.max_range_tmp

    def _parse_result_dict(self):
        # 1. u,v
        for each in self.dict_u.values():
            if each is not None:
                self.result_tmp_u.extend(each)

        for each in self.dict_v.values():
            if each is not None:
                self.result_tmp_v.extend(each)

        # 2. calculate max {u,v}
        if len(self.result_tmp_u) > 0 and len(self.result_tmp_v) > 0:
            temp1 = np.max(np.abs(self.result_tmp_u))
            temp2 = np.max(np.abs(self.result_tmp_v))
            temp = max(temp1, temp2)
            if self.max_range_tmp < temp:
                self.max_range_tmp = temp

    def _func_uv(self):
        # according to the baseline type, calculate the corresponding uv coverage
        if (self.baseline_type & 1) != 0:  # ground to ground
            lst_u, lst_v, bl_sta1_name, bl_sta2_name, bl_duration \
                = self._func_uv_gg()
            self.dict_u["gg"] = lst_u
            self.dict_v["gg"] = lst_v
            self.dict_bl_sta1["gg"] = bl_sta1_name
            self.dict_bl_sta2["gg"] = bl_sta2_name
            self.dict_bl_duration["gg"] = bl_duration

        if (self.baseline_type & 2) != 0:  # ground to ground
            lst_u, lst_v, bl_sta1_name, bl_sta2_name, bl_duration \
                = self._func_uv_gg()
            self.dict_u["gs"] = lst_u
            self.dict_v["gs"] = lst_v
            self.dict_bl_sta1["gs"] = bl_sta1_name
            self.dict_bl_sta2["gs"] = bl_sta2_name
            self.dict_bl_duration["gs"] = bl_duration

        if (self.baseline_type & 4) != 0:  # ground to ground
            lst_u, lst_v, bl_sta1_name, bl_sta2_name, bl_duration \
                = self._func_uv_gg()
            self.dict_u["ss"] = lst_u
            self.dict_v["ss"] = lst_v
            self.dict_bl_sta1["ss"] = bl_sta1_name
            self.dict_bl_sta2["ss"] = bl_sta2_name
            self.dict_bl_duration["ss"] = bl_duration

    def _get_uv_coordination(self, mat_uv, pos_sta1, pos_sta2):
        d = np.array(pos_sta1) - np.array(pos_sta2)
        dtran = np.array([d])
        uvc = np.dot(mat_uv, dtran.T)
        if self.flag_unit == 0:
            return uvc[0][0] * 1000 / self.obs_wlen, uvc[1][0] * 1000 / self.obs_wlen, uvc[2][0] * 1000 / self.obs_wlen
        else:
            return uvc[0][0] * 1000, uvc[1][0] * 1000, uvc[2][0] * 1000

    def _get_max_baseline(self):
        max_baseline = 0
        lst_ground = self.lst_ground
        lst_space = self.lst_space
        if (self.baseline_type & 1) != 0:
            for i in np.arange(len(lst_ground)):
                for j in np.arange(i + 1, len(lst_ground)):
                    delta_x = lst_ground[i][1] - lst_ground[j][1]
                    delta_y = lst_ground[i][2] - lst_ground[j][2]
                    delta_z = lst_ground[i][3] - lst_ground[j][3]
                    distance = delta_x ** 2 + delta_y ** 2 + delta_z ** 2
                    baseline = np.sqrt(distance)
                    if max_baseline < baseline:
                        max_baseline = baseline

        if (self.baseline_type & 2) != 0:
            for m in range(len(lst_space)):
                baseline = lst_space[m][1] * (1 + lst_space[m][2])
                if baseline > max_baseline:
                    max_baseline = baseline
            max_baseline = max_baseline + lc.earth_radius

        elif (self.baseline_type & 4) != 0:
            max_apogee = lc.earth_radius  # 卫星的最大远地点距离
            second_max_apogee = 0
            for m in range(len(lst_space)):
                temp = lst_space[m][1] * (1 + lst_space[m][2])  # 半长轴 偏心率
                if temp > max_apogee:
                    second_max_apogee = max_apogee
                    max_apogee = temp
                elif temp > second_max_apogee:
                    second_max_apogee = temp
            max_baseline = max_apogee + second_max_apogee

        return max_baseline

    def _func_uv_gg(self):
        # define output
        lst_u = []
        lst_v = []
        lst_w = []
        baseline_sta1_name = []  # 一条地地基线对应的两个站名
        baseline_sta2_name = []
        baseline_duration = []  # 基线存在的时间

        # traverse all the time period
        for timestamp in np.arange(self.start_mjd, self.stop_mjd, self.time_step):
            active_station = mo.obs_all_active_vlbi(timestamp, self.src_ra, self.src_dec, self.pos_mat_vlbi,
                                                    self.cutoff_mode)
            uv_matrix = ut.trans_matrix_uv_itrf(timestamp, self.src_ra, self.src_dec)
            # traverse all the combinations of ground stations
            for i in np.arange(len(self.pos_mat_vlbi)):
                for j in np.arange(i + 1, len(self.pos_mat_vlbi)):
                    if active_station[2 * i + 1] is True and active_station[2 * j + 1] is True:
                        sta1_pos = self.pos_mat_vlbi[i][1:4]
                        sta2_pos = self.pos_mat_vlbi[j][1:4]
                        u, v, w = self._get_uv_coordination(uv_matrix, sta1_pos, sta2_pos)  # 单位为m
                        u /= 1000
                        v /= 1000
                        lst_u.extend([u, -u])
                        lst_v.extend([-v, v])
                        lst_w.extend([w, -w])
                        baseline_sta1_name.extend([self.pos_mat_vlbi[i][0]])
                        baseline_sta2_name.extend([self.pos_mat_vlbi[j][0]])
                        baseline_duration.extend([timestamp])

        # return the value
        return lst_u, lst_v, baseline_sta1_name, baseline_sta2_name, baseline_duration

    def _func_uv_gs(self, start_mjd, stop_mjd, time_step, src_ra, src_dec, pos_mat_sat, pos_mat_telemetry,
                    pos_mat_vlbi, obs_freq, flag_unit, cutoff_mode, precession_mode):
        if len(self.lst_space) < 1:
            return None, None, None, None, None

        lst_u, lst_v, baseline_sta1_name, baseline_sta2_name, baseline_duration = \
            None, None, None, None, None
        # return the value
        return lst_u, lst_v, baseline_sta1_name, baseline_sta2_name, baseline_duration

    def _func_uv_ss(self, start_mjd, stop_mjd, time_step, src_ra, src_dec, pos_mat_sat, pos_mat_telemetry,
                    obs_freq, flag_unit, cutoff_mode, precession_mode):
        if len(self.lst_space) < 2:
            return None, None, None, None, None

        lst_u, lst_v, baseline_sta1_name, baseline_sta2_name, baseline_duration = \
            None, None, None, None, None
        # return the value
        return lst_u, lst_v, baseline_sta1_name, baseline_sta2_name, baseline_duration


class FuncYearUv(object):

    def __init__(self, start_t, step_t, p_src, p_sat, p_vlbi, p_tele,
                 freq, bl_type, f_unit, cutoff_type, precession_type):
        # # input parameters
        # self.start_mjd = start_t
        # self.time_step = step_t
        # self.pos_src = p_src
        # self.pos_mat_sat = p_sat
        # self.pos_mat_vlbi = p_vlbi
        # self.pos_mat_telemetry = p_tele
        # self.obs_freq = freq
        # self.baseline_type = bl_type
        # self.flag_unit = f_unit
        # self.cutoff_mode = cutoff_type
        # self.precession_mode = precession_type

        # create FuncUV
        self.myFuncUv = FuncUv(start_t, 0, step_t, p_src, p_src, p_sat, p_vlbi, p_tele,
                               freq, bl_type, f_unit, cutoff_type, precession_type)

        # result
        self.result_time_u = []
        self.result_time_v = []
        self.max_range = 1

    def _func_all_year_uv(self):
        # generated 12 all year time, and calculate u,v
        date = ut.mjd_2_time(self.myFuncUv.start_mjd)
        year = date[1]
        month = date[2]
        # print("year", year, "month", month)
        for _ in range(0, 12):
            # generate time
            if month > 13:
                year += 1
                month -= 12
                temp_start = ut.time_2_mjd(year, month, 1, 0, 0, 0, 0)
                temp_end = ut.time_2_mjd(year, month, 2, 0, 0, 0, 0)
            else:
                temp_start = ut.time_2_mjd(year, month, 1, 0, 0, 0, 0)
                temp_end = ut.time_2_mjd(year, month, 2, 0, 0, 0, 0)
            month += 1
            # print("temp_start", temp_start, "temp_end", temp_end)
            # invoke u,v
            temp_u, temp_v, temp_max = self.myFuncUv._get_reset_time_info(temp_start, temp_end, self.myFuncUv.time_step)

            self.result_time_u.append(temp_u)
            self.result_time_v.append(temp_v)
            # calculate max {u,v}
            if self.max_range < temp_max:
                self.max_range = temp_max

    def get_result_year_uv(self):
        self._func_all_year_uv()
        return self.result_time_u, self.result_time_v, self.max_range * 1.3


class FuncSrcUv(object):
    def __init__(self, start_t, stop_t, step_t, p_mat_src, p_sat, p_vlbi, p_tele,
                 freq, bl_type, f_unit, cutoff_type, precession_type):
        # 1. input parameter
        self.pos_multi_src = p_mat_src

        self.start_mjd = start_t
        self.stop_mjd = stop_t
        self.time_step = step_t
        self.pos_mat_sat = p_sat
        self.pos_mat_vlbi = p_vlbi
        self.pos_mat_telemetry = p_tele
        self.obs_freq = freq
        self.baseline_type = bl_type
        self.flag_unit = f_unit  # 0->lambda, 1->km
        self.cutoff_mode = cutoff_type
        self.precession_mode = precession_type

        # 2. invoke func UV

    def func_multi_source_uv(self):
        pass

    def get_result_multi_src_uv(self):
        pass


# 计算最大基线长度
# whichbaselines:0001 -> GroundToGround (1)
#               0010 -> GroundToSpace (2)
#               0100 -> SpaceToSpace(4)
#               0011 -> GroundToGround, GroundToSpace(3)
#               0110 -> GroundToSpace, SpaceToSpace,(6)
#               0111 -> GroundToGround,GroundToSpace,SpaceToSpace(7)


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
    # invoke the calculation functions
    myFuncUV = FuncUv(start_time, stop_time, time_step, lc.pos_mat_src[0], lc.pos_mat_src, lc.pos_mat_sat,
                      lc.pos_mat_vlbi, lc.pos_mat_telemetry, lc.obs_freq, lc.baseline_type, lc.unit_flag,
                      lc.cutoff_mode, lc.precession_mode)

    # single uv
    x, y, max_xy = myFuncUV.get_result_single_uv_with_update()
    plt.figure(1)
    plt.subplot(111, aspect='equal')
    if x is not None and y is not None:
        x = np.array(x)
        y = np.array(y)
        max_range = max_xy * 1.3
        plt.scatter(x, y, s=1, marker='.', color='brown')
        plt.xlim([-12500, 12500])
        plt.ylim([-12500, 12500])
        # plt.xlim(-max_range, +max_range)
        # plt.ylim(-max_range, +max_range)
        plt.title("UV Plot")
        if lc.unit_flag == 0:
            plt.xlabel("u$(\lambda)$")
            plt.ylabel("v$(\lambda)$")
        else:
            plt.xlabel("u$(km)$")
            plt.ylabel("v$(km)$")

        ax = plt.gca()  # 获取当前图像的坐标轴信息
        # ax.get_major_formatter().set_powerlimits((0, 1))  # 将坐标轴的base number设置为一位。
        # ax.get_xaxis().get_major_formatter().set_powerlimits((0, 1))
        # ax.get_yaxis().get_major_formatter().set_powerlimits((0, 1))

        plt.grid()

    # # time uv
    # result_time_u, result_time_v, max_range = myFuncUV.get_result_year_uv_with_update()
    # # print(max_range, result_time_u)
    # plt.figure(2)
    # if len(result_time_u) != 0 and len(result_time_v) != 0:
    #     # 横着有3个是6分, 纵轴有4个是8分, 所以取最小公倍数, 24
    #     k = 0
    #     for irow in (21, 15, 9, 3):  # 24份对应的画点的位置
    #         for icol in (20, 12, 4):
    #             if len(result_time_u[k]) > 0 and len(result_time_v[k]) > 0:
    #                 temp_u = result_time_u[k] / max_range * 4
    #                 temp_v = result_time_v[k] / max_range * 3
    #                 temp_u += icol
    #                 temp_v += irow
    #                 plt.scatter(temp_u, temp_v, s=3, marker='.', color='b')
    #             k += 1
    #     plt.title("All Year Round UV Plot")
    #     plt.xlim(0, 24)
    #     plt.ylim(0, 24)
    #     plt.xticks([4, 12, 20], [1, 2, 3])
    #     plt.yticks([3, 9, 15, 21], [4, 3, 2, 1])
    #     plt.xlabel(r"$i_{th}$\ month")
    #     plt.ylabel(r"Quarter")
    #     plt.grid()

    # # sky uv
    # result_mat_u, result_mat_v, max_range = myFuncUV.get_result_sky_uv_with_update()
    # plt.figure(3)
    # if len(result_mat_u) != 0 and len(result_mat_v) != 0:
    #     k = 0
    #     # print(len(result_mat_u))
    #     for i in (2, 6, 10, 14, 18, 22):
    #         for j in (-60, -30, 0, 30, 60):
    #             if len(result_mat_u[k]) > 0 and len(result_mat_v[k]) > 0:
    #                 temp_u = np.array(result_mat_u[k]) / max_range
    #                 temp_v = np.array(result_mat_v[k]) / max_range * 10
    #                 temp_u += i
    #                 temp_v += j
    #                 plt.scatter(temp_u, temp_v, s=3, marker='.', color='b')
    #             k += 1
    #
    #     # for i in (-60, -30, 0, 30, 60):
    #     #     for j in (2, 6, 10, 14, 18, 22):
    #     #         if len(result_mat_u[k]) > 0 and len(result_mat_v[k]) > 0:
    #     #             temp_u = np.array(result_mat_u[k]) / max_range
    #     #             temp_v = np.array(result_mat_v[k]) / max_range * 10
    #     #             temp_u += i
    #     #             temp_v += j
    #     #             plt.scatter(temp_u, temp_v, s=3, marker='.', color='b')
    #     #         k += 1
    #
    #     # plot sun position
    #     sun_ra, sun_dec = me.sun_ra_dec_cal(start_time, stop_time, time_step)
    #     plt.plot(np.array(sun_ra), np.array(sun_dec), '.k', linewidth=2)
    #     plt.plot(sun_ra[0], sun_dec[0], 'or', alpha=0.5, markersize=20)
    #     # ticks
    #     plt.title("All Sky UV Plot")
    #     plt.xlabel(r"Ra($H$)")
    #     plt.ylabel(r'Dec ($^\circ$)')
    #     plt.xticks([0, 2, 6, 10, 14, 18, 22, 24])
    #     plt.yticks([-90, -60, -30, 0, 30, 60, 90])
    #     plt.xlim(0, 24)
    #     plt.ylim(-90, +90)
    #
    #     # flip
    #     # plt.ylabel(r"Ra($H$)")
    #     # plt.xlabel(r'Dec ($^\circ$)')
    #     # plt.yticks([0, 2, 6, 10, 14, 18, 22, 24])
    #     # plt.xticks([-90, -60, -30, 0, 30, 60, 90])
    #     # plt.ylim(0, 24)
    #     # plt.xlim(-90, +90)
    #
    #     plt.grid()

    # multi src
    result_src_name, result_src_u, result_src_v, max_range = myFuncUV.get_result_multi_src_with_update()

    plt.figure(4)
    num_src = len(result_src_name)
    if num_src > 0:
        # num_col = int(np.ceil(np.sqrt(num_src)))
        # num_row = int(np.ceil(num_src/num_col))
        num_col = 4
        num_row = 7
        for k in range(num_src):
            # tmp_pos = '%d%d%d' % (num_row, num_col, k+1)
            # print(tmp_pos)
            # plt.subplot(tmp_pos)
            plt.subplot(num_row, num_col, k+1, aspect='equal')
            if len(result_src_u[k]) > 0 and len(result_src_v[k]) > 0:
                plt.scatter(result_src_u[k], result_src_v[k], s=1, marker='.', color='brown')
                plt.xlim([-max_range, max_range])
                plt.ylim([-max_range, max_range])
            plt.title(result_src_name[k])
        plt.xlabel('u (km)')
        plt.ylabel('v (km)')
        # plt.text(0,0, 'normalized by %f'%max_range)

        # plot_postion = [x for x in range(num_row * 2) if x % 2 == 1]
        # k = 0
        # for i in plot_postion:
        #     for j in plot_postion:
        #         if len(result_src_u[k]) > 0 and len(result_src_v[k]) > 0:
        #             temp_u = np.array(result_src_u[k]) / max_range
        #             temp_v = np.array(result_src_v[k]) / max_range * 10
        #             temp_u += i
        #             temp_v += j
        #             plt.scatter(temp_u, temp_v, s=3, marker='.', color='b')
        #         k += 1

    plt.show()


def test_new_all_sky():
    # load data from configurations
    start_time = ut.time_2_mjd(lc.StartTimeGlobalYear, lc.StartTimeGlobalMonth,
                               lc.StartTimeGlobalDay, lc.StartTimeGlobalHour,
                               lc.StartTimeGlobalMinute, lc.StartTimeGlobalSecond, 0)
    stop_time = ut.time_2_mjd(lc.StopTimeGlobalYear, lc.StopTimeGlobalMonth,
                              lc.StopTimeGlobalDay, lc.StopTimeGlobalHour,
                              lc.StopTimeGlobalMinute, lc.StopTimeGlobalSecond, 0)
    time_step = ut.time_2_day(lc.TimeStepGlobalDay, lc.TimeStepGlobalHour, lc.TimeStepGlobalMinute,
                              lc.TimeStepGlobalSecond)
    # invoke the calculation functions
    myFuncUV = FuncUv(start_time, stop_time, time_step, lc.pos_mat_src[0], lc.pos_mat_src, lc.pos_mat_sat,
                      lc.pos_mat_vlbi, lc.pos_mat_telemetry, lc.obs_freq, lc.baseline_type, lc.unit_flag,
                      lc.cutoff_mode, lc.precession_mode)
    # sky uv
    result_mat_u, result_mat_v, max_range = myFuncUV.get_result_sky_uv_with_update()
    plt.figure(1)
    if len(result_mat_u) != 0 and len(result_mat_v) != 0:
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

        # flip
        # plt.ylabel(r"Ra($H$)")
        # plt.xlabel(r'Dec ($^\circ$)')
        # plt.yticks([0, 2, 6, 10, 14, 18, 22, 24])
        # plt.xticks([-90, -60, -30, 0, 30, 60, 90])
        # plt.ylim(0, 24)
        # plt.xlim(-90, +90)

    plt.grid()
    plt.show()


if __name__ == "__main__":
    test()
    # test_new_all_sky()
