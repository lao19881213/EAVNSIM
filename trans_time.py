"""
@functions: Utility library, including all the used time transformations
@author: Zhen ZHAO
@date: April 24, 2018
"""
import numpy as np


def time_2_jde(year, month, day, hour, minute, sec):
    """
    儒略日的计算
    :return: jde time
    """
    if month == 1 or month == 2:
        f = year - 1
        g = month + 12
    else:  # month >= 3
        f = year
        g = month
    mid1 = np.floor(365.25 * f)
    mid2 = np.floor(30.6001 * (g+1))
    para_a = 2-np.floor(f/100)+np.floor(f/400)
    para_j = mid1 + mid2 + day + para_a + 1720994.5
    jde_time = para_j + hour / 24 + minute / 1440 + sec / 86400
    return jde_time


def time_2_mjd(year, month, day, hour, minute, sec, d_sec):
    """
    得到修正儒略日
    :return: mjd
    """
    YP = year
    MP = month
    if month <= 2:
        month += 12
        year = year - 1
    if (YP < 1582) or (YP == 1582 and MP < 10) or (YP == 1582 and MP == 10 and day <= 4):
        B = -2 + int((year + 4716) / 4) - 1179
    elif (YP > 1582) or (YP == 1582 and MP > 10) or (YP == 1582 and MP == 10 and day > 10):
        B = int(year / 400) - int(year / 100) + int(year / 4)

    mjd = 365.0 * np.double(year) - 679004.0 + np.double(B) + np.floor(30.6001 * np.double(month + 1)) + np.double(day)
    mjd += (np.double(3600 * hour + 60 * minute + sec) + d_sec) / 86400.00
    return mjd


def mjd_2_time(mjd_time):
    month_array = ((31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31), (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31))
    year = int(0.0027379093 * mjd_time + 1858.877)
    day = int(mjd_time - time_2_mjd(year, 1, 0, 0, 0, 0, 0.0))
    if (year % 4 == 0 and year % 400 == 0) or (year % 4 == 0 and year % 100 != 0):
        m_flag = 0
    else:
        m_flag = 1
    month = 1
    for i in range(0, 12):
        day = day - month_array[m_flag][i]
        if day <= 0:
            day = day + month_array[m_flag][i]
            month = i+1
            break
        else:
            continue
    week = (int(mjd_time)-5) % 7
    mjd_time = mjd_time - time_2_mjd(year, month, day, 0, 0, 0, 0.0)
    mjd_time = mjd_time * 86400.0
    F = int(mjd_time)
    if np.fabs(mjd_time-np.floor(mjd_time)) >= 0.5:
        F = F+1
    hour = F//3600
    minute = np.mod(F, 3600)//60
    sec = np.mod(F, 3600)
    sec = np.mod(sec, 60)
    return week, year, month, day, hour, minute, sec  # W返回0代表星期一，返回6代表星期天


def time_2_day(day, hour, minute, sec):
    """
    将一段时间单位，转化为天数
    :param day:
    :param hour:
    :param minute:
    :param sec:
    :return:
    """
    day_num = np.double(3600 * hour + 60 * minute + sec) / 86400.00
    day_num += day
    return day_num


def time_2_rad(hour, minute, sec):
    """
    将时角转换为弧度
    :param hour:
    :param minute:
    :param sec:
    :return:
    """
    if hour < 0:
        flag = -1
    else:
        flag = 1
    hour = np.abs(hour)
    angle_rad = (hour + (60.0 * minute + sec) / 3600.0) / 12 * np.pi
    return angle_rad*flag


def time_str_2_rad(time_st):
    """
    将时间字符串转换为弧度
    :param time_st:"21h33m26s"
    :return: radian
    """
    time_str = time_st
    time_str = time_str.replace('h', ':')
    time_str = time_str.replace('m', ':')
    time_str = time_str.replace('s', '')
    time_str = time_str.split(':')
    time_h = int(time_str[0])
    time_m = int(time_str[1])
    time_s = float(time_str[2])
    time_rad = time_2_rad(time_h, time_m, time_s)
    return time_rad


def time_str_2_mjd(time_st):
    time_str = time_st
    time_year = int(time_str[0:4])
    time_month = int(time_str[4:6])
    time_day = int(time_str[6:8])
    time_hour = int(time_str[8:10])
    time_minute = int(time_str[10:12])
    time_second = int(time_str[12:14])
    mjd_time = time_2_mjd(time_year, time_month, time_day, time_hour, time_minute, time_second, 0)
    return mjd_time


def mjd_2_julian(mjd_time):
    """
    #J2000 2000年1月1日12时
    :param mjd_time:
    :return:
    """
    julian_time = (mjd_time-51544.5)/36525     # part4-1 p31
    return julian_time     # 以J2000作为参考，计算MJD和Julian时间


def mjd_2_gmst(mjd_time):
    """
    格林尼治平均恒星时
    :param mjd_time:
    :return:
    """
    jutime = mjd_2_julian(mjd_time)
    gmst = 67310.548 + 8640184.812866 * jutime + (mjd_time + 0.5 - int(mjd_time + 0.5)) * 86400
    gmst = gmst * np.pi / 43200
    return gmst


def mjd_2_gast(mjd_time):
    """
    格林尼治视恒星时
    :param mjd_time:
    :return:
    """
    gmst = mjd_2_gmst(mjd_time)
    eq_e = equinox_equation(mjd_time)  # part4-1 p31
    gast = gmst + eq_e
    return gast


def mjd_2_gst(time_mjd, delta_t, utc_ut1):
    dpi = 3.141592653589793238462643
    gst_offset = 0.7790572732640
    gst_factor = 1.00273781191135448
    t = time_mjd - 51544.5
    t = t * gst_factor
    t = t - np.double(np.int(t))  # 取其小数部分
    t = t + (delta_t + utc_ut1) / 86400.0 * gst_factor
    theta = 2.0 * dpi * (gst_offset + t)
    return theta


def ecliptic_obliquity(mjd_time):
    """
    计算黄赤交角
    :param mjd_time:
    :return:
    """
    ju_time = mjd_2_julian(mjd_time)

    # 4709636#将秒的单位转化为弧度1s/3600/180*np.pi=1/206264.80624709636
    epsilon = (84381.448 - 46.815 * ju_time - 0.00059 * ju_time ** 2 + 0.001813 * ju_time ** 3) / 206264.8062
    return epsilon


def nutation_omega(ju_time):
    """
    简化章动模型的基本参数Ω的计算
    :param ju_time:
    :return:
    """
    omega = (450160.28 - 6962890.539 * ju_time)  # 平均的月球轨道升交点经度
    omega = omega / 206264.8062
    return omega


def longitude_nutation(mjd_time):
    """
    黄经章动
    :param mjd_time:
    :return:
    """
    tim = (mjd_time - 51544.5) / 36525
    omega = nutation_omega(tim)  # Omega的单位为弧度
    delta_psi = -(17.1996 + 0.01742 * tim) * np.sin(omega) / 206264.8062
    return delta_psi


def equinox_equation(mjd_time):
    """
    春分方程
    :param mjd_time:
    :return:
    """
    epsilon = ecliptic_obliquity(mjd_time)
    delta_psi = longitude_nutation(mjd_time)
    e_e = delta_psi * np.cos(epsilon)
    return e_e
