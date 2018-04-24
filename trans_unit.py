"""
@functions: Utility library, including all the used unit transformations
@author: Zhen ZHAO
@date: April 24, 2018
"""
import numpy as np
import load_conf as lc


def freq_2_wavelength(obs_freq):
    """
    freq to wavelength
    :param obs_freq:
    :return: wavelength
    """
    wavelength = lc.light_speed / obs_freq
    return wavelength


def angle_str_2_rad(angle_str):
    """
    sometimes the source info is given in the 'dms' format
    we transform it into the radian to facilitate the calculation
    :param angle_str: "23d43m54s"
    :return: 0.414195720319121
    """
    angle_str = angle_str.replace('d', ':')
    angle_str = angle_str.replace('m', ':')
    angle_str = angle_str.replace('s', '')
    angle_str = angle_str.split(':')
    angle_d = int(angle_str[0])
    angle_m = int(angle_str[1])
    angle_s = float(angle_str[2])
    angle_rad = angle_2_rad(angle_d, angle_m, angle_s)
    return angle_rad


def angle_2_rad(dd, mm, ss):
    """
    transform angle to radian
    :param dd:
    :param mm:
    :param ss:
    :return: radian
    """
    if dd < 0:
        flag = -1
        dd = np.abs(dd)
    else:
        flag = 1
    angle_rad = (dd + ((mm * 60.0 + ss) / 3600.0)) / 180 * np.pi

    return angle_rad * flag


def rad_2_angle(rad):
    """
    transform radian to angle
    :param rad:
    :return: angle [0,180]
    """
    return rad * 180 / np.pi


def sgn(x):
    """
    sign function
    :param x: an integer
    :return: the sign
    """
    return np.sign(x)


def angle_btw_vec(vec_x, vec_y):
    """
    calculate the included angle
    :param vec_x: 3x1 vector, in rad unit
    :param vec_y: 3x1 vector, in rad unit
    :return: included angle between two vectors, belongs to [0,pi]
    """
    arc = vec_x[0][0] * vec_y[0][0] + vec_x[1][0] * vec_y[1][0] + vec_x[2][0] * vec_y[2][0]
    if arc > 1 or arc < -1:
        arc = sgn(arc)
    arc = np.arccos(arc)
    return arc  # 两个单位向量的夹角，范围0-pi
