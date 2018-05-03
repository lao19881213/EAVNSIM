"""
@functions: Utility library, including all the used coordinate transformations
@author: Zhen ZHAO
@date: April 24, 2018
"""
import numpy as np
import load_conf as lc
import trans_time as tt


def trans_matrix_uv_itrf(mjd_time, ra, dec):
    """
    生成从ITRF到UV坐标系的转换矩阵
    :param mjd_time:
    :param ra:
    :param dec:
    :return:
    """
    gast = tt.mjd_2_gast(mjd_time)
    hour_angle = gast - ra
    hour_angle = np.mod(hour_angle, np.pi * 2)
    matrix = np.array([[np.sin(hour_angle), np.cos(hour_angle), 0],
                       [-np.sin(dec) * np.cos(hour_angle), np.sin(dec) * np.sin(hour_angle), np.cos(dec)],
                       [np.cos(dec) * np.cos(hour_angle), -np.cos(dec) * np.sin(hour_angle), np.sin(dec)]
                       ])
    return matrix


def geographic_2_itrf(longitude, latitude, height):
    """
    地理坐标系专为ITRF坐标
    :param longitude: 地理坐标的经度
    :param latitude: 纬度
    :param height:  高度
    :return: ITRF坐标位置(x,y,z)
    """
    e_square = lc.eccentricity_square
    temp = lc.earth_radius / np.sqrt(1 - e_square * (np.sin(latitude) ** 2))
    # 计算笛卡尔坐标(x,y,z)
    x = (temp + height) * np.cos(latitude) * np.cos(longitude)
    y = (temp + height) * np.cos(latitude) * np.sin(longitude)
    z = ((1 - e_square) * temp + height) * np.sin(latitude)
    return x, y, z


def itrf_2_geographic(cor_x, cor_y, cor_z):
    p = np.sqrt(cor_x ** 2 + cor_y ** 2)
    f = lc.earth_flattening
    e_square = lc.eccentricity_square

    # calculate longitude
    if (cor_x == 0) and (cor_y == 0):
        longitude = 0
        if cor_z == 0:
            latitude = 0
            height = -1 * lc.earth_radius
    else:
        longitude = np.arctan2(cor_y, cor_x)
    # calculate latitude
    if p == 0:
        if cor_z > 0:
            latitude = np.pi / 2
        elif cor_z < 0:
            latitude = -np.pi / 2
    else:
        r = np.sqrt(p ** 2 + cor_z ** 2)

        temp = cor_z / p * ((1 - f) + e_square * lc.earth_radius / r)
        u = np.arctan(temp)

        temp = (cor_z * (1 - f) + e_square * lc.earth_radius * ((np.sin(u)) ** 3)) / (
                (1 - f) * (p - e_square * lc.earth_radius * ((np.cos(u)) ** 3)))
        latitude = np.arctan(temp)  # -90度到+90度
    # calculate height
    if cor_z != 0:
        height = p * np.cos(latitude) + cor_z * np.sin(latitude) - lc.earth_radius * np.sqrt(
            1 - e_square * (np.sin(latitude) ** 2))
    return longitude, latitude, height


def rect_2_polar(x):
    """
    直角坐标系转换为极坐标（Long,Lat)
    :param x: 3维直角坐标
    :return:
    """
    r = np.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2)
    x[0] = x[0] / r
    x[1] = x[1] / r
    x[2] = x[2] / r
    if x[0] == 0 and x[1] == 0:
        Long = 0
    else:
        Long = np.arctan2(x[0], x[1])  # -180~180
        if Long < 0:
            Long = Long + np.pi * 2
    Lat = np.arcsin(x[2])
    return Long, Lat  # part4-2 p16


def polar_2_rect(long, lat):
    """
    极坐标到直角坐标的3维单位向量x
    :param long:
    :param lat:
    :return:
    """
    x1 = np.cos(lat) * np.cos(long)
    x2 = np.cos(lat) * np.sin(long)
    x3 = np.sin(lat)
    return x1, x2, x3


def equatorial_2_horizontal(time_mjd, ra_src, dec_src, long_station, lat_station):
    x, y, z = polar_2_rect(ra_src, dec_src)
    gast = tt.mjd_2_gast(time_mjd)
    rz_pi = np.array([[np.cos(np.pi), np.sin(np.pi), 0],
                      [-np.sin(np.pi), np.cos(np.pi), 0],
                      [0, 0, 1]
                      ])
    ry_latitude = np.array([[np.cos(np.pi / 2 - lat_station), 0, -np.sin(np.pi / 2 - lat_station)],
                            [0, 1, 0],
                            [np.sin(np.pi / 2 - lat_station), 0, np.cos(np.pi / 2 - lat_station)]
                            ])
    matrix1 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
    rz_longitude = np.array([[np.cos(gast + long_station), np.sin(gast + long_station), 0],
                             [-np.sin(gast + long_station), np.cos(gast + long_station), 0],
                             [0, 0, 1]
                             ])
    vec_xyz = np.array([[x], [y], [z]])
    temp_mat = np.dot(rz_pi, ry_latitude)
    temp_mat = np.dot(temp_mat, matrix1)
    temp_mat = np.dot(temp_mat, rz_longitude)
    horizon_xyz = np.dot(temp_mat, vec_xyz)
    horizon_xlst = [horizon_xyz[0][0], horizon_xyz[1][0], horizon_xyz[2][0]]
    azimuth, elevation = rect_2_polar(horizon_xlst)
    return azimuth, elevation


def drotate(x, e, axis):
    """
    :param x: 待旋转的向量
    :param e: 旋转的角度
    :param axis: 旋转轴
    :return: 旋转后的向量
    """
    u = x[0]
    v = x[1]
    w = x[2]
    cos = np.cos
    sin = np.sin
    if axis == 'x' or axis == 'X':
        x[1] = v * cos(e) - w * sin(e)
        x[2] = v * sin(e) + w * cos(e)
        return x
    elif axis == 'y' or axis == 'Y':
        x[2] = w * cos(e) - u * sin(e)
        x[0] = w * sin(e) + u * cos(e)
        return x
    elif axis == 'z' or axis == 'Z':
        x[0] = u * cos(e) - v * sin(e)
        x[1] = u * sin(e) + v * cos(e)
        return x
    else:
        print("drotate:bad flag to rotate %s\n" % axis)


def itrf_2_horizontal(satellite_lst, long_sta, lat_sta, height_sta):
    """
    从地面坐标到水平系统的转换
    :param satellite_lst: 依次存放的是数据是卫星位置和速度：x y z vx vy vz
    :param long_sta:  遥测站的经度
    :param lat_sta:   遥测站的维度
    :param height_sta:  遥测站的高度
    :return:
    """

    x0, y0, z0 = geographic_2_itrf(long_sta, lat_sta, height_sta)
    x = satellite_lst[0] - x0
    y = satellite_lst[1] - y0
    z = satellite_lst[2] - z0
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    # 从地面坐标到水平系统的位置坐标转换
    matrix1 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
    Rz_pi = np.array([[np.cos(np.pi), np.sin(np.pi), 0],
                      [-np.sin(np.pi), np.cos(np.pi), 0],
                      [0, 0, 1]
                      ])
    Ry_Latitude = np.array([[np.cos(np.pi / 2 - lat_sta), 0, -np.sin(np.pi / 2 - lat_sta)],
                            [0, 1, 0],
                            [np.sin(np.pi / 2 - lat_sta), 0, np.cos(np.pi / 2 - lat_sta)]
                            ])
    Rz_Longitude = np.array([[np.cos(long_sta), np.sin(long_sta), 0],
                             [-np.sin(long_sta), np.cos(long_sta), 0],
                             [0, 0, 1]
                             ])
    xyzITRF = np.array([[x, y, z]])
    xyzITRF = xyzITRF.T
    TempMatrix = np.dot(matrix1, Rz_pi)
    TempMatrix = np.dot(TempMatrix, Ry_Latitude)
    TempMatrix = np.dot(TempMatrix, Rz_Longitude)
    xyzHorizon = np.dot(TempMatrix, xyzITRF)
    # 从地面坐标到水平坐标的速度坐标转换
    velocitymatrix = np.array([[satellite_lst[3], satellite_lst[4], satellite_lst[5]]])
    velocitymatrix = velocitymatrix.T
    VxVyVzHorizon = np.dot(TempMatrix, velocitymatrix)
    XVector = [xyzHorizon[0][0], xyzHorizon[1][0], xyzHorizon[2][0]]
    Azimuth, Elevation = rect_2_polar(XVector)  # 仰角和方位角
    # Radial,azimuthal and vertical nelocity componengts
    Ry_Elevation = np.array([[np.cos(-1 * Elevation), 0, -np.sin(-1 * Elevation)],
                             [0, 1, 0],
                             [np.sin(-1 * Elevation), 0, np.cos(-1 * Elevation)]
                             ])
    Rz_Azimuth = np.array([[np.cos(Azimuth), np.sin(Azimuth), 0],
                           [-np.sin(Azimuth), np.cos(Azimuth), 0],
                           [0, 0, 1]
                           ])
    TempMatrix = np.dot(Ry_Elevation, Rz_Azimuth)
    VelocityVector = np.dot(TempMatrix, VxVyVzHorizon)
    AzimuthVelocity = VelocityVector[1][0]
    ElevationVelocity = VelocityVector[2][0]
    AzimuthVelocity = AzimuthVelocity / r
    ElevationVelocity = ElevationVelocity / r  # part4-2 p11
    return [Azimuth, Elevation, AzimuthVelocity, ElevationVelocity]  # 返回元祖


def equatorial_2_ecliptic(equ, epsilon):
    """
    将赤道源坐标系转换为黄道坐标系
    :param equ: 赤道系统单位矢量[[x],[y],[z]]
    :param epsilon: 黄道倾角
    :return: 黄道系统单位矢量
    """
    rx_epsilon=np.array([[1, 0, 0],
                         [0, np.cos(epsilon), np.sin(epsilon)],
                         [0, -np.sin(epsilon), np.cos(epsilon)]
                        ])
    ecu = np.dot(rx_epsilon, equ)
    return ecu


def itrf_2_icrf(time_mjd, itrf_sat_x, itrf_sat_y, itrf_sat_z, itrf_sat_vx,
                itrf_sat_vy, itrf_sat_vz):
    gast = tt.mjd_2_gast(time_mjd)
    rz_gast = np.array([[np.cos(-gast), np.sin(-gast), 0],
                        [-np.sin(-gast), np.cos(-gast), 0],
                        [0, 0, 1]
                        ])
    itrf_pos_vec = np.array([[itrf_sat_x], [itrf_sat_y], [itrf_sat_z]])
    icrf_pos_vec = np.dot(rz_gast, itrf_pos_vec)
    # 恒星时/平均时间比
    k = 1.002737909350795  # d(gast)/dt
    k = k * np.pi / 43200
    itrf_veliocity_vec = np.array([[itrf_sat_vx], [itrf_sat_vy], [itrf_sat_vz]])
    temp_mat_1 = np.dot(rz_gast, itrf_veliocity_vec)
    temp_mat_2 = np.array([[k * (-np.sin(gast)), k * (-np.cos(gast)), 0],
                            [k * np.cos(gast), k * (-np.sin(gast)), 0],
                            [0, 0, 0]
                            ])
    temp_mat_3 = np.dot(temp_mat_2, itrf_pos_vec)
    icrf_velocity_vec = temp_mat_1 + temp_mat_3
    return icrf_pos_vec, icrf_velocity_vec


def icrf_2_itrf(time_mjd, icrf_sat_x, icrf_sat_y, icrf_sat_z, icrf_sat_vx,
                icrf_sat_vy, icrf_sat_vz):
    gast = tt.mjd_2_gast(time_mjd)
    rz_gast = np.array([[np.cos(gast), np.sin(gast), 0],
                        [-np.sin(gast), np.cos(gast), 0],
                        [0, 0, 1]
                        ])
    icrf_pos_vec = np.array([[icrf_sat_x], [icrf_sat_y], [icrf_sat_z]])
    itrf_pos_vec = np.dot(rz_gast, icrf_pos_vec)
    # 恒星时/平均时间比
    k = 1.002737909350795  # d(gast)/dt
    k = k * np.pi / 43200
    icrf_veliocity_mat = np.array([[icrf_sat_vx], [icrf_sat_vy], [icrf_sat_vz]])
    temp_mat_1 = np.dot(rz_gast, icrf_veliocity_mat)
    temp_mat_2 = np.array([[k * (-np.sin(gast)), k * np.cos(gast), 0],
                           [-k * np.cos(gast), k * (-np.sin(gast)), 0],
                           [0, 0, 0]
                           ])
    temp_mat_3 = np.dot(temp_mat_2, icrf_pos_vec)
    itrf_velocity_mat = temp_mat_1 + temp_mat_3
    return [itrf_pos_vec[0][0], itrf_pos_vec[1][0], itrf_pos_vec[2][0],
            itrf_velocity_mat[0][0], itrf_velocity_mat[1][0], itrf_velocity_mat[2][0]]
