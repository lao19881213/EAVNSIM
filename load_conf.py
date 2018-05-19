"""
@functions: list all the parameters for testing
@author: Zhen ZHAO
@date: April 23, 2018
"""

# velocity of light
light_speed = 299792458.8
# Earth radius [km]
earth_radius = 6378.1363
# Earth flattening
earth_flattening = 1 / 298.257
# square of ellipsoid eccentricity
eccentricity_square = earth_flattening * (2 - earth_flattening)
# GM constant     # GM=3.986004418*1e14    #地球（包括大气）引力常数  单位为m^3/s^-2 折合3.986004418*1e5
GM = 3.986004418 * 1e5  # [km^3/s^-2]


###########################################
# 1. 观测时间的设置
###########################################

# 起始时间全局变量
StartTime = "2020/01/01/00/00/00"
StartTimeGlobalYear = 2020
StartTimeGlobalMonth = 1
StartTimeGlobalDay = 1
StartTimeGlobalHour = 0
StartTimeGlobalMinute = 0
StartTimeGlobalSecond = 0


# 结束时间全局变量
StopTime = "2020/01/02/00/00/00"
StopTimeGlobalYear = 2020
StopTimeGlobalMonth = 1
StopTimeGlobalDay = 2
StopTimeGlobalHour = 0
StopTimeGlobalMinute = 0
StopTimeGlobalSecond = 0

TimeStep = "00:00:05:00"
TimeStepGlobalDay = 0
TimeStepGlobalHour = 0
TimeStepGlobalMinute = 5
TimeStepGlobalSecond = 0


###########################################
# 2. 观测参数的设置
###########################################

# 三种基线类型的选择标志
baseline_flag_gg = 1
baseline_flag_gs = 0
baseline_flag_ss = 0
baseline_type = baseline_flag_gg | baseline_flag_gs | baseline_flag_ss
# 001(1)->select GtoG 010(2)->SELECT GtoS, 100(4)->StoS

# 观测频率和带宽
obs_freq = 43e9
bandwidth = 3.2e7

# 单位选择标志
unit_flag = 1

# cutoff_mode=1 #截止模式选择
cutoff_mode = {'flag': 1, 'CutAngle': -10}  # 截止模式选择，flag:0->取数据库中设置的水平角，1->取界面上设置的水平角 2->取大者，3->取小者
precession_mode = {'flag': 0}  # 进动模型选择，0->Two-Body,1->J2

###########################################
# 3. 源，观测站，卫星的信息
###########################################
# 源信息
pos_mat_src = [['0134+329', 0.42624576, 0.57874696]]
# VLBI站信息
pos_mat_test = [['Effelsberg', 4033.94775, 486.99037, 4900.430, 20],
                ['Jodrell Bank(Lovell)', 3822.6264970, -154.1055889, 5086.4862618, 20],
                ]
pos_mat_vlbi = [['Effelsberg', 4033.94775, 486.99037, 4900.430, 20],
                ['Jodrell Bank(Lovell)', 3822.6264970, -154.1055889, 5086.4862618, 20],
                ['Green Bank', 882.87995, -4924.48234, 2944.13065, 20],
                ['Shanghai', -2831.6869201, 4675.7336809, 3275.3276821, 20]
                ]
# 遥测站信息
pos_mat_telemetry = [['Goldstone', -2353.62000, -4641.34000, 3677.05000]]
# 卫星列表信息,每一元祖数据对应的信息为Name,a,e,i,w,Ω,M0,Epoch
pos_mat_sat = [['VSOP', 17367.457, 0.60150, 31.460, 106.755, 16.044, 66.210, 50868.00000],
               ['RadioAstron', 46812.900, 0.8200, 51.000, 285.000, 255.000, 280.000, 50449.000000]]
# M0未知，这里设置为0，假设2020年3月1日通过近地点

###########################################
# 4. 数据库的读取
###########################################

# 数据库中的表
# Soutable = 0
# Teletable = []
# Satetable = []
# VLStable = [1]

# 与数据库相关的变量
# hostname="localhost"
# username="root"
# password=""
# dbname="astro"

# 数据库表数据路径的装载
# SourceFilePath = "Database/source.txt"
# VLBIStationFilePath = "Database/VLBIStation.txt"
# SatelliteFilePath = "Database/satellite.txt"
# TrackStationFilePath = "Database/trackstation.txt"


def print_setting():
    info = []
    label1 = "Start Time: %d/%d/%d %d:%d:%d UT" % (StartTimeGlobalYear, StartTimeGlobalMonth, StartTimeGlobalDay,
                                                   StartTimeGlobalHour, StartTimeGlobalMinute, StartTimeGlobalSecond)
    info.append(label1)

    label2 = "Stop Time: %d/%d/%d %d:%d:%d UT" % (StopTimeGlobalYear, StopTimeGlobalMonth, StopTimeGlobalDay,
                                                  StopTimeGlobalHour, StopTimeGlobalMinute, StopTimeGlobalSecond)
    info.append(label2)

    label3 = "Time step: %dd %dh %dm %ds" % (TimeStepGlobalDay, TimeStepGlobalHour,
                                             TimeStepGlobalMinute, TimeStepGlobalSecond)
    info.append(label3)

    label4 = "Wavelength: %f" % (light_speed * 100 / obs_freq)
    info.append(label4 + '\n')

    label5 = "Source:\n\t"
    for item in pos_mat_src:
       label5 = label5 + item[0]
       label5 = label5 + '\n\tRA:     '
       label5 = label5 + str(item[1])
       label5 = label5 + '\n\tDEC:    '
       label5 = label5 + str(item[2])
       label5 = label5 + '\n'
    info.append(label5)

    label6 = "Satellite:\n\t"
    for item in pos_mat_sat:
       label6 = label6+item[0]
       label6 = label6+'\n\t'
    info.append(label6)

    label7 = 'VLBI Stations:\n\t'
    for item in pos_mat_vlbi:
       label7 = label7+item[0]
       label7 = label7+'\n\t'
    info.append(label7)
    print("=" * 30, '\n')
    print("\n".join(info))
    print("=" * 30)


if __name__ == '__main__':
    print_setting()

