import os
import sys
sys.path.append("/nas/user/wyh/TNC")
# from pybind.test_funcs import test_map
from pybind.test_funcs import Map

import numpy as np
import pandas as pd
import re
import tqdm

from datetime import date, time, datetime
from time import mktime

traj_root = "/nas/user/wyh/TNC/traj_dealer/30w_section_mode/"
SH_map = Map("/nas/user/wyh/dataset/roadnet/Shanghai", zone_range=[31.17491, 121.439492, 31.305073, 121.507001])

# service for ETA
def create_datetime(timestamp):
    timestamp = str(timestamp)
    if int(timestamp) < 100000000:  # for Shanghai Dataset
        mon = int(timestamp[0])
        day = int(timestamp[1]) * 10 + int(timestamp[2])
        hour = int(timestamp[3:]) // 3600
        min = (int(timestamp[3:]) % 3600) // 60
        sec = int(timestamp[3:]) % 60
        return int(mktime(datetime.combine(date(year=2015, month=mon, day=day),
                                           time(hour=hour, minute=min, second=sec))
                          .timetuple()))
    else:
        return int(timestamp)

def get_day(timestamp):
    d = datetime.fromtimestamp(timestamp)
    return d.strftime("%Y-%m-%d")

def getEdges(file_path):
    ret_records = []
    data = open(file_path, 'r')
    for item in data:
        ret_records.append([int(_) for _ in re.split(' |,', item.strip())])
    return ret_records

def getTrajs(file_path):
    ret_records = []
    buffer_records = []
    data = open(file_path, 'r')
    for item in data:
        line = item.strip()
        line = re.split(' |,', line)
        if line[0][0] == '-':
            ret_records.append(buffer_records)
            buffer_records = []
        else:
            line = [int(float(line[0])), float(line[1]), float(line[2]), int(float(line[3]))]
            buffer_records.append(line)
    return ret_records

# print(getEdges("/nas/user/wyh/TNC/traj_dealer/30w_section_mode/new_30w_route.txt"))
# print(getTrajs("/nas/user/wyh/TNC/traj_dealer/30w_section_mode/30w_traj.txt"))


# 定义一个函数，输入参数为轨迹和路线，输出为路线上每个路段的到达时间
def in_road_time(traj, route):
    traj_len = len(traj)
    route_time = {}
    route_end_time = {}
    route_time_list = []

    for i in range(traj_len):
        rid = traj[i][-1]
        if rid != -999:
            if rid not in route_time:
                route_time[rid] = traj[i][0] - traj[0][0]
                route_end_time[rid] = traj[i][0] - traj[0][0]
            route_end_time[rid] = max(route_end_time[rid], traj[i][0] - traj[0][0])

    route_time[-999] = -999

    # 初始化两个指针，分别指向路线上的第一个和第二个路段
    left = 0
    right = 0
    # 遍历路线上的每个路段
    for cid, rid in enumerate(route):
        right = max(right, cid)
        if rid != -999:
            if rid not in route_time:
                while (route[right] not in route_time):
                    right += 1
                if route[right] != -999:
                    # 用线性插值的方法
                    tmp = (cid - left - 1) / (right - left - 1) * (route_time[route[right]] - route_end_time[route[left]]) + route_end_time[route[left]]
                    route_time_list.append(int(tmp + traj[0][0]))
            else:
                left = cid
                route_time_list.append(int(route_time[rid] + traj[0][0]))
        else:
            route_time_list.append(-999)

    # return route_time_list
    # print(route_time_list)
    return route_time_list

def output_traj(trajs, edges, out_file_path, map, sec_mode = True):
    out_file_path.write('id;path;tlist;usr_id;traj_id;vflag;start_time\n')
    if sec_mode == True:
        id = 0
        for traj, edge in zip(trajs, edges):
            # 指示器
            print("id = ", id)

            sec_id = 0
            a = []
            b = []
            r0 = [int(_) for _ in edge]
            t0 = [int(_) for _ in in_road_time(traj, edge)]
            for r, t in zip(r0, t0):
                if r != -999:
                    if sec_id == 0:
                        a.append(map.edgeNode[r][0])
                        a.append(map.edgeNode[r][1])
                        b.append(t)
                    else:
                        a.append(map.edgeNode[r][1])
                        b.append(t)
                else:
                    # FLAG = 1
                    if len(a) != 0:
                        a.pop()
                    a.append(-999)
                    b.append(-999)
                sec_id += 1

            if a.pop() == -999:
                a.append(-999)


            print(len(a), len(b))
            out_file_path.write(';'.join([
                str(id), str(a), str(b), '0', '0', '1', get_day(create_datetime(traj[0][0]))
                ]))
            out_file_path.write('\n')
            id += 1       
    else:
        id = 0
        for traj, edge in zip(trajs, edges):
            # 指示器
            # print(id)
            a = [int(_) for _ in edge]
            b = [int(_) for _ in in_road_time(traj, edge)]
            print(len(a), len(b))
            out_file_path.write(';'.join([
                str(id), str(a), str(b), '0', '0', '1', get_day(create_datetime(traj[0][0]))
                ]))
            out_file_path.write('\n')
            id += 1

# 定义数据集并划分
trajs = getTrajs(os.path.join(traj_root, "30w_traj.txt"))
edges = getEdges(os.path.join(traj_root, "new_30w_route.txt"))
MAX = 300000

# MAX要更改
# trajs = getTrajs(os.path.join(traj_root, "5_traj.txt"))
# edges = getEdges(os.path.join(traj_root, "1_route.txt"))
# MAX = 5


# train:test:valid = 7:1:2
train_trajs = trajs[:int(MAX * 0.7)]
valid_trajs = trajs[int(MAX * 0.7):int(MAX * 0.8)]
test_trajs = trajs[int(MAX * 0.8):MAX]
train_edges = edges[:int(MAX * 0.7)]
valid_edges = edges[int(MAX * 0.7):int(MAX * 0.8)]
test_edges = edges[int(MAX * 0.8):MAX]


# output_traj(train_trajs, train_edges, open( "/nas/user/wyh/TNC/traj_dealer/30w_section_mode/test.csv", 'w+'), SH_map, True)
# output_traj(train_trajs, train_edges, open( "/nas/user/wyh/TNC/traj_dealer/30w_section_mode/SHmap_train.csv", 'w+'), SH_map, True)
output_traj(valid_trajs, valid_edges, open( "/nas/user/wyh/TNC/traj_dealer/30w_section_mode/SHmap_valid.csv", 'w+'), SH_map, True)
output_traj(test_trajs, test_edges, open( "/nas/user/wyh/TNC/traj_dealer/30w_section_mode/SHmap_test.csv", 'w+'), SH_map, True)
# output_traj(trajs, edges, open( "/nas/user/wyh/TNC/traj_dealer/30w_section_mode/new_r_test.csv", 'w+'), SH_map, False)

# def output_trajs(trajs, edges, file):
#     file.write('id;path;tlist;usr_id;traj_id;vflag;start_time\n')
#     if True:
#         file.write(';'.join([
#             '0', str([rid for rid in maps.valid_edge]), '[-1]', '0', '0', '1', '2015-01-01'
#         ]))
#         file.write('\n')
#     id = 0
#     path = []
#     for rid in maps.valid_edge:
#         path.append(rid)

#     for traj, edge in zip(trajs, edges):
#         file.write(';'.join([
#             str(id), str([int(_) for _ in edge]), str([int(_) for _ in in_road_time(traj, edge)]), '0', '0', '1', get_day(create_datetime(traj[0][0]))
#         ]))
#         file.write('\n')
#         id += 1
# print(in_road_time(trajs[0], edges[0]))
# show_output_traj(trajs, edges)

# output_trajs(train_trajs, train_edges, open(f'./raw_data/{args.dataset}/{args.dataset}_train.csv', 'w+'))
# output_trajs(valid_trajs, valid_edges, open(f'./raw_data/{args.dataset}/{args.dataset}_eval.csv', 'w+'))
# output_trajs(test_trajs, test_edges, open(f'./raw_data/{args.dataset}/{args.dataset}_test.csv', 'w+'))