import sys
sys.path.append("/nas/user/wyh/TNC")
# from pybind.test_funcs import test_map
from pybind.test_funcs import Map

import numpy as np
import pandas as pd
import re

from datetime import date, time, datetime
from time import mktime


traj_root = "traj_dealer/30w_section_mode/30w_traj.txt"
SH_map = Map("/nas/user/wyh/dataset/roadnet/Shanghai", zone_range=[31.17491, 121.439492, 31.305073, 121.507001])


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

print(getEdges("/nas/user/wyh/TNC/traj_dealer/30w_section_mode/1_route.txt"))
