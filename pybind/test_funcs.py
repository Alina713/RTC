import mm
import math
from queue import PriorityQueue, Queue
import numpy as np
# import seaborn
import sys
# 最大递归深度
sys.setrecursionlimit(100000)
import pdb
# from scipy.sparse import csr_matrix
# from sklearn.cluster import SpectralClustering
import random
import pickle

# import matplotlib.pyplot as plt
from tqdm import tqdm
# import jgraph
import time
# import config
import os

from multiprocessing import Pool

import torch as th
import numpy
import dgl
# import scipy.sparse as sp
# from dgl.nn.pytorch.conv.appnpconv import APPNPConv
# from scipy.linalg import fractional_matrix_power, inv
# from sklearn.preprocessing import MinMaxScaler

class Map:
    def __init__(self, dir, zone_range):
        edgeFile = open(dir + '/edgeOSM.txt')
        # 处理waytype信息，对应edgeosm中第一个参数
        wayFile = open(dir + '/wayTypeOSM.txt')
        # for line in wayFile.readlines():
        #     item_list = line.strip().split()
        #     roadId = int(item_list[0])
        #     wayId = int(item_list[-1])
        #     self.wayType[roadId] = wayId


        self.edgeDis = []
        # 保存边的路口编号对
        self.edgeNode = []
        self.edgeCord = []
        self.nodeSet = set()
        self.nodeDict = {}
        self.edgeDict = {}
        self.edgeRevDict = {}
        self.nodeEdgeDict = {}
        self.nodeEdgeRevDict = {}
        self.minLat = 1e18
        self.maxLat = -1e18
        self.minLon = 1e18
        self.maxLon = -1e18
        self.edgeNum = 0
        self.nodeNum = 0
        self.zone_range = zone_range
        self.valid_edge = {}
        self.valid_to_origin = {}
        self.valid_edge_num = 0

        self.edge_to_cluster = {}
        self.cluster_to_edge = {}
        self.cluster_neighbor = {}
        self.cluster_neighbor_edge = {}
        self.cluster_neighbor_cluster = {}

        # service for recover_dgl()
        self.tmp_edgeDict = {}
        self.tmp_edgeRevDict = {}

        # 保存所有信息，service for valid_map_trajmm(self) &
        self.info = []

        for line in edgeFile.readlines():
            item_list = line.strip().split()
            my_tuple = ()
            my_tuple += tuple(item_list)
            # print(my_tuple)
            self.info.append(my_tuple)
            a = int(item_list[1])
            b = int(item_list[2])
            self.edgeNode.append((a, b))
            self.nodeDict[a] = b
            if a not in self.nodeEdgeDict:
                self.nodeEdgeDict[a] = []
            if b not in self.nodeEdgeRevDict:
                self.nodeEdgeRevDict[b] = []
            self.nodeEdgeDict[a].append(self.edgeNum)
            self.nodeEdgeRevDict[b].append(self.edgeNum)
            self.nodeSet.add(a)
            self.nodeSet.add(b)
            num = int(item_list[3])
            dist = 0
            # print (item_list)
            self.edgeCord.append(list(map(float, item_list[4:])))
            inzone_flag = True
            for i in range(num):
                tmplat = float(item_list[4 + i * 2])
                tmplon = float(item_list[5 + i * 2])
                self.minLat = min(self.minLat, tmplat)
                self.maxLat = max(self.maxLat, tmplat)
                self.minLon = min(self.minLon, tmplon)
                self.maxLon = max(self.maxLon, tmplon)
                inzone_flag = inzone_flag and self.inside_zone(tmplat, tmplon)

            if inzone_flag:
                self.valid_edge[self.edgeNum] = self.valid_edge_num
                self.valid_to_origin[self.valid_edge_num] = self.edgeNum
                self.valid_edge_num += 1

            for i in range(num - 1):
                dist += self.calSpatialDistance(float(item_list[4 + i * 2]), float(item_list[5 + i * 2]),
                                                float(item_list[6 + i * 2]), float(item_list[7 + i * 2]))
            self.edgeDis.append(dist)
            self.edgeNum += 1

        # 读入数据集结束
        # e用于储存有效边
        e = []
        for i in range(0, self.valid_edge_num):
            e.append(self.valid_to_origin[i])

        # 交叉路口数目
        self.nodeNum = len(self.nodeSet)

        # 路段ID
        # 路段-路段表示开始
        for veid in range(self.valid_edge_num):
            self.edgeRevDict[veid] = []
            # self.RNodeRevDict[eid] = []

        # 构建图; edgeDict存储了路段的连通关系
        # edgeDict[a] = b 代表路段a指向路段b的有向连接
        for veid in range(self.valid_edge_num):
            eid = self.valid_to_origin[veid]
            # a，b存放路段的起止ID
            a, b = self.edgeNode[eid]
            self.edgeDict[veid] = []
            if b in self.nodeEdgeDict:
                for nid in self.nodeEdgeDict[b]:
                    if nid in e:
                        vnid = self.valid_edge[nid]
                        self.edgeDict[veid].append(vnid)
                        self.edgeRevDict[vnid].append(veid)

        # print(self.edgeDict)

        # print(self.edgeDict[0])
        # self.igraph = igraph.Graph(directed=True)
        #  self.igraph.add_vertices(self.nodeNum)
        edge_list = []
        edge_weight_list = []
        for eid in range(self.edgeNum):
            a, b = self.edgeNode[eid]
            if (a == b):
                continue
            edge_list.append((a, b))
            edge_weight_list.append(self.edgeDis[eid])

        # self.igraph.add_edges(edge_list)
        #   self.igraph.es['dis'] = edge_weight_list

        print('edge Num: ', self.edgeNum)
        print('node Num: ', self.nodeNum)
        print('valid_edge_num: ', self.valid_edge_num)

        self.wayType = {}

    
    # start与end均为valid_id
    def shortestPathAll(self, start, end=-1, with_route=False):
        pq = PriorityQueue()
        pq.put((0, start))
        ans = []
        dist = [1e18 for i in range(self.edgeNum)]
        pred = [1e18 for i in range(self.edgeNum)]
        dist[start] = self.edgeDis[start]
        pred[start] = -1
        nodeset = {}
        ans.append(start)
        while (pq.qsize()):
            dis, id = pq.get()
            if id == end:
                break
            if id not in nodeset:
                nodeset[id] = 1
            else:
                continue
            for nid in self.edgeDict[id]:
                # if (nid in self.valid_edge):
                if dist[nid] > dist[id] + self.edgeDis[nid]:
                    dist[nid] = dist[id] + self.edgeDis[nid]
                    pred[nid] = id
                    pq.put((dist[nid], nid))
        if not with_route:
            pred = []
        if end != -1:
            return dist[end], pred
        return dist, pred

    def shortestPath(self, start, end, with_route=True):
        dis, pred = self.shortestPathAll(start, end, with_route)

        if end == -1:
            return dis, pred

        if with_route:
            id = end
            # print ('route: ')
            arr = [id]
            while (pred[id] >= 0 and pred[id] < 1e18):
                #  print (pred[id],end=',')
                id = pred[id]
                arr.append(id)
            arr = list(reversed(arr))
            # 去掉首尾元素
            new_arr = arr[1:-1]
        #    arr = [self.valid_edge[item] for item in arr]
            return dis, new_arr
        else:
            return dis

    # G(V, E): intersection为V点，road segment为E边
    def dgl_valid_map(self):
        u = []
        v = []
        for eid in range(self.valid_edge_num):
            rsid = self.valid_to_origin[eid]
            # a，b存放路段的起止ID
            a, b = self.edgeNode[rsid]
            u.append(a)
            v.append(b)

        graph = dgl.graph((u, v))
        return graph


    # 将dgl格式的图片进行还原，将邻接关系保存至self.tmp_edgeDict & self.tmp_edgeRevDict
    def recover_dgl(self, graph_):
        adj = sp.csr_matrix(dgl.to_scipy_sparse_matrix(graph_))

        self.tmp_edgeDict = {}
        self.tmp_edgeRevDict = {}
        for a in range(adj.shape[0]):
            self.tmp_edgeRevDict[a] = []
            self.tmp_edgeDict[a] = []

        for i in range(adj.shape[0]):
            # for j in range(i, adj.shape[1]):
            for j in range(adj.shape[1]):
                if adj[i, j] == 1:
                    self.tmp_edgeDict[i].append(j)
                    self.tmp_edgeRevDict[j].append(i)
                    # edges.append((i, j))

    # 将valid图进行mm
    def valid_map_show(self):
        valid_cnt = self.valid_num()
        # 定义传入cpp中的roadnet信息为inp变量,txt中的一整行都要读入
        inp = []
        for item in valid_cnt:
            inp.append(self.info[item])
            # print("stop")

        return inp
        # print(type(inp))
        # print(len(inp))


    # 第一种增强方式：p代表要删去边的百分比，取值范围为[0,1]，暂定为15%，即0.15
    # 原diff_map1s
    def diff_map1_show(self, p):
        # 输入至数组中，并传入cpp文件
        valid_cnt = self.valid_num()
        tmp = 1-p
        # 定义传入cpp中的roadnet信息为inp变量,txt中的一整行都要读入
        inp = []
        for item in valid_cnt:
            inp.append(self.info[item])

        n = int(len(inp) * tmp)
        diff_inp = random.sample(inp, n)

        return diff_inp


    # 获取最大的路口编号
    def maxid(self):
        File = open("/nas/user/wyh/dataset/roadnet/Shanghai/edgeOSM.txt")
        maxid = 0

        for line in File.readlines():
            item_list = line.strip().split()
            id = int(item_list[0])
            for veid in range(self.valid_edge_num):
                eid = self.valid_to_origin[veid]
                if id == eid:
                    maxid = max(maxid, int(item_list[1]))
                    maxid = max(maxid, int(item_list[2]))

        return maxid

    # 获取有效边的编号
    def valid_num(self):
        v_id = []
        for veid in range(self.valid_edge_num):
            eid = self.valid_to_origin[veid]
            v_id.append(eid)

        return v_id
        # print(v_id)

    def convert2_valid_num(self, idd):
        eid = self.valid_edge[idd]
        return eid

    def convert2_edge_num(self, idd):
        edgeNum = self.valid_to_origin[idd]
        return edgeNum

    def calSpatialDistance(self, x1, y1, x2, y2):
        lat1 = (math.pi / 180.0) * x1
        lat2 = (math.pi / 180.0) * x2
        lon1 = (math.pi / 180.0) * y1
        lon2 = (math.pi / 180.0) * y2
        R = 6378.137
        t = math.sin(lat1) * math.sin(lat2) + math.cos(lat1) * math.cos(lat2) * math.cos(lon2 - lon1)
        if t > 1.0:
            t = 1.0
        d = math.acos(t) * R * 1000
        return d

    def inside_zone(self, lat, lon):
        return self.zone_range[0] <= lat and lat <= self.zone_range[2] and self.zone_range[1] <= lon and lon <= self.zone_range[3]


    def shortest_route(self, t):
        tmp_e = -1
        eid = []
        n = 1
        route_ans = []

        for item in t:
            id_time = int(item[0])
            if id_time < 0:
                # 保存每条轨迹至tmp_ans
                tmp_ans = []
                tmp_ans.append(str(self.convert2_edge_num(eid[0])))
                length = len(eid)
                for i in range(1,length):
                    d, temp = self.shortestPath(eid[i-1], eid[i])
                    for elem in temp:
                        tmp_ans.append(str(self.convert2_edge_num(elem)))
                    tmp_ans.append(str(self.convert2_edge_num(eid[i])))
                route_ans.append(tmp_ans)
                eid.clear()
                continue

            e = int(item[3])
            e = self.convert2_valid_num(e)
            if e != tmp_e:
                tmp_e = e
                eid.append(e)

        # return route_ans

        # intersection_mode
        mmtraj_intersection_mode = []
        for item in route_ans:
            row = []
            for i in item:
                a, b = self.edgeNode[int(i)]
                row.append((a,b))
            mmtraj_intersection_mode.append(row)
        return mmtraj_intersection_mode


class diff_Map:
    def __init__(self, inp):
        self.edgeDis = []
        self.edgeNode = []
        self.edgeCord = []
        self.nodeSet = set()
        self.nodeDict = {}
        self.edgeDict = {}
        self.edgeRevDict = {}
        self.nodeEdgeDict = {}
        self.nodeEdgeRevDict = {}
        self.minLat = 1e18
        self.maxLat = -1e18
        self.minLon = 1e18
        self.maxLon = -1e18
        self.edgeNum = 0
        self.nodeNum = 0

        self.edge_to_cluster = {}
        self.cluster_to_edge = {}
        self.cluster_neighbor = {}
        self.cluster_neighbor_edge = {}
        self.cluster_neighbor_cluster = {}

        # 用于转换item_list[0]与eid之间的关系
        cnt = 0
        # small2big
        self.tmp_arr = []
        # big2small
        self.temp_arr = {}

        for item_list in inp:
            self.tmp_arr.append(int(item_list[0]))
            self.temp_arr[int(item_list[0])] = cnt
            a = int(item_list[1])
            b = int(item_list[2])
            self.edgeNode.append((a, b))
            self.nodeDict[a] = b
            if a not in self.nodeEdgeDict:
                self.nodeEdgeDict[a] = []
            if b not in self.nodeEdgeRevDict:
                self.nodeEdgeRevDict[b] = []
            self.nodeEdgeDict[a].append(self.edgeNum)
            self.nodeEdgeRevDict[b].append(self.edgeNum)
            self.nodeSet.add(a)
            self.nodeSet.add(b)
            num = int(item_list[3])
            dist = 0
            # print (item_list)
            self.edgeCord.append(list(map(float, item_list[4:])))
            for i in range(num):
                tmplat = float(item_list[4 + i * 2])
                tmplon = float(item_list[5 + i * 2])
                self.minLat = min(self.minLat, tmplat)
                self.maxLat = max(self.maxLat, tmplat)
                self.minLon = min(self.minLon, tmplon)
                self.maxLon = max(self.maxLon, tmplon)

            for i in range(num - 1):
                dist += self.calSpatialDistance(float(item_list[4 + i * 2]), float(item_list[5 + i * 2]),
                                                float(item_list[6 + i * 2]), float(item_list[7 + i * 2]))
            self.edgeDis.append(dist)
            self.edgeNum += 1

            cnt+=1

        self.nodeNum = len(self.nodeSet)

        for eid in range(self.edgeNum):
            self.edgeRevDict[eid] = []
        for eid in range(self.edgeNum):
            a, b = self.edgeNode[eid]
            self.edgeDict[eid] = []
            if b in self.nodeEdgeDict:
                for nid in self.nodeEdgeDict[b]:
                    self.edgeDict[eid].append(nid)
                    self.edgeRevDict[nid].append(eid)

        # self.igraph = igraph.Graph(directed=True)
        #  self.igraph.add_vertices(self.nodeNum)
        edge_list = []
        edge_weight_list = []
        for eid in range(self.edgeNum):
            a, b = self.edgeNode[eid]
            if (a == b):
                continue
            edge_list.append((a, b))
            edge_weight_list.append(self.edgeDis[eid])

        # self.igraph.add_edges(edge_list)
        #   self.igraph.es['dis'] = edge_weight_list

        print('edge Num: ', self.edgeNum)
        print('node Num: ', self.nodeNum)

        self.wayType = {}

    def calSpatialDistance(self, x1, y1, x2, y2):
        lat1 = (math.pi / 180.0) * x1
        lat2 = (math.pi / 180.0) * x2
        lon1 = (math.pi / 180.0) * y1
        lon2 = (math.pi / 180.0) * y2
        R = 6378.137
        t = math.sin(lat1) * math.sin(lat2) + math.cos(lat1) * math.cos(lat2) * math.cos(lon2 - lon1)
        if t > 1.0:
            t = 1.0
        d = math.acos(t) * R * 1000
        return d

    # valid到原图
    def s2b(self, eid):
        return self.tmp_arr[eid]
    
    def b2s(self, item_id):
        return self.temp_arr[item_id]

    # G(V, E): intersection为V点，road segment为E边
    def dgl_valid_map(self):
        u = []
        v = []
        for eid in range(self.edgeNum):
            # a，b存放路段的起止ID
            a, b = self.edgeNode[eid]
            u.append(a)
            v.append(b)

        graph = dgl.graph((u, v))
        return graph
    
    # start与end均为valid_id
    def shortestPathAll(self, start, end=-1, with_route=False):
        pq = PriorityQueue()
        pq.put((0, start))
        ans = []
        dist = [1e18 for i in range(self.edgeNum)]
        pred = [1e18 for i in range(self.edgeNum)]
        dist[start] = self.edgeDis[start]
        pred[start] = -1
        nodeset = {}
        ans.append(start)
        while (pq.qsize()):
            dis, id = pq.get()
            if id == end:
                break
            if id not in nodeset:
                nodeset[id] = 1
            else:
                continue
            for nid in self.edgeDict[id]:
                # if (nid in self.valid_edge):
                if dist[nid] > dist[id] + self.edgeDis[nid]:
                    dist[nid] = dist[id] + self.edgeDis[nid]
                    pred[nid] = id
                    pq.put((dist[nid], nid))
        if not with_route:
            pred = []
        if end != -1:
            return dist[end], pred
        return dist, pred

    def shortestPath(self, start, end, with_route=True):
        dis, pred = self.shortestPathAll(start, end, with_route)

        if end == -1:
            return dis, pred

        if with_route:
            id = end
            # print ('route: ')
            arr = [id]
            while (pred[id] >= 0 and pred[id] < 1e18):
                #  print (pred[id],end=',')
                id = pred[id]
                arr.append(id)
            arr = list(reversed(arr))
            # 去掉首尾元素
            new_arr = arr[1:-1]
        #    arr = [self.valid_edge[item] for item in arr]
            return dis, new_arr
        else:
            return dis


    def shortest_route(self, t):
        tmp_e = -1
        eid = []
        n = 1
        route_ans = []

        for item in t:
            id_time = int(item[0])
            if id_time < 0:
                # 保存每条轨迹至tmp_ans
                tmp_ans = []
                tmp_ans.append(str(self.s2b(eid[0])))
                length = len(eid)
                for i in range(1,length):
                    d, temp = self.shortestPath(eid[i-1], eid[i])
                    for elem in temp:
                        tmp_ans.append(str(self.s2b(elem)))
                    tmp_ans.append(str(self.s2b(eid[i])))
                route_ans.append(tmp_ans)
                eid.clear()
                continue

            e = int(item[3])
            e = self.b2s(e)
            if e != tmp_e:
                tmp_e = e
                eid.append(e)

        # return route_ans

        # intersection_mode
        mmtraj_intersection_mode = []
        for item in route_ans:
            row = []
            for i in item:
                i0 = self.b2s(int(i))
                a, b = self.edgeNode[i0]
                row.append((a,b))
            mmtraj_intersection_mode.append(row)
        return mmtraj_intersection_mode




# map为Map类数据，traj为轨迹文件路径string
def mmtraj_route(map, traj):
    x = map.valid_map_show()
    y = mm.avail_mm(x, traj)
    mmtraj = map.shortest_route(y)
    return mmtraj

# diff_map为数组[]形式的地图（转变为为diff_Map类数据），traj为轨迹文件路径string
def diff_mmtraj_route(diff_map, traj):
    x = diff_Map(diff_map)
    y = mm.avail_mm(diff_map, traj)
    mmtraj = x.shortest_route(y)
    return mmtraj

if __name__ == "__main__":
    SH_map = Map("/nas/user/wyh/dataset/roadnet/Shanghai", zone_range=[31.17491, 121.439492, 31.305073, 121.507001])
    print("valid")
    mmtraj = mmtraj_route(SH_map, "/nas/user/wyh/TNC/data/validtraj_20150401_ShangHai.txt")
    print(mmtraj)
    print("分割##################")
    # print("valid")
    # mmtraj = mmtraj_route(SH_map, "/nas/user/wyh/TNC/data/validtraj_20150401_ShangHai.txt")
    # print(mmtraj)
    print("diff")
    diff_SH_map_inp = SH_map.diff_map1_show(0.15)
    diff_mmtraj = diff_mmtraj_route(diff_SH_map_inp, "/nas/user/wyh/TNC/data/validtraj_20150401_ShangHai.txt")
    print(diff_mmtraj)
    # ans = SH_map.dgl_valid_map()
    # print(ans)


    # diff_SH_map = diff_Map(diff_SH_map_inp)

    # mmtraj = mmtraj_route(SH_map, "/nas/user/wyh/TNC/data/validtraj_20150401_ShangHai.txt")

    print("endd")

    # maps = Map("/nas/user/wyh/dataset/roadnet/Shanghai", zone_range = [31.17491, 121.439492, 31.305073, 121.507001]).diffmap_appnp(0.05)
    # maps = Map("/nas/user/wyh/dataset/roadnet/Shanghai").diffmap_appnp()
    # new_adj = Map("/nas/user/wyh/dataset/roadnet/Shanghai", zone_range = [31.17491, 121.439492, 31.305073, 121.507001]).diffmap_appnp(0.08)
    # recnt(new_adj)
    # v_num = Map("/nas/user/wyh/dataset/roadnet/Shanghai", zone_range=[31.17491, 121.439492, 31.305073, 121.507001]).valid_num()
    # Map("/nas/user/wyh/dataset/roadnet/Shanghai", zone_range=[31.17491, 121.439492, 31.305073, 121.507001]).valid_num()
    # maxid = Map("/nas/user/wyh/dataset/roadnet/Shanghai", zone_range=[31.17491, 121.439492, 31.305073, 121.507001]).maxid()
    # print(maxid) 57253
    # Map("/nas/user/wyh/dataset/roadnet/Shanghai", zone_range=[31.17491, 121.439492, 31.305073, 121.507001]).diff_map()
    # 随机删除20%的路段
    # map1 = Map("/nas/user/wyh/dataset/roadnet/Shanghai", zone_range=[31.17491, 121.439492, 31.305073, 121.507001])
    # map1.diff_map1(0.2)
    # 测试读入编码乱的路网 重新设计一个MMap类

    # 4验证合理性
    # MMap("/nas/user/wyh/essential_generate/validmap_ShangHai.txt").route(
    #     "/nas/user/wyh/essential_generate/draw/10_mmtraj_SH0401.txt")
    # MMap("/nas/user/wyh/essential_generate/SH_map1.txt").diff_route("/nas/user/wyh/essential_generate/draw/diff_10_mmtraj_SH0401.txt")


