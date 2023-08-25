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
# import dgl
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


    # def diffmap(self, vid):
    #     edgeFile = open("/nas/user/wyh/dataset/roadnet/Shanghai/edgeOSM.txt")
    #     for line in edgeFile.readlines():
    #         item_list = line.strip().split()
    #         id = int(item_list[0])
    #         if id in vid:
    #             with open("/nas/user/wyh/pre_contrastive/validmap_ShangHai.txt", 'a') as f:
    #                 f.write(line)

    # 以dgl.graph的格式存储有效原图（暂时仅存储点边关系不涉及具体属性）
    def dgl_valid_map(self):
        u = []
        v = []
        for eid in range(self.valid_edge_num):
            for nid in self.edgeDict[eid]:
                u.append(eid)
                v.append(nid)

        graph = dgl.graph((u, v))
        return graph

    # 以dgl.graph的格式记录新生成的增强地图（增强方式随机，暂时仅存储点边关系不涉及具体属性）
    def aug_dgl_valid_map(self, p):
        u = []
        v = []

        tmp = 10 * (1 - p) + 1
        # print(tmp)
        # cnt = 0

        for eid in range(self.valid_edge_num):
            for nid in self.edgeDict[eid]:
                # 以1-p的概率保留边
                r = random.randint(1, 10)
                if r < tmp:
                    u.append(eid)
                    v.append(nid)

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
    def valid_map_trajmm(self):
        valid_cnt = self.valid_num()
        # 定义传入cpp中的roadnet信息为inp变量,txt中的一整行都要读入
        inp = []
        for item in valid_cnt:
            inp.append(self.info[item])
            # print("stop")

        return inp
        # print(type(inp))
        # print(inp)

    # aug后的rn进行mm
    def aug_map_trajmm(self):
        return 0

    # 输出新地图为txt
    def valid_map(self):
        File = open("/nas/user/wyh/dataset/roadnet/Shanghai/edgeOSM.txt")

        for line in File.readlines():
            item_list = line.strip().split()
            id = int(item_list[0])
            for veid in range(self.valid_edge_num):
                eid = self.valid_to_origin[veid]
                if id == eid:
                    with open("/nas/user/wyh/pre_contrastive/validmap_ShangHai.txt", 'a') as f:
                        f.write(line)
                        break

    # 第一种增强方式：p代表要删去边的百分比，取值范围为[0,1]，输出为txt
    def diff_map1(self, p):
        File = open("/nas/user/wyh/dataset/roadnet/Shanghai/edgeOSM.txt")
        tmp = 10*(1-p)+1
        # print(tmp)
        cnt = 0

        for line in File.readlines():
            item_list = line.strip().split()
            id = int(item_list[0])
            for veid in range(self.valid_edge_num):
                eid = self.valid_to_origin[veid]
                if id == eid:
                    r = random.randint(1, 10)
                    if r < tmp:
                        with open("/nas/user/wyh/essential_generate/SH_map1.txt", 'a') as f:
                            f.write(line)
                            cnt += 1
                            print(cnt)
                            break

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


    def draw_map(self):
        line = []
        for eid in tqdm(range(self.edgeNum)):
            a1, b1 = self.edgeCord[eid][1], self.edgeCord[eid][0]
            a2, b2 = self.edgeCord[eid][-1], self.edgeCord[eid][-2]
            line.append([a1, a2])
            line.append([b1, b2])

        plt.plot(*line)

        plt.savefig('sh_global_map.pdf')

    def draw_valid_map(self):
        line = []
        for veid in tqdm(range(self.valid_edge_num)):
            eid = self.valid_to_origin[veid]
            a1, b1 = self.edgeCord[eid][1], self.edgeCord[eid][0]
            a2, b2 = self.edgeCord[eid][-1], self.edgeCord[eid][-2]
            line.append([a1, a2])
            line.append([b1, b2])

        plt.plot(*line)

        plt.savefig('valid_sh_global_map.pdf')

class MMap:
    def __init__(self, dir):
        edgeFile = open(dir)
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
        # self.RNodeDict = {}
        # self.RNodeRevDict = {}

        self.edge_to_cluster = {}
        self.cluster_to_edge = {}
        self.cluster_neighbor = {}
        self.cluster_neighbor_edge = {}
        self.cluster_neighbor_cluster = {}

        self.o_to_n = {}
        self.n_to_o = {}

        for line in edgeFile.readlines():
            item_list = line.strip().split()

            oid = int(item_list[0])
            self.o_to_n[oid] = self.edgeNum
            self.n_to_o[self.edgeNum] = oid

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

        # 读入数据集结束
        # 交叉路口数目
        self.nodeNum = len(self.nodeSet)

        # 路段ID
        # 路段-路段表示开始
        for eid in range(self.edgeNum):
            self.edgeRevDict[eid] = []
            # self.RNodeRevDict[eid] = []

        # 构建图; edgeDict存储了路段的连通关系
        # edgeDict[a] = b 代表路段a指向路段b的有向连接
        for eid in range(self.edgeNum):
            # a，b存放路段的起止ID
            a, b = self.edgeNode[eid]
            self.edgeDict[eid] = []
            # self.RNodeDict[eid] = []
            if b in self.nodeEdgeDict:
                for nid in self.nodeEdgeDict[b]:
                    self.edgeDict[eid].append(nid)
                    self.edgeRevDict[nid].append(eid)
                    # if nid in self.nodeEdgeRevDict:
                    #     for Rnid in self.nodeEdgeRevDict[nid]:
                    #         self.RNodeDict[eid].append(Rnid)
                    #         self.RNodeRevDict[Rnid].append(eid)

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

    def route(self, mmdir):
        mmFile = open(mmdir)
        # route_arr = []
        tmp_e = -1
        eid = []
        n = 1
        org_map = Map("/nas/user/wyh/dataset/roadnet/Shanghai",
                      zone_range=[31.17491, 121.439492, 31.305073, 121.507001])

        for line in mmFile.readlines():
            item_list = line.strip().split()
            id_time = int(item_list[0])
            if id_time < 0:
                with open("/nas/user/wyh/essential_generate/draw/try.txt", 'a') as f:
                    f.write(str(org_map.convert2_edge_num(eid[0])) + " ")
                    length = len(eid)
                    for i in range(1, length):
                        d, temp = self.shortestPath(eid[i-1], eid[i])
                        for elem in temp:
                            f.write(str(org_map.convert2_edge_num(elem)) + " ")
                        f.write(str(org_map.convert2_edge_num(eid[i])) + " ")
                    f.write("\n" + "-" + str(n) + "\n")
                    n += 1
                    eid.clear()
                    continue
            e = int(item_list[3])
            e = org_map.convert2_valid_num(e)
            if e != tmp_e:
                tmp_e = e
                eid.append(e)

    def diff_route(self, mmdir):
        mmFile = open(mmdir)
        # route_arr = []
        tmp_e = -1
        eid = []
        n = 1
        # org_map = Map("/nas/user/wyh/dataset/roadnet/Shanghai",
        #               zone_range=[31.17491, 121.439492, 31.305073, 121.507001])

        for line in mmFile.readlines():
            item_list = line.strip().split()
            id_time = int(item_list[0])
            if id_time < 0:
                with open("/nas/user/wyh/essential_generate/draw/diff_try.txt", 'a') as f:
                    f.write(str(self.n_to_o[eid[0]]) + " ")
                    length = len(eid)
                    for i in range(1, length):
                        d, temp = self.shortestPath(eid[i-1], eid[i])
                        for elem in temp:
                            f.write(str(self.n_to_o[elem]) + " ")
                        f.write(str(self.n_to_o[eid[i]]) + " ")
                    f.write("\n" + "-" + str(n) + "\n")
                    n += 1
                    eid.clear()
                    continue
            e = int(item_list[3])
            # e = org_map.convert2_valid_num(e)
            e = self.o_to_n[e]
            if e != tmp_e:
                tmp_e = e
                eid.append(e)

if __name__ == "__main__":
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


    inp = Map("/nas/user/wyh/dataset/roadnet/Shanghai", zone_range=[31.17491, 121.439492, 31.305073, 121.507001]).valid_map_trajmm()
    print("分割##################")
    # print(mm.de_read(inp, "/nas/user/wyh/dataset/traj/Shanghai/20150401_cleaned_mm_trajs.txt"))
    print(mm.avail_mm(inp, "/nas/user/wyh/TNC/data/validtraj_20150401_ShangHai.txt"))
    print("endd")
    



