import mm
import ast
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

import torch
import dgl
import scipy
import networkx as nx
import folium
import folium.plugins
from rtree import index
# import scipy.sparse as sp
# from dgl.nn.pytorch.conv.appnpconv import APPNPConv
# from scipy.linalg import fractional_matrix_power, inv
# from sklearn.preprocessing import MinMaxScaler

# for folium：创建颜色列表
colors = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet', 'pink', 'brown', 'black']
# global dfs_time
dfs_time = 0 # 时间戳
# m = folium.Map(location=[31.2389, 121.4992], zoom_start=12)

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
    
    def nx_valid_map(self):
        G = self.dgl_valid_map()
        # 将dgl.Graph对象转换为network稀疏矩阵对象，此时不用考虑映射问题
        nx_G = dgl.to_networkx(G)

        return nx_G

    def valid_adj(self):
        G = self.dgl_valid_map()
        # 将dgl.Graph对象转换为network稀疏矩阵对象，此时不用考虑映射问题
        nx_g = dgl.to_networkx(G)
        # 使用networkx.to_numpy_matrix()函数，将一个networkx.Graph对象转换为一个numpy.ndarray对象，表示图的邻接矩阵
        adj = nx.to_numpy_matrix(nx_g)
        # 打印邻接矩阵
        return adj

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
        # [('121238', '57252', '57253', '2', '31.1841042', '121.452009', '31.1859025', '121.4541695'), ('121239', '57253', '57252', '2', '31.1859025', '121.4541695', '31.1841042', '121.452009')]

    def v_section_mode_map(self):
        sec_v = []
        map = self.valid_map_show()
        for m in map:
            sec_v.append(int(m[1]))
            sec_v.append(int(m[2]))

        # print(sec_e)
        return sec_v

    def e_section_mode_map(self):
        sec_e = []
        map = self.valid_map_show()
        for m in map:
            sec_e.append((int(m[1]), int(m[2])))

        # print(sec_e)
        return sec_e


    # 第一种增强方式：p代表要删去边的百分比，取值范围为[0,1]，暂定为15%，即0.15（TODO：不可删除割边）
    # 原diff_map1s
    def diff_map1_show(self, p):
        bridge_edge = self.find_bridge()
        # 记录为割边的valid_num
        bridge_edge_Num = []

        for e in bridge_edge:
            edgeNum = self.nodeEdgeDict[e[0]]
            for item in edgeNum:
                if item in self.nodeEdgeRevDict[e[1]]:
                    bridge_edge_Num.append(item)

        valid_cnt = self.valid_num()
        non_bridge_cnt = []
        for v in valid_cnt:
            if v in bridge_edge_Num:
                continue
            else:
                non_bridge_cnt.append(v)

        # 非割边删除
        tmp = 1-p
        n = int(len(valid_cnt) * tmp)
        assert n<len(non_bridge_cnt), "check diff_map1!"
        remain_non_bridge_num = random.sample(non_bridge_cnt, n)

        diff_map_num = bridge_edge_Num + remain_non_bridge_num
        # 升序排列
        diff_map_num.sort()

        # 定义传入cpp中的roadnet信息为inp变量,txt中的一整行都要读入
        diff_inp = []
        for item in diff_map_num:
            diff_inp.append(self.info[item])
            
        return diff_inp
    
    def find_bridge(self):
        g = self.dgl_valid_map()
        # 定义DFS算法中用到的变量
        low = [0] * g.num_nodes() # 每个节点能够回溯到的最早的祖先节点的时间戳
        disc = [-1] * g.num_nodes() # 每个节点被发现的时间戳
        parent = [-1] * g.num_nodes() # 每个节点的父节点
        bridge = [] # 桥边列表

        # 定义DFS算法
        def dfs(u):
            global dfs_time
            # 标记当前节点已被发现，并记录时间戳
            disc[u] = dfs_time
            low[u] = dfs_time
            dfs_time += 1
            # 遍历当前节点的所有邻居节点
            for v in g.successors(u).tolist():
                # 如果邻居节点未被发现，则将当前节点作为其父节点，并继续DFS
                if disc[v] == -1:
                    parent[v] = u
                    dfs(v)
                    # 回溯时更新当前节点能够回溯到的最早祖先节点的时间戳
                    low[u] = min(low[u], low[v])
                    # 如果当前节点能够回溯到的最早祖先节点的时间戳小于邻居节点被发现的时间戳，则说明当前节点和邻居节点之间是桥边
                    if low[v] > disc[u]:
                        bridge.append((u, v))
                # 如果邻居节点已被发现，且不是当前节点的父节点，则更新当前节点能够回溯到的最早祖先节点的时间戳
                elif v != parent[u]:
                    low[u] = min(low[u], disc[v])

        # 对图中的每个节点执行DFS算法，找出所有的桥边
        for i in range(g.num_nodes()):
            if disc[i] == -1:
                dfs(i)

        for b in bridge:
            pass
        return bridge

    # 第二种增强方式：p代表要增加边的百分比，取值范围为[0,1]，暂定为10%，即0.1，但未考虑图的连通性（感觉上肯定连通）
    def diff_map2_show(self, p):
        a =self.valid_edge_num
        add_edge_num = int(p * a)
        f = 0
        new_RS = []

        # 创建一个空的rtree对象
        idx = index.Index()

        VM = self.valid_map_show()
        sec_id = []
        sec = []

        for i in VM:
            tmp = int(i[3])
            in_sec_id = int(i[1])
            out_sec_id = int(i[2])
            if in_sec_id in sec_id:
                pass
            else:
                sec_id.append(in_sec_id)
                sec.append({"id": in_sec_id, "coords": (float(i[4]), float(i[5]))})


            if out_sec_id in sec_id:
                pass
            else:
                sec_id.append(out_sec_id)
                sec.append({"id": out_sec_id, "coords": (float(i[4+(tmp-1)*2]), float(i[5+(tmp-1)*2]))})

        for i in sec:
            idx.insert(i["id"], i["coords"])

        # # 创建一个线对象列表，每个线对象是一个字典，包含两个端点的id和坐标
        # lines = []

        # 遍历每个点对象
        for i in sec:
            if f > add_edge_num:
                break
            # 查询最近邻的点对象，返回结果数量为2（包括自身）
            nearest = list(idx.nearest(i["coords"], 2))
            
            # 如果查询结果有两个，则说明存在最近邻的点对象（不包括自身）
            if len(nearest) == 2:
                # 获取最近邻的点对象的整数标识符
                n_id = nearest[1]
                
                # 根据整数标识符从点对象列表中找到对应的点对象
                n = [q for q in sec if q["id"] == n_id][0]

                new_RS.append((str(a+f), str(i["id"]), str(n["id"]), str(2), str(i["coords"][0]), str(i["coords"][1]), str(n["coords"][0]), str(n["coords"][1])))

                f+=1
        
        # return new_RS
        diff_map2_inp = VM + new_RS

        return diff_map2_inp
    
    # 第三种增强方式：使用噪声图加法对图进行扰动，考虑图的连通性，效果并不好
    def diff_map3_show(self):
        diff_RS = []
        a =self.edgeNum
        f = 0

        VM = self.valid_map_show()
        sec_id = []
        sec = []

        for i in VM:
            tmp = int(i[3])
            in_sec_id = int(i[1])
            out_sec_id = int(i[2])
            if in_sec_id in sec_id:
                pass
            else:
                sec_id.append(in_sec_id)
                sec.append({"id": in_sec_id, "coords": (float(i[4]), float(i[5]))})


            if out_sec_id in sec_id:
                pass
            else:
                sec_id.append(out_sec_id)
                sec.append({"id": out_sec_id, "coords": (float(i[4+(tmp-1)*2]), float(i[5+(tmp-1)*2]))})

        # print("sec_init_done")
        # print(sec)
        # return 0

        G = self.nx_valid_map()
        # 根据原图的度分布生成一个噪声图
        l = len(G.nodes())

        p = np.array(list(dict(G.degree()).values())) / l # 每个节点的度数占总度数的比例
        Q_ = nx.fast_gnp_random_graph(l, p.mean(), directed=True) # 生成一个随机有向图，每条边的存在概率为平均度数比例
        Q = nx.MultiDiGraph(Q_)

        # 将原图与噪声图相加，得到一个扰动后的图
        H = nx.compose(G, Q) # 合并两个图，相同的边只保留一条

        edges = H.edges()
        edges_list = list(edges)


        # 迭代EdgeView对象，得到每一条边的起点和终点
        for e in edges_list:
            if e in self.edgeNode:
                v = self.edgeNode.index(e)
                diff_RS.append(self.info[v])
                # print(e)
            else:
                flag = False
                for i in sec:
                    # print(i)
                    if e[0] == i["id"]:
                        # print("find")
                        in_id = str(i["id"])
                        in_x = str(i["coords"][0])
                        in_y = str(i["coords"][1])
                        for n in sec:
                            if e[1] == n["id"]:
                                diff_RS.append((str(a+f), in_id, str(n["id"]), str(2), in_x, in_y, str(n["coords"][0]), str(n["coords"][1])))
                                f+=1
                                flag = True
                                break
                        if flag:
                            print("f=", f)
                            break

        
        return diff_RS
                

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


    def shortest_route(self, t, sec_mode=True):
        tmp_e = -1
        eid = []
        n = 1
        route_ans = []

        print("最短路补全开始")

        for item in tqdm(t):
            id_time = int(item[0])
            if id_time < 0:
                # 保存每条轨迹至tmp_ans
                tmp_ans = []
                if len(eid) > 1:
                    tmp_ans.append(str(self.convert2_edge_num(eid[0])))
                    length = len(eid)
                    for i in range(1,length):
                        if eid[i] != -2:
                            d, temp = self.shortestPath(eid[i-1], eid[i])
                            for elem in temp:
                                tmp_ans.append(str(self.convert2_edge_num(elem)))
                            tmp_ans.append(str(self.convert2_edge_num(eid[i])))
                        else:
                            tmp_ans.append(-999)
                else:
                    tmp_ans.append(-999)

                route_ans.append(tmp_ans)
                eid = []
                tmp_e = -1 
                continue

            e = int(item[3])
            # 匹配路段为0的情况
            if e!=0:
                e = self.convert2_valid_num(e)
            else:
                e = -2
            if e != tmp_e:
                tmp_e = e
                eid.append(e)

        if sec_mode==True:
            # intersection_mode
            mmtraj_intersection_mode = []
            for item in route_ans:
                row = []
                for i in item:
                    a, b = self.edgeNode[int(i)]
                    row.append((a,b))
                mmtraj_intersection_mode.append(row)
            return mmtraj_intersection_mode
        else:
            return route_ans
    
    # folium画图
    def draw_traj_on_map(self, T):
        # zoom_start参数：缩放级别越高，地图显示的范围越小，细节越清晰
        # m = folium.Map(location=[31.2389, 121.4992], zoom_start=12)
        num = 0
        for ts in T:
            m = folium.Map(location=[31.2389, 121.4992], zoom_start=12)
            points = []
            for t in ts:
                item = self.info[int(t)]
                cnt = int(item[3])
                for n in range(cnt):
                    points.append([float(item[4+2*n]), float(item[5+2*n])])
            
            # 添加轨迹线对象
            color = random.choice(colors)
            line = folium.plugins.AntPath(points, color='brown', weight=5, opacity=0.8, pulseColor=None, delay=1000).add_to(m)
            for i in range(len(points) - 1):
                folium.Marker(points[i], icon=folium.Icon(color='green'), popup=f'{i}').add_to(line)

            m.save('../folium_figure/sh_'+str(num)+'.html')
            num+=1

            # folium.PolyLine(points, color=color, weight=10, opacity=0.8).add_to(m)

    def rs2sec(self, rs_id):
        if rs_id >= 0:
            a, b = self.edgeNode[rs_id]
        else:
            a = -999
            b = -999
        return a, b


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
        self.info = inp

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


    def shortest_route(self, t, sec_mode=True):
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
                eid = []
                tmp_e = -1
                continue

            e = int(item[3])
            e = self.b2s(e)
            if e != tmp_e:
                tmp_e = e
                eid.append(e)

        if sec_mode==True:
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
        else:
            return route_ans
        
    # folium画图
    def draw_traj_on_map(self, T):
        # zoom_start参数：缩放级别越高，地图显示的范围越小，细节越清晰
        # m = folium.Map(location=[31.2389, 121.4992], zoom_start=12)
        num = 0
        for ts in T:
            m = folium.Map(location=[31.2389, 121.4992], zoom_start=12)
            points = []
            for t in ts:
                t = self.b2s(int(t))
                item = self.info[t]
                cnt = int(item[3])
                for n in range(cnt):
                    points.append([float(item[4+2*n]), float(item[5+2*n])])
            
            # 添加轨迹线对象
            color = random.choice(colors)
            line = folium.plugins.AntPath(points, color='black', weight=5, opacity=0.8, pulseColor=None, delay=1000).add_to(m)
            for i in range(len(points) - 1):
                folium.Marker(points[i], icon=folium.Icon(color='green'), popup=f'{i}').add_to(line)

            m.save('../folium_figure/new_2_sh_'+str(num)+'.html')
            num+=1

            # folium.PolyLine(points, color=color, weight=10, opacity=0.8).add_to(m)
        # m.save('../folium_figure/diff_sh_1.html')

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


# map为Map类数据，traj为轨迹文件路径string, 现在变为traj_inp格式
def mmtraj_route(map, traj, T = True):
    x = map.valid_map_show()
    y = mm.avail_mm(x, traj)
    # 删去False输出路口序列
    mmtraj = map.shortest_route(y, T)
    return mmtraj

# map为Map类数据，traj_file_path = "/nas/user/wyh/TNC/traj_dealer/30w_section_mode/30w_traj.txt"  
#  route_file_path = "/nas/user/wyh/TNC/traj_dealer/30w_section_mode/30w_route.txt" 
# 生成route文件
def generate_route_file(map, traj_file_path, route_file_path): 
    mmtrajs = map.shortest_route(test_traj(traj_file_path), False)

    for mmtraj in tqdm(mmtrajs):
        with open(route_file_path, 'a') as f:
            for item in mmtraj:
                f.write(str(item) + ' ')
            f.write('\n')

# diff_map为数组[]形式的地图（转变为为diff_Map类数据），traj为轨迹文件路径string
def diff_mmtraj_route(diff_map, traj):
    x = diff_Map(diff_map)
    y = mm.avail_mm(diff_map, traj)
    # 删去False输出路口序列
    mmtraj = x.shortest_route(y, False)
    return mmtraj, x

# graph是一个networkx.Graph对象，表示图数据；返回一个torch.Tensor对象，shape为[n_nodes, 2]，表示每个节点的入度和出度特征向量
def compute_degree_features(graph):
    nodes = list(graph.nodes())
    n_nodes = len(nodes)
    # print(n_nodes)
    # 初始化特征矩阵
    features = torch.zeros(n_nodes, 2)
    # 遍历每个节点
    for i, node in enumerate(nodes):
        # 计算节点的入度和出度
        in_degree = graph.in_degree(node)
        out_degree = graph.out_degree(node)
        # 将入度和出度的值归一化到[0, 1]区间
        in_degree = in_degree / (n_nodes - 1)
        out_degree = out_degree / (n_nodes - 1)
        # 将入度和出度的值赋给特征矩阵
        features[i, 0] = in_degree
        features[i, 1] = out_degree
    # 返回特征矩阵
    return features

# r 为traj_file_route，e.g."/nas/user/wyh/TNC/traj_dealer/10_valid_traj_ShangHai.txt"
def traj_inp(r):
    traj_inp = []
    trajFile = open(r)

    for line in trajFile.readlines():
        str_data = line.strip()
        list_data = ast.literal_eval(str_data)
        if type(list_data)==int:
            continue
        else:
            traj_inp.append(list_data)

    # print(len(traj_inp))
    # [[['1', '1.1', '1.2', '123'], ['1', '1.1', '1.2', '123']],   [['1', '1.1', '1.2', '123'], ['1', '1.1', '1.2', '123']]]
            
    return traj_inp

# 文件r格式："/nas/user/wyh/TNC/traj_dealer/30w_section_mode/error_30w_traj.txt"
def traj_inp0(r):
    traj_inp = []
    traj_ = []
    traj_path = open(r)

    for line in traj_path.readlines():
        item_list = line.strip().split()
        if len(item_list) <= 1:
            traj_inp.append(traj_)
            traj_ = []
            continue
        else:
            tsmp = str(int(float(item_list[0])))
            lat = item_list[1]
            lon = item_list[2]
            rid = item_list[3]
            traj_.append([tsmp, lat, lon, rid])


    return traj_inp


def test_traj(r):
    traj_ = []
    traj_path = open(r)
    for line in traj_path.readlines():
        item_list = line.strip().split()
        if len(item_list) <= 1:
            traj_.append([int(float(item_list[0]))])
            continue
        else:
            tsmp = int(float(item_list[0]))
            lat = float(item_list[1])
            lon = float(item_list[2])
            rid = int(float(item_list[3]))
            traj_.append([tsmp, lat, lon, rid])

    return traj_

    
def data_process():
    n = 0
    SH_map = Map("/nas/user/wyh/dataset/roadnet/Shanghai", zone_range=[31.17491, 121.439492, 31.305073, 121.507001])
    trajinp = traj_inp("/nas/user/wyh/TNC/traj_dealer/30w_valid_traj_ShangHai.txt")
    # mmtrajs = mmtraj_route(SH_map, trajinp, False)

    x = SH_map.valid_map_show()
    mmtrajs = mm.avail_mm(x, trajinp)

    # print(trajinp)

    for mmtraj in tqdm(mmtrajs):
        n+=1
        with open("/nas/user/wyh/TNC/traj_dealer/30w_section_mode/30w_traj.txt", 'a') as f:
            for item in mmtraj:
                f.write(str(item) + ' ')
            if n>0:
                f.write('\n')

    
    # for mmtraj in tqdm(mmtrajs):
    #     n+=1
    #     if n>240000:
    #         print("val", n)
    #         with open("/nas/user/wyh/TNC/traj_dealer/30w_section_mode/30w_val.txt", 'a') as f_val:
    #             if n>240001:
    #                 f_val.write('\n')
    #             for sec_pair in mmtraj:
    #                 for sec in sec_pair:
    #                     f_val.write(str(sec) + ' ')

    #     elif n>210000:
    #         print("test", n)
    #         with open("/nas/user/wyh/TNC/traj_dealer/30w_section_mode/30w_test.txt", 'a') as f_test:
    #             if n>210001:
    #                 f_test.write('\n')
    #             for sec_pair in mmtraj:
    #                 for sec in sec_pair:
    #                     f_test.write(str(sec) + ' ')

    #     else:
    #         print("train", n)
    #         with open("/nas/user/wyh/TNC/traj_dealer/30w_section_mode/30w_train.txt", 'a') as f_train:
    #             if n>1:
    #                 f_train.write('\n')
    #             for sec_pair in mmtraj:
    #                 for sec in sec_pair:
    #                     f_train.write(str(sec) + ' ')

    print("finish")
    return 0


def test_map():
    # m = folium.Map(location=[31.2389, 121.4992], zoom_start=12)
    SH_map = Map("/nas/user/wyh/dataset/roadnet/Shanghai", zone_range=[31.17491, 121.439492, 31.305073, 121.507001])
    print(SH_map.diff_map1_show(0.2))
    # print(SH_map.valid_map_show())
    # trajinp0 = traj_inp("/nas/user/wyh/TNC/traj_dealer/10_valid_traj_ShangHai.txt")

    # trajinp = traj_inp0("/nas/user/wyh/TNC/traj_dealer/30w_section_mode/1_traj_test.txt")

    # x = SH_map.valid_map_show()
    # mmtrajs = mm.avail_mm(x, trajinp)

    # # # print(trajinp)

    # for mmtraj in tqdm(mmtrajs):
    #     with open("/nas/user/wyh/TNC/traj_dealer/30w_section_mode/5_traj.txt", 'a') as f:
    #         for item in mmtraj:
    #             f.write(str(item) + ' ')
    #         f.write('\n')


    # traj_file_path = "/nas/user/wyh/TNC/traj_dealer/30w_section_mode/30w_traj.txt"  
    # route_file_path = "/nas/user/wyh/TNC/traj_dealer/30w_section_mode/new_30w_route.txt" 
    # generate_route_file(SH_map, traj_file_path, route_file_path)


    print("endd")


if __name__ == "__main__":
    # print(traj_inp("/nas/user/wyh/TNC/traj_dealer/10_valid_traj_ShangHai.txt"))
    test_map()
    # draw_traj_on_map()
    # pass

    # data_process()


