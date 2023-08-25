#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h> 
#include <pybind11/functional.h> 
#include <pybind11/chrono.h>

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <list>
#include <set>
#include <map>
#include <queue>
#include <random>
#include <sstream>
// #include "progressbar.hpp"

#define PI_Divided_by_180 0.0174532925199432957694
#define Length_Per_Rad 111226.29021121707545
#define INF 1e17
#define SIGMAZ 4.07
#define HLAF_SIGMAZ2 -0.0301843053
#define FRAC_SQR_2PI_SIGMAZ 0.0980202163
using namespace std;

////////////////Data structs///////////////////
class Point {
public:
    void set_lon(double lon){this->lon = lon;}
    void set_lat(double lat){this->lat = lat;}
    void set_time(int time){this->time = time;}

    double lon;
    double lat;
    int time;

//    Point() : time(-1) {}
};

class Road {
public:
    vector<Point> path;
    int orgID;
    int startPointId;
    int endPointId;
    // int wayType;
    // string wayName;
    double Len;
} r[130000];

class Traj {
public:
    vector<Point> path;
} t[150000];

class Score {
public:
    int RoadID;
    long double score;
    int preIndex;
    double distFromStart;
    Score() {}
    Score(int RoadID, long double score, int pre, double distLeft) {
        this->RoadID = RoadID;
        this->score = score;
        this->preIndex = pre;
        this->distFromStart = distLeft;
    }
};

class Node {
public:
    int end;
    double dist;

    bool operator<(const Node& rhs) const { return dist > rhs.dist; }
};

struct AdjNode {
    int endPointId;
    int edgeId;
    AdjNode* next;
};

///////////////////////////////////////////////////

///////////////////////The globals////////////////////
int n = 0;
// int m = 150000;
//定义m->轨迹条数，暂时读2条
// int m = 80928;
int m = 1;
int k = 60000;
// double maxLon, maxLat;
// double minLon, minLat;
double minLat = 31.17491;
double minLon = 121.439492;
double maxLat = 31.305073;
double maxLon = 121.507001;
list<set<int> > Candidates;
list<int> ans[150000];
map<pair<int, int>, pair<double, double> > shortestDistPair;
int gridWidth, gridHeight;
double gridSize;
vector<AdjNode*> adjList;

//调参区域
list<int> grid[5000][5000];
int L = 50;
long double BETA[31] = { 0,           0.49037673,  0.82918373,  1.24364564,  1.67079581,  2.00719298,
                         2.42513007,  2.81248831,  3.15745473,  3.52645392,  4.09511775,  4.67319795,
                         5.41088180,  6.47666590,  6.29010734,  7.80752112,  8.09074504,  8.08550528,
                         9.09405065,  11.09090603, 11.87752824, 12.55107715, 15.82820829, 17.69496773,
                         18.07655652, 19.63438911, 25.40832185, 23.76001877, 28.43289797, 32.21683062,
                         34.56991141 };
//调参区域到此结束^_^!

////////////////////////////////////////////////////

////////////////////functions///////////////////

//点到点的距离
double distance(double lon1, double lat1, double lon2, double lat2) {
    double delta_lat = lat1 - lat2;
    double delta_lon = (lon2 - lon1) * cos(lat1 * PI_Divided_by_180);
    return Length_Per_Rad * sqrt(delta_lat * delta_lat + delta_lon * delta_lon);
}

//计算路径长度
double RoadLength(int i) {
    double length = 0;
    vector<Point>::iterator p, nxtp;
    p = nxtp = r[i].path.begin();
    nxtp++;
    while (nxtp != r[i].path.end()) {
        length += distance(p->lon, p->lat, nxtp->lon, nxtp->lat);
        p++;
        nxtp++;
    }
    return length;
}


double arccos(double lon, double lat, Point a, Point b) {
    double v1x = a.lon - lon;
    double v1y = a.lat - lat;
    double v2x = b.lon - a.lon;
    double v2y = b.lat - a.lat;
    return (v1x * v2x + v1y * v2y) / sqrt((v1x * v1x + v1y * v1y) * (v2x * v2x + v2y * v2y));
}

//点到边的距离
double distance(double lon, double lat, vector<Point>& path) {
    double ans = INF;
    vector<Point>::iterator p, nxtp;
    for (p = path.begin(); p != path.end(); ++p) {
        double tmp = distance(lon, lat, p->lon, p->lat);
        if (tmp < ans)
            ans = tmp;
    }
    p = path.begin();
    nxtp = path.begin();
    nxtp++;
    while (nxtp != path.end()) {
        if (arccos(lon, lat, *p, *nxtp) <= 0 && arccos(lon, lat, *nxtp, *p) <= 0) {
            double A = (nxtp->lat - p->lat);
            double B = -(nxtp->lon - p->lon);
            double C = p->lat * (nxtp->lon - p->lon) - p->lon * (nxtp->lat - p->lat);
            double tmp = abs(A * lon + B * lat + C) / sqrt(A * A + B * B) * Length_Per_Rad;
            if (tmp < ans)
                ans = tmp;
        }
        p++;
        nxtp++;
    }
    return ans;
}

//点到线距离，且返回距离起点的长度
double distance(double lon, double lat, vector<Point>& path, double& Len) {
    double tmpLen = 0;
    double ans = INF;
    double tmpd = 0;
    double x1 = -1, y1 = -1;
    for (vector<Point>::iterator Iter = path.begin(); Iter != path.end(); Iter++) {
        if (x1 != -1 && y1 != -1) {
            double x2 = Iter->lat;
            double y2 = Iter->lon;
            double dist = distance(y1, x1, lon, lat);
            if (dist < ans) {
                ans = dist;
                tmpLen = tmpd;
            }
            double v1x = x2 - x1;
            double v1y = y2 - y1;
            double v2x = lat - x1;
            double v2y = lon - y1;
            double v3x = lat - x2;
            double v3y = lon - y2;
            if (v1x * v2x + v1y * v2y > 0 && -v1x * v3x - v1y * v3y > 0 && (v1x != 0 || v1y != 0)) {
                double rate = ((lat - x2) * v1x + (lon - y2) * v1y) / (-v1x * v1x - v1y * v1y);
                double x = rate * x1 + (1 - rate) * x2;
                double y = rate * y1 + (1 - rate) * y2;
                double dist = distance(y, x, lon, lat);
                if (dist < ans) {
                    ans = dist;
                    tmpLen = tmpd + distance(y1, x1, y, x);
                }
            }
            tmpd += distance(y1, x1, y2, x2);
        }
        x1 = Iter->lat;
        y1 = Iter->lon;
    }
    Len = tmpLen;
    return ans;
}

bool cmp1(pair<double, double>& p1, pair<double, double>& p2) { return p1.first < p2.first; }

double EmissionPro(double distance) { return exp(distance * distance * HLAF_SIGMAZ2 * FRAC_SQR_2PI_SIGMAZ); }

double shortestPathLength(int Id1, int Id2, double deltaT) {
    vector<double> dist = vector<double>(k);
    vector<bool> flag = vector<bool>(k);
    for (int i = 0; i < k; ++i) {
        dist[i] = INF;
        flag[i] = false;
    }
    dist[Id1] = 0;
    priority_queue<Node> s;
    s.push((Node){ Id1, 0.0 });
    while (!s.empty()) {
        Node x = s.top();
        s.pop();
        int u = x.end;
        if (flag[u])
            continue;
        flag[u] = true;
        if (u == Id2)
            break;
        for (AdjNode* i = adjList[u]->next; i != NULL; i = i->next) {
            if (dist[i->endPointId] > dist[u] + r[i->edgeId].Len) {
                dist[i->endPointId] = dist[u] + r[i->edgeId].Len;
                s.push((Node){ i->endPointId, dist[i->endPointId] });
            }
        }
    }
    return dist[Id2];
}

int GetIndex(vector<Score>& row) {
    int res = -1;
    long double prob = -1;
    for (size_t i = 0; i < row.size(); i++) {
        if (prob < row[i].score) {
            prob = row.at(i).score;
            res = i;
        }
    }
    return res;
}

/////////////////////////////////////////////

////////GridIndex////////////

// service for gridmaking

void Grid_Segment(int id, vector<Point>::iterator pt1, vector<Point>::iterator pt2) {
    double x1 = pt1->lon - minLon;
    double y1 = pt1->lat - minLat;
    double x2 = pt2->lon - minLon;
    double y2 = pt2->lat - minLat;
    int row = (int)(y1 / gridSize);
    int nxtrow = (int)(y2 / gridSize);
    int col = (int)(x1 / gridSize);
    int nxtcol = (int)(x2 / gridSize);
    int i;

    if (row == nxtrow && col == nxtcol) {
        if (row >= gridHeight || row < 0 || col >= gridWidth || col < 0 ||
            (grid[row][col].size() > 0 && grid[row][col].back() == id))
            return;
        grid[row][col].push_back(id);
        return;
    }

    if (row == nxtrow) {
        for (i = min(col, nxtcol); i <= max(col, col); ++i) {
            if (row >= gridHeight || row < 0 || col >= gridWidth || col < 0 ||
                (grid[row][col].size() > 0 && grid[row][col].back() == id))
                continue;
            grid[row][i].push_back(id);
        }
        return;
    }

    if (col == nxtcol) {
        for (i = min(row, nxtrow); i < max(row, nxtrow); ++i) {
            if (row >= gridHeight || row < 0 || col >= gridWidth || col < 0 ||
                (grid[row][col].size() > 0 && grid[row][col].back() == id))
                continue;
            grid[i][col].push_back(id);
        }
        return;
    }

    double A = y2 - y1;
    double B = x2 - x1;
    double C = x1 * y2 - x2 * y1;
    pair<double, double> plots[5000];
    int in = 0;
    for (i = min(row, nxtrow); i <= max(row, nxtrow); ++i) {
        plots[in++] = make_pair((C + B * i * gridSize) / A, i * gridSize);
    }
    for (i = min(col, nxtcol); i <= max(col, nxtcol); ++i) {
        plots[in++] = make_pair(i * gridSize, (-C + A * i * gridSize) / B);
    }
    plots[in++] = make_pair(x1, y1);
    plots[in++] = make_pair(x1, y1);
    sort(plots, plots + in, cmp1);

    for (i = 0; i < in - 1; ++i) {
        // always left-down
        int pts_row = (int)(plots[i].second / gridSize + 1e-9);
        int pts_col = (int)(plots[i].first / gridSize + 1e-9);
        int pts_nxtrow = (int)(plots[i + 1].second / gridSize + 1e-9);
        int pts_nxtcol = (int)(plots[i + 1].first / gridSize + 1e-9);
        int row = min(pts_row, pts_nxtrow);
        int col = min(pts_col, pts_nxtcol);
        if (row >= gridHeight || row < 0 || col >= gridWidth || col < 0 ||
            (grid[row][col].size() > 0 && grid[row][col].back() == id))
            continue;
        grid[row][col].push_back(id);
    }
    return;
}

/////////////////////////////////////////////////

/////////////Algorithm/////////////

//第i条轨迹的候选 
// service for matching_hmm
void getCandidate(int i) {
    Candidates.clear();
    // 有显示
    // cout<<"t[i].path.begin().lon = "<<t[i].path.begin()->lon<<endl;
    // cout<<"t[i].path.end().lon = "<<t[i].path.end()->lon<<endl;
    for (auto j = t[i].path.begin(); j != t[i].path.end(); j++) {
        double x = j->lon - minLon;
        double y = j->lat - minLat;
        set<int> temp;
        set<int>::iterator q;
        int row = int(y / gridSize);
        int col = int(x / gridSize);
        // 有显示
        cout<<"row: "<<row<<endl;
        cout<<"col: "<<col<<endl;

        for (int r = row - 1; r <= row + 1; ++r)
            for (int c = col - 1; c <= col + 1; ++c) {
                // 经过此处
                // cout<<"@ "<<endl;
                list<int>::iterator p;
                for (p = grid[r][c].begin(); p != grid[r][c].end(); p++) {
                    if (temp.find(*p) != temp.end())
                        continue;
                    temp.insert(*p);
                    cout<<"$ ";
                    // 完全没经过这里
                }
            }
        Candidates.push_back(temp);
    }
    // candidate中没有东西
    // for (list<set<int> >::iterator it = Candidates.begin (); it != Candidates.end (); ++it) {
    //     for (set<int>::iterator s = it->begin (); s != it->end (); ++s) {
    //                     cout << *s << " ";
    //     }
    //     cout << endl;
    // }
}
/////////////////////////////////////////////////

// 【交互】读入路网 & 轨迹数据
int de_read(std::vector<std::vector<string>>& input, string route_trajfile) {
    //初始化AdjList
    for (int i = 0; i < k; ++i) {
        AdjNode* head = new AdjNode();
        head->endPointId = i;
        head->next = NULL;
        adjList.push_back(head);
    }
    // 路网行数 相当于原来的n
    int rows = input.size();

    // Read in roads保存于r[i]中
    for (int i = 0; i < rows; ++i) {
        int id = std::stoi (input [i] [0]);
        int p1 = std::stoi (input [i] [1]);
        int p2 = std::stoi (input [i] [2]);
        int c = std::stoi (input [i] [3]);
        r[i].orgID = id;
        r[i].startPointId = p1;
        r[i].endPointId = p2;
        // r[id].wayName = way_string;
        // r[id].wayType = way_type;

        // Read in adjList
        AdjNode* current = adjList[p1];
        while (current->next != NULL) current = current->next;
        AdjNode* tmpAdjNode = new AdjNode();
        tmpAdjNode->endPointId = p2;
        tmpAdjNode->edgeId = i;
        tmpAdjNode->next = NULL;
        current->next = tmpAdjNode;
        
        
        for (int j = 0; j < c; ++j) {
            double x = std::stod(input[i][2*j+4]);
            double y = std::stod(input[i][2*j+5]);
            Point* p = new Point();
            p->lat = x;
            p->lon = y;
            r[i].path.push_back(*p);
            delete p;
        }
        r[i].Len = RoadLength(i);
    }

    // Read in Trajs
    std::ifstream trajfile(route_trajfile);
    // 因为经纬度为小数所以这里应为double而不是int
    std::vector<std::vector<double> > traj_info; // 创建保存数据的容器
    if (trajfile.is_open()) { // 判断文件是否成功打开
        std::string line;
        while (getline(trajfile, line)) { // 读取文件的每一行
            std::istringstream ss(line); // 创建字符串流，用于分离每个整数
            std::vector<double> roww; // 创建每一行数据的向量
            double num;
            while (ss >> num) { // 将每个整数从字符串中提取并添加到行向量中
                roww.push_back(num);
            }
            traj_info.push_back(roww);
        }
        trajfile.close(); // 关闭文件
    } else {
        std::cout << "无法打开轨迹文件。" << std::endl;
    }

    int j = 0;
    for (int i = 0; i < m; ++i) {
        while (1) {
            int time;
            time = int(traj_info[j][0]);
            if (time <= m){
                j++;
                break;
            }
            double x, y;
            x = traj_info[j][1];
            y = traj_info[j][2];
            Point* p = new Point();
            p->lat = x;
            p->lon = y;
            p->time = time;
            t[i].path.push_back(*p);
            j++;
        }
    }

    // cout<<traj_info[0][0]<<endl;

// gridmaking
    gridWidth = int(distance(maxLon, minLat, minLon, minLat) / L) + 1;
    cout<<gridWidth<<endl;
    gridHeight = int((maxLat - minLat) / (maxLon - minLon) * double(gridWidth)) + 1;
    gridSize = (maxLon - minLon) / double(gridWidth);
    cout<<gridSize<<endl;
    for (int i = 0; i < n; ++i) {
        vector<Point>::iterator p = r[i].path.begin();
        vector<Point>::iterator nxtp = r[i].path.begin();
        nxtp++;
        while (nxtp != r[i].path.end()) {
            Grid_Segment(i, p, nxtp);
            p++;
            nxtp++;
        }
    }

    // matching_hmm
    for (int i = 0; i < m; ++i) {
        // 对于一条轨迹，获取每个采样点的候选路段
        // 整体思路如下：首先找到每个采样点所在的格子（每个格子大小是50m*50m的）
        // 所以如果只选择所载格子的所有路段会有遗漏
        // 根据论文中的做法，我们选取当前GPS所在的格子以及周围3*3区域的格子中所有路段作为候选路段
        // 经过这里了
        // cout<<"# ";
        getCandidate(i);
        list<set<int> >::iterator in = Candidates.begin();
        // 这句话获取这条轨迹平均采样率，用于得到转移概率中的Beta参数
        int sampleRate = (int)(t[i].path.size()) > 1 ? (t[i].path.back().time - t[i].path.front().time) /
                                                           ((int)(t[i].path.size()) - 1)
                                                     : (t[i].path.back().time - t[i].path.front().time);
        
        // cout<<(t[i].path.back().time - t[i].path.front().time) /
        //                                                    ((int)(t[i].path.size()) - 1)<<endl;
        // 做一个越界的判断
        if (sampleRate > 30)
           sampleRate = 30;
        long double BT = BETA[sampleRate];
        vector<vector<Score> > scoreMatrix;
        vector<Point>::iterator formerTrajPoint = t[i].path.end();
        bool flag = true;
        int currentTrajPointIndex = 0;
        for (vector<Point>::iterator trajectoryIterator = t[i].path.begin();
             trajectoryIterator != t[i].path.end(); trajectoryIterator++) {
            // 定义相邻两个采样点之间的距离
            double distBetweenTwoTrajPoints;
            double deltaT = -1;
            if (formerTrajPoint != t[i].path.end()) {
                deltaT = trajectoryIterator->time - formerTrajPoint->time;
                distBetweenTwoTrajPoints = distance(trajectoryIterator->lon, trajectoryIterator->lat,
                                                    formerTrajPoint->lon, formerTrajPoint->lat);
            }
            long double currentMaxProb = -1e10;
            vector<Score> scores;
            long double* emissionProbs = new long double[in->size()];
            int currentCanadidateRoadIndex = 0;
            for (set<int>::iterator p = (*in).begin(); p != (*in).end(); p++) {
                int preColumnIndex = -1;
                double currentDistfromStart = 0;
                // 计算采样点到候选路段的距离
                double DistBetweenTrajPointAndRoad = distance(
                    trajectoryIterator->lon, trajectoryIterator->lat, r[*p].path, currentDistfromStart);
                emissionProbs[currentCanadidateRoadIndex] = EmissionPro(DistBetweenTrajPointAndRoad);
                // flag判定是不是第一个采样点
                if (!flag) {
                    long double currentMaxProbTmp = -1e10;
                    int formerCanadidateRoadIndex = 0;
                    // 遍历上一个采样点得到的Score矩阵
                    for (vector<Score>::iterator sp = scoreMatrix.back().begin();
                         sp != scoreMatrix.back().end(); sp++) {
                        double formerDistfromStart = sp->distFromStart;
                        double formerDistToEnd = r[sp->RoadID].Len - formerDistfromStart;
                        double routeNetworkDistBetweenTwoRoads;
                        double routeNetworkDistBetweenTwoTrajPoints;
                        if (*p == sp->RoadID)
                            // 如果上一个候选点和下一个候选点在同一个距离，可以直接求出距离
                            // 这是因为Score里面保存了这个GPS点到路段起始的距离
                            // 这个其实是一个时间的优化
                            routeNetworkDistBetweenTwoTrajPoints =
                                fabs(currentDistfromStart - formerDistfromStart);
                        else {
                            // 这个对最短路进行记忆化
                            // 就是说如果两个点计算过了，下一次就不用再算了
                            pair<int, int> ind = make_pair(r[sp->RoadID].endPointId, r[*p].startPointId);
                            // 找到了之前算过的最短路
                            if (shortestDistPair.find(ind) != shortestDistPair.end() &&
                                shortestDistPair[ind].first < INF)
                                routeNetworkDistBetweenTwoRoads = shortestDistPair[ind].first;
                            else {
                                routeNetworkDistBetweenTwoRoads = shortestPathLength(
                                        r[sp->RoadID].endPointId, r[*p].startPointId, deltaT);
                                shortestDistPair[ind] =
                                        make_pair(routeNetworkDistBetweenTwoRoads, deltaT);
                                }
                            // }
                            routeNetworkDistBetweenTwoTrajPoints =
                                routeNetworkDistBetweenTwoRoads + currentDistfromStart + formerDistToEnd;
                        }
                        long double transactionProb = exp(-fabs((long double)distBetweenTwoTrajPoints -
                                      (long double)routeNetworkDistBetweenTwoTrajPoints) / BT) / BT;
                        // 概率=转移概率*发射概率
                        long double tmpTotalProbForTransaction = sp->score * transactionProb;
                        if (currentMaxProbTmp < tmpTotalProbForTransaction) {
                            currentMaxProbTmp = tmpTotalProbForTransaction;
                            preColumnIndex = formerCanadidateRoadIndex;
                        }
                        formerCanadidateRoadIndex++;
                    }
                    emissionProbs[currentCanadidateRoadIndex] *= currentMaxProbTmp;
                }
                scores.push_back(Score(*p, emissionProbs[currentCanadidateRoadIndex], preColumnIndex,
                                       currentDistfromStart));
                // 记录当前最大的概率
                if (currentMaxProb < emissionProbs[currentCanadidateRoadIndex])
                    currentMaxProb = emissionProbs[currentCanadidateRoadIndex];
                currentCanadidateRoadIndex++;
            }
            delete[] emissionProbs;
            formerTrajPoint = trajectoryIterator;
            currentTrajPointIndex++;
            // 这里做了一个归一化，就是除了最大的概率
            // 这样可以保证计算过程中不会出现精度问题
            // 这是一个不错的优化！
            // 不会影响准确率，但是可以提高精度
            for (int j = 0; j < scores.size(); ++j) scores[j].score /= currentMaxProb;
            scoreMatrix.push_back(scores);
            // 如果没有候选点，那么就将轨迹断开。
            if (scores.size() == 0) {
                flag = true;
                formerTrajPoint = t[i].path.end();
            } else
                flag = false;
            in++;
            // 经过这里了
            // cout << "第" << currentTrajPointIndex << "个候选路段匹配成功" << "\n";
        }
        // 这一段是从最后开始倒叙输出最后的答案
        int startColumnIndex = GetIndex(scoreMatrix.back());
        int temp0 = 0;
        int temp = 0;
        for (int j = scoreMatrix.size() - 1; j >= 0; j--) {
            if (startColumnIndex != -1) {
                cout<<1<<" ";
                // ans默认为0，完全没经过这里
                temp0 = scoreMatrix[j][startColumnIndex].RoadID;
                // 将坐标代回
                temp = r[temp0].orgID;
                ans[i].push_front(temp);
                startColumnIndex = scoreMatrix[j][startColumnIndex].preIndex;
            } else  //断点处理
            {
                ans[i].push_front(temp);
                if (j > 0)
                    startColumnIndex = GetIndex(scoreMatrix[j - 1]);
            }
        }
        cout << "第" << i << "条路正常" << "\n";
    }

    // main function
    list<int>::iterator p;
    ofstream file;
    file.open("diff_valid.txt");
    // progressbar bar(m);
    // bar.set_todo_char(" ");
    // bar.set_done_char("█");
    // bar.set_opening_bracket_char("{");
    // bar.set_closing_bracket_char("}");
    for (int i = 0; i < m; ++i) {
        // bar.update(i);
        for (int j = 0; j < t[i].path.size(); ++j) {
            // cout << t[i].path[j].time << ' ' << t[i].path[j].lat << ' ' << t[i].path[j].lon << ' ' << ans[i].front() << endl;
            file << t[i].path[j].time << ' ' << t[i].path[j].lat << ' ' << t[i].path[j].lon << ' ' << ans[i].front() << endl;
            ans[i].pop_front();
        }
        cout<<"第"<<i<<"条轨迹匹配完毕"<<endl;
        // cout << "-" << i + 1 << endl;
        file << "-" << i + 1 << endl;
    }
    file.close();


    return 0;
}

    // int main() {
    //     read();
    //     gridMaking();
    //     matching_hmm();
    //     list<int>::iterator p;
    //     ofstream file;
    //     file.open("diff_valid.txt");
    //     // progressbar bar(m);
    //     // bar.set_todo_char(" ");
    //     // bar.set_done_char("█");
    //     // bar.set_opening_bracket_char("{");
    //     // bar.set_closing_bracket_char("}");
    //     for (int i = 0; i < m; ++i) {
    //         // bar.update(i);
    //         for (int j = 0; j < t[i].path.size(); ++j) {
    //             // cout << t[i].path[j].time << ' ' << t[i].path[j].lat << ' ' << t[i].path[j].lon << ' ' << ans[i].front() << endl;
    //             file << t[i].path[j].time << ' ' << t[i].path[j].lat << ' ' << t[i].path[j].lon << ' ' << ans[i].front() << endl;
    //             ans[i].pop_front();
    //         }
    //         cout<<"第"<<i<<"条轨迹匹配完毕"<<endl;
    //         // cout << "-" << i + 1 << endl;
    //         file << "-" << i + 1 << endl;
    //     }
    //     file.close();
    //     return 0;
    // }


// void gridMaking() {
//     gridWidth = int(distance(maxLon, minLat, minLon, minLat) / L) + 1;
//     // cout<<gridWidth<<endl;
//     gridHeight = int((maxLat - minLat) / (maxLon - minLon) * double(gridWidth)) + 1;
//     gridSize = (maxLon - minLon) / double(gridWidth);
//     for (int i = 0; i < n; ++i) {
//         vector<Point>::iterator p = r[i].path.begin();
//         vector<Point>::iterator nxtp = r[i].path.begin();
//         nxtp++;
//         while (nxtp != r[i].path.end()) {
//             Grid_Segment(i, p, nxtp);
//             p++;
//             nxtp++;
//         }
//     }
// }



// 隐马尔科夫
// void matching_hmm() {
//     for (int i = 0; i < m; ++i) {
//         // 对于一条轨迹，获取每个采样点的候选路段
//         // 整体思路如下：首先找到每个采样点所在的格子（每个格子大小是50m*50m的）
//         // 所以如果只选择所载格子的所有路段会有遗漏
//         // 根据论文中的做法，我们选取当前GPS所在的格子以及周围3*3区域的格子中所有路段作为候选路段
//         // cout<<"# ";经过这里了
//         getCandidate(i);
//         list<set<int> >::iterator in = Candidates.begin();
//         // 这句话获取这条轨迹平均采样率，用于得到转移概率中的Beta参数
//         int sampleRate = (int)(t[i].path.size()) > 1 ? (t[i].path.back().time - t[i].path.front().time) /
//                                                            ((int)(t[i].path.size()) - 1)
//                                                      : (t[i].path.back().time - t[i].path.front().time);
        
//         // cout<<(t[i].path.back().time - t[i].path.front().time) /
//         //                                                    ((int)(t[i].path.size()) - 1)<<endl;
//         // 做一个越界的判断
//         if (sampleRate > 30)
//            sampleRate = 30;
//         long double BT = BETA[sampleRate];
//         vector<vector<Score> > scoreMatrix;
//         vector<Point>::iterator formerTrajPoint = t[i].path.end();
//         bool flag = true;
//         int currentTrajPointIndex = 0;
//         for (vector<Point>::iterator trajectoryIterator = t[i].path.begin();
//              trajectoryIterator != t[i].path.end(); trajectoryIterator++) {
//             // 定义相邻两个采样点之间的距离
//             double distBetweenTwoTrajPoints;
//             double deltaT = -1;
//             if (formerTrajPoint != t[i].path.end()) {
//                 deltaT = trajectoryIterator->time - formerTrajPoint->time;
//                 distBetweenTwoTrajPoints = distance(trajectoryIterator->lon, trajectoryIterator->lat,
//                                                     formerTrajPoint->lon, formerTrajPoint->lat);
//             }
//             long double currentMaxProb = -1e10;
//             vector<Score> scores;
//             long double* emissionProbs = new long double[in->size()];
//             int currentCanadidateRoadIndex = 0;
//             for (set<int>::iterator p = (*in).begin(); p != (*in).end(); p++) {
//                 int preColumnIndex = -1;
//                 double currentDistfromStart = 0;
//                 // 计算采样点到候选路段的距离
//                 double DistBetweenTrajPointAndRoad = distance(
//                     trajectoryIterator->lon, trajectoryIterator->lat, r[*p].path, currentDistfromStart);
//                 emissionProbs[currentCanadidateRoadIndex] = EmissionPro(DistBetweenTrajPointAndRoad);
//                 // flag判定是不是第一个采样点
//                 if (!flag) {
//                     long double currentMaxProbTmp = -1e10;
//                     int formerCanadidateRoadIndex = 0;
//                     // 遍历上一个采样点得到的Score矩阵
//                     for (vector<Score>::iterator sp = scoreMatrix.back().begin();
//                          sp != scoreMatrix.back().end(); sp++) {
//                         double formerDistfromStart = sp->distFromStart;
//                         double formerDistToEnd = r[sp->RoadID].Len - formerDistfromStart;
//                         double routeNetworkDistBetweenTwoRoads;
//                         double routeNetworkDistBetweenTwoTrajPoints;
//                         if (*p == sp->RoadID)
//                             // 如果上一个候选点和下一个候选点在同一个距离，可以直接求出距离
//                             // 这是因为Score里面保存了这个GPS点到路段起始的距离
//                             // 这个其实是一个时间的优化
//                             routeNetworkDistBetweenTwoTrajPoints =
//                                 fabs(currentDistfromStart - formerDistfromStart);
//                         else {
//                             // 这个对最短路进行记忆化
//                             // 就是说如果两个点计算过了，下一次就不用再算了
//                             pair<int, int> ind = make_pair(r[sp->RoadID].endPointId, r[*p].startPointId);
//                             // 找到了之前算过的最短路
//                             if (shortestDistPair.find(ind) != shortestDistPair.end() &&
//                                 shortestDistPair[ind].first < INF)
//                                 routeNetworkDistBetweenTwoRoads = shortestDistPair[ind].first;
//                             else {
//                                 routeNetworkDistBetweenTwoRoads = shortestPathLength(
//                                         r[sp->RoadID].endPointId, r[*p].startPointId, deltaT);
//                                 shortestDistPair[ind] =
//                                         make_pair(routeNetworkDistBetweenTwoRoads, deltaT);
//                                 }
//                             // }
//                             routeNetworkDistBetweenTwoTrajPoints =
//                                 routeNetworkDistBetweenTwoRoads + currentDistfromStart + formerDistToEnd;
//                         }
//                         long double transactionProb = exp(-fabs((long double)distBetweenTwoTrajPoints -
//                                       (long double)routeNetworkDistBetweenTwoTrajPoints) / BT) / BT;
//                         // 概率=转移概率*发射概率
//                         long double tmpTotalProbForTransaction = sp->score * transactionProb;
//                         if (currentMaxProbTmp < tmpTotalProbForTransaction) {
//                             currentMaxProbTmp = tmpTotalProbForTransaction;
//                             preColumnIndex = formerCanadidateRoadIndex;
//                         }
//                         formerCanadidateRoadIndex++;
//                     }
//                     emissionProbs[currentCanadidateRoadIndex] *= currentMaxProbTmp;
//                 }
//                 scores.push_back(Score(*p, emissionProbs[currentCanadidateRoadIndex], preColumnIndex,
//                                        currentDistfromStart));
//                 // 记录当前最大的概率
//                 if (currentMaxProb < emissionProbs[currentCanadidateRoadIndex])
//                     currentMaxProb = emissionProbs[currentCanadidateRoadIndex];
//                 currentCanadidateRoadIndex++;
//             }
//             delete[] emissionProbs;
//             formerTrajPoint = trajectoryIterator;
//             currentTrajPointIndex++;
//             // 这里做了一个归一化，就是除了最大的概率
//             // 这样可以保证计算过程中不会出现精度问题
//             // 这是一个不错的优化！
//             // 不会影响准确率，但是可以提高精度
//             for (int j = 0; j < scores.size(); ++j) scores[j].score /= currentMaxProb;
//             scoreMatrix.push_back(scores);
//             // 如果没有候选点，那么就将轨迹断开。
//             if (scores.size() == 0) {
//                 flag = true;
//                 formerTrajPoint = t[i].path.end();
//             } else
//                 flag = false;
//             in++;
//             // cout << "第" << currentTrajPointIndex << "个候选路段匹配成功" << "\n";
//         }
//         // 这一段是从最后开始倒叙输出最后的答案
//         int startColumnIndex = GetIndex(scoreMatrix.back());
//         int temp0 = 0;
//         int temp = 0;
//         for (int j = scoreMatrix.size() - 1; j >= 0; j--) {
//             if (startColumnIndex != -1) {
//                 // cout<<1<<" ";
//                 // ans默认为0，完全没经过这里
//                 temp0 = scoreMatrix[j][startColumnIndex].RoadID;
//                 // 将坐标代回
//                 temp = r[temp0].orgID;
//                 ans[i].push_front(temp);
//                 startColumnIndex = scoreMatrix[j][startColumnIndex].preIndex;
//             } else  //断点处理
//             {
//                 ans[i].push_front(temp);
//                 if (j > 0)
//                     startColumnIndex = GetIndex(scoreMatrix[j - 1]);
//             }
//         }
//         // cout << "第" << i << "条路正常" << "\n";
//     }
// }







// void work(Trajectory t) {
//     Trajectory new_t;
// }


int add(int a, int b) {
    return a + b;
}

namespace py = pybind11;
PYBIND11_MODULE(mm, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring
    m.def("de_read", &de_read, "A function which trys receiving list type parameter");
}
