// #include<boost/python.hpp>
// valid地图匹配
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

using namespace boost::python;

////////////////Data structs///////////////////
class Point {
public:
    void set_lon(std::double lon){this->lon = lon;}
    void set_lat(std::double lat){this->lat = lat;}
    void set_time(std::int time){this->time = time;}

    double lon;
    double lat;
    int time;

//    Point() : time(-1) {}
};

BOOST_PYTHON_MODULE(point)
{
    class_<Point>("Point")
        .def("set_lon", &Point::set_lon)
        .def("set_lat", &Point::set_lat)
        .def("set_time", &Point::set_time)
    ;
}

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

BOOST_PYTHON_MODULE(road)
{
    class_<Road>("Road")
        .def("set_lon", &Point::set_lon)
        .def("set_lat", &Point::set_lat)
        .def("set_time", &Point::set_time)
    ;
}

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

