#pragma once

#ifndef PCL_UTILS_H_
#define PCL_UTILS_H_

#include "./python_utils.h"
#include <pcl/pcl_base.h>
#include <pcl/point_types.h>

typedef  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr PCXYZRGBAPtr;
typedef  pcl::PointCloud<pcl::PointXYZRGBA> PCXYZRGBA;
typedef  pcl::PointCloud<pcl::PointXYZRGB>::Ptr PCXYZRGBPtr;
typedef  pcl::PointCloud<pcl::PointXYZRGB> PCXYZRGB;
typedef  pcl::PointCloud<pcl::PointXYZ>::Ptr PCXYZPtr;
typedef  pcl::PointCloud<pcl::PointXYZ> PCXYZ;
typedef  pcl::PointCloud<pcl::PointXYZRGBNormal> PCXYZRGBN;
typedef  pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr PCXYZRGBNPtr;
typedef  pcl::PointCloud<pcl::Normal> PCN;
typedef  pcl::PointCloud<pcl::Normal>::Ptr PCNPtr;


int ndarrayToPCL(bpn::ndarray np_cloud, PCNPtr &cloud){
    #ifdef _WIN32 || __linux__
    const uint32_t n_points = bp::len(np_cloud);
    const uint32_t n_feat = bp::len(np_cloud[0]);
    #else
    const uint32_t n_points = np_cloud.shape(0);
    const uint32_t n_feat = np_cloud.shape(1);
    #endif

    cloud->points.reserve(n_points);
    cloud->width=n_points;
    cloud->height=1;

    int x_idx = n_feat - 3;
    int y_idx = n_feat - 2;
    int z_idx = n_feat - 1;

    for(int i = 0; i < n_points; i++){
        pcl::Normal p;

        float nx = bp::extract<float>(np_cloud[i][x_idx]);
        float ny = bp::extract<float>(np_cloud[i][y_idx]);
        float nz = bp::extract<float>(np_cloud[i][z_idx]);

        p.normal_x = nx;
        p.normal_y = ny;
        p.normal_z = nz;

        //cout << unsigned(p.r) << " " << unsigned(p.g) << " " << unsigned(p.b) << endl;

        cloud->points.push_back(p);
    }
    cloud->is_dense = true;
    return n_points;
}


int ndarrayToPCL(bpn::ndarray np_cloud, PCXYZRGBNPtr &cloud){
    #ifdef _WIN32 || __linux__
    const uint32_t n_points = bp::len(np_cloud);
    const uint32_t n_feat = bp::len(np_cloud[0]);
    #else
    const uint32_t n_points = np_cloud.shape(0);
    const uint32_t n_feat = np_cloud.shape(1);
    #endif

    cloud->points.reserve(n_points);
    cloud->width=n_points;
    cloud->height=1;

    for(int i = 0; i < n_points; i++){
        pcl::PointXYZRGBNormal p;
        float x = bp::extract<float>(np_cloud[i][0]);
        float y = bp::extract<float>(np_cloud[i][1]);
        float z = bp::extract<float>(np_cloud[i][2]);

        unsigned int r = static_cast<unsigned int>(bp::extract<float>(np_cloud[i][3]));
        unsigned int g = static_cast<unsigned int>(bp::extract<float>(np_cloud[i][4]));
        unsigned int b = static_cast<unsigned int>(bp::extract<float>(np_cloud[i][5]));

        float nx = bp::extract<float>(np_cloud[i][6]);
        float ny = bp::extract<float>(np_cloud[i][7]);
        float nz = bp::extract<float>(np_cloud[i][8]);

        p.x=x;
        p.y=y;
        p.z=z;
        //printf("%f %f %f\n", p.x, p.y, p.z);
        p.r=r;
        p.g=g;
        p.b=b;

        p.normal_x = nx;
        p.normal_y = ny;
        p.normal_z = nz;

        //cout << unsigned(p.r) << " " << unsigned(p.g) << " " << unsigned(p.b) << endl;

        cloud->points.push_back(p);
    }
    cloud->is_dense = true;
    return n_points;
}


int ndarrayToPCL(bpn::ndarray np_cloud, PCXYZRGBPtr &cloud){
    #ifdef _WIN32 || __linux__
    const uint32_t n_points = bp::len(np_cloud);
    const uint32_t n_feat = bp::len(np_cloud[0]);
    #else
    const uint32_t n_points = np_cloud.shape(0);
    const uint32_t n_feat = np_cloud.shape(1);
    #endif

    cloud->points.reserve(n_points);
    cloud->width=n_points;
    cloud->height=1;

    for(int i = 0; i < n_points; i++){
        pcl::PointXYZRGB p;
        float x = bp::extract<float>(np_cloud[i][0]);
        float y = bp::extract<float>(np_cloud[i][1]);
        float z = bp::extract<float>(np_cloud[i][2]);

        unsigned int r = static_cast<unsigned int>(bp::extract<float>(np_cloud[i][3]));
        unsigned int g = static_cast<unsigned int>(bp::extract<float>(np_cloud[i][4]));
        unsigned int b = static_cast<unsigned int>(bp::extract<float>(np_cloud[i][5]));

        p.x=x;
        p.y=y;
        p.z=z;
        //printf("%f %f %f\n", p.x, p.y, p.z);
        p.r=r;
        p.g=g;
        p.b=b;

        //cout << unsigned(p.r) << " " << unsigned(p.g) << " " << unsigned(p.b) << endl;

        cloud->points.push_back(p);
    }
    cloud->is_dense = true;
    return n_points;
}


int ndarrayToPCL(bpn::ndarray np_cloud, PCXYZPtr &cloud){
    #ifdef _WIN32 || __linux__
    const uint32_t n_points = bp::len(np_cloud);
    const uint32_t n_feat = bp::len(np_cloud[0]);
    #else
    const uint32_t n_points = np_cloud.shape(0);
    const uint32_t n_feat = np_cloud.shape(1);
    #endif

    cloud->points.reserve(n_points);
    cloud->width=n_points;
    cloud->height=1;

    for(int i = 0; i < n_points; i++){
        pcl::PointXYZ p;
        float x = bp::extract<float>(np_cloud[i][0]);
        float y = bp::extract<float>(np_cloud[i][1]);
        float z = bp::extract<float>(np_cloud[i][2]);

        //unsigned int r = static_cast<unsigned int>(bp::extract<float>(P[i][3]));
        //unsigned int g = static_cast<unsigned int>(bp::extract<float>(P[i][4]));
        //unsigned int b = static_cast<unsigned int>(bp::extract<float>(P[i][5]));

        p.x=x;
        p.y=y;
        p.z=z;
        //printf("%f %f %f\n", p.x, p.y, p.z);
        //p.r=r;
        //p.g=g;
        //p.b=b;

        //cout << unsigned(p.r) << " " << unsigned(p.g) << " " << unsigned(p.b) << endl;

        cloud->points.push_back(p);
    }
    cloud->is_dense = true;
    return n_points;
}


#endif