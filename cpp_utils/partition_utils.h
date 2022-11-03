#pragma once

#ifndef PARTITION_UTILS_H_
#define PARTITION_UTILS_H_

#include <vector>

std::vector<int> create_partition(std::vector<std::vector<int> >& clusters_points_idx, int n_points){
    std::vector<int> partition(n_points,0);
    int clusters_num = clusters_points_idx.size();

    //printf("Num Clusters %i\n ", clusters_num);
    for (int i = 0; i < n_points; i++)
    {
        partition[i] = -1;
    }
    for(int i=0;i<clusters_num;i++)
    {
        std::vector<int > points_idx;
        points_idx=clusters_points_idx.at(i);
        int points_num=points_idx.size();
        //printf("Cluster %i has %i points, ", i, points_num);
        for(int j=0;j<points_num;j++)
        {
            int point_idx = points_idx.at(j);
            partition[point_idx] = i;
        }   
    }
    return partition;
}

#endif