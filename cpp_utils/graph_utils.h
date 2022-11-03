#include "./python_utils.h"
#include <Eigen/Dense>

Eigen::MatrixXi st2neighbours(bpn::ndarray target, int n_points, int k){
    Eigen::MatrixXi neighbours(n_points, k);
    const uint32_t n_elems = get_len(target);
    int row_idx = 0;
    int col_idx = 0;

    #ifdef _WIN32 || __linux__
    for (int i=0; i < n_elems; i++){
        if ( (i % k) == 0 && i > 0 ){
            row_idx++;
            col_idx = 0;
        }
        neighbours(row_idx, col_idx) = bp::extract<uint32_t>(target[i]);
        col_idx++;
    }
    #else
    uint32_t* target_data = reinterpret_cast<uint32_t*>(target.get_data());
    for (int i=0; i < n_elems; i++){
        if ( (i % k) == 0 && i > 0 ){
            row_idx++;
            col_idx = 0;
        }
        neighbours(row_idx, col_idx) = target_data[i];
        col_idx++;
    }
    #endif
    return neighbours;
}

Eigen::MatrixXd convert_normals(bpn::ndarray normals, int n_points){
    Eigen::MatrixXd eig_normals(n_points, 3);
    #ifdef _WIN32 || __linux__
    for (int i=0; i < n_points; i++){
        eig_normals(i, 0) = (double) bp::extract<float>(normals[i][0]);
        eig_normals(i, 1) = (double) bp::extract<float>(normals[i][1]);
        eig_normals(i, 2) = (double) bp::extract<float>(normals[i][2]);
        //printf("%f, %f, %f\n", eig_normals(i, 0), eig_normals(i, 1), eig_normals(i, 2));
    }
    #else
    float* normals_data = reinterpret_cast<float*>(normals.get_data());
    int j = 0;
    for (int i=0; i < n_points; i++){
        eig_normals(i, 0) = (double) normals_data[j];
        eig_normals(i, 1) = (double) normals_data[j+1];
        eig_normals(i, 2) = (double) normals_data[j+2];
        j += 3;
        //printf("%f, %f, %f\n", eig_normals(i, 0), eig_normals(i, 1), eig_normals(i, 2));
    }
    #endif
    return eig_normals;
}