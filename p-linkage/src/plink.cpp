#include <chrono>
#include "../../cpp_utils/python_utils.h"
#include "../../cpp_utils/partition_utils.h"
#include "../../cpp_utils/graph_utils.h"

#include "PCAFunctions.h"
#include "ClusterGrowPLinkage.h"
#include "utils.h"


namespace bp = boost::python;
namespace bpn = boost::python::numpy;


int ndarrayToPC(bpn::ndarray np_cloud, PointCloud<double> &pointData){
    const uint32_t n_points = get_len(np_cloud);
    const uint32_t n_feat = get_len_dim2(np_cloud);

    pointData.pts.reserve(n_points);

    #ifdef _WIN32 || __linux__
    for(int i = 0; i < n_points; i++){
        double x = (double)bp::extract<float>(np_cloud[i][0]);
        double y = (double)bp::extract<float>(np_cloud[i][1]);
        double z = (double)bp::extract<float>(np_cloud[i][2]);
        pointData.pts.push_back(PointCloud<double>::PtData(x, y, z));
    }
    #else
    float* cloud_data = reinterpret_cast<float*>(np_cloud.get_data());
    int next_col_idx = 0;
    for(int i = 0; i < n_points * n_feat; i+=3){
        if (next_col_idx == 0){
            double x = (double) cloud_data[i];
            double y = (double) cloud_data[i+1];
            double z = (double) cloud_data[i+2];
            pointData.pts.push_back(PointCloud<double>::PtData(x, y, z));
        }
        next_col_idx += 3;
        if (next_col_idx == n_feat){
            next_col_idx = 0;
        }
    }
    #endif
    return n_points;
}


#ifdef _WIN32 || __linux__
PyObject * plinkage(const bpn::ndarray & cloud, const int k, const float angle, const int min_cluster_size, const float angle_dev)
#else // apple
bp::list plinkage(const bpn::ndarray & cloud, const int k, const float angle, const int min_cluster_size, const float angle_dev)
#endif
{
    double theta = (double)angle / 180.0 * M_PI;

    PointCloud<double> pointData;

    int n_points = ndarrayToPC(cloud, pointData);
    //std::cout << "extracted " << n_points << " points" << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    // step2: build kd-tree
    std::vector<PCAInfo> pcaInfos;
    PCAFunctions pcaer;
    pcaer.PCA(pointData, k, pcaInfos);

    // step3: run point segmentation algorithm
    std::vector<std::vector<int>> clusters;
    PLANE_MODE planeMode = SURFACE;               // PLANE  SURFACE
    //std::cout << "init segmenter" << std::endl; 
    ClusterGrowPLinkage segmenter(k, theta, planeMode);
    segmenter.setData(pointData, pcaInfos);
    //std::cout << "run segmentation" << std::endl; 
    segmenter.run(clusters, min_cluster_size, angle_dev, n_points);
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;

    std::vector<int> partition = create_partition(clusters, n_points);

    #ifdef _WIN32 || __linux__
    return to_py_tuple::convert(Components_tuple(clusters, partition, (float)elapsed.count()));
    #else // apple
    bp::list pyobject = to_py_tuple::convert_lists(Components_tuple(clusters, partition, (float)elapsed.count()));
    return pyobject;
    #endif
}

#ifdef _WIN32 || __linux__
PyObject * plinkage_geo(const bpn::ndarray & cloud, const bpn::ndarray & target, const bpn::ndarray & normals,
        const int k, const float angle, const int min_cluster_size, const float angle_dev, const bool use_normals)
#else // apple
bp::list plinkage_geo(const bpn::ndarray & cloud, const bpn::ndarray & target, const bpn::ndarray & normals,
        const int k, const float angle, const int min_cluster_size, const float angle_dev, const bool use_normals)
#endif
{
    double theta = (double)angle / 180.0 * M_PI;
    // TODO use normals of mesh
    PointCloud<double> pointData;

    int n_points = ndarrayToPC(cloud, pointData);
    Eigen::MatrixXi neighbours = st2neighbours(target, n_points, k);
    //printf("conv normals\n");
    //printf("%f, %f, %f\n", (double)bp::extract<float>(normals[0][0]), (double)bp::extract<float>(normals[0][1]), (double)bp::extract<float>(normals[0][2]));
        
    Eigen::MatrixXd eig_normals = convert_normals(normals, n_points);
    //std::cout << "extracted " << n_points << " points" << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    // step2: build kd-tree
    std::vector<PCAInfo> pcaInfos;
    PCAFunctions pcaer;
    //pcaer.PCA(pointData, k, pcaInfos);
    //printf("PCA GEO\n");
    pcaer.PCAGeo(pointData, neighbours, pcaInfos, eig_normals, use_normals);
    //printf("Done\n");

    // step3: run point segmentation algorithm
    std::vector<std::vector<int>> clusters;
    PLANE_MODE planeMode = SURFACE;               // PLANE  SURFACE
    //std::cout << "init segmenter" << std::endl; 
    ClusterGrowPLinkage segmenter(k, theta, planeMode);
    //printf("set data\n");
    segmenter.setData(pointData, pcaInfos);
    //printf("run\n");
    segmenter.run(clusters, min_cluster_size, angle_dev, n_points);
    //printf("success\n");
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;

    std::vector<int> partition = create_partition(clusters, n_points);

    #ifdef _WIN32 || __linux__
    return to_py_tuple::convert(Components_tuple(clusters, partition, (float)elapsed.count()));
    #else // apple
    bp::list pyobject = to_py_tuple::convert_lists(Components_tuple(clusters, partition, (float)elapsed.count()));
    return pyobject;
    #endif
}


//using namespace boost::python;
BOOST_PYTHON_MODULE(libplink)
{
    #ifdef _WIN32 || __linux__
    _import_array();
    #endif
    Py_Initialize();
    bpn::initialize();
    
    def("plinkage", plinkage);
    def("plinkage", plinkage,
        (
            bp::args("k")=100,
            bp::args("angle")=90.0f,
            bp::args("min_cluster_size")=10,
            bp::args("angle_dev")=10.0f
        )
    );
    def("plinkage_geo", plinkage_geo);
    def("plinkage_geo", plinkage_geo, 
        (   
            bp::args("k")=100,
            bp::args("angle")=90.0f,
            bp::args("min_cluster_size")=10,
            bp::args("angle_dev")=10.0f,
            bp::args("use_normals")=true
        )
    );
}