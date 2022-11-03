#include <chrono>
//#define BOOST_PYTHON_STATIC_LIB
#include "../cpp_utils/python_utils.h"
#include "../cpp_utils/pcl_utils.h"

//#define PCL_NO_PRECOMPILE
#include "m_supervoxel_clustering.hpp"
#include "supervoxel_clustering_geo.h"

namespace bp = boost::python;
namespace bpn = boost::python::numpy;


PyObject * vccs(const bpn::ndarray & cloud, const float voxel_resolution, const float seed_resolution, 
    const float color_importance, const float spatial_importance, const float normal_importance, const int refinementIter)
{

    PCXYZRGBPtr input_cloud(new PCXYZRGB);
    //PCXYZPtr input_cloud(new PCXYZ);
    //pcl::SupervoxelClusteringGeo<pcl::PointXYZRGB> test(voxel_resolution, seed_resolution);
    int n_points = ndarrayToPCL(cloud, input_cloud);
    /*
    printf("n_points: %i\n", n_points);
    for (int i = 0; i < n_points; i++) {
        printf("%f, %f, %f\n", input_cloud->points[i].x, input_cloud->points[i].y, input_cloud->points[i].z);
    }
    */

    auto start = std::chrono::high_resolution_clock::now();
    pcl::MSupervoxelClustering<pcl::PointXYZRGB> super (voxel_resolution, seed_resolution);
    //pcl::MSupervoxelClustering<pcl::PointXYZ> super(voxel_resolution, seed_resolution);
    super.setInputCloud (input_cloud);
    super.setColorImportance (color_importance);
    super.setSpatialImportance (spatial_importance);
    super.setNormalImportance (normal_importance);
    std::map <uint32_t, pcl::MSupervoxel<pcl::PointXYZRGB>::Ptr > supervoxel_clusters;
    //std::map <uint32_t, pcl::MSupervoxel<pcl::PointXYZ>::Ptr > supervoxel_clusters;
    super.extract (supervoxel_clusters);
    if(refinementIter > 0)
        super.refineSupervoxels(refinementIter, supervoxel_clusters);
    //std::cout << "Found " << supervoxel_clusters.size () << " Supervoxels!\n";
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;

    pcl::PointCloud<pcl::PointXYZL>::Ptr labeled_cloud = super.getLabeledCloud ();

    std::vector<std::vector<int> > clusters_points_idx;
    std::vector<int> partition;
    int max_label = super.getMaxLabel();
    clusters_points_idx.reserve(max_label + 1);
    for(int g = 0; g < max_label + 1; g++)
    {
        std::vector<int> idxs;
        clusters_points_idx.push_back(idxs);
    }
    for (int i = 0; i < labeled_cloud->points.size(); i++){  
        int point_label = labeled_cloud->points[i].label;
        partition.push_back(point_label);
        clusters_points_idx[point_label].push_back(i);
        //if (point_label > 0)
        //{
        //}  
    }

    return to_py_tuple::convert(Components_tuple(clusters_points_idx, partition, (float)elapsed.count()));
}

PyObject * vccs_mesh(
        const bpn::ndarray & cloud,
        const bpn::ndarray & uni_source,
        const bpn::ndarray & uni_index,
        const bpn::ndarray & uni_counts,
        const bpn::ndarray & source,
        const bpn::ndarray & target,
        const bpn::ndarray & distances,
        const float voxel_resolution,
        const float seed_resolution, 
        const float color_importance,
        const float spatial_importance,
        const float normal_importance,
        const int refinementIter,
        const float r_search_gain,
        const bool precalc)
{
    PCXYZRGBPtr input_cloud(new PCXYZRGB);
    int n_points = ndarrayToPCL(cloud, input_cloud);
    //pcl::SupervoxelClusteringGeo<pcl::PointXYZRGB> test(voxel_resolution, seed_resolution);
    PCNPtr normal_cloud(new PCN);
    ndarrayToPCL(cloud, normal_cloud);

    auto start = std::chrono::high_resolution_clock::now();
    pcl::SupervoxelClusteringGeo<pcl::PointXYZRGB> super (
        voxel_resolution,
        seed_resolution,
        uni_source,
        uni_index,
        uni_counts,
        source,
        target,
        distances,
        r_search_gain,
        precalc);
    super.setInputCloud (input_cloud);
    super.setNormalCloud(normal_cloud);
    super.setColorImportance (color_importance);
    super.setSpatialImportance (spatial_importance);
    super.setNormalImportance (normal_importance);
    std::map <uint32_t, pcl::MSupervoxel<pcl::PointXYZRGB>::Ptr > supervoxel_clusters;
    //printf("extract\n");
    super.extract (supervoxel_clusters);
    //printf("done\n");
    if(refinementIter > 0)
        super.refineSupervoxels(refinementIter, supervoxel_clusters);
    //std::cout << "Found " << supervoxel_clusters.size () << " Supervoxels!\n";
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;

    pcl::PointCloud<pcl::PointXYZL>::Ptr labeled_cloud = super.getLabeledCloud ();

    std::vector<std::vector<int> > clusters_points_idx;
    std::vector<int> partition;
    int max_label = super.getMaxLabel();
    clusters_points_idx.reserve(max_label + 1);
    for(int g = 0; g < max_label + 1; g++)
    {
        std::vector<int> idxs;
        clusters_points_idx.push_back(idxs);
    }
    for (int i = 0; i < labeled_cloud->points.size(); i++){  
        int point_label = labeled_cloud->points[i].label;
        partition.push_back(point_label);
        clusters_points_idx[point_label].push_back(i);
        //if (point_label > 0)
        //{
        //}  
    }

    return to_py_tuple::convert(Components_tuple(clusters_points_idx, partition, (float)elapsed.count()));
}


//using namespace boost::python;
BOOST_PYTHON_MODULE(libvccs)
{
    _import_array();
    Py_Initialize();
    bpn::initialize();
    //bp::to_python_converter<std::vector<std::vector<float>, std::allocator<std::vector<float> > >, VecvecToArray<float> >();
    //bp::to_python_converter< Components_tuple, to_py_tuple>();
    def("vccs", vccs);
    def("vccs", vccs, 
        (
            bp::args("voxel_resolution")=0.008f,
            bp::args("seed_resolution")=0.1f,
            bp::args("color_importance")=0.2f,
            bp::args("spatial_importance")=0.4f,
            bp::args("normal_importance")=1.0f,
            bp::args("refinementIter")=2
        )
    );
    
    def("vccs_mesh", vccs_mesh);
    def("vccs_mesh", vccs_mesh, 
        (
            bp::args("voxel_resolution")=0.008f,
            bp::args("seed_resolution")=0.1f,
            bp::args("color_importance")=0.2f,
            bp::args("spatial_importance")=0.4f,
            bp::args("normal_importance")=1.0f,
            bp::args("refinementIter")=2,
            bp::args("r_search_gain")=0.5f,
            bp::args("precalc")=false
        )
    );
}