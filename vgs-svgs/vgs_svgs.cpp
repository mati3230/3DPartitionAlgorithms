#include <chrono>
#include "../cpp_utils/python_utils.h"
#include "../cpp_utils/pcl_utils.h"
#include "../cpp_utils/partition_utils.h"

#include "voxel_segmentation.h"
#include "supervoxel_segmentation.h"

namespace bp = boost::python;
namespace bpn = boost::python::numpy;

PyObject * svgs(const bpn::ndarray & cloud, const float voxel_size, const float seed_size, const float graph_size, 
        const float sig_p, const float sig_n, const float sig_o, const float sig_f, const float sig_e, const float sig_w,
        const float sig_a, const float sig_b, const float sig_c, const float cut_thred,
        const int points_min, const int adjacency_min, const int voxels_min){
    PCXYZPtr input_cloud(new PCXYZ);
    
    int n_points = ndarrayToPCL(cloud, input_cloud);

    auto start = std::chrono::high_resolution_clock::now();
    unsigned int node_ID = 0;
    double min_x = 0, min_y = 0, min_z = 0, max_x = 0, max_y = 0, max_z = 0;

    std::vector<std::vector<int> > clusters_points_idx;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_voxels(new PCXYZRGB);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_supervoxels(new PCXYZRGB);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr clustered_cloud(new PCXYZRGB);
    pcl::PolygonMesh::Ptr normes_spvoxels(new pcl::PolygonMesh);

    //Super Voxelization
    pcl::SuperVoxelBasedSegmentation<pcl::PointXYZ> supervoxel_structure(voxel_size);//Building octree structure 

    supervoxel_structure.setInputCloud(input_cloud); //Input point cloud to octree
    supervoxel_structure.getCloudPointNum(input_cloud);
    supervoxel_structure.addPointsFromInputCloud();

    supervoxel_structure.setVoxelSize(voxel_size, points_min);
    supervoxel_structure.setSupervoxelSize(seed_size, voxels_min, points_min, adjacency_min);
    supervoxel_structure.setGraphSize(graph_size);

    supervoxel_structure.getBoundingBox(min_x, min_y, min_z, max_x, max_y, max_z);
    supervoxel_structure.setBoundingBox(min_x, min_y, min_z, max_x, max_y, max_z);

    //Get center points of all occupied voxels
    supervoxel_structure.setSupervoxelCentersCentroids();
    supervoxel_structure.getVoxelNum();

    //printf("SVGS: SupervoxelGeneration\n");
    //Supervoxel generation
    supervoxel_structure.segmentSupervoxelCloudWithGraphModel(sig_a, sig_b, sig_f, cut_thred, sig_p, sig_n, sig_o, sig_e, sig_c, sig_w);

    //printf("SVGS: Results\n");
    //Results
    //supervoxel_structure.drawColorMapofPointsinClusters(clustered_cloud);

    //printf("SVGS: Done\n");
    clusters_points_idx = supervoxel_structure.getClusterIdx();
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;

    int clusters_num = clusters_points_idx.size();
    std::vector<int> partition = create_partition(clusters_points_idx, n_points);
    return to_py_tuple::convert(Components_tuple(clusters_points_idx, partition, (float)elapsed.count()));
}


PyObject * svgs_mesh(
        const bpn::ndarray & cloud,
        const bpn::ndarray & source_nei,
        const bpn::ndarray & target_nei,
        const bpn::ndarray & uni_source, 
        const bpn::ndarray & uni_index,
        const bpn::ndarray & uni_counts, 
        const bpn::ndarray & distances,
        const float voxel_size, const float seed_size, const float graph_size,
        const float sig_p, const float sig_n, const float sig_o, const float sig_f, const float sig_e, const float sig_w,
        const float sig_a, const float sig_b, const float sig_c, const float cut_thred,
        const int points_min, const int adjacency_min, const int voxels_min, const float r_search_gain, const bool precalc){
    PCXYZPtr input_cloud(new PCXYZ);
    
    int n_points = ndarrayToPCL(cloud, input_cloud);
    
    PCNPtr normal_cloud(new PCN);
    ndarrayToPCL(cloud, normal_cloud);

    auto start = std::chrono::high_resolution_clock::now();
    unsigned int node_ID = 0;
    double min_x = 0, min_y = 0, min_z = 0, max_x = 0, max_y = 0, max_z = 0;

    std::vector<std::vector<int> > clusters_points_idx;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_voxels(new PCXYZRGB);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_supervoxels(new PCXYZRGB);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr clustered_cloud(new PCXYZRGB);
    pcl::PolygonMesh::Ptr normes_spvoxels(new pcl::PolygonMesh);

    //Super Voxelization
    pcl::SuperVoxelBasedSegmentation<pcl::PointXYZ> supervoxel_structure(voxel_size);//Building octree structure 

    supervoxel_structure.setInputCloud(input_cloud); //Input point cloud to octree
    supervoxel_structure.getCloudPointNum(input_cloud);
    supervoxel_structure.addPointsFromInputCloud();

    supervoxel_structure.setVoxelSize(voxel_size, points_min);
    supervoxel_structure.setGraphSize(graph_size);
    supervoxel_structure.setSupervoxelSize(seed_size, voxels_min, points_min, adjacency_min);
    //supervoxel_structure.setGraphSize(graph_size);

    supervoxel_structure.getBoundingBox(min_x, min_y, min_z, max_x, max_y, max_z);
    supervoxel_structure.setBoundingBox(min_x, min_y, min_z, max_x, max_y, max_z);

    //Get center points of all occupied voxels
    supervoxel_structure.setSupervoxelCentersCentroids();
    supervoxel_structure.getVoxelNum();

    //printf("SVGS: SupervoxelGeneration\n");
    //Supervoxel generation
    supervoxel_structure.segmentSupervoxelCloudWithGraphModelGeo(
        source_nei, target_nei, uni_source, uni_index, uni_counts, distances, normal_cloud,
        sig_a, sig_b, sig_f, cut_thred, sig_p, sig_n, sig_o, sig_e, sig_c, sig_w, r_search_gain, precalc);

    //printf("SVGS: Results\n");
    //Results
    //supervoxel_structure.drawColorMapofPointsinClusters(clustered_cloud);

    //printf("SVGS: Done\n");
    clusters_points_idx = supervoxel_structure.getClusterIdx();
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;

    int clusters_num = clusters_points_idx.size();
    std::vector<int> partition = create_partition(clusters_points_idx, n_points);
    return to_py_tuple::convert(Components_tuple(clusters_points_idx, partition, (float)elapsed.count()));
}


PyObject * vgs(const bpn::ndarray & cloud, const float voxel_size, const float graph_size, const float sig_p, const float sig_n, 
        const float sig_o, const float sig_e, const float sig_c, const float sig_w, const float cut_thred,
        const int points_min, const int adjacency_min, const int voxels_min){
    PCXYZPtr input_cloud(new PCXYZ);
    
    int n_points = ndarrayToPCL(cloud, input_cloud);
    auto start = std::chrono::high_resolution_clock::now();
    unsigned int node_ID = 0;
    double min_x = 0, min_y = 0, min_z = 0, max_x = 0, max_y = 0, max_z = 0;

    //pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud(new PCXYZRGB);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr clustered_cloud(new PCXYZRGB);
    //pcl::PolygonMesh::Ptr colored_voxels(new pcl::PolygonMesh);
    //pcl::PolygonMesh::Ptr frames_voxels(new pcl::PolygonMesh);
    //pcl::PolygonMesh::Ptr Normes_voxels(new pcl::PolygonMesh);
    //pcl::PolygonMesh::Ptr clustered_voxels(new pcl::PolygonMesh);
    std::vector<pcl::PointXYZ, Eigen::aligned_allocator<pcl::PointXYZ>> voxel_centers;

    //printf("Voxelization\n");
    //Voxelization
    pcl::VoxelBasedSegmentation<pcl::PointXYZ> voxel_structure(voxel_size);//Building octree structure 
    voxel_structure.setInputCloud(input_cloud); //Input point cloud to octree
    voxel_structure.getCloudPointNum(input_cloud);
    voxel_structure.addPointsFromInputCloud();
    voxel_structure.setVoxelSize(voxel_size, points_min, voxels_min, adjacency_min);
    voxel_structure.getBoundingBox(min_x, min_y, min_z, max_x, max_y, max_z);
    voxel_structure.setBoundingBox(min_x, min_y, min_z, max_x, max_y, max_z);

    //printf("Center\n");
    //Get center points of all occupied voxels
    voxel_structure.setVoxelCenters();
    voxel_centers = voxel_structure.getVoxelCenters();
    voxel_structure.getVoxelNum();

    //printf("Features\n");
    //Features calculation
    voxel_structure.calcualteVoxelCloudAttributes(input_cloud);

    //printf("Adjacencies\n");
    //Find adjacencies 
    voxel_structure.findAllVoxelAdjacency(graph_size);

    //printf("Segmentation\n");
    //Segmentation
    voxel_structure.segmentVoxelCloudWithGraphModel(cut_thred, sig_p, sig_n, sig_o, sig_e, sig_c, sig_w);

    //Output segmented color draw
    voxel_structure.drawColorMapofPointsinClusters(clustered_cloud);
    std::vector<std::vector<int> > clusters_points_idx;

    clusters_points_idx = voxel_structure.getClusterIdx();
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    std::vector<int> partition = create_partition(clusters_points_idx, n_points);
    return to_py_tuple::convert(Components_tuple(clusters_points_idx, partition, (float)elapsed.count()));
}


PyObject * vgs_mesh(
        const bpn::ndarray & cloud,
        const bpn::ndarray & target,
        const bpn::ndarray & uni_index,
        const bpn::ndarray & uni_counts,
        const bpn::ndarray & distances,
        const bpn::ndarray & normals,
        const float voxel_size, const float sig_p, const float sig_n, 
        const float sig_o, const float sig_e, const float sig_c, const float sig_w, const float cut_thred,
        const int points_min, const int adjacency_min, const int voxels_min, const bool use_normals){
    PCXYZPtr input_cloud(new PCXYZ);
    
    int n_points = ndarrayToPCL(cloud, input_cloud);
    auto start = std::chrono::high_resolution_clock::now();
    unsigned int node_ID = 0;
    double min_x = 0, min_y = 0, min_z = 0, max_x = 0, max_y = 0, max_z = 0;

    //pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud(new PCXYZRGB);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr clustered_cloud(new PCXYZRGB);
    //pcl::PolygonMesh::Ptr colored_voxels(new pcl::PolygonMesh);
    //pcl::PolygonMesh::Ptr frames_voxels(new pcl::PolygonMesh);
    //pcl::PolygonMesh::Ptr Normes_voxels(new pcl::PolygonMesh);
    //pcl::PolygonMesh::Ptr clustered_voxels(new pcl::PolygonMesh);
    std::vector<pcl::PointXYZ, Eigen::aligned_allocator<pcl::PointXYZ>> voxel_centers;

    //printf("Voxelization\n");
    //Voxelization
    uint32_t n_verts = bp::len(cloud);
    //pcl::VoxelBasedSegmentation<pcl::PointXYZ> voxel_structure(voxel_size, source, target, distances, n_verts);//Building octree structure 
    pcl::VoxelBasedSegmentation<pcl::PointXYZ> voxel_structure(voxel_size);
    //printf("Constructor calculated\n");
    voxel_structure.setInputCloud(input_cloud); //Input point cloud to octree
    voxel_structure.getCloudPointNum(input_cloud);
    voxel_structure.addPointsFromInputCloud();
    voxel_structure.setVoxelSize(voxel_size, points_min, voxels_min, adjacency_min);
    voxel_structure.getBoundingBox(min_x, min_y, min_z, max_x, max_y, max_z);
    voxel_structure.setBoundingBox(min_x, min_y, min_z, max_x, max_y, max_z);

    //printf("Center\n");
    //Get center points of all occupied voxels
    voxel_structure.setVoxelCenters();
    voxel_centers = voxel_structure.getVoxelCenters();
    voxel_structure.getVoxelNum();

    //printf("Features\n");
    //Features calculation
    if (use_normals)
    {
        voxel_structure.calcualteVoxelCloudAttributesGeo(input_cloud, normals); // leads more 'normal' behaviour in comination with findAllVoxelAdjacency(1.0f)
        //printf("use_normals\n");
    }
    else
    {
        voxel_structure.calcualteVoxelCloudAttributes(input_cloud);
        //printf("ignore normals\n");
    }

    //printf("Adjacencies\n");
    //Find adjacencies 
    voxel_structure.findAllVoxelAdjacencyGeo(input_cloud, target, uni_index, uni_counts, distances); // leads to vccs like partition in combination with calcualteVoxelCloudAttributes(input_cloud)
    //voxel_structure.findAllVoxelAdjacency(1.0f); // 0.5f leads to vccs like partition

    //printf("Segmentation\n");
    //Segmentation
    // Cannot calculate geodesic adjacency matrix between voxels because it is too slow
    voxel_structure.segmentVoxelCloudWithGraphModelGeo(cut_thred, sig_p, sig_n, sig_o, sig_e, sig_c, sig_w);

    //printf("Output\n");
    //Output segmented color draw
    voxel_structure.drawColorMapofPointsinClusters(clustered_cloud);
    std::vector<std::vector<int> > clusters_points_idx;

    clusters_points_idx = voxel_structure.getClusterIdx();
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    std::vector<int> partition = create_partition(clusters_points_idx, n_points);
    return to_py_tuple::convert(Components_tuple(clusters_points_idx, partition, (float)elapsed.count()));
}


//using namespace boost::python;
BOOST_PYTHON_MODULE(libvgs)
{
    _import_array();
    Py_Initialize();
    bpn::initialize();
    //bp::to_python_converter<std::vector<std::vector<float>, std::allocator<std::vector<float> > >, VecvecToArray<float> >();
    //bp::to_python_converter< Components_tuple, to_py_tuple>();
    def("vgs", vgs);
    def("vgs", vgs,
        (
            bp::args("voxel_size")=0.15f,
            bp::args("graph_size")=0.5f,
            bp::args("sig_p")=0.2f,
            bp::args("sig_n")=0.2f,
            bp::args("sig_o")=0.2f,
            bp::args("sig_e")=0.2f,
            bp::args("sig_c")=0.2f,
            bp::args("sig_w")=2.0f,
            bp::args("cut_thred")=0.3f,
            bp::args("points_min")=10,
            bp::args("adjacency_min")=3,
            bp::args("voxels_min")=3
        )
    );
    def("vgs_mesh", vgs_mesh);
    def("vgs_mesh", vgs_mesh,
        (
            bp::args("voxel_size")=0.15f,
            bp::args("sig_p")=0.2f,
            bp::args("sig_n")=0.2f,
            bp::args("sig_o")=0.2f,
            bp::args("sig_e")=0.2f,
            bp::args("sig_c")=0.2f,
            bp::args("sig_w")=2.0f,
            bp::args("cut_thred")=0.3f,
            bp::args("points_min")=10,
            bp::args("adjacency_min")=3,
            bp::args("voxels_min")=3,
            bp::args("use_normals")=false
        )
    );
    def("svgs", svgs);
    def("svgs", svgs,
        (
            bp::args("voxel_size")=0.05f,
            bp::args("seed_size")=0.25f,
            bp::args("graph_size")=0.75f,
            bp::args("sig_p")=0.2f,
            bp::args("sig_n")=0.2f,
            bp::args("sig_o")=0.2f,
            bp::args("sig_f")=0.2f,
            bp::args("sig_e")=0.2f,
            bp::args("sig_w")=1.0f,
            bp::args("sig_a")=0.0f,
            bp::args("sig_b")=0.25f,
            bp::args("sig_c")=0.75f,
            bp::args("cut_thred")=0.3f,
            bp::args("points_min")=10,
            bp::args("adjacency_min")=3,
            bp::args("voxels_min")=3
        )
    );
    def("svgs_mesh", svgs_mesh);
    def("svgs_mesh", svgs_mesh,
        (
            bp::args("voxel_size")=0.05f,
            bp::args("seed_size")=0.25f,
            bp::args("graph_size")=0.25f,
            bp::args("sig_p")=0.2f,
            bp::args("sig_n")=0.2f,
            bp::args("sig_o")=0.2f,
            bp::args("sig_f")=0.2f,
            bp::args("sig_e")=0.2f,
            bp::args("sig_w")=1.0f,
            bp::args("sig_a")=0.0f,
            bp::args("sig_b")=0.25f,
            bp::args("sig_c")=0.75f,
            bp::args("cut_thred")=0.3f,
            bp::args("points_min")=10,
            bp::args("adjacency_min")=3,
            bp::args("voxels_min")=3,
            bp::args("r_search_gain")=0.5f,
            bp::args("precalc")=false
        )
    );
}