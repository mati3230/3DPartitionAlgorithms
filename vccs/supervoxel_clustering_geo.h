#ifdef _WIN32
//#define PCL_NO_PRECOMPILE
#include "m_supervoxel_clustering.h"
//#include <pcl/segmentation/impl/supervoxel_clustering.hpp>
#else
#include <pcl/segmentation/impl/supervoxel_clustering.hpp>
#endif
#include <boost/shared_ptr.hpp>

#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/property_map/property_map.hpp>

#include <map>

#include "../cpp_utils/python_utils.h"
#include "../cpp_utils/geodesic_knn.h"
namespace bp = boost::python;
namespace bpn = boost::python::numpy;

#ifndef SUPERVOXEL_CLUSTERING_GEO
#define SUPERVOXEL_CLUSTERING_GEO

typedef boost::shared_ptr<bpn::ndarray> ndarrayPtr;
typedef std::pair<int, int> Edge;
typedef boost::adjacency_list < boost::listS, boost::vecS, boost::directedS, boost::no_property, boost::property < boost::edge_weight_t, float > > graph_t;
typedef boost::graph_traits < graph_t >::vertex_descriptor vertex_descriptor;
typedef boost::shared_ptr<graph_t> grapthPtr;


namespace pcl{
    template<typename PointT>
    class SupervoxelClusteringGeo : public MSupervoxelClustering<PointT>{
    using PCLBase <PointT>::input_;
    public:
        SupervoxelClusteringGeo(double voxel_resolution, double seed_resolution) 
            : MSupervoxelClustering<PointT>(voxel_resolution, seed_resolution)
        { 
            r_search_gain_ = 0.5f;
        };
        SupervoxelClusteringGeo(
                double voxel_resolution,
                double seed_resolution,
                bpn::ndarray uni_source,
                bpn::ndarray uni_index,
                bpn::ndarray uni_counts,
                bpn::ndarray source,
                bpn::ndarray target,
                bpn::ndarray distances,
                float r_search_gain,
                bool precalc) 
            : MSupervoxelClustering<PointT>(voxel_resolution, seed_resolution)
        {
            r_search_gain_ = r_search_gain;
            precalc_ = precalc;
            spatial_distances_.clear();
            uni_source_ = boost::make_shared<bpn::ndarray>(uni_source);
            uni_index_ = boost::make_shared<bpn::ndarray>(uni_index);
            uni_counts_ = boost::make_shared<bpn::ndarray>(uni_counts);
            source_ = boost::make_shared<bpn::ndarray>(source);
            target_ = boost::make_shared<bpn::ndarray>(target);
            distances_ = boost::make_shared<bpn::ndarray>(distances);
            //printf("added pointers\n");


            uint32_t n_edges = bp::len(source);
            uint32_t n_verts = bp::len(uni_source);
            // Warning: Memory Exception can occur
            //printf("n_edges %i, n_verts %i\n", (int) n_edges, (int) n_verts);
#ifdef _WIN32
            edges_ = new Edge[n_edges];
            weights_ = new float[n_edges];
#else
            Edge edges[(int)n_edges];
            float weights[(int)n_edges];
#endif
            //printf("Array created\n");
            for (uint32_t i=0; i < n_edges; i++){
                //printf("extract u ");
                uint32_t u = bp::extract<uint32_t>(source[i]);
                uint32_t v = bp::extract<uint32_t>(target[i]);
                //printf("Iter %i: Add Edge (%i, %i) ", (int) i, (int) u, (int) v);
#ifdef _WIN32
                edges_[i] = Edge(u, v);
                float w = bp::extract<float>(distances[i]);
                //printf("with weight %f\n", (double)w);
                weights_[i] = w;
#else
                edges[i] = Edge(u, v);
                float w = bp::extract<float>(distances[i]);
                //printf("with weight %f\n", (double)w);
                weights[i] = w;
#endif
            }
            //printf("added weights\n");
#ifdef _WIN32
            graph_t g(edges_, edges_ + n_edges, weights_, n_verts);
#else
            graph_t g(edges, edges + n_edges, weights, n_verts);
#endif
            mesh_ = boost::make_shared<graph_t>(g);
            //printf("graph done\n");
        };
#ifdef _WIN32
        ~SupervoxelClusteringGeo() 
        {
            delete[] edges_;
            delete[] weights_;
        };
#endif
        void setInputCloud (const typename pcl::PointCloud<PointT>::ConstPtr& cloud) override
        {
            MSupervoxelClustering<PointT>::setInputCloud(cloud);

            if(point_kdtree_ == 0)
            {
                point_kdtree_.reset (new pcl::KdTreeFLANN<PointT>);
                point_kdtree_ ->setInputCloud (input_);
            }
        };
    private:
        ndarrayPtr uni_source_; 
        ndarrayPtr uni_index_;
        ndarrayPtr uni_counts_;
        ndarrayPtr source_; 
        ndarrayPtr target_; 
        ndarrayPtr distances_;
        //bpn::ndarray uni_source_, uni_index_, uni_counts_, source_, target_, distances_;
#ifdef _WIN32
        boost::shared_ptr<pcl::KdTreeFLANN<PointT>> point_kdtree_;
#else
        typename pcl::KdTreeFLANN<PointT>::Ptr point_kdtree_;
#endif
        grapthPtr mesh_;
        std::map<int, std::vector<float>> spatial_distances_;
        float r_search_gain_;
#ifdef _WIN32
        Edge* edges_;
        float* weights_;
#endif
        bool precalc_;

        void selectInitialSupervoxelSeeds(std::vector<int> &seed_indices) override
        {
            //printf("SupervoxelClusteringGeo: Select initial seeds\n");
            //MSupervoxelClustering<PointT>::selectInitialSupervoxelSeeds(seed_indices);

            pcl::octree::OctreePointCloudSearch <PointT> seed_octree (MSupervoxelClustering<PointT>::seed_resolution_);
            seed_octree.setInputCloud (MSupervoxelClustering<PointT>::voxel_centroid_cloud_);
            seed_octree.addPointsFromInputCloud ();
            std::vector<PointT, Eigen::aligned_allocator<PointT> > voxel_centers; 
            int num_seeds = seed_octree.getOccupiedVoxelCenters(voxel_centers);

            seed_indices.clear ();
            std::vector<int> closest_index;
            std::vector<float> distance;
            closest_index.resize(1,0);
#ifdef _WIN32
            if (voxel_kdtree_ == 0)
            {
                voxel_kdtree_.reset(new pcl::search::KdTree<PointT>);
                voxel_kdtree_->setInputCloud(voxel_centroid_cloud_);
            }
#else
            if (MSupervoxelClustering<PointT>::voxel_kdtree_ == 0)
            {
                MSupervoxelClustering<PointT>::voxel_kdtree_.reset (new pcl::search::KdTree<PointT>);
                MSupervoxelClustering<PointT>::voxel_kdtree_ ->setInputCloud (MSupervoxelClustering<PointT>::voxel_centroid_cloud_);
            }
#endif
            //printf("init point kdtree\n");
            if(point_kdtree_ == 0)
            {
                point_kdtree_.reset (new pcl::KdTreeFLANN<PointT>);
                point_kdtree_ ->setInputCloud (input_);
            }

            float search_radius = r_search_gain_ * MSupervoxelClustering<PointT>::seed_resolution_; // R_search in the paper
            float min_points = 0.05f * (search_radius)*(search_radius) * 3.1415926536f  / (MSupervoxelClustering<PointT>::resolution_*MSupervoxelClustering<PointT>::resolution_);
 
            std::vector<int> neighbors;
            std::vector<float> sqr_distances;
            //printf("start seed filtering with %i seeds\n", (int)num_seeds);
            for (int i = 0; i < num_seeds; i++)  
            {
                //printf("search k neighbours\n");
                point_kdtree_->nearestKSearch (voxel_centers[i], 1, closest_index, distance);
#ifdef _WIN32
                voxel_kdtree_->nearestKSearch(voxel_centers[i], 1, neighbors, sqr_distances);
#else
                MSupervoxelClustering<PointT>::voxel_kdtree_->nearestKSearch (voxel_centers[i], 1 , neighbors, sqr_distances);
#endif
                //seed_indices_orig[i] = closest_index[0];
                //printf("done\n");
                int search_index = closest_index[0];
                int ref_index = neighbors[0];
                std::vector<uint32_t> out_source, out_target;
                std::vector<float> out_distances;
                //printf("search bfs\n");
                if(this->precalc_){
                    get_neighbours_radius((uint32_t)search_index, *uni_index_, *uni_counts_, 
                        *target_, *distances_, search_radius, out_target, out_distances);
                }
                else{
                    search_bfs_radius_single((uint32_t)search_index, *uni_source_, *uni_index_, *uni_counts_, 
                        *target_, *distances_, search_radius, out_source, out_target, out_distances, true);
                }
                //printf("create set\n");
                std::set<int> vidxs;
                for (int j = 0; j < out_target.size(); j++){
                    uint32_t t_idx = out_target[j];
                    PointT t_point = input_->points[t_idx];
                    typename MSupervoxelClustering<PointT>::LeafContainerT* leaf = MSupervoxelClustering<PointT>::adjacency_octree_->getLeafContainerAtPoint (t_point);
                    typename MSupervoxelClustering<PointT>::MVoxelData& voxel_data = leaf->getData ();

                    vidxs.emplace(voxel_data.idx_);
                }

                uint32_t n_neighbours = vidxs.size();
                if ( (float)n_neighbours > min_points)
                {
                    seed_indices.push_back (ref_index);
                }
            }
        };

        float voxelDataDistance (const typename MSupervoxelClustering<PointT>::MVoxelData &v1, const typename MSupervoxelClustering<PointT>::MVoxelData &v2) override
        {
            //printf("SupervoxelClusteringGeo: voxelDataDistance\n");
            std::vector<int> closest_index;
            std::vector<float> distance;
            PointT query;
            query.x = v1.xyz_(0);
            query.y = v1.xyz_(1);
            query.z = v1.xyz_(2);
            point_kdtree_->nearestKSearch (query, 1, closest_index, distance);
            //seed_indices_orig[i] = closest_index[0];
            int v1_pindex = closest_index[0];
            
            closest_index.clear();
            distance.clear();

            query.x = v2.xyz_(0);
            query.y = v2.xyz_(1);
            query.z = v2.xyz_(2);
            point_kdtree_->nearestKSearch (query, 1, closest_index, distance);
            //seed_indices_orig[i] = closest_index[0];
            int v2_pindex = closest_index[0];

            float spatial_distance = 0.0f;
            bool contains = false;
            if(spatial_distances_.count(v2_pindex))
            {
                std::vector<float> dists = spatial_distances_[v2_pindex];
                spatial_distance = dists[v1_pindex];
                contains = true;
            }
            if(spatial_distances_.count(v1_pindex) && !contains)
            {
                std::vector<float> dists = spatial_distances_[v1_pindex];
                spatial_distance = dists[v2_pindex];
                contains = true;
            }
            if(!contains){
                std::vector<float> d(boost::num_vertices(*mesh_));
                //boost::property_map<graph_t, boost::edge_weight_t>::type weightmap = boost::get(boost::edge_weight, *mesh_);
                std::vector<vertex_descriptor> p(boost::num_vertices(*mesh_));
                vertex_descriptor s = boost::vertex(v1_pindex, *mesh_);
                //printf("calc dijkstra\n");
                dijkstra_shortest_paths(*mesh_, s,
                    boost::predecessor_map(boost::make_iterator_property_map(p.begin(), get(boost::vertex_index, *mesh_))).
                    distance_map(boost::make_iterator_property_map(d.begin(), get(boost::vertex_index, *mesh_))));
                //printf("done\n");
                spatial_distance = d[v2_pindex];

                //spatial_distances_[v1_pindex] = d;
                //spatial_distances_.emplace(v1_pindex, d);
                this->spatial_distances_.emplace(v1_pindex, d);
            }


            float color_dist =  (v1.rgb_ - v2.rgb_).norm () / 255.0f;
            float cos_angle_normal = 1.0f - std::abs (v1.normal_.dot (v2.normal_));

            float dist = cos_angle_normal * MSupervoxelClustering<PointT>::normal_importance_ + color_dist * MSupervoxelClustering<PointT>::color_importance_+ spatial_distance * MSupervoxelClustering<PointT>::spatial_importance_;
            //printf("%f\n", dist);
            return dist;
        };
    };
}

#endif