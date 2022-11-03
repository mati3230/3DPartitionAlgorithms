////////////////////////////////////////////////////////////////////////////////
//  File:	 voxel_segmentation.h
//  Author:	 Yusheng Xu, PF_Technische Universitaet Muechen (yusheng.xu@tum.de)
//  Description: The voxel based segmentation for point cloud 
//  Modified:    27.05.2018, By Dong Lin, TU Dresden, bugs fixed
//
//  Copyright (c) 2015-2018  Yusheng Xu (yusheng.xu@tum.de)
//
//  This library is free software; you can redistribute it and/or
//  modify it under the terms of the GNU General Public
//  License as published by the Free Software Foundation; either
//  Version 3 of the License, or (at your option) any later version.
//
//  This library is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
//  General Public License for more details.
////////////////////////////////////////////////////////////////////////////////
#pragma once

#ifndef PCL_SEGMENTATION_VOXEL_H_
#define PCL_SEGMENTATION_VOXEL_H_

#include <vector>
#include <algorithm>
#include <map>
#include <math.h>
#include <limits.h>
#include <queue>

#include <unsupported/Eigen/MatrixFunctions>

#include <pcl/point_cloud.h>
#include <pcl/features/fpfh.h>
#ifdef _WIN32
#include <pcl/kdtree/kdtree_flann.h>
#else
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#endif

#include <pcl/octree/octree.h>

#include <pcl/PolygonMesh.h>
#include <pcl/conversions.h>

#include "../cpp_utils/python_utils.h"

#include <boost/shared_ptr.hpp>

#include "../cpp_utils/geodesic_knn.h"
/*
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/property_map/property_map.hpp>
*/

namespace bp = boost::python;
namespace bpn = boost::python::numpy;

//Type definition
typedef  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr PCXYZRGBAPtr;
typedef  pcl::PointCloud<pcl::PointXYZRGBA> PCXYZRGBA;
typedef  pcl::PointCloud<pcl::PointXYZRGB>::Ptr PCXYZRGBPtr;
typedef  pcl::PointCloud<pcl::PointXYZRGB> PCXYZRGB;
typedef  pcl::PointCloud<pcl::PointXYZ>::Ptr PCXYZPtr;
typedef  pcl::PointCloud<pcl::PointXYZ> PCXYZ;

typedef  pcl::PointCloud<pcl::Normal>::Ptr PTNORMPtr;
typedef  pcl::PointCloud<pcl::Normal> PTNORM;

/*
typedef std::pair<int, int> Edge;
typedef boost::adjacency_list < boost::listS, boost::vecS, boost::directedS, boost::no_property, boost::property < boost::edge_weight_t, float > > graph_t;
typedef boost::graph_traits < graph_t >::vertex_descriptor vertex_descriptor;
*/
typedef boost::shared_ptr<graph_t> grapthPtr;

 namespace pcl
 {
   template<typename PointT>
   class  VoxelBasedSegmentation: public pcl::octree::OctreePointCloud<PointT>
   {
	//Inherited 
    using pcl::octree::OctreePointCloud<PointT>::input_;
	
	//New ones
	public:
		struct
		Weight_Index
		{
			float Weight;
			int Index;
		};

		static bool
		godown(const Weight_Index & a, const Weight_Index & b)
		{
			return a.Weight > b.Weight;
		}

		static bool
		riseup(const Weight_Index & a, const Weight_Index & b)
		{
			return a.Weight < b.Weight;
		}

		//Construction
		VoxelBasedSegmentation(double input_resolution)
			: pcl::octree::OctreePointCloud<PointT>(input_resolution)
		{
		}


		//Construction
		VoxelBasedSegmentation(double input_resolution, bpn::ndarray source, bpn::ndarray target, bpn::ndarray distances, uint32_t n_verts)
			: pcl::octree::OctreePointCloud<PointT>(input_resolution)
		{
			/*
      uint32_t n_edges = bp::len(source);
      Edge edges[n_edges];
      float weights[n_edges];
      for (uint32_t i=0; i < n_edges; i++){
          uint32_t u = bp::extract<uint32_t>(source[i]);
          uint32_t v = bp::extract<uint32_t>(target[i]);
          edges[i] = Edge(u, v);
          float w = bp::extract<float>(distances[i]);
          weights[i] = w;
      }

      graph_t g(edges, edges + n_edges, weights, n_verts);
      mesh_ = boost::make_shared<graph_t>(g);
      */
		}

		//Destrction
		~VoxelBasedSegmentation()
		{
		}

		//Member functions
		int
		getCloudPointNum(PCXYZPtr input_data)
		{
			this->points_num_=input_data->points.size();

			this->points_cloud_= input_data;

			return (points_num_);
		}

		int
		getVoxelNum()
		{
			voxels_num_=this->voxel_centers_.size();
			return (voxels_num_);
		}

		int
		getClusterNum()
		{
			return (clusters_num_);
		}

		std::vector<std::vector<int> >
		getClusterIdx()
		{
			return(clusters_point_idx_);
		}

		//Voxel
		void
		setVoxelSize(double input_resolution,int points_num_min, int voxels_num_min, int voxels_adj_min)
		{
			this->voxel_resolution_=input_resolution;
			this->voxel_points_min_=points_num_min;
			this->cluster_voxels_min_=voxels_num_min;
			this->voxel_adjacency_min_=voxels_adj_min;
		}

		void
		setBoundingBox(double min_x, double min_y, double min_z, double max_x, double max_y, double max_z)
		{
			this->min_x_=min_x;
			this->min_y_=min_y;
			this->min_z_=min_z;

			this->max_x_=max_x;
			this->max_y_=max_y;
			this->max_z_=max_z;

		}

		void
		setVoxelCenters()
		{
			//Setting
			VoxelBasedSegmentation<pcl::PointXYZ>::LeafNodeIterator iter_leaf (this); //Iteration index for all leaf nodes in the tree

			uint32_t voxel_id = 0;
			//Tranverse the octree leaves
			while (*++iter_leaf)
			{
				//Search centers from the key of the leaf node
				pcl::PointXYZ node_center;
				pcl::octree::OctreeKey leaf_key;
				leaf_key=iter_leaf.getCurrentOctreeKey();
				this->getVoxelCenterFromOctreeKey(leaf_key, node_center);
				voxel_centers_.push_back(node_center);

				//Search idx all the points in this leaf node
				std::vector<int> points_idx;
				points_idx=iter_leaf.getLeafContainer().getPointIndicesVector();
				this->voxels_point_idx_.push_back(points_idx);

				for(uint32_t i = 0; i < points_idx.size(); i++){
					uint32_t p_idx = points_idx[i];
					point2vox_.insert(std::pair<uint32_t, uint32_t>(p_idx, voxel_id));
				}

				voxel_id++;
			}

			//voxel_centers_=input_voxel_centers;
			
			//Build cloud
			int voxel_centers_cloud_size=voxel_centers_.size();
			PCXYZPtr temp_voxel_centers_cloud (new PCXYZ);
			pcl::PointXYZ voxel_center_pt;

			centers_num_=voxel_centers_cloud_size;

			for (int i=0;i<voxel_centers_cloud_size;i++) 
			{
				voxel_center_pt=voxel_centers_[i];
				temp_voxel_centers_cloud->points.push_back(voxel_center_pt);
			}

			temp_voxel_centers_cloud->width = (int)temp_voxel_centers_cloud->points.size();  
			temp_voxel_centers_cloud->height = 1; 

			//Assignment
			//voxel_centers_cloud_=PCXYZPtr(temp_voxel_centers_cloud);
			//voxel_centers_cloud_=temp_voxel_centers_cloud;
			voxel_centers_cloud_ = PCXYZPtr(new PCXYZ);
			copyPointCloud(*temp_voxel_centers_cloud, *voxel_centers_cloud_);
			/*printf("Have %i voxel centres\n", (int)voxel_centers_cloud_->points.size());
			for(int i=0; i < 5; i++){
				pcl::PointXYZ p = voxel_centers_cloud_->points[i];
				printf("%f %f %f\n", p.x, p.y, p.z);
			}*/
		}

		std::vector<pcl::PointXYZ, Eigen::aligned_allocator<pcl::PointXYZ>>
		getVoxelCenters()
		{
			return(voxel_centers_);
		}

		void
		setVoxelChosen(int voxel_idx,  int chosen_ornot)
		{
			if(chosen_ornot==0)
			{
				this->voxel_used_.push_back(true);
			}
			else
			{
				this->voxel_used_.push_back(false);			
			}
		}

		void
		setVoxelClustered(int voxel_idx,  int clustered_ornot)
		{
			if(clustered_ornot==0)
			{
				this->voxel_clustered_.push_back(true);
			}
			else
			{
				this->voxel_clustered_.push_back(false);			
			}
		}

		void
		findAllVoxelAdjacency(float graph_size)
		{
			//Build kd-tree of voxels' centers
			//printf("KDTree: Assign %i points\n", (int)this->voxel_centers_cloud_->size());
			
			//printf("Init Tree\n");
			pcl::KdTreeFLANN<pcl::PointXYZ> voxel_centers_kdtree;
			//printf("Set Input Cloud\n");
   		/*
   		std::vector<int> idxs((int)this->voxel_centers_cloud_->size());
   		for (int i=0; i<(int)this->voxel_centers_cloud_->size(); i++){
   			idxs[i]=i;
   		}
   		boost::shared_ptr< std::vector<int> >  idxsPtr = boost::make_shared< std::vector<int> >(idxs);
			voxel_centers_kdtree.setInputCloud (this->voxel_centers_cloud_, idxsPtr);	
   		*/
			voxel_centers_kdtree.setInputCloud (this->voxel_centers_cloud_);	
			// call of destructor flann nn_index
			//printf("Set Input Cloud Success\n");
			this->graph_resolution_=graph_size;

			//Search radius
			double search_radius;
			search_radius=graph_size;

			//Tranverse all the centers in the tree
			//printf("Search centers\n");
			pcl::PointXYZ search_point;
			//int n_neighbours[centers_num_];
			for (int i=0; i<centers_num_;i++)
			{
				//Searching points
				search_point=voxel_centers_cloud_->points[i];

				//Defining search radius
				std::vector<int> pointIdxRadiusSearch;
				std::vector<float> pointRadiusSquaredDistance;

				//Searching results
				//printf("Apply radius search\n");
				//n_neighbours[i] = (int)pointIdxRadiusSearch.size();
				if ( voxel_centers_kdtree.radiusSearch (search_point, search_radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0 )
				{
					//printf("pointRadiusSquaredDistance: %f\n", pointRadiusSquaredDistance[0]);
					//Add elements in the vector
					voxels_adjacency_idx_.push_back(std::vector<int>(pointIdxRadiusSearch.size()+1,0));
					voxels_adjacency_idx_[i][0]=pointIdxRadiusSearch.size();//The first element records the num of adjacencies
					//printf("Voxel %i has %i neighbours\n", i, (int)pointIdxRadiusSearch.size());

					//Test
					//std::cout	<< "   "<<std::endl;
					//std::cout	<< " Node:   "<< i<< "  having  "<< voxels_adjacency_idx_[i][0] <<" adjacent voxels "<<std::endl;
					//n_neighbours[i] = (int)pointIdxRadiusSearch.size();
					for (size_t j = 1; j < pointIdxRadiusSearch.size()+1; j++)//The rest records the idx 
					{
						//Records
						voxels_adjacency_idx_[i][j]=pointIdxRadiusSearch[j-1];
						//std::cout	<< "    "  << voxels_adjacency_idx_[i][j]<< std::endl;
					}
				}
				//printf("Nothing found\n");
			}
			/*
			float sum = 0.0f;
			int zeros = 0;
			for (int i=0; i<centers_num_;i++)
			{
				if (n_neighbours[i] == 0){
					zeros++;
					continue;
				}
				sum += (float)n_neighbours[i];
			}
			sum /= (float)centers_num_;
			printf("average number of neighbours: %f\n", (double) sum);
			printf("%i/%i voxels have no neighbours\n", zeros, centers_num_);
			*/
		}


		void
		findAllVoxelAdjacencyGeo(PCXYZPtr input_cloud, bpn::ndarray target, bpn::ndarray uni_index, bpn::ndarray uni_counts, bpn::ndarray distances)
		{
			// TODO: Store distances to neighbour voxels so that we can get rid of the calculation of djiskstra which is quiet time consuming 
			//The first element of each adjacency list records the num of adjacencies
			//std::vector<std::vector<int> > voxels_adjacency_idx_
			//voxels_adjacency_idx_.push_back(std::vector<int>());

			pcl::KdTreeFLANN<pcl::PointXYZ> point_kdtree;
			point_kdtree.setInputCloud (input_cloud);	

			for(uint32_t i = 0; i < getVoxelNum(); i++){
				std::vector<int> neighbours;
				voxels_adjacency_idx_.push_back(neighbours);
			}

			std::map<int, std::set<int>> vox_adj;

			//int n_neighbours[centers_num_];

      std::vector<int> closest_index;
      std::vector<float> distance;
			for(uint32_t i = 0; i < centers_num_; i++){
				//n_neighbours[i] = 0;
				point_kdtree.nearestKSearch (voxel_centers_cloud_->points[i], 1, closest_index, distance);
				uint32_t pindex = closest_index[0];
				uint32_t start = bp::extract<uint32_t>(uni_index[pindex]);
				uint32_t stop = start + bp::extract<uint32_t>(uni_counts[pindex]);
				for(uint32_t j = start; j < stop; j++){
					uint32_t tindex = bp::extract<uint32_t>(target[j]);
					float distance = bp::extract<float>(distances[j]);
					int vox_idx_v = (int)point2vox_[pindex];
					int vox_idx_u = (int)point2vox_[tindex];
					if(vox_idx_v == vox_idx_u)
						continue;
					if(vox_adj.count(vox_idx_v)) // contains
					{
						vox_adj[vox_idx_v].emplace(vox_idx_u);
					}
					else
					{
						std::set<int> neighbours;
						neighbours.emplace(vox_idx_u);
						std::pair<int, std::set<int>> pair (vox_idx_v, neighbours);  
						vox_adj.insert(pair);
					}

					if(vox_adj.count(vox_idx_u)) // contains
					{
						vox_adj[vox_idx_u].emplace(vox_idx_v);
					}
					else
					{
						std::set<int> neighbours;
						neighbours.emplace(vox_idx_v);
						std::pair<int, std::set<int>> pair (vox_idx_u, neighbours);  
						vox_adj.insert(pair);
					}
					spatial_distances_[vox_idx_u][vox_idx_v] = distance;
					spatial_distances_[vox_idx_v][vox_idx_u] = distance;
				}
			}
			//int m = 0;
			for (std::map<int, std::set<int>>::iterator it = vox_adj.begin(); it != vox_adj.end(); ++it){
				//voxels_adjacency_idx_.push_back()
				int vox_idx = it->first;
				std::set<int> neighbours = it->second;
				voxels_adjacency_idx_[vox_idx].push_back((int)neighbours.size());
				//n_neighbours[m] = (int)neighbours.size();
				for(std::set<int>::iterator setit = neighbours.begin(); setit != neighbours.end(); ++setit)
				{
					voxels_adjacency_idx_[vox_idx].push_back(*setit);
				} 
				//m++;
			}
			/*
			float sum = 0.0f;
			int zeros = 0;
			for (int i=0; i<centers_num_;i++)
			{
				if (n_neighbours[i] == 0){
					zeros++;
					continue;
				}
				sum += (float)n_neighbours[i];
			}
			sum /= (float)centers_num_;
			printf("average number of neighbours: %f\n", (double) sum);
			printf("%i/%i voxels have no neighbours\n", zeros, centers_num_);
			*/
		}



		std::vector<int>
		getOneVoxelAdjacency(int voxel_id)
		{
			////Test
			//std::cout	<< " "<< std::endl;
			//std::cout	<< " Find adjacent voxels of voxel  "<< voxel_id <<std::endl;
			//std::cout	<< " Node:   "<< voxel_id<< "  having  "<< voxels_adjacency_idx_[voxel_id][0] <<" adjacent voxels "<<std::endl;
			std::vector<int> adjacency_idx;//=Eigen::VectorXi::Zero(voxels_adjacency_idx_[voxel_id][0]);
			if (voxels_adjacency_idx_[voxel_id].size() == 0)
				return adjacency_idx;
			//Traverse
			for (size_t j = 1; j < voxels_adjacency_idx_[voxel_id][0]+1; j++)//Find the idx of recorded adjacent voxels 
			{
				////Records
				//std::cout	<< "    "  << voxels_adjacency_idx_[voxel_id][j]<< std::endl;					
				adjacency_idx.push_back(voxels_adjacency_idx_[voxel_id][j]);
				//std::cout	<< "    "  << adjacency_idx[j-1]<< std::endl;
			}

			return(adjacency_idx);
		}

		//Features
		void
		calcualteVoxelCloudAttributes(PCXYZPtr input_cloud)
		{
			//Iteration index
			VoxelBasedSegmentation<pcl::PointXYZ>::LeafNodeIterator iter_leaf (this); //Iteration index for all leaf nodes in the tree
	
			//Initial feature
			this->initialVoxelAttributes();		
		
			int leaf_num = 0;
			std::vector<float> voxel_features;
			std::vector<float> voxel_eigens;
			std::vector<float> voxel_colors;
			pcl::Normal voxel_norm;
			pcl::PointXYZ voxel_centroid;

			//Tranverse the octree leaves
			while (*++iter_leaf)// The initial node is neglected!
			{
				int point_num=0;
				int points_size=0;
				int point_idx=0;
				pcl::PointXYZ node_center;

				std::vector<int> index_vector;
				PCXYZPtr voxel_cloud (new PCXYZ);

				//Find centers of the leaf node			
				points_size=iter_leaf.getLeafContainer().getSize();
				iter_leaf.getLeafContainer().getPointIndices(index_vector);

				//Traverse the points in node
				if(points_size>this->voxel_points_min_)
				{
					//Get the points in the voxel
					while(point_num<points_size)
					{
						point_idx=index_vector[point_num];

						voxel_cloud->points.push_back(input_cloud->points[point_idx]);

						point_num++;
					}
		
					//Output points in leaf node
					voxel_cloud->width = (int)voxel_cloud->points.size();  
					voxel_cloud->height = 1;

					//Centroid calculation
					voxel_centroid=this->calculateVoxelCentroid(voxel_cloud);//Centroid
					this->setVoxelCentroid(leaf_num,voxel_centroid);

					//Norm calculation
					voxel_norm=this->calculateVoxelNorms(voxel_cloud);//Norms
					//printf("%f\n", voxel_norm.size());
					this->setVoxelNorms(leaf_num,voxel_norm);

					//Eigen calculation
					voxel_eigens=this->calculateEigenFeatures(voxel_cloud);//Eigen features
					this->setVoxelEigens(leaf_num,voxel_eigens);
					
					//Set voxel chosen or not
					this->setVoxelChosen(leaf_num, 0);
					//Set voxel clustered or not
					this->setVoxelClustered(leaf_num,1);

				}
				else
				{
					//Set voxel chosen or not
					this->setVoxelChosen(leaf_num, 1);
					//Set voxel clustered or not
					this->setVoxelClustered(leaf_num,1);
				}
		
				++leaf_num;

				//std::cout<<leaf_num<<std::endl;
			}
		
		}


		//Features
		void
		calcualteVoxelCloudAttributesGeo(PCXYZPtr input_cloud, bpn::ndarray normals)
		{
			//Iteration index
			VoxelBasedSegmentation<pcl::PointXYZ>::LeafNodeIterator iter_leaf (this); //Iteration index for all leaf nodes in the tree
	
			//Initial feature
			this->initialVoxelAttributes();		
		
			int leaf_num = 0;
			std::vector<float> voxel_features;
			std::vector<float> voxel_eigens;
			std::vector<float> voxel_colors;
			pcl::Normal voxel_norm;
			pcl::PointXYZ voxel_centroid;

			//Tranverse the octree leaves
			while (*++iter_leaf)// The initial node is neglected!
			{
				int point_num=0;
				int points_size=0;
				int point_idx=0;
				pcl::PointXYZ node_center;

				std::vector<int> index_vector;
				PCXYZPtr voxel_cloud (new PCXYZ);

				//Find centers of the leaf node			
				points_size=iter_leaf.getLeafContainer().getSize();
				iter_leaf.getLeafContainer().getPointIndices(index_vector);

				//Traverse the points in node
				if(points_size>this->voxel_points_min_)
				{
					Eigen::Vector3f normal;
					//Get the points in the voxel
					float nx = 0.0f;
					float ny = 0.0f;
					float nz = 0.0f;
					while(point_num<points_size)
					{
						point_idx=index_vector[point_num];

						nx += bp::extract<float>(normals[point_idx][0]);
						ny += bp::extract<float>(normals[point_idx][1]);
						nz += bp::extract<float>(normals[point_idx][2]);

						voxel_cloud->points.push_back(input_cloud->points[point_idx]);

						point_num++;
					}

					nx /= (float)points_size;
					ny /= (float)points_size;
					nz /= (float)points_size;

					float mag_n = sqrtf(nx*nx + ny*ny + nz*nz);
					nx /= mag_n;
					ny /= mag_n;
					nz /= mag_n;
		
					voxel_norm.normal_x = nx;
					voxel_norm.normal_y = ny;
					voxel_norm.normal_z = nz;

					//Output points in leaf node
					voxel_cloud->width = (int)voxel_cloud->points.size();  
					voxel_cloud->height = 1;

					//Centroid calculation
					voxel_centroid=this->calculateVoxelCentroid(voxel_cloud);//Centroid
					this->setVoxelCentroid(leaf_num,voxel_centroid);

					//Norm calculation
					//voxel_norm=this->calculateVoxelNorms(voxel_cloud);//Norms
					this->setVoxelNorms(leaf_num,voxel_norm);

					//Eigen calculation of the points in a voxel
					voxel_eigens=this->calculateEigenFeatures(voxel_cloud);//Eigen features
					this->setVoxelEigens(leaf_num,voxel_eigens);
					
					//Set voxel chosen or not
					this->setVoxelChosen(leaf_num, 0);
					//Set voxel clustered or not
					this->setVoxelClustered(leaf_num,1);

				}
				else
				{
					//Set voxel chosen or not
					this->setVoxelChosen(leaf_num, 1);
					//Set voxel clustered or not
					this->setVoxelClustered(leaf_num,1);
				}
		
				++leaf_num;

				//std::cout<<leaf_num<<std::endl;
			}
		
		}


		//Segmentation
		void
		segmentVoxelCloudWithGraphModel(float cut_thred,float sig_p, float sig_n, float sig_o, float sig_e, float sig_c, float sig_w)      //Segmentation VGS
		{
			/*
			// TODO Delete!!!
			int n_empty = 0;
			pcl::Normal temp_norm;
			for (int i=0; i<voxel_norms_.size(); i++){
				temp_norm=voxel_norms_[i];
				if(temp_norm.normal_x==0 || temp_norm.normal_y==0 || temp_norm.normal_z==0)
				{
					n_empty++;
				}
			}
			double percent = (double)n_empty / (double) voxel_norms_.size();
			printf("%f voxels have empty normals\n", percent);
			*/

			//Connectivity calculation
			for(int i=0;i<voxels_num_;i++)
			{
				//Judgement
				if(this->voxel_used_[i])//Voxel used or not
				{
					Eigen::MatrixXf voxel_adjgraph;
					std::vector<int> voxel_connect;
					std::vector<int> voxel_adjidx;
					
					//Get adjacency
					//printf("Get adjacency\n");
					voxel_adjidx=this->getOneVoxelAdjacency(i);

					if(voxel_adjidx.size() == 0)
					{
						//Store the connection informaiton
						std::vector<int> voxel_connect;
						//voxel_connect.push_back(0);
						//printf("voxel_connect.push_back(0)\n");
						voxels_connect_idx_.push_back(voxel_connect);
						continue;
					}

					//Graph building
					//printf("Graph building\n");
					voxel_adjgraph=this->buildAdjacencyGraph(voxel_adjidx,sig_p,sig_n,sig_o,sig_e,sig_c,sig_w);

					//Graph cut
					//printf("Graph cut\n");
					//voxel_connect=this->cutGraphThreshold(cut_thred,voxel_adjgraph,voxel_adjidx);
					//voxel_connect=this->cutGraphKMeans(cut_thred,voxel_adjgraph,voxel_adjidx);
					//voxel_connect=this->cutGraphMaxFlow(cut_thred,voxel_adjgraph,voxel_adjidx);
					//voxel_connect=this->cutGraphMinCut(cut_thred,voxel_adjgraph,voxel_adjidx);
					//voxel_connect=this->cutGraphNormCut(cut_thred,voxel_adjgraph,voxel_adjidx);
					voxel_connect=this->cutGraphSegmentation(cut_thred,voxel_adjgraph,voxel_adjidx);

					//Store the connection information
					//printf("Store the connection information\n");
					voxels_connect_idx_.push_back(voxel_connect);

				}
				else
				{
					//Store the connection informaiton
					std::vector<int> voxel_connect;
					//voxel_connect.push_back(0);
					//printf("voxel_connect.push_back(0)\n");
					voxels_connect_idx_.push_back(voxel_connect);
				}

				//std::cout<<i<<std::endl;
			}
			//Cross validation
			//printf("Cross validation\n");
 			this->crossValidation();
			//Closest checking
			//printf("Closest checking\n");
			this->closestCheck(sig_p, sig_n, sig_o, sig_e, sig_c, sig_w);
			//Voxel Clustering
			//printf("Voxel Clustering\n");
			this->clusteringVoxels();

		}


		void
		segmentVoxelCloudWithGraphModelGeo(float cut_thred,float sig_p, float sig_n, float sig_o, float sig_e, float sig_c, float sig_w)      //Segmentation VGS
		{
			/*
			// TODO Delete!!!
			int n_empty = 0;
			pcl::Normal temp_norm;
			for (int i=0; i<voxel_norms_.size(); i++){
				temp_norm=voxel_norms_[i];
				if(temp_norm.normal_x==0 || temp_norm.normal_y==0 || temp_norm.normal_z==0)
				{
					n_empty++;
				}
			}
			double percent = (double)n_empty / (double) voxel_norms_.size();
			printf("%f voxels have empty normals\n", percent);
			*/
			//Connectivity calculation
			for(int i=0;i<voxels_num_;i++)
			{
				//Judgement
				if(this->voxel_used_[i])//Voxel used or not
				{
					Eigen::MatrixXf voxel_adjgraph;
					std::vector<int> voxel_connect;
					std::vector<int> voxel_adjidx;
					
					//printf("Get Adjacency\n");
					//Get adjacency
					voxel_adjidx=this->getOneVoxelAdjacency(i);

					//printf("Build Graph Geo\n");
					//Graph building
					voxel_adjgraph=this->buildAdjacencyGraphGeo(voxel_adjidx,sig_p,sig_n,sig_o,sig_e,sig_c,sig_w);

					//Graph cut
					//voxel_connect=this->cutGraphThreshold(cut_thred,voxel_adjgraph,voxel_adjidx);
					//voxel_connect=this->cutGraphKMeans(cut_thred,voxel_adjgraph,voxel_adjidx);
					//voxel_connect=this->cutGraphMaxFlow(cut_thred,voxel_adjgraph,voxel_adjidx);
					//voxel_connect=this->cutGraphMinCut(cut_thred,voxel_adjgraph,voxel_adjidx);
					//voxel_connect=this->cutGraphNormCut(cut_thred,voxel_adjgraph,voxel_adjidx);
					voxel_connect=this->cutGraphSegmentation(cut_thred,voxel_adjgraph,voxel_adjidx);

					//Store the connection information
					voxels_connect_idx_.push_back(voxel_connect);

				}
				else
				{
					//Store the connection informaiton
					std::vector<int> voxel_connect;
					//voxel_connect.push_back(0);
					voxels_connect_idx_.push_back(voxel_connect);
				}

				//std::cout<<i<<std::endl;
			}
	
			//Cross validation
 			this->crossValidation();
			//Closest checking
			this->closestCheck(sig_p, sig_n, sig_o, sig_e, sig_c, sig_w);
			//Voxel Clustering
			this->clusteringVoxels();
			spatial_distances_.clear();
		}


		//Display
		void
		drawColorMapofPointsinVoxels(pcl::PointCloud<pcl::PointXYZRGB>::Ptr output_cloud)
		{
			//Setting
			int node_ID;
			int points_size;
			int points_min=voxel_points_min_;
			unsigned int leaf_count = 0, leaf_num = 0;

			pcl::PointXYZ original_point; 
			pcl::PointXYZRGB colored_point;
			std::vector<int> color_map(3);
			srand(static_cast<unsigned int> (time(0)));

			VoxelBasedSegmentation<pcl::PointXYZ>::LeafNodeIterator iter_leaf (this); //Iteration index for all leaf nodes in the tree

			//Tranverse the octree leaves
			while (*++iter_leaf)
			{
				int point_num=0;
				int point_idx=0;
				int node_depth=0;
				pcl::PointXYZ node_center;

				//Set random color of points in this voxel
				color_map[0]=static_cast<unsigned int> (rand()%256);
				color_map[1]=static_cast<unsigned int> (rand()%256);
				color_map[2]=static_cast<unsigned int> (rand()%256);

				std::vector<int> index_vector;
				PCXYZPtr voxel_cloud (new PCXYZ);

				//Find the index of the node
				node_ID=iter_leaf.getNodeID();				
				points_size=iter_leaf.getLeafContainer().getSize();
				iter_leaf.getLeafContainer().getPointIndices(index_vector);

				//Search centers from the key of the node leaf
				pcl::octree::OctreeKey leaf_key;
				leaf_key=iter_leaf.getCurrentOctreeKey();
				this->getVoxelCenterFromOctreeKey(leaf_key, node_center);

				//Traverse the points in node and color them
				bool temp_bool=true;
				//if(leaf_num>176 && leaf_num<178)
				//	temp_bool=true;
				//else if(leaf_num>181 && leaf_num<186)
				//	temp_bool=true;
				//else if(leaf_num>186 && leaf_num<188)
				//	temp_bool=true;
				//else if(leaf_num>188 && leaf_num<196)
				//	temp_bool=true;
				//else if(leaf_num==428)
				//	temp_bool=true;

				if(points_size>points_min && temp_bool==true)
				{
					//Get the points in the voxel
					while(point_num<points_size)
					{
						point_idx=index_vector[point_num];
						original_point=points_cloud_->points[point_idx];//Find original point
						
						colored_point.x=original_point.x;
						colored_point.y=original_point.y;
						colored_point.z=original_point.z;
						colored_point.r=color_map[0];
						colored_point.g=color_map[1];
						colored_point.b=color_map[2];

						//Coloring point
						output_cloud->points.push_back(colored_point);//Put colored point in the cloud

						point_num++;
					}
		
					++leaf_count;
				}

				++leaf_num;
			}

			output_cloud->width = (int)output_cloud->points.size();  
			output_cloud->height = 1;	
		}

		void
		drawColorMapofVoxels(pcl::PolygonMesh::Ptr output_mesh)//Draw the colored boxes of the voxels
		{

			//Setting
			std::vector<pcl::Vertices> polygons; 
			pcl::PointCloud<pcl::PointXYZRGB> clouds_vertices;

			int node_ID;
			int points_size;
			int points_min=voxel_points_min_;
			unsigned int leaf_count = 0;

			pcl::PointXYZRGB colored_vertex;
			std::vector<int> color_map(3);
			srand(static_cast<unsigned int> (time(0)));

			VoxelBasedSegmentation<pcl::PointXYZ>::LeafNodeIterator iter_leaf (this); //Iteration index for all leaf nodes in the tree

			//Tranverse the octree leaves
			int i=0;
			while (*++iter_leaf)
			{
				int point_num=0;
				int point_idx=0;
				int node_depth=0;
				pcl::PointXYZ node_center;

				//Set random color of points in this voxel
				color_map[0]=static_cast<unsigned int> (rand()%256);
				color_map[1]=static_cast<unsigned int> (rand()%256);
				color_map[2]=static_cast<unsigned int> (rand()%256);

				//Vertices of the voxel
				PCXYZRGBPtr voxel_vertices (new PCXYZRGB);
				voxel_vertices->points.resize(8);

				//Find the index of the node
				node_ID=iter_leaf.getNodeID();				
				points_size=iter_leaf.getLeafContainer().getSize();

				//Search centers from the key of the node leaf
				pcl::octree::OctreeKey leaf_key;
				leaf_key=iter_leaf.getCurrentOctreeKey();
				this->getVoxelCenterFromOctreeKey(leaf_key, node_center);
				//node_center=voxel_centers_[leaf_count];

				//Traverse the points in node and color them
				if(points_size>points_min)
				{
					//Get the points in the voxel	
					voxel_vertices->points[0].x=node_center.x-0.5*voxel_resolution_;//X
					voxel_vertices->points[1].x=node_center.x+0.5*voxel_resolution_;
					voxel_vertices->points[2].x=node_center.x+0.5*voxel_resolution_;
					voxel_vertices->points[3].x=node_center.x-0.5*voxel_resolution_;
					voxel_vertices->points[4].x=node_center.x-0.5*voxel_resolution_;					
					voxel_vertices->points[5].x=node_center.x+0.5*voxel_resolution_;
					voxel_vertices->points[6].x=node_center.x+0.5*voxel_resolution_;					
					voxel_vertices->points[7].x=node_center.x-0.5*voxel_resolution_;					
					
					voxel_vertices->points[0].y=node_center.y-0.5*voxel_resolution_;//Y
					voxel_vertices->points[1].y=node_center.y-0.5*voxel_resolution_;
					voxel_vertices->points[2].y=node_center.y+0.5*voxel_resolution_;
					voxel_vertices->points[3].y=node_center.y+0.5*voxel_resolution_;
					voxel_vertices->points[4].y=node_center.y-0.5*voxel_resolution_;					
					voxel_vertices->points[5].y=node_center.y-0.5*voxel_resolution_;
					voxel_vertices->points[6].y=node_center.y+0.5*voxel_resolution_;					
					voxel_vertices->points[7].y=node_center.y+0.5*voxel_resolution_;
	
					for(int j=0;j<4;j++)//Z
					{
						voxel_vertices->points[j].z=node_center.z-0.5*voxel_resolution_;
						voxel_vertices->points[j+4].z=node_center.z+0.5*voxel_resolution_;
					}
					
					//Color
					for(int j=0;j<8;j++)
					{
						voxel_vertices->points[j].r=color_map[0];
						voxel_vertices->points[j].g=color_map[1];
						voxel_vertices->points[j].b=color_map[2];
					}

					//Input vertices points
					for(int j=0;j<8;j++)
					{
						clouds_vertices.points.push_back(voxel_vertices->points[j]);
					}

					//Input vertices topology
					pcl::Vertices vertice1;
					vertice1.vertices.push_back(i*8+0);vertice1.vertices.push_back(i*8+1);vertice1.vertices.push_back(i*8+2);//012
					output_mesh->polygons.push_back(vertice1);
					pcl::Vertices vertice2;
					vertice2.vertices.push_back(i*8+0);vertice2.vertices.push_back(i*8+2);vertice2.vertices.push_back(i*8+3);//023
					output_mesh->polygons.push_back(vertice2);
					//4567
					pcl::Vertices vertice3;
					vertice3.vertices.push_back(i*8+4);vertice3.vertices.push_back(i*8+5);vertice3.vertices.push_back(i*8+6);//456
					output_mesh->polygons.push_back(vertice3);
					pcl::Vertices vertice4;
					vertice4.vertices.push_back(i*8+4);vertice4.vertices.push_back(i*8+6);vertice4.vertices.push_back(i*8+7);//467
					output_mesh->polygons.push_back(vertice4);
					//0145
					pcl::Vertices vertice5;
					vertice5.vertices.push_back(i*8+0);vertice5.vertices.push_back(i*8+1);vertice5.vertices.push_back(i*8+5);//015
					output_mesh->polygons.push_back(vertice5);
					pcl::Vertices vertice6;
					vertice6.vertices.push_back(i*8+0);vertice6.vertices.push_back(i*8+5);vertice6.vertices.push_back(i*8+4);//054
					output_mesh->polygons.push_back(vertice6);
					//1265
					pcl::Vertices vertice7;
					vertice7.vertices.push_back(i*8+1);vertice7.vertices.push_back(i*8+2);vertice7.vertices.push_back(i*8+5);//125
					output_mesh->polygons.push_back(vertice7);
					pcl::Vertices vertice8;
					vertice8.vertices.push_back(i*8+2);vertice8.vertices.push_back(i*8+6);vertice8.vertices.push_back(i*8+5);//265
					output_mesh->polygons.push_back(vertice8);
					//0374
					pcl::Vertices vertice9;
					vertice9.vertices.push_back(i*8+0);vertice9.vertices.push_back(i*8+3);vertice9.vertices.push_back(i*8+7);//012
					output_mesh->polygons.push_back(vertice9);
					pcl::Vertices vertice10;
					vertice10.vertices.push_back(i*8+0);vertice10.vertices.push_back(i*8+7);vertice10.vertices.push_back(i*8+4);//074
					output_mesh->polygons.push_back(vertice10);
					//2376
					pcl::Vertices vertice11;
					vertice11.vertices.push_back(i*8+2);vertice11.vertices.push_back(i*8+3);vertice11.vertices.push_back(i*8+7);//237
					output_mesh->polygons.push_back(vertice11);
					pcl::Vertices vertice12;
					vertice12.vertices.push_back(i*8+2);vertice12.vertices.push_back(i*8+7);vertice12.vertices.push_back(i*8+6);//276
					output_mesh->polygons.push_back(vertice12);

					i++;
				}
				
				++leaf_count; 

			}

			//Create polygonmesh
			pcl::toPCLPointCloud2(clouds_vertices,output_mesh->cloud);
		}

		void
		drawColorMapofClusteredVoxels(pcl::PolygonMesh::Ptr output_mesh)//Draw the colored clusters with the voxels in them sharing the same color
		{
			//Geometry
			std::vector<pcl::Vertices> polygons; 
			pcl::PointCloud<pcl::PointXYZRGB> clouds_vertices;
			std::vector<int> use_or_not;

			//Setting
			int i=0;
			int points_size;
			int points_min=voxel_points_min_;
			unsigned int leaf_count = 0;

			//pcl::PointXYZ colored_vertex;
			pcl::PointXYZRGB colored_vertex;
			std::vector<int> color_map(3);
			srand(static_cast<unsigned int> (time(0)));

			//Tranverse the octree leaves
			for (int m=0;m<this->clusters_num_;m++)
			{
				color_map[0]=static_cast<unsigned int> (rand()%256);
				color_map[1]=static_cast<unsigned int> (rand()%256);
				color_map[2]=static_cast<unsigned int> (rand()%256);

				for (int n=0;n<this->clusters_voxel_idx_[m].size();n++)
				{
					int point_num=0;
					int point_idx=0;
					int voxel_idx=0;
					pcl::PointXYZ node_center;

					//Corresponding voxel idx
					voxel_idx=this->clusters_voxel_idx_[m][n];

					//Vertices of the voxel
					PCXYZRGBPtr voxel_vertices (new PCXYZRGB);
					voxel_vertices->points.resize(8);

					//Find the index of the node
					node_center=voxel_centers_[voxel_idx];

					//Traverse the points in node and color them
					if(voxel_used_[voxel_idx])
					{
						//Get the points in the voxel	
						voxel_vertices->points[0].x=node_center.x-0.5*voxel_resolution_;//X
						voxel_vertices->points[1].x=node_center.x+0.5*voxel_resolution_;
						voxel_vertices->points[2].x=node_center.x+0.5*voxel_resolution_;
						voxel_vertices->points[3].x=node_center.x-0.5*voxel_resolution_;
						voxel_vertices->points[4].x=node_center.x-0.5*voxel_resolution_;					
						voxel_vertices->points[5].x=node_center.x+0.5*voxel_resolution_;
						voxel_vertices->points[6].x=node_center.x+0.5*voxel_resolution_;					
						voxel_vertices->points[7].x=node_center.x-0.5*voxel_resolution_;					
					
						voxel_vertices->points[0].y=node_center.y-0.5*voxel_resolution_;//Y
						voxel_vertices->points[1].y=node_center.y-0.5*voxel_resolution_;
						voxel_vertices->points[2].y=node_center.y+0.5*voxel_resolution_;
						voxel_vertices->points[3].y=node_center.y+0.5*voxel_resolution_;
						voxel_vertices->points[4].y=node_center.y-0.5*voxel_resolution_;					
						voxel_vertices->points[5].y=node_center.y-0.5*voxel_resolution_;
						voxel_vertices->points[6].y=node_center.y+0.5*voxel_resolution_;					
						voxel_vertices->points[7].y=node_center.y+0.5*voxel_resolution_;
	
						for(int i=0;i<4;i++)//Z
						{
							voxel_vertices->points[i].z=node_center.z-0.5*voxel_resolution_;
							voxel_vertices->points[i+4].z=node_center.z+0.5*voxel_resolution_;
						}
					
						//Color
						for(int i=0;i<8;i++)
						{
							voxel_vertices->points[i].r=color_map[0];
							voxel_vertices->points[i].g=color_map[1];
							voxel_vertices->points[i].b=color_map[2];
						}

		
						for(int i=0;i<8;i++)
						{
							clouds_vertices.points.push_back(voxel_vertices->points[i]);
						}

						//Record
						i=leaf_count;
						//0123
						pcl::Vertices vertice1;
						vertice1.vertices.push_back(i*8+0);vertice1.vertices.push_back(i*8+1);vertice1.vertices.push_back(i*8+2);//012
						output_mesh->polygons.push_back(vertice1);
						pcl::Vertices vertice2;
						vertice2.vertices.push_back(i*8+0);vertice2.vertices.push_back(i*8+2);vertice2.vertices.push_back(i*8+3);//023
						output_mesh->polygons.push_back(vertice2);
						//4567
						pcl::Vertices vertice3;
						vertice3.vertices.push_back(i*8+4);vertice3.vertices.push_back(i*8+5);vertice3.vertices.push_back(i*8+6);//456
						output_mesh->polygons.push_back(vertice3);
						pcl::Vertices vertice4;
						vertice4.vertices.push_back(i*8+4);vertice4.vertices.push_back(i*8+6);vertice4.vertices.push_back(i*8+7);//467
						output_mesh->polygons.push_back(vertice4);
						//0145
						pcl::Vertices vertice5;
						vertice5.vertices.push_back(i*8+0);vertice5.vertices.push_back(i*8+1);vertice5.vertices.push_back(i*8+5);//015
						output_mesh->polygons.push_back(vertice5);
						pcl::Vertices vertice6;
						vertice6.vertices.push_back(i*8+0);vertice6.vertices.push_back(i*8+5);vertice6.vertices.push_back(i*8+4);//054
						output_mesh->polygons.push_back(vertice6);
						//1265
						pcl::Vertices vertice7;
						vertice7.vertices.push_back(i*8+1);vertice7.vertices.push_back(i*8+2);vertice7.vertices.push_back(i*8+5);//125
						output_mesh->polygons.push_back(vertice7);
						pcl::Vertices vertice8;
						vertice8.vertices.push_back(i*8+2);vertice8.vertices.push_back(i*8+6);vertice8.vertices.push_back(i*8+5);//265
						output_mesh->polygons.push_back(vertice8);
						//0374
						pcl::Vertices vertice9;
						vertice9.vertices.push_back(i*8+0);vertice9.vertices.push_back(i*8+3);vertice9.vertices.push_back(i*8+7);//012
						output_mesh->polygons.push_back(vertice9);
						pcl::Vertices vertice10;
						vertice10.vertices.push_back(i*8+0);vertice10.vertices.push_back(i*8+7);vertice10.vertices.push_back(i*8+4);//074
						output_mesh->polygons.push_back(vertice10);
						//2376
						pcl::Vertices vertice11;
						vertice11.vertices.push_back(i*8+2);vertice11.vertices.push_back(i*8+3);vertice11.vertices.push_back(i*8+7);//237
						output_mesh->polygons.push_back(vertice11);
						pcl::Vertices vertice12;
						vertice12.vertices.push_back(i*8+2);vertice12.vertices.push_back(i*8+7);vertice12.vertices.push_back(i*8+6);//276
						output_mesh->polygons.push_back(vertice12);

						++leaf_count;
					}

				}

			}

			//Create polygonmesh
			pcl::toPCLPointCloud2(clouds_vertices,output_mesh->cloud);
		}

		void
		drawFrameMapofVoxels(pcl::PolygonMesh::Ptr output_mesh)
		{
			//Setting
			std::vector<pcl::Vertices> polygons; 
			pcl::PointCloud<pcl::PointXYZRGB> clouds_vertices;

			int node_ID;
			int points_size;
			int points_min=voxel_points_min_;
			unsigned int leaf_count = 0;

			pcl::PointXYZRGB colored_vertex;
			std::vector<int> color_map(3);
			srand(static_cast<unsigned int> (time(0)));

			VoxelBasedSegmentation<pcl::PointXYZ>::LeafNodeIterator iter_leaf (this); //Iteration index for all leaf nodes in the tree

			//Tranverse the octree leaves
			int i=0;
			while (*++iter_leaf)
			{
				int point_num=0;
				int point_idx=0;
				int node_depth=0;
				pcl::PointXYZ node_center;

				//Set random color of points in this voxel
				color_map[0]=static_cast<unsigned int> (rand()%256);
				color_map[1]=static_cast<unsigned int> (rand()%256);
				color_map[2]=static_cast<unsigned int> (rand()%256);

				//Vertices of the voxel
				PCXYZRGBPtr voxel_vertices (new PCXYZRGB);
				voxel_vertices->points.resize(8);

				//Find the index of the node
				node_ID=iter_leaf.getNodeID();				
				points_size=iter_leaf.getLeafContainer().getSize();

				//Search centers from the key of the node leaf
				pcl::octree::OctreeKey leaf_key;
				leaf_key=iter_leaf.getCurrentOctreeKey();
				this->getVoxelCenterFromOctreeKey(leaf_key, node_center);
				//node_center=voxel_centers_[leaf_count];

				//Test figure draw
				bool temp_bool=true;
				//if(leaf_count>176 && leaf_count<178)
				//	temp_bool=true;
				//else if(leaf_count>181 && leaf_count<186)
				//	temp_bool=true;
				//else if(leaf_count>186 && leaf_count<188)
				//	temp_bool=true;
				//else if(leaf_count>188 && leaf_count<196)
				//	temp_bool=true;
				//else if(leaf_count==428)
				//	temp_bool=true;


				//Traverse the points in node and color them
				if(points_size>points_min && temp_bool==true)
				{
					//Get the points in the voxel	
					voxel_vertices->points[0].x=node_center.x-0.5*voxel_resolution_;//X
					voxel_vertices->points[1].x=node_center.x+0.5*voxel_resolution_;
					voxel_vertices->points[2].x=node_center.x+0.5*voxel_resolution_;
					voxel_vertices->points[3].x=node_center.x-0.5*voxel_resolution_;
					voxel_vertices->points[4].x=node_center.x-0.5*voxel_resolution_;					
					voxel_vertices->points[5].x=node_center.x+0.5*voxel_resolution_;
					voxel_vertices->points[6].x=node_center.x+0.5*voxel_resolution_;					
					voxel_vertices->points[7].x=node_center.x-0.5*voxel_resolution_;					
					
					voxel_vertices->points[0].y=node_center.y-0.5*voxel_resolution_;//Y
					voxel_vertices->points[1].y=node_center.y-0.5*voxel_resolution_;
					voxel_vertices->points[2].y=node_center.y+0.5*voxel_resolution_;
					voxel_vertices->points[3].y=node_center.y+0.5*voxel_resolution_;
					voxel_vertices->points[4].y=node_center.y-0.5*voxel_resolution_;					
					voxel_vertices->points[5].y=node_center.y-0.5*voxel_resolution_;
					voxel_vertices->points[6].y=node_center.y+0.5*voxel_resolution_;					
					voxel_vertices->points[7].y=node_center.y+0.5*voxel_resolution_;
	
					for(int j=0;j<4;j++)//Z
					{
						voxel_vertices->points[j].z=node_center.z-0.5*voxel_resolution_;
						voxel_vertices->points[j+4].z=node_center.z+0.5*voxel_resolution_;
					}
					
					//Color
					for(int j=0;j<8;j++)
					{
						voxel_vertices->points[j].r=color_map[0];
						voxel_vertices->points[j].g=color_map[1];
						voxel_vertices->points[j].b=color_map[2];
					}

					//Input vertices points
					for(int j=0;j<8;j++)
					{
						clouds_vertices.points.push_back(voxel_vertices->points[j]);
					}

					//Input vertices topology
					//0123
					pcl::Vertices vertice01;
					vertice01.vertices.push_back(i*8+0);vertice01.vertices.push_back(i*8+1);vertice01.vertices.push_back(i*8+0);//01
					output_mesh->polygons.push_back(vertice01);
					pcl::Vertices vertice03;
					vertice03.vertices.push_back(i*8+0);vertice03.vertices.push_back(i*8+3);vertice03.vertices.push_back(i*8+0);//03
					output_mesh->polygons.push_back(vertice03);
					pcl::Vertices vertice12;
					vertice12.vertices.push_back(i*8+1);vertice12.vertices.push_back(i*8+2);vertice12.vertices.push_back(i*8+1);//12
					output_mesh->polygons.push_back(vertice12);
					pcl::Vertices vertice23;
					vertice23.vertices.push_back(i*8+2);vertice23.vertices.push_back(i*8+3);vertice23.vertices.push_back(i*8+2);//23
					output_mesh->polygons.push_back(vertice23);
					//4567
					pcl::Vertices vertice45;
					vertice45.vertices.push_back(i*8+4);vertice45.vertices.push_back(i*8+5);vertice45.vertices.push_back(i*8+4);//45
					output_mesh->polygons.push_back(vertice45);
					pcl::Vertices vertice56;
					vertice56.vertices.push_back(i*8+5);vertice56.vertices.push_back(i*8+6);vertice56.vertices.push_back(i*8+5);//56
					output_mesh->polygons.push_back(vertice56);
					pcl::Vertices vertice67;
					vertice67.vertices.push_back(i*8+6);vertice67.vertices.push_back(i*8+7);vertice67.vertices.push_back(i*8+6);//67
					output_mesh->polygons.push_back(vertice67);
					pcl::Vertices vertice74;
					vertice74.vertices.push_back(i*8+7);vertice74.vertices.push_back(i*8+4);vertice74.vertices.push_back(i*8+7);//74
					output_mesh->polygons.push_back(vertice74);
					//Rest
					pcl::Vertices vertice04;
					vertice04.vertices.push_back(i*8+0);vertice04.vertices.push_back(i*8+4);vertice04.vertices.push_back(i*8+0);//04
					output_mesh->polygons.push_back(vertice04);
					pcl::Vertices vertice15;
					vertice15.vertices.push_back(i*8+1);vertice15.vertices.push_back(i*8+5);vertice15.vertices.push_back(i*8+1);//15
					output_mesh->polygons.push_back(vertice15);
					pcl::Vertices vertice26;
					vertice26.vertices.push_back(i*8+2);vertice26.vertices.push_back(i*8+6);vertice26.vertices.push_back(i*8+2);//26
					output_mesh->polygons.push_back(vertice26);
					pcl::Vertices vertice37;
					vertice37.vertices.push_back(i*8+3);vertice37.vertices.push_back(i*8+7);vertice37.vertices.push_back(i*8+3);//37
					output_mesh->polygons.push_back(vertice37);
					i++;
				}
				
				++leaf_count; 

			}

			//Create polygonmesh
			pcl::toPCLPointCloud2(clouds_vertices,output_mesh->cloud);
		}

		void
		drawColorMapofPointsinClusters(pcl::PointCloud<pcl::PointXYZRGB>::Ptr output_cloud)//This is obligatory!
		{
			//Setting
			int i=0;
			int points_size=0;
			int out_num=0;
			unsigned int leaf_count = 0;

			//Random color
			pcl::PointXYZRGB colored_point;
			pcl::PointXYZ original_point;
			std::vector<int> color_map(3), cluster_points;
			srand(static_cast<unsigned int> (time(0)));

			//Tranverse the octree leaves
			for (int m=0;m<this->clusters_num_;m++)
			{
				color_map[0]=static_cast<unsigned int> (rand()%256);
				color_map[1]=static_cast<unsigned int> (rand()%256);
				color_map[2]=static_cast<unsigned int> (rand()%256);

				if(this->clusters_voxel_idx_[m].size()>this->cluster_voxels_min_)
				{

					for (int n=0;n<this->clusters_voxel_idx_[m].size();n++)
					{

						int voxel_idx=0;
						//Corresponding voxel idx
						voxel_idx=this->clusters_voxel_idx_[m][n];

						//Traverse the points in node and color them
						int point_idx=0;
						std::vector<int> points_idx;
						points_idx=this->voxels_point_idx_[voxel_idx];
						points_size=points_idx.size();
					
						//Coloring
						for(int i=0;i<points_size;i++)
						{
							original_point=this->points_cloud_->points[points_idx[i]];
							colored_point.x=original_point.x;						
							colored_point.y=original_point.y;
							colored_point.z=original_point.z;
							colored_point.r=color_map[0];
							colored_point.g=color_map[1];
							colored_point.b=color_map[2];
							if(clusters_voxel_idx_[m].size()>this->cluster_voxels_min_)
							{
								cluster_points.push_back(points_idx[i]);
							}
							output_cloud->push_back(colored_point);
						}

						//std::cout<<leaf_count<<std::endl;
						leaf_count++;
					}

					this->clusters_point_idx_.push_back(cluster_points);
					cluster_points.clear();
				}
			}

			output_cloud->width=output_cloud->points.size();
			output_cloud->height=1;

		}

		void
		drawNormofVoxels(pcl::PolygonMesh::Ptr output_mesh)
		{
			//Setting
			std::vector<pcl::Vertices> polygons; 
			pcl::PointCloud<pcl::PointXYZRGB> clouds_vertices;

			int node_ID;
			int points_size;
			int points_min=voxel_points_min_;
			unsigned int leaf_count = 0;

			//Set random color of points in this voxel
			pcl::PointXYZRGB colored_vertex;
			std::vector<int> color_map(3);
			srand(static_cast<unsigned int> (time(0)));

			color_map[0]=static_cast<unsigned int> (rand()%256);
			color_map[1]=static_cast<unsigned int> (rand()%256);
			color_map[2]=static_cast<unsigned int> (rand()%256);

			VoxelBasedSegmentation<pcl::PointXYZ>::LeafNodeIterator iter_leaf (this); //Iteration index for all leaf nodes in the tree

			//Tranverse the octree leaves
			int i=0;
			while (*++iter_leaf)
			{
				int point_num=0;
				int point_idx=0;
				int node_depth=0;
				pcl::PointXYZ node_center;

				//Vertices of the voxel
				PCXYZRGBPtr voxel_vertices (new PCXYZRGB);
				voxel_vertices->points.resize(2);

				//Find the index of the node
				node_ID=iter_leaf.getNodeID();				
				points_size=iter_leaf.getLeafContainer().getSize();

				//Search centers from the key of the node leaf
				pcl::octree::OctreeKey leaf_key;
				leaf_key=iter_leaf.getCurrentOctreeKey();
				this->getVoxelCenterFromOctreeKey(leaf_key, node_center);

				//Traverse the points in node and color them
				if(points_size>points_min)
				{
					//Get the points in the voxel	
					voxel_vertices->points[0].x=node_center.x;//X
					voxel_vertices->points[1].x=node_center.x+voxel_resolution_*voxel_norms_[leaf_count].normal_x;
									
					voxel_vertices->points[0].y=node_center.y;//Y
					voxel_vertices->points[1].y=node_center.y+voxel_resolution_*voxel_norms_[leaf_count].normal_y;

					voxel_vertices->points[0].z=node_center.z;//Z
					voxel_vertices->points[1].z=node_center.z+voxel_resolution_*voxel_norms_[leaf_count].normal_z;

					//Color
					for(int j=0;j<2;j++)
					{
						voxel_vertices->points[j].r=color_map[0];
						voxel_vertices->points[j].g=color_map[1];
						voxel_vertices->points[j].b=color_map[2];
					}

					//Input vertices points
					for(int j=0;j<2;j++)
					{
						clouds_vertices.points.push_back(voxel_vertices->points[j]);
					}

					//Input vertices topology
					//center to norm
					pcl::Vertices vertice0;
					vertice0.vertices.push_back(i*2+0);vertice0.vertices.push_back(i*2+1);vertice0.vertices.push_back(i*2+0);
					output_mesh->polygons.push_back(vertice0);
		
					i++;
				}
				
				++leaf_count; 

			}

			//Create polygonmesh
			pcl::toPCLPointCloud2(clouds_vertices,output_mesh->cloud);

		}

    private:
		
		//Member variables 
		int points_num_;				//Num of input points
		int voxels_num_;				//Num of generated voxels
		int clusters_num_;				//Num of aggregated clusters
		int segments_num_;				//Num of output segments
		int centers_num_;				//Num of centers of all the voxels
		int voxel_points_min_;			//Min Num of points in a voxel
		int voxel_points_max_;			//Max num of points in a voxel
		int voxel_adjacency_min_;		//Min num of adjacent voxels
		int voxel_adjacency_max_;		//Max num of adjacent voxels
		int cluster_voxels_min_;		//Min num of voxels in a cluster
		int cluster_voxels_max_;		//Max num of voxels in a cluster

		float voxel_resolution_;		//Resolution(size) of the voxel
		float graph_resolution_;		//Resolution(size) of the local affinity graph
		float min_x_, min_y_, min_z_, max_x_, max_y_, max_z_;

		std::vector<pcl::PointXYZ, Eigen::aligned_allocator<pcl::PointXYZ> > voxel_centers_;    //Center point of the voxels
		std::vector<pcl::PointXYZ, Eigen::aligned_allocator<pcl::PointXYZ> > voxel_centroids_;  //Center point of the voxels
		std::vector<bool> voxel_used_;															//The voxel is chosen or not
		std::vector<bool> voxel_clustered_;														//The voxel has been clutered or not

		std::vector<std::vector<float> > voxel_eigens_;											//Eigen features of all the voxels
		std::vector<pcl::Normal> voxel_norms_;													//Norms of all the voxels

		std::vector<std::vector<int> > voxels_adjacency_idx_, voxels_regional_idx_;				//Idx of the adjacent voxels
		std::vector<std::vector<int> > voxels_connect_idx_;										//Idx of the connected voxels
		std::vector<std::vector<int> > voxels_point_idx_, clusters_point_idx_;					//Idx of all the points in one voxel
		std::vector<std::vector<int> > clusters_voxel_idx_;										//Idx of voxels segmented in one cluster
		
		PCXYZPtr points_cloud_;			//PT Ptr of all points
		PCXYZPtr voxel_centers_cloud_;  //PT Ptr of all center points of voxels

		std::map<uint32_t, uint32_t> point2vox_;
		//grapthPtr mesh_;
    //std::map<int, std::vector<float>> spatial_distances_;
    std::map<int, std::map<int, float>> spatial_distances_;

		//Member functions
		//Features
		std::vector<float>
		calculateEigenFeatures(PCXYZPtr input_cloud)//Eigen features
		{
			//Parameter setting
			Eigen::Vector3f eig_values;
			Eigen::Matrix3f eig_vectors;
			Eigen::Matrix3f *cor_matrix=new Eigen::Matrix3f;
			std::vector<float> output_features;

			int point_num=0;
			point_num=input_cloud->points.size();
			float eig_e1=0,eig_e2=0,eig_e3=0;
			//float *features=new float[8];

			//Weighted corvarance matrix
			//this->calculateWeightedCorvariance(input_cloud,cor_matrix);
			this->calculateCorvariance(input_cloud,cor_matrix);

			//EVD
			pcl::eigen33 (*cor_matrix, eig_vectors, eig_values);

			//Eigen values (normalized)
			if(eig_values[0]==0 && eig_values[1]==0 && eig_values[2]==0)
			{
				for(int i=0;i<8;i++)
				{
					output_features.push_back(float(0));
				}
			}
			else
			{
				eig_e3=(float)eig_values[0]/sqrt(pow(eig_values[0],2)+pow(eig_values[1],2)+pow(eig_values[2],2));
				eig_e2=(float)eig_values[1]/sqrt(pow(eig_values[0],2)+pow(eig_values[1],2)+pow(eig_values[2],2));
				eig_e1=(float)eig_values[2]/sqrt(pow(eig_values[0],2)+pow(eig_values[1],2)+pow(eig_values[2],2));


				//Feature calculation
				if(eig_e1==0)
				{
					output_features.push_back(float(0));//Linearity
					output_features.push_back(float(1));//Planarity
					output_features.push_back(float(0));//Scattering
				}
				else
				{
					output_features.push_back(float(eig_e1-eig_e2)/eig_e1);//Linearity
					output_features.push_back(float(eig_e2-eig_e3)/eig_e1);//Planarity
					output_features.push_back(float(eig_e3)/eig_e1);//Scattering
				}

				output_features.push_back(float(eig_e3)/(eig_e1+eig_e2+eig_e3));//Change of curvature

				if(eig_e2==0)
				{
					output_features.push_back(float(0));//Anisotropy
				}
				else
				{
					output_features.push_back(float(eig_e1-eig_e3)/eig_e1);//Anisotropy
				}

				if(eig_e1*eig_e2*eig_e3==0)
				{
					output_features.push_back(float(0));//Eigenentropy
				}
				else
				{
					output_features.push_back(-1*(eig_e1*log(eig_e1)+eig_e2*log(eig_e2)+eig_e3*log(eig_e3)));//Eigenentropy
				}
				
				output_features.push_back(eig_e1+eig_e2+eig_e3);//Sum of eigen values
				output_features.push_back(pow(float(eig_e1*eig_e2*eig_e3),float(1.0/3)));//Omnivariance
			}

			////Test 
			//std::cout<<"Eigen Features: "<<std::endl;
			//std::cout<<" "<<output_features[0]<<" "<<output_features[1]<<" "<<output_features[2]<<" "<<output_features[3]<<std::endl;
			//std::cout<<" "<<output_features[4]<<" "<<output_features[5]<<" "<<output_features[6]<<" "<<output_features[7]<<std::endl;
			//std::cout<<""<<std::endl;
			
			return(output_features);
		}

		pcl::PointXYZ
		calculateVoxelCentroid(PCXYZPtr input_cloud)
		{
			//Parameter setting
			pcl::PointXYZ output_centroid;
			int point_num=input_cloud->points.size();
			float x_sum=0,y_sum=0,z_sum=0;

			for(int i=0;i<point_num;i++)
			{
				x_sum=x_sum+input_cloud->points[i].x;
				y_sum=y_sum+input_cloud->points[i].y;
				z_sum=z_sum+input_cloud->points[i].z;
			}

			output_centroid.x=x_sum/point_num;
			output_centroid.y=y_sum/point_num;
			output_centroid.z=z_sum/point_num;

			return(output_centroid);
		}

		pcl::Normal
		calculateVoxelNorms(PCXYZPtr input_cloud)
		{
			//Parameter setting
			Eigen::Vector3f eig_values;
			Eigen::Matrix3f eig_vectors;
			Eigen::Matrix3f *cor_matrix=new Eigen::Matrix3f;
			pcl::Normal output_normal;
			pcl::PointXYZ view_point;

			int point_num=input_cloud->points.size();
			//view_point.x=0-input_cloud->points[0].x;
			//view_point.y=0-input_cloud->points[0].y;
			//view_point.z=1.5-input_cloud->points[0].z;
			view_point.x = 0 - input_cloud->points[0].x;
			view_point.y = 0 - input_cloud->points[0].y;
			view_point.z = 1.5 - input_cloud->points[0].z; // TODO hier wird getrickst!
			float eig_e1=0,eig_e2=0,eig_e3=0;

			//Weighted corvarance matrix
			this->calculateCorvariance(input_cloud,cor_matrix);
			//this->calculateWeightedCorvariance(input_cloud,cor_matrix);			
			//EVD
			pcl::eigen33 (*cor_matrix, eig_vectors, eig_values);

			//Eigen values
			eig_e1=eig_values[0];
			eig_e2=eig_values[1];
			eig_e3=eig_values[2];

			//Feature calculation
			output_normal.normal_x=eig_vectors(0,0);
			output_normal.normal_y=eig_vectors(1,0);
			output_normal.normal_z=eig_vectors(2,0);

			//Direction judgement
			if((output_normal.normal_x*view_point.x+output_normal.normal_y*view_point.y+output_normal.normal_z*view_point.z)<0)
			{
				output_normal.normal_x=output_normal.normal_x*-1;
				output_normal.normal_y=output_normal.normal_y*-1;
				output_normal.normal_z=output_normal.normal_z*-1;		
				//printf("cor\n");	
			}

			//Test 
			//std::cout<<"Norms: "<<std::endl;
			//std::cout<<" "<<output_normal.normal_x<<" "<<output_normal.normal_y<<" "<<output_normal.normal_z<<std::endl;
			//std::cout<<""<<std::endl;
			
			return(output_normal);
		}

		void
		setVoxelCentroid(int voxel_id, pcl::PointXYZ voxel_centroid)
		{
			voxel_centroids_.at(voxel_id)=voxel_centroid;
		}

		void
		setVoxelNorms(int voxel_id, pcl::Normal voxel_norm)
		{
			voxel_norms_.at(voxel_id)=voxel_norm;
		}

		void
		setVoxelEigens(int voxel_id, std::vector<float> input_eigens)
		{
			voxel_eigens_[voxel_id].clear();
			voxel_eigens_[voxel_id]=input_eigens;
		}

		void
		initialVoxelAttributes()
		{
			std::vector<float> empty_attribute(1,0);
			pcl::Normal empty_norm;
			pcl::PointXYZ empty_point;
			
			for(int i=0;i<this->voxels_num_;i++)
			{
				voxel_eigens_.push_back(empty_attribute);//Eigens
				voxel_norms_.push_back(empty_norm);//Norm
				voxel_centroids_.push_back(empty_point);//Centroid
			}
		}

		void
		calculateWeightedCorvariance(PCXYZPtr input_cloud, Eigen::Matrix3f *output_cor)
		{
			//Settings 
			pcl::PointXYZ point_can;	// Candidate point
			Eigen::Vector3f key_coor=Eigen::Vector3f::Zero(3,1);	// Coordinates of key point 
			Eigen::Vector3f can_coor=Eigen::Vector3f::Zero(3,1);	// Coordinates of candidate point
			Eigen::Vector3f sum_coor=Eigen::Vector3f::Zero(3,1);	// Sum of coordinates
			Eigen::Vector3f diff_coor=Eigen::Vector3f::Zero(3,1);	// Coordinates difference
			Eigen::Matrix3f cor_single=Eigen::Matrix3f::Zero(3,3);	// CTC for a point
			Eigen::Matrix3f cor_sum=Eigen::Matrix3f::Zero(3,3);		// Sum of all CTC

			std::vector<int> points_id_support; 	// Neighbors within radius search
			std::vector<float> points_dis_support; 	// Distance of these neighbors 

			int num_support=input_cloud->points.size();	//Num of input point
			int num_support_min=3;
			float point_dis=0;
			float sum_dis=0;

			//Tranverse in the support region
			if(num_support>num_support_min)
			{
				for (size_t i = 0; i < num_support; i++)
				{
					point_can=input_cloud->points[i];
					sum_coor[0]=sum_coor[0]+point_can.x;
					sum_coor[1]=sum_coor[1]+point_can.y;
					sum_coor[2]=sum_coor[2]+point_can.z;
				}

				//key point
				key_coor[0]=sum_coor[0]/num_support;
				key_coor[1]=sum_coor[1]/num_support;
				key_coor[2]=sum_coor[2]/num_support;

				for (size_t j = 0; j < num_support; j++)
				{
					//Get candidate point in support
					point_can=input_cloud->points[j]; 
					can_coor[0]=point_can.x;
					can_coor[1]=point_can.y;
					can_coor[2]=point_can.z;

					//Coordinate differences
					diff_coor=can_coor-key_coor;

					//Distance between the candidate and key points
					point_dis=diff_coor.norm();
					sum_dis=sum_dis+point_dis;

					//CTC
					cor_single=point_dis*diff_coor*diff_coor.transpose();
					cor_sum=cor_sum+cor_single;
				}
			}
			else
			{
				sum_dis=1;
				cor_sum=Eigen::Matrix3f::Zero(3,3);
			}

			//Final covariance matrix
			cor_sum=cor_sum/sum_dis;
			*output_cor=cor_sum;

		}

		void
		calculateCorvariance(PCXYZPtr input_cloud, Eigen::Matrix3f *output_cor)
		{
			//Settings 
			pcl::PointXYZ point_can;	// Candidate point
			Eigen::Vector3f key_coor=Eigen::Vector3f::Zero(3,1);	// Coordinates of key point 
			Eigen::Vector3f can_coor=Eigen::Vector3f::Zero(3,1);	// Coordinates of candidate point
			Eigen::Vector3f sum_coor=Eigen::Vector3f::Zero(3,1);	// Sum of coordinates
			Eigen::Vector3f diff_coor=Eigen::Vector3f::Zero(3,1);	// Coordinates difference
			Eigen::Matrix3f cor_single=Eigen::Matrix3f::Zero(3,3);	// CTC for a point
			Eigen::Matrix3f cor_sum=Eigen::Matrix3f::Zero(3,3);		// Sum of all CTC

			std::vector<int> points_id_support; 	// Neighbors within radius search
			std::vector<float> points_dis_support; 	// Distance of these neighbors 

			int num_support=input_cloud->points.size();	//Num of input point
			int num_support_min=3;
			float point_dis=0;
			float sum_dis=0;

			//Tranverse in the support region
			if(num_support>num_support_min)
			{
				for (size_t i = 0; i < num_support; i++)
				{
					point_can=input_cloud->points[i];
					sum_coor[0]=sum_coor[0]+point_can.x;
					sum_coor[1]=sum_coor[1]+point_can.y;
					sum_coor[2]=sum_coor[2]+point_can.z;
				}

				//key point
				key_coor[0]=sum_coor[0]/num_support;
				key_coor[1]=sum_coor[1]/num_support;
				key_coor[2]=sum_coor[2]/num_support;

				for (size_t j = 0; j < num_support; j++)
				{
					//Get candidate point in support
					point_can=input_cloud->points[j]; 
					can_coor[0]=point_can.x;
					can_coor[1]=point_can.y;
					can_coor[2]=point_can.z;

					//Coordinate differences
					diff_coor=can_coor-key_coor;

					//CTC
					cor_single=diff_coor*diff_coor.transpose();
					cor_sum=cor_sum+cor_single;
				}
			}
			else
			{
				sum_dis=1;
				cor_sum=Eigen::Matrix3f::Zero(3,3);
			}

			//Final covariance matrix
			*output_cor=cor_sum;		
		
		}

		//Graph based method
		std::vector<float>
		measuringDistance(std::vector<float> v1_center, std::vector<float> v2_center, std::vector<float> v1_norm, std::vector<float> v2_norm, std::vector<float> v1_eigen, std::vector<float> v2_eigen)
		{
			//Settings
			std::vector<float> similarity_dist;
			float dist_space=100;	//S:space distance
			float dist_angle=100;	//A:norm angle
			float dist_stair=100;	//T:stair distance
			float dist_eigen=100;	//E:eigen feature
			float dist_convx=100;	//C:convex

			float cos_v1_dist=0, cos_v2_dist=0,dist_v1_v2=0, cos_v1_v2=0, cos_d_s=0;
			float dist_cos_dist1=0,dist_cos_dist2=0, thred_singular=0, max_singular=0;
			float dist_v1 = 0, dist_v2 = 0, dist_o1 = 0, dist_o2 = 0;
			double a_1=0, a_2=0, a_1_2=0, a_d_s=0, a_d_s1=0, a_d_s2=0, PI=3.1415926;
			std::vector<float> norm_v1_v2, product_v1_v2, norm_v2_v1;

			//Distance calculation
			if(v1_center.size()>1 && v2_center.size()>1) //space: euclidean distance
			{
				dist_v1_v2=sqrt(pow(v1_center[0]-v2_center[0],2)+pow(v1_center[1]-v2_center[1],2)+pow(v1_center[2]-v2_center[2],2));

				dist_space=dist_v1_v2;
				if(dist_v1_v2!=0)
				{
					norm_v1_v2.clear();
					norm_v1_v2.push_back((v1_center[0]-v2_center[0])/dist_v1_v2);
					norm_v1_v2.push_back((v1_center[1]-v2_center[1])/dist_v1_v2); 
					norm_v1_v2.push_back((v1_center[2]-v2_center[2])/dist_v1_v2);

					norm_v2_v1.clear();
					norm_v2_v1.push_back((v2_center[0]-v1_center[0])/dist_v1_v2);
					norm_v2_v1.push_back((v2_center[1]-v1_center[1])/dist_v1_v2);
					norm_v2_v1.push_back((v2_center[2]-v1_center[2])/dist_v1_v2);

					product_v1_v2.clear();
					product_v1_v2.push_back(v1_center[1]*v2_center[2]-v1_center[2]*v2_center[1]);
					product_v1_v2.push_back(v1_center[2]*v2_center[0]-v1_center[0]*v2_center[2]);
					product_v1_v2.push_back(v1_center[0]*v2_center[1]-v1_center[1]*v2_center[0]);
				}
			}
			
			if(v1_norm.size()>1 && v2_norm.size()>1)   //norm: cos angle
			{

				if(dist_space!=0)
				{
					// cos(angle)
					cos_v1_v2=( v1_norm[0]*v2_norm[0]+v1_norm[1]*v2_norm[1]+v1_norm[2]*v2_norm[2]);
					cos_v1_dist=( v1_norm[0]*norm_v1_v2[0]+v1_norm[1]*norm_v1_v2[1]+v1_norm[2]*norm_v1_v2[2]);
					cos_v2_dist=( v2_norm[0]*norm_v1_v2[0]+v2_norm[1]*norm_v1_v2[1]+v2_norm[2]*norm_v1_v2[2]);
					
					cos_d_s=( product_v1_v2[0]*norm_v1_v2[0]+product_v1_v2[1]*norm_v1_v2[1]+product_v1_v2[2]*norm_v1_v2[2]);
					
					a_1=acos(cos_v1_dist);
					a_2=acos(cos_v2_dist);
					a_1_2=acos(cos_v1_v2);

					a_d_s1=acos(cos_d_s);
					a_d_s2=PI-a_d_s1;

					// angle = acos(cos(angle))
					dist_angle = acos(cos_v1_v2);// norm angle

					dist_v1 = v1_norm[0] * v1_center[0] + v1_norm[1] * v1_center[1] + v1_norm[2] * v1_center[2];
					dist_v2 = v2_norm[0] * v2_center[0] + v2_norm[1] * v2_center[1] + v2_norm[2] * v2_center[2];
					dist_o1 = v1_norm[0] * v2_center[0] + v1_norm[1] * v2_center[1] + v1_norm[2] * v2_center[2];
					dist_o2 = v2_norm[0] * v1_center[0] + v2_norm[1] * v1_center[1] + v2_norm[2] * v1_center[2];
					if (dist_v1_v2 != 0) //stair-like
					{
						dist_stair = sqrt(pow((dist_o1 - dist_v1), 2) + pow((dist_o2 - dist_v2), 2));
					}
					else
					{
						dist_stair = 0;
					}
				}

				//Singular judgement
				double temp_a=0.5, temp_off=PI/6, 
				max_singular=PI/2;
				thred_singular=(float)max_singular/(1+exp(-1*temp_a*(a_1_2-temp_off)));

				a_d_s=a_d_s1;
				if(a_d_s1>a_d_s2)
				{
					a_d_s=a_d_s2;
				}

				if(a_d_s>thred_singular) //closure: convexity 
				{
					dist_convx=abs(a_1-a_2);	
				}
				else
				{
					dist_convx = PI;
				}

			}

			if(v1_eigen.size()>1 && v2_eigen.size()>1)  //eigen: Pearson cross correlation
			{
				float eigen_cos=0, eigen_abs1=0, eigen_abs2=0; 

				for(int i=4;i<v1_eigen.size();i++)
				{
					eigen_cos=eigen_cos + v1_eigen[i] * v2_eigen[i];
					eigen_abs1=eigen_abs1 + v1_eigen[i] * v1_eigen[i];
					eigen_abs2=eigen_abs2 + v2_eigen[i] * v2_eigen[i];
				}

				if(eigen_abs1!=0 && eigen_abs2!=0)
				{
					dist_eigen=1.0f - eigen_cos / (sqrt(eigen_abs1)*sqrt(eigen_abs2));
				}
			}
			
			//printf("%f, %f, %f, %f, %f\n", (double) dist_space, (double) dist_angle, (double) dist_stair, (double) dist_eigen, (double) dist_convx);
			//Output similarity distances
			similarity_dist.push_back(dist_space);	//S
			similarity_dist.push_back(dist_angle);	//A
			similarity_dist.push_back(dist_stair);	//T
			similarity_dist.push_back(dist_eigen);	//E
			similarity_dist.push_back(dist_convx);	//C

			return(similarity_dist);		
		}


		int findClosestPoint(pcl::PointXYZ center, std::vector<int> points_idxs, PCXYZPtr input_cloud){
			float min_dist = std::numeric_limits<float>::max();;
			int idx = points_idxs[0];
			for(int i=0; i<points_idxs.size(); i++){
				int p_idx = points_idxs[i];
				pcl::PointXYZ p = input_cloud->points[p_idx];
				float diff_x = center.x - p.x;
				float diff_y = center.y - p.y;
				float diff_z = center.z - p.z;
				float dist = sqrt(diff_x*diff_x + diff_y*diff_y + diff_z*diff_z);
				if (dist < min_dist){
					min_dist = dist;
					idx = p_idx;
				}
			}
			return idx;
		}

		/*
		//Graph based method
		std::vector<float>
		measuringDistanceGeo(PCXYZPtr input_cloud, int v_idx_i, int v_idx_j, std::vector<float> v1_center, std::vector<float> v2_center, std::vector<float> v1_norm, std::vector<float> v2_norm, std::vector<float> v1_eigen, std::vector<float> v2_eigen)
		{
			//Settings
			std::vector<float> similarity_dist;
			float dist_space=100;	//S:space distance
			float dist_angle=100;	//A:norm angle
			float dist_stair=100;	//T:stair distance
			float dist_eigen=100;	//E:eigen feature
			float dist_convx=100;	//C:convex


			float cos_v1_dist=0, cos_v2_dist=0,dist_v1_v2=0, cos_v1_v2=0, cos_d_s=0;
			float dist_cos_dist1=0,dist_cos_dist2=0, thred_singular=0, max_singular=0;
			float dist_v1 = 0, dist_v2 = 0, dist_o1 = 0, dist_o2 = 0;
			double a_1=0, a_2=0, a_1_2=0, a_d_s=0, a_d_s1=0, a_d_s2=0, PI=3.1415926;
			std::vector<float> norm_v1_v2, product_v1_v2, norm_v2_v1;

			//Distance calculation
			if(v1_center.size()>1 && v2_center.size()>1) //space: euclidean distance
			{
				//printf("Find closest i\n");
				std::vector<int> i_points = voxels_point_idx_[v_idx_i];
				pcl::PointXYZ vc_i = voxel_centers_[v_idx_i];
				int close_i = findClosestPoint(vc_i, i_points, input_cloud);
				//printf("Found %i\n", close_i);

				//printf("Find closest j\n");
				std::vector<int> j_points = voxels_point_idx_[v_idx_j];
				pcl::PointXYZ vc_j = voxel_centers_[v_idx_j];
				int close_j = findClosestPoint(vc_j, j_points, input_cloud);
				//printf("Found %i\n", close_j);

	      bool contains = false;
	      if(spatial_distances_.count(close_i))
	      {
	      		//printf("Contains i\n");
	          std::vector<float> dists = spatial_distances_[close_i];
	          dist_v1_v2 = dists[close_j];
	          contains = true;
	      }
	      if(spatial_distances_.count(close_j) && !contains)
	      {
	      		//printf("Contains j\n");
	          std::vector<float> dists = spatial_distances_[close_j];
	          dist_v1_v2 = dists[close_i];
	          contains = true;
	      }
	      if(!contains){
	      		//printf("Calc paths\n");
			      std::vector<vertex_descriptor> p(boost::num_vertices(*mesh_));
			      std::vector<float> d(boost::num_vertices(*mesh_));
			      vertex_descriptor s = boost::vertex(close_i, *mesh_);

			      dijkstra_shortest_paths(*mesh_, s,
			          boost::predecessor_map(boost::make_iterator_property_map(p.begin(), get(boost::vertex_index, *mesh_))).
			          distance_map(boost::make_iterator_property_map(d.begin(), get(boost::vertex_index, *mesh_))));
	          //printf("done\n");
	          dist_v1_v2 = d[close_j];

	          //spatial_distances_[close_i] = d;
	          //spatial_distances_.emplace(close_i, d);
	          spatial_distances_.emplace(close_i, d);
	      }

				dist_space = dist_v1_v2;

				if(dist_v1_v2 != 0)
				{
					norm_v1_v2.clear();
					norm_v1_v2.push_back((v1_center[0]-v2_center[0])/dist_v1_v2);
					norm_v1_v2.push_back((v1_center[1]-v2_center[1])/dist_v1_v2); 
					norm_v1_v2.push_back((v1_center[2]-v2_center[2])/dist_v1_v2);

					norm_v2_v1.clear();
					norm_v2_v1.push_back((v2_center[0]-v1_center[0])/dist_v1_v2);
					norm_v2_v1.push_back((v2_center[1]-v1_center[1])/dist_v1_v2);
					norm_v2_v1.push_back((v2_center[2]-v1_center[2])/dist_v1_v2);

					product_v1_v2.clear();
					product_v1_v2.push_back(v1_center[1]*v2_center[2]-v1_center[2]*v2_center[1]);
					product_v1_v2.push_back(v1_center[2]*v2_center[0]-v1_center[0]*v2_center[2]);
					product_v1_v2.push_back(v1_center[0]*v2_center[1]-v1_center[1]*v2_center[0]);
				}
			}
			
			if(v1_norm.size()>1 && v2_norm.size()>1)   //norm: cos angle
			{
				if(dist_space!=0)
				{
					cos_v1_v2=( v1_norm[0]*v2_norm[0]+v1_norm[1]*v2_norm[1]+v1_norm[2]*v2_norm[2]);
					cos_v1_dist=( v1_norm[0]*norm_v1_v2[0]+v1_norm[1]*norm_v1_v2[1]+v1_norm[2]*norm_v1_v2[2]);
					cos_v2_dist=( v2_norm[0]*norm_v1_v2[0]+v2_norm[1]*norm_v1_v2[1]+v2_norm[2]*norm_v1_v2[2]);
					
					cos_d_s=( product_v1_v2[0]*norm_v1_v2[0]+product_v1_v2[1]*norm_v1_v2[1]+product_v1_v2[2]*norm_v1_v2[2]);
					
					a_1=acos(cos_v1_dist);
					a_2=acos(cos_v2_dist);
					a_1_2=acos(cos_v1_v2);

					a_d_s1=acos(cos_d_s);
					a_d_s2=PI-a_d_s1;

					dist_angle = acos(cos_v1_v2);// norm angle

					dist_v1 = v1_norm[0] * v1_center[0] + v1_norm[1] * v1_center[1] + v1_norm[2] * v1_center[2];
					dist_v2 = v2_norm[0] * v2_center[0] + v2_norm[1] * v2_center[1] + v2_norm[2] * v2_center[2];
					dist_o1 = v1_norm[0] * v2_center[0] + v1_norm[1] * v2_center[1] + v1_norm[2] * v2_center[2];
					dist_o2 = v2_norm[0] * v1_center[0] + v2_norm[1] * v1_center[1] + v2_norm[2] * v1_center[2];
					if (dist_v1_v2 != 0) //stair-like
					{
						dist_stair = sqrt(pow((dist_o1 - dist_v1), 2) + pow((dist_o2 - dist_v2), 2));
					}
					else
					{
						dist_stair = 0;
					}
				}

				//Singular judgement
				double temp_a=0.5, temp_off=PI/6, 
				max_singular=PI/2;
				thred_singular=(float)max_singular/(1+exp(-1*temp_a*(a_1_2-temp_off)));

				a_d_s=a_d_s1;
				if(a_d_s1>a_d_s2)
				{
					a_d_s=a_d_s2;
				}

				if(a_d_s>thred_singular) //closure: convexity 
				{
					dist_convx=abs(a_1-a_2);	
				}
				else
				{
					dist_convx = PI;
				}

			}

			if(v1_eigen.size()>1 && v2_eigen.size()>1)  //eigen: Pearson cross correlation
			{
				float eigen_cos=0, eigen_abs1=0, eigen_abs2=0; 

				for(int i=4;i<v1_eigen.size();i++)
				{
					eigen_cos=eigen_cos + v1_eigen[i] * v2_eigen[i];
					eigen_abs1=eigen_abs1 + v1_eigen[i] * v1_eigen[i];
					eigen_abs2=eigen_abs2 + v2_eigen[i] * v2_eigen[i];
				}

				if(eigen_abs1!=0 && eigen_abs2!=0)
				{
					dist_eigen=1.0f - eigen_cos / (sqrt(eigen_abs1)*sqrt(eigen_abs2));
				}
			}

			//printf("%f, %f, %f, %f, %f\n", (double) dist_space, (double) dist_angle, (double) dist_stair, (double) dist_eigen, (double) dist_convx);
			//Output similarity distances
			similarity_dist.push_back(dist_space);	//S
			similarity_dist.push_back(dist_angle);	//A
			similarity_dist.push_back(dist_stair);	//T
			similarity_dist.push_back(dist_eigen);	//E
			similarity_dist.push_back(dist_convx);	//C

			return(similarity_dist);		
		}
		*/


		//Graph based method
		std::vector<float>
		measuringDistanceGeo(int v_idx_i, int v_idx_j, std::vector<float> v1_center, std::vector<float> v2_center, std::vector<float> v1_norm, std::vector<float> v2_norm, std::vector<float> v1_eigen, std::vector<float> v2_eigen)
		{
			//Settings
			std::vector<float> similarity_dist;
			float dist_space=100;	//S:space distance
			float dist_angle=100;	//A:norm angle
			float dist_stair=100;	//T:stair distance
			float dist_eigen=100;	//E:eigen feature
			float dist_convx=100;	//C:convex


			float cos_v1_dist=0, cos_v2_dist=0,dist_v1_v2=0, cos_v1_v2=0, cos_d_s=0;
			float dist_cos_dist1=0,dist_cos_dist2=0, thred_singular=0, max_singular=0;
			float dist_v1 = 0, dist_v2 = 0, dist_o1 = 0, dist_o2 = 0;
			double a_1=0, a_2=0, a_1_2=0, a_d_s=0, a_d_s1=0, a_d_s2=0, PI=3.1415926;
			std::vector<float> norm_v1_v2, product_v1_v2, norm_v2_v1;

			//Distance calculation
			if(v1_center.size()>1 && v2_center.size()>1) //space: euclidean distance
			{
				dist_v1_v2 = spatial_distances_[v_idx_i][v_idx_j];
				dist_space = dist_v1_v2;

				if(dist_v1_v2 != 0)
				{
					norm_v1_v2.clear();
					norm_v1_v2.push_back((v1_center[0]-v2_center[0])/dist_v1_v2);
					norm_v1_v2.push_back((v1_center[1]-v2_center[1])/dist_v1_v2); 
					norm_v1_v2.push_back((v1_center[2]-v2_center[2])/dist_v1_v2);

					norm_v2_v1.clear();
					norm_v2_v1.push_back((v2_center[0]-v1_center[0])/dist_v1_v2);
					norm_v2_v1.push_back((v2_center[1]-v1_center[1])/dist_v1_v2);
					norm_v2_v1.push_back((v2_center[2]-v1_center[2])/dist_v1_v2);

					product_v1_v2.clear();
					product_v1_v2.push_back(v1_center[1]*v2_center[2]-v1_center[2]*v2_center[1]);
					product_v1_v2.push_back(v1_center[2]*v2_center[0]-v1_center[0]*v2_center[2]);
					product_v1_v2.push_back(v1_center[0]*v2_center[1]-v1_center[1]*v2_center[0]);
				}
			}
			
			if(v1_norm.size()>1 && v2_norm.size()>1)   //norm: cos angle
			{
				if(dist_space!=0)
				{
					cos_v1_v2=( v1_norm[0]*v2_norm[0]+v1_norm[1]*v2_norm[1]+v1_norm[2]*v2_norm[2]);
					cos_v1_dist=( v1_norm[0]*norm_v1_v2[0]+v1_norm[1]*norm_v1_v2[1]+v1_norm[2]*norm_v1_v2[2]);
					cos_v2_dist=( v2_norm[0]*norm_v1_v2[0]+v2_norm[1]*norm_v1_v2[1]+v2_norm[2]*norm_v1_v2[2]);
					
					cos_d_s=( product_v1_v2[0]*norm_v1_v2[0]+product_v1_v2[1]*norm_v1_v2[1]+product_v1_v2[2]*norm_v1_v2[2]);
					
					a_1=acos(cos_v1_dist);
					a_2=acos(cos_v2_dist);
					a_1_2=acos(cos_v1_v2);

					a_d_s1=acos(cos_d_s);
					a_d_s2=PI-a_d_s1;

					dist_angle = acos(cos_v1_v2);// norm angle

					dist_v1 = v1_norm[0] * v1_center[0] + v1_norm[1] * v1_center[1] + v1_norm[2] * v1_center[2];
					dist_v2 = v2_norm[0] * v2_center[0] + v2_norm[1] * v2_center[1] + v2_norm[2] * v2_center[2];
					dist_o1 = v1_norm[0] * v2_center[0] + v1_norm[1] * v2_center[1] + v1_norm[2] * v2_center[2];
					dist_o2 = v2_norm[0] * v1_center[0] + v2_norm[1] * v1_center[1] + v2_norm[2] * v1_center[2];
					if (dist_v1_v2 != 0) //stair-like
					{
						dist_stair = sqrt(pow((dist_o1 - dist_v1), 2) + pow((dist_o2 - dist_v2), 2));
					}
					else
					{
						dist_stair = 0;
					}
				}

				//Singular judgement
				double temp_a=0.5, temp_off=PI/6, 
				max_singular=PI/2;
				thred_singular=(float)max_singular/(1+exp(-1*temp_a*(a_1_2-temp_off)));

				a_d_s=a_d_s1;
				if(a_d_s1>a_d_s2)
				{
					a_d_s=a_d_s2;
				}

				if(a_d_s>thred_singular) //closure: convexity 
				{
					dist_convx=abs(a_1-a_2);	
				}
				else
				{
					dist_convx = PI;
				}

			}

			if(v1_eigen.size()>1 && v2_eigen.size()>1)  //eigen: Pearson cross correlation
			{
				float eigen_cos=0, eigen_abs1=0, eigen_abs2=0; 

				for(int i=4;i<v1_eigen.size();i++)
				{
					eigen_cos=eigen_cos + v1_eigen[i] * v2_eigen[i];
					eigen_abs1=eigen_abs1 + v1_eigen[i] * v1_eigen[i];
					eigen_abs2=eigen_abs2 + v2_eigen[i] * v2_eigen[i];
				}

				if(eigen_abs1!=0 && eigen_abs2!=0)
				{
					dist_eigen=1.0f - eigen_cos / (sqrt(eigen_abs1)*sqrt(eigen_abs2));
				}
			}

			//printf("%f, %f, %f, %f, %f\n", (double) dist_space, (double) dist_angle, (double) dist_stair, (double) dist_eigen, (double) dist_convx);
			//Output similarity distances
			similarity_dist.push_back(dist_space);	//S
			similarity_dist.push_back(dist_angle);	//A
			similarity_dist.push_back(dist_stair);	//T
			similarity_dist.push_back(dist_eigen);	//E
			similarity_dist.push_back(dist_convx);	//C

			return(similarity_dist);		
		}

		
		float
		distanceWeight(std::vector<float> dist_all, float sig_p, float sig_n, float sig_o, float sig_e, float sig_c, float sig_w)
		{
			//Parameters and settings
			float dist_s, dist_c, dist_a, dist_t, dist_e;
			float similarity_weight = 0, similarity_dist = 0;

			dist_s = dist_all[0]; //Space
			dist_a = dist_all[1]; //Angle
			dist_t = dist_all[2]; //sTair
			dist_e = dist_all[3]; //Eigen
			dist_c = dist_all[4]; //Convex

			//Distance calculation			
			similarity_dist=sqrt(pow(dist_s/sig_p,2)+ pow(dist_a / sig_n, 2)+ pow(dist_t / sig_o, 2)+pow(dist_c/sig_c,2)+pow(dist_e/sig_e,2));
			similarity_weight=exp(-0.5*similarity_dist/pow(sig_w,2));
			//printf("%f\n", (double)similarity_weight);

			return(similarity_weight);		
		}

		//Affinity graph
		std::vector<std::vector<float> >
		voxelAttributes(int v_idx)
		{
			//Parameters and settings
			std::vector<std::vector<float> > v_attributes;
			std::vector<float> v_position, v_normal, v_eigen; //Attributes of supervoxels			
			pcl::PointXYZ temp_center, empty_center;
			pcl::Normal temp_norm, empty_norm;

			empty_center.x=0;empty_center.y=0;empty_center.z=0;
			empty_norm.normal_x=0;empty_norm.normal_y=0;empty_norm.normal_z=0;

			temp_center=this->voxel_centroids_[v_idx];
			temp_norm=this->voxel_norms_[v_idx];

			if(temp_center.x!=empty_center.x && temp_center.y!=empty_center.y && temp_center.z!=empty_center.z)
			{		
				v_position.push_back(temp_center.x);
				v_position.push_back(temp_center.y);
				v_position.push_back(temp_center.z);
			}
			else
			{
				v_position.push_back(0);
			}

			if(temp_norm.normal_x!=empty_norm.normal_x && temp_norm.normal_y!=empty_norm.normal_y && temp_norm.normal_z!=empty_norm.normal_z)
			{
				v_normal.push_back(temp_norm.normal_x);
				v_normal.push_back(temp_norm.normal_y);
				v_normal.push_back(temp_norm.normal_z);
			}
			else
			{
				v_normal.push_back(0);
			}
			
			v_eigen=this->voxel_eigens_[v_idx];

			//Assigning
			v_attributes.push_back(v_position);
			v_attributes.push_back(v_normal);
			v_attributes.push_back(v_eigen);
			return(v_attributes);
		}

		Eigen::MatrixXf
		buildAdjacencyGraph(std::vector<int> voxels_idx, float sig_p, float sig_n, float sig_o, float sig_e, float sig_c, float sig_w)		//Building simple adjacency Graph: static graph only connectivity considered
		{
			//Parameters and settings
			Eigen::MatrixXf adj_matrix; //Adjacency in a n*n*n cube
			std::vector<float> voxel1_position, voxel1_normal, voxel1_eigen;
			std::vector<float> voxel2_position, voxel2_normal, voxel2_eigen;
			pcl::PointXYZ temp_center, empty_center;
			pcl::Normal temp_norm, empty_norm;
			
			std::vector<float> dist_all;
			float dist_s, dist_e, dist_n, dist_a, dist_r; //For the last three, convex, normal angles, stair-case

			double similarity_voxels=0;
			int adjacency_num=0;
			empty_center.x=0;empty_center.y=0;empty_center.z=0;
			empty_norm.normal_x=0;empty_norm.normal_y=0;empty_norm.normal_z=0;

			//Find idx of adjacent voxels
			adjacency_num=voxels_idx.size();

			//Initialization
			adj_matrix=Eigen::MatrixXf::Zero(adjacency_num,adjacency_num); 

			//Assigining
			for (int i=0;i<adjacency_num;i++)
			{
				voxel1_position.clear();
				voxel1_normal.clear();
				voxel1_eigen.clear();
				temp_center=voxel_centroids_[voxels_idx[i]];
				temp_norm=voxel_norms_[voxels_idx[i]];

				if(temp_center.x!=empty_center.x && temp_center.y!=empty_center.y && temp_center.z!=empty_center.z)
				{		
					voxel1_position.push_back(temp_center.x);
					voxel1_position.push_back(temp_center.y);
					voxel1_position.push_back(temp_center.z);
				}
				else
				{
					voxel1_position.push_back(0);
				}

				//printf("%f, %f, %f\n", empty_norm.normal_x, empty_norm.normal_y, empty_norm.normal_z);
				// if every element of the normal vector is unequal to zero
				if(temp_norm.normal_x!=empty_norm.normal_x && temp_norm.normal_y!=empty_norm.normal_y && temp_norm.normal_z!=empty_norm.normal_z)
				{
					voxel1_normal.push_back(temp_norm.normal_x);
					voxel1_normal.push_back(temp_norm.normal_y);
					voxel1_normal.push_back(temp_norm.normal_z);
					//printf("%f, %f, %f\n", temp_norm.normal_x, temp_norm.normal_y, temp_norm.normal_z);
				}
				else
				{
					voxel1_normal.push_back(0);
					//printf("empty\n");
				}

				voxel1_eigen=voxel_eigens_[voxels_idx[i]];

				for(int j=0;j<adjacency_num;j++)
				{
					voxel2_position.clear();
					voxel2_normal.clear();
					voxel2_eigen.clear();
					temp_center=voxel_centroids_[voxels_idx[j]];
					temp_norm=voxel_norms_[voxels_idx[j]];

					if(temp_center.x!=empty_center.x && temp_center.y!=empty_center.y && temp_center.z!=empty_center.z)
					{		
						voxel2_position.push_back(temp_center.x);
						voxel2_position.push_back(temp_center.y);
						voxel2_position.push_back(temp_center.z);
					}
					else
					{
						voxel2_position.push_back(0);
					}

					if(temp_norm.normal_x!=empty_norm.normal_x && temp_norm.normal_y!=empty_norm.normal_y && temp_norm.normal_z!=empty_norm.normal_z)
					{
						voxel2_normal.push_back(temp_norm.normal_x);
						voxel2_normal.push_back(temp_norm.normal_y);
						voxel2_normal.push_back(temp_norm.normal_z);
					}
					else
					{
						voxel2_normal.push_back(0);
					}

					voxel2_eigen=voxel_eigens_[voxels_idx[j]];

					//Similarity measuring
					if(i!=j)
					{
						dist_all=measuringDistance(voxel1_position, voxel2_position, voxel1_normal, voxel2_normal, voxel1_eigen, voxel2_eigen);
						similarity_voxels=distanceWeight(dist_all, sig_p, sig_n, sig_o, sig_e, sig_c, sig_w);
						//similarity_voxels=distanceProbability(dist_all, sig_p, sig_n, sig_o, sig_e, sig_c, sig_w);

						adj_matrix(i,j)=similarity_voxels;
						//printf("%f\n", (double)similarity_voxels);
					}
					else
					{
						adj_matrix(i,j)=1;
					}

					//Test
					//std::cout<< adj_matrix(i,j)<<"  ";
				}
			
				//Test
				//std::cout<<""<<std::endl;
			}

			//Test
			//std::cout<<""<<std::endl;
			return(adj_matrix);
		}


		Eigen::MatrixXf
		buildAdjacencyGraphGeo(std::vector<int> voxels_idx, float sig_p, float sig_n, float sig_o, float sig_e, float sig_c, float sig_w)		//Building simple adjacency Graph: static graph only connectivity considered
		{
			//Parameters and settings
			Eigen::MatrixXf adj_matrix; //Adjacency in a n*n*n cube
			std::vector<float> voxel1_position, voxel1_normal, voxel1_eigen;
			std::vector<float> voxel2_position, voxel2_normal, voxel2_eigen;
			pcl::PointXYZ temp_center, empty_center;
			pcl::Normal temp_norm, empty_norm;
			
			std::vector<float> dist_all;
			float dist_s, dist_e, dist_n, dist_a, dist_r; //For the last three, convex, normal angles, stair-case

			double similarity_voxels=0;
			int adjacency_num=0;
			empty_center.x=0;empty_center.y=0;empty_center.z=0;
			empty_norm.normal_x=0;empty_norm.normal_y=0;empty_norm.normal_z=0;

			//Find idx of adjacent voxels
			adjacency_num=voxels_idx.size();

			//Initialization
			adj_matrix=Eigen::MatrixXf::Zero(adjacency_num,adjacency_num); 

			//Assigining
			for (int i=0;i<adjacency_num;i++)
			{
				voxel1_position.clear();
				voxel1_normal.clear();
				voxel1_eigen.clear();
				temp_center=voxel_centroids_[voxels_idx[i]];
				temp_norm=voxel_norms_[voxels_idx[i]];

				if(temp_center.x!=empty_center.x && temp_center.y!=empty_center.y && temp_center.z!=empty_center.z)
				{		
					voxel1_position.push_back(temp_center.x);
					voxel1_position.push_back(temp_center.y);
					voxel1_position.push_back(temp_center.z);
				}
				else
				{
					voxel1_position.push_back(0);
				}

				if(temp_norm.normal_x!=empty_norm.normal_x && temp_norm.normal_y!=empty_norm.normal_y && temp_norm.normal_z!=empty_norm.normal_z)
				{
					voxel1_normal.push_back(temp_norm.normal_x);
					voxel1_normal.push_back(temp_norm.normal_y);
					voxel1_normal.push_back(temp_norm.normal_z);
					//printf("%f, %f, %f\n", temp_norm.normal_x, temp_norm.normal_y, temp_norm.normal_z);
				}
				else
				{
					voxel1_normal.push_back(0);
					//printf("empty\n");
				}

				voxel1_eigen=voxel_eigens_[voxels_idx[i]];

				for(int j=0;j<adjacency_num;j++)
				{
					voxel2_position.clear();
					voxel2_normal.clear();
					voxel2_eigen.clear();
					temp_center=voxel_centroids_[voxels_idx[j]];
					temp_norm=voxel_norms_[voxels_idx[j]];

					if(temp_center.x!=empty_center.x && temp_center.y!=empty_center.y && temp_center.z!=empty_center.z)
					{		
						voxel2_position.push_back(temp_center.x);
						voxel2_position.push_back(temp_center.y);
						voxel2_position.push_back(temp_center.z);
					}
					else
					{
						voxel2_position.push_back(0);
					}

					if(temp_norm.normal_x!=empty_norm.normal_x && temp_norm.normal_y!=empty_norm.normal_y && temp_norm.normal_z!=empty_norm.normal_z)
					{
						voxel2_normal.push_back(temp_norm.normal_x);
						voxel2_normal.push_back(temp_norm.normal_y);
						voxel2_normal.push_back(temp_norm.normal_z);
					}
					else
					{
						voxel2_normal.push_back(0);
					}

					voxel2_eigen=voxel_eigens_[voxels_idx[j]];

					//Similarity measuring
					if(i!=j)
					{
						//printf("Similarity Measuring\n");
						//dist_all=measuringDistanceGeo(points_cloud_, voxels_idx[i], voxels_idx[j], voxel1_position, voxel2_position, voxel1_normal, voxel2_normal, voxel1_eigen, voxel2_eigen);
						dist_all=measuringDistanceGeo(voxels_idx[i], voxels_idx[j], voxel1_position, voxel2_position, voxel1_normal, voxel2_normal, voxel1_eigen, voxel2_eigen);
						similarity_voxels=distanceWeight(dist_all, sig_p, sig_n, sig_o, sig_e, sig_c, sig_w);
						//similarity_voxels=distanceProbability(dist_all, sig_p, sig_n, sig_o, sig_e, sig_c, sig_w);

						adj_matrix(i,j)=similarity_voxels;
						//printf("%f\n", (double)similarity_voxels);
					}
					else
					{
						adj_matrix(i,j)=1;					
					}

					//Test
					//std::cout<< adj_matrix(i,j)<<"  ";
				}
			
				//Test
				//std::cout<<""<<std::endl;
			}

			//Test
			//std::cout<<"clear dists"<<std::endl;
			//spatial_distances_.clear();
			return(adj_matrix);
		}


		//Graph segmentation
		std::vector<int>
		cutGraphSegmentation(float thrd_value, Eigen::MatrixXf graph_matrix, std::vector<int> voxel_adjidx)       //Simple test, cut the graph with a given threshold
		{
			//Parameters and settings
			int graph_size = 0, weight_size = 0;
			float initial_thrd = 1;
			graph_size = graph_matrix.rows();
			weight_size = graph_size*graph_size;
			std::vector<int> connected_voxels;
			graph_matrix.resize(1, weight_size);

			// Sort, descending using sort function
			std::vector<Weight_Index> weight_index_array;
			Weight_Index temp;
			for (int i = 0; i < weight_size; i++)
			{
				temp.Weight = graph_matrix(0, i);
				temp.Index = i;
				weight_index_array.push_back(temp);
			}
			sort(weight_index_array.begin(), weight_index_array.end(), godown); // riseup

			//Merging
			std::vector<float> seg_int;
			std::vector< std::vector<int> > seg_ver_idx;
			std::vector<int> seg_size, ver_seg_idx, vec_temp;
			for (int i = 0; i < graph_size; i++)
			{
				vec_temp.clear();//Temp
				vec_temp.push_back(i);

				seg_ver_idx.push_back(vec_temp);	// segment has which vertice, if the vertice has been merged, than empty
				ver_seg_idx.push_back(i);			// vertice belong to which segment
				seg_size.push_back(1);				// How many vertex in the segment
				seg_int.push_back(initial_thrd);	// Initial of mint
			}

			// 边权值从大到小进行合并...
			// seg_ver_idx [面号]: 当前面包含的所有点
			// ver_seg_idx [点号]: 当前点所属的面片
			// seg_size [面号]: 当前面具有点的数量
			// seg_int [面号]: 当前面的异质性测度(初始值是 1)
			for (int i = 0; i < weight_size; i++)
			{
				float w_temp, thred_temp;
				int v1_idx, v2_idx, mint_idx, mint_not_idx;
				v1_idx = floor(double(weight_index_array[i].Index / graph_size));
				v2_idx = weight_index_array[i].Index - v1_idx*graph_size;
				w_temp = weight_index_array[i].Weight;

				if (ver_seg_idx[v1_idx] != ver_seg_idx[v2_idx])
				{
					float v1_mint, v2_mint;
					int v1_size, v2_size;
					v1_size = seg_size[ver_seg_idx[v1_idx]];
					v2_size = seg_size[ver_seg_idx[v2_idx]];
					v1_mint = seg_int[ver_seg_idx[v1_idx]] - thrd_value / v1_size;
					v2_mint = seg_int[ver_seg_idx[v2_idx]] - thrd_value / v2_size;

					if (v1_mint >= v2_mint)
					{
						mint_idx = ver_seg_idx[v1_idx];
						mint_not_idx = ver_seg_idx[v2_idx];
						thred_temp = v1_mint;//seg_mint[ver_seg_idx[v1_idx]];
					}
					else
					{
						mint_idx = ver_seg_idx[v2_idx];
						mint_not_idx = ver_seg_idx[v1_idx];
						thred_temp = v2_mint;//seg_mint[ver_seg_idx[v2_idx]];
					}
					//printf("%f, %f\n", (double) thred_temp, (double)w_temp);
					if (w_temp > thred_temp)//Judgement, merge or not
					{
						seg_int[mint_idx] = w_temp;

						for (int j = 0; j < seg_size[mint_not_idx]; j++)
						{
							seg_ver_idx[mint_idx].push_back(seg_ver_idx[mint_not_idx][j]);  //Rearrange vertex of merged one
							ver_seg_idx[seg_ver_idx[mint_not_idx][j]] = mint_idx;			// Remark the vertex of merged one					
						}

						seg_size[mint_idx] = seg_size[mint_idx] + seg_size[mint_not_idx]; //Change sizes
						seg_size[mint_not_idx] = 0;

						seg_ver_idx[mint_not_idx].clear();
					}
				}
			}

			//Select
			int seg_con = 0;
			for (int i = 0; i<graph_size; i++)//Start from 1, because zero is itself
			{
				if (seg_ver_idx[i].size() != 0)
				{
					for (int j = 0; j<seg_ver_idx[i].size(); j++)
					{
						// 令 seg_ver_idx[i][j]==0 的意思是找到当前图模型中含有中心体素的图模型
						// 因为实际情况下, 该图模型可能含有多个不相连的图模型, 其中一般最大的那个含有目标体素...
						if (seg_ver_idx[i][j] == 0)
						{
							seg_con = i;
							// 感觉找到了就可以不同继续循环了....
							break;
						}
					}
				}
			}

			for (int k = 0; k<seg_ver_idx[seg_con].size(); k++)
			{
				connected_voxels.push_back(voxel_adjidx[seg_ver_idx[seg_con][k]]);
			}

			return(connected_voxels);
		}

		//Clustering
		void
		recursionSearch(std::vector<int> connects_idx, std::vector<int> *inthis_cluster_idx )//recursion search for finding voxels belonging to one cluster
		{
			int connect_id;

			for(int i=0;i<connects_idx.size();i++)// loop + recrusion
			{

				if(!voxel_clustered_[connects_idx[i]])
				{	
				
					inthis_cluster_idx->push_back(connects_idx[i]);		//Record the idx of this voxel
					
					voxel_clustered_[connects_idx[i]]=true;				//change the status
					//recursion
					this->recursionSearch(voxels_connect_idx_[connects_idx[i]],inthis_cluster_idx);	
					
				}

			}

		}

		void
		clusteringVoxels()		//Clustering of voxels		
		{
			//Setting
			int clusters_num = 0;
			std::vector<int> cluster_voxel;
			 
			//Clustering
			int i = 0;
			for (int i = 0; i<this->voxels_num_; i++)// loop + recrusion
			{
				if (!this->voxel_clustered_[i])
				{
					//std::cout<<i<<std::endl;

					std::vector<int> *inthis_cluster_idx = new std::vector<int>;

					this->voxel_clustered_.at(i) = true;

					this->recursionSearch(this->voxels_connect_idx_[i], inthis_cluster_idx);

					inthis_cluster_idx->push_back(i); //Don't forget to 

					this->clusters_voxel_idx_.push_back(*inthis_cluster_idx);
				}
			}


			//Output
			this->clusters_num_=clusters_voxel_idx_.size();

			//std::cout<<" In total " << this->clusters_num_ <<std::endl;

			//for(int m=0;m<this->clusters_num_;m++)
			//{
			//	int cluster_size=clusters_voxel_idx_[m].size();
			//	//std::cout<<"Cluster "<< m <<" has voxels: "<<std::endl;
			//	for(int n=0;n<cluster_size;n++)
			//	{
			//		std::cout<<clusters_voxel_idx_[m][n] << ", ";
			//	}
			//	std::cout<< "      "<<std::endl;
			//}
		
		}

		//Other
		void
		getVoxelCenterFromOctreeKey (pcl::octree::OctreeKey key, pcl::PointXYZ & point)
		{
			// define point to leaf node voxel center
			point.x = static_cast<float> ((static_cast<double> (key.x) + 0.5f) * this->voxel_resolution_ + this->min_x_);
			point.y = static_cast<float> ((static_cast<double> (key.y) + 0.5f) * this->voxel_resolution_ + this->min_y_);
			point.z = static_cast<float> ((static_cast<double> (key.z) + 0.5f) * this->voxel_resolution_ + this->min_z_);
		}

		void
		crossValidation()
		{
			//Traverse
			for(int i=0;i<voxels_num_;i++)// loop all the voxels
			{
				int inthis_num=voxels_connect_idx_[i].size();
				std::vector<int> new_connect_idx;

				if(inthis_num>1)
				{
					for(int j=0;j<inthis_num;j++)// check its connections
					{
						bool find_ornot=false;
						int find_idx=i;
						int search_idx=voxels_connect_idx_[i][j];
						int search_size=voxels_connect_idx_[search_idx].size();
						
						if(search_size>0)
						{
							for(int k=0;k<search_size;k++)
							{
								if(voxels_connect_idx_[search_idx][k]==find_idx)
								{
									find_ornot=true;
								}						
							}
						}
						//Judgement
						if(find_ornot==true)
						{
							new_connect_idx.push_back(search_idx);
						}
					}

					//Replace
					voxels_connect_idx_[i].clear();
					voxels_connect_idx_[i]=new_connect_idx;
				}
				//else if(inthis_num>0) //Closet neighbor check
				//{
				//	if(voxels_adjacency_idx_[i].size()>0)
				//	{
				//		float min_dis=0, temp_dis=0;
				//		int min_idx=voxels_adjacency_idx_[i][0];
				//		pcl::PointXYZ temp_p1, temp_p2;
				//		temp_p1=voxel_centroids_[i];
				//		temp_p2=voxel_centroids_[voxels_adjacency_idx_[i][0]];	
				//		min_dis=sqrt(pow(temp_p1.x-temp_p2.x,2)+pow(temp_p1.y-temp_p2.y,2)+pow(temp_p1.z-temp_p2.z,2));
				//		for (int j=0;j<voxels_adjacency_idx_[i].size();j++)
				//		{
				//			if(voxels_connect_idx_[voxels_adjacency_idx_[i][j]].size()>1)
				//			{
				//				temp_p2=voxel_centroids_[voxels_adjacency_idx_[i][j]];						
				//				temp_dis=sqrt(pow(temp_p1.x-temp_p2.x,2)+pow(temp_p1.y-temp_p2.y,2)+pow(temp_p1.z-temp_p2.z,2));
				//				if(temp_dis<min_dis)
				//				{
				//					min_dis=temp_dis;
				//					min_idx=voxels_adjacency_idx_[i][j];
				//				}
				//			}
				//		}
				//		//new_connect_idx.push_back(min_idx);
				//		voxels_connect_idx_[i].push_back(min_idx);
				//		voxels_connect_idx_[min_idx].push_back(i);
				//	}
				//}
			}		
		}

		void
		closestCheck(float sig_p, float sig_n, float sig_o, float sig_e, float sig_c, float sig_w)
		{
			//Settings
			std::vector<float> voxel1_position, voxel1_normal, voxel1_eigen;
			std::vector<float> voxel2_position, voxel2_normal, voxel2_eigen;
			pcl::PointXYZ temp_center, empty_center;
			pcl::Normal temp_norm, empty_norm;
			std::vector<float> dist_vector;
			empty_center.x=0;empty_center.y=0;empty_center.z=0;
			empty_norm.normal_x=0;empty_norm.normal_y=0;empty_norm.normal_z=0;

			//Closet neighbor check
			for(int i=0;i<voxels_num_;i++)
			{
				int inthis_num=voxels_connect_idx_[i].size();
				std::vector<int> new_connect_idx;
				
				if(inthis_num>0 && inthis_num<2)// The voxels have zero connection will be reconnected. 
				{
					if(voxels_adjacency_idx_[i].size()>this->voxel_adjacency_min_)// But if this "single" voxel has only few adjacencies, we will give it up.
					{
						//Test
						//std::cout<<"Single voxel: "<<i<<std::endl;
											
						float min_dis=0, temp_dis=0;
						int min_idx;
						pcl::PointXYZ temp_p1, temp_p2;
						
						voxel1_position.clear();
						voxel1_normal.clear();
						voxel1_eigen.clear();
						temp_center=voxel_centroids_[i];
						temp_norm=voxel_norms_[i];

						if(temp_center.x!=empty_center.x && temp_center.y!=empty_center.y && temp_center.z!=empty_center.z)
						{
							voxel1_position.push_back(temp_center.x);
							voxel1_position.push_back(temp_center.y);
							voxel1_position.push_back(temp_center.z);
						}
						else
						{
							voxel1_position.push_back(0);
						}

						if(temp_norm.normal_x!=empty_norm.normal_x && temp_norm.normal_y!=empty_norm.normal_y && temp_norm.normal_z!=empty_norm.normal_z)
						{
							voxel1_normal.push_back(temp_norm.normal_x);
							voxel1_normal.push_back(temp_norm.normal_y);
							voxel1_normal.push_back(temp_norm.normal_z);
						}
						else
						{
							voxel1_normal.push_back(0);
						}
						voxel1_eigen=voxel_eigens_[i];

						//Initialization
						min_dis=0; min_idx=-1;

						//Check neighbors
						for (int j=0;j<voxels_adjacency_idx_[i].size();j++)
						{
							if(voxels_connect_idx_[voxels_adjacency_idx_[i][j]].size()>1) // We cannot merge a single voxel to another single voxel
							{
								temp_p2=voxel_centroids_[voxels_adjacency_idx_[i][j]];

								voxel2_position.clear();
								voxel2_normal.clear();
								voxel2_eigen.clear();
								temp_center=voxel_centroids_[voxels_adjacency_idx_[i][j]];
								temp_norm=voxel_norms_[voxels_adjacency_idx_[i][j]];

								if(temp_center.x!=empty_center.x && temp_center.y!=empty_center.y && temp_center.z!=empty_center.z)	
								{
									voxel2_position.push_back(temp_center.x);
									voxel2_position.push_back(temp_center.y);
									voxel2_position.push_back(temp_center.z);
								}
								else
								{
									voxel2_position.push_back(0);
								}

								if(temp_norm.normal_x!=empty_norm.normal_x && temp_norm.normal_y!=empty_norm.normal_y && temp_norm.normal_z!=empty_norm.normal_z)
								{
									voxel2_normal.push_back(temp_norm.normal_x);
									voxel2_normal.push_back(temp_norm.normal_y);
									voxel2_normal.push_back(temp_norm.normal_z);
								}
								else
								{
									voxel2_normal.push_back(0);
								}
								voxel2_eigen=voxel_eigens_[voxels_adjacency_idx_[i][j]];

								dist_vector=measuringDistance(voxel1_position, voxel2_position, voxel1_normal, voxel2_normal, voxel1_eigen, voxel2_eigen);		
								temp_dis= distanceWeight(dist_vector, sig_p, sig_n, sig_o, sig_e, sig_c, sig_w);

								if(temp_dis>=min_dis)
								{
									min_dis=temp_dis;
									min_idx=voxels_adjacency_idx_[i][j];
								}

							}
						}

						//new_connect_idx.push_back(min_idx);
						if(min_idx!=-1)
						{
							voxels_connect_idx_[i].push_back(min_idx);
							voxels_connect_idx_[min_idx].push_back(i);
						}

					}

				}
			
			}		
				
		}
   };
 }

 #endif // PCL_SEGMENTATION_VOXEL_H_
