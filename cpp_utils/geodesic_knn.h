#pragma once

#ifndef GEODESIC_KNN_H_
#define GEODESIC_KNN_H_

#include <Eigen/Dense>
#include <math.h>
#include <vector>
#include <map>
#include <limits>
#include <algorithm>
#include <omp.h>

#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/property_map/property_map.hpp>

#include "./python_utils.h"

typedef std::pair<int, int> Edge;
typedef boost::adjacency_list < boost::listS, boost::vecS, boost::directedS, boost::no_property, boost::property < boost::edge_weight_t, float > > graph_t;
typedef boost::graph_traits < graph_t >::vertex_descriptor vertex_descriptor;

struct ReplaceEdge{
    uint32_t s, t, is, it;
    uint32_t replace;

    ReplaceEdge() : s(0), t(0), is(0), it(0), replace(0)
    {
    }
};

struct VertexDistTuple{
    uint32_t v;
    float d;

    VertexDistTuple() : v(0), d(0)
    {

    }

    VertexDistTuple(uint32_t v, float d) : v(v), d(d)
    {

    }
};


bool smaller(float a, float b)
{
    return (a < b);
}


std::vector<std::vector<float>> all_to_all_k_shortest_paths(bpn::ndarray source, bpn::ndarray target, bpn::ndarray distances, uint32_t n_verts, uint32_t k){
    std::vector<std::vector<float>> dist_mat;
    dist_mat.reserve(n_verts);
    for(uint32_t i = 0; i < n_verts; i++){
        std::vector<float> dist_vec (k, 0.0f);
        dist_mat.push_back(dist_vec);
    }

    uint32_t n_edges = get_len(source);

#ifdef _WIN32
    Edge* edges = new Edge[n_edges];
    float* weights = new float[n_edges];
#else
    Edge edges[n_edges];
    float weights[n_edges];
#endif
    for (uint32_t i=0; i < n_edges; i++){
        uint32_t u = bp::extract<uint32_t>(source[i]);
        uint32_t v = bp::extract<uint32_t>(target[i]);
        edges[i] = Edge(u, v);
        float w = bp::extract<float>(distances[i]);
        weights[i] = w;
    }

    graph_t g(edges, edges + n_edges, weights, n_verts);
    std::vector<float> d(boost::num_vertices(g));
    //boost::property_map<graph_t, boost::edge_weight_t>::type weightmap = boost::get(boost::edge_weight, *mesh_);
    std::vector<vertex_descriptor> p(boost::num_vertices(g));
    for(uint32_t i = 0; i < n_verts; i++){
        vertex_descriptor s = boost::vertex(i, g);
        dijkstra_shortest_paths(g, s,
            boost::predecessor_map(boost::make_iterator_property_map(p.begin(), get(boost::vertex_index, g))).
            distance_map(boost::make_iterator_property_map(d.begin(), get(boost::vertex_index, g))));
        std::sort(d.begin(), d.end(), smaller);
        for (uint32_t j = 0; j < k; j++){
            dist_mat[i][j] = d[j];
        }
    }

#ifdef _WIN32
    delete[] edges;
    delete[] weights;
#endif
    return dist_mat;
}


std::vector<std::vector<float>> all_to_all_shortest_paths(bpn::ndarray source, bpn::ndarray target, bpn::ndarray distances, uint32_t n_verts){
    std::vector<std::vector<float>> dist_mat;
    dist_mat.reserve(n_verts);
    for(uint32_t i = 0; i < n_verts; i++){
        std::vector<float> dist_vec (n_verts, 0.0f);
        dist_mat.push_back(dist_vec);
    }

    uint32_t n_edges = get_len(source);
#ifdef _WIN32
    Edge* edges = new Edge[n_edges];
    float* weights = new float[n_edges];
#else
    Edge edges[n_edges];
    float weights[n_edges];
#endif
    for (uint32_t i=0; i < n_edges; i++){
        uint32_t u = bp::extract<uint32_t>(source[i]);
        uint32_t v = bp::extract<uint32_t>(target[i]);
        edges[i] = Edge(u, v);
        float w = bp::extract<float>(distances[i]);
        weights[i] = w;
    }

    graph_t g(edges, edges + n_edges, weights, n_verts);
    std::vector<float> d(boost::num_vertices(g));
    //boost::property_map<graph_t, boost::edge_weight_t>::type weightmap = boost::get(boost::edge_weight, *mesh_);
    std::vector<vertex_descriptor> p(boost::num_vertices(g));
    for(uint32_t i = 0; i < n_verts; i++){
        vertex_descriptor s = boost::vertex(i, g);
        dijkstra_shortest_paths(g, s,
            boost::predecessor_map(boost::make_iterator_property_map(p.begin(), get(boost::vertex_index, g))).
            distance_map(boost::make_iterator_property_map(d.begin(), get(boost::vertex_index, g))));
        dist_mat[i] = d;
    }
#ifdef _WIN32
    delete[] edges;
    delete[] weights;
#endif
    return dist_mat;
}


std::vector<float> one_to_all_shortest_paths(uint32_t s_idx, bpn::ndarray source, bpn::ndarray target, bpn::ndarray distances, uint32_t n_verts){
    uint32_t n_edges = get_len(source);
#ifdef _WIN32
    Edge* edges = new Edge[n_edges];
    float* weights = new float[n_edges];
#else
    Edge edges[n_edges];
    float weights[n_edges];
#endif
    for (uint32_t i=0; i < n_edges; i++){
        uint32_t u = bp::extract<uint32_t>(source[i]);
        uint32_t v = bp::extract<uint32_t>(target[i]);
        edges[i] = Edge(u, v);
        float w = bp::extract<float>(distances[i]);
        weights[i] = w;
    }

    graph_t g(edges, edges + n_edges, weights, n_verts);

    std::vector<float> d(boost::num_vertices(g));
    //boost::property_map<graph_t, boost::edge_weight_t>::type weightmap = boost::get(boost::edge_weight, *mesh_);
    std::vector<vertex_descriptor> p(boost::num_vertices(g));
    vertex_descriptor s = boost::vertex(s_idx, g);
    dijkstra_shortest_paths(g, s,
        boost::predecessor_map(boost::make_iterator_property_map(p.begin(), get(boost::vertex_index, g))).
        distance_map(boost::make_iterator_property_map(d.begin(), get(boost::vertex_index, g))));
    //printf("%i, %i\n", (int)s_idx, p[s_idx]);
#ifdef _WIN32
    delete[] edges;
    delete[] weights;
#endif
    return d;
}


uint32_t binary_search(bpn::ndarray arr, uint32_t low, uint32_t high, uint32_t x){
    uint32_t mid = (uint32_t) floor((high + low) / 2);
    if (bp::extract<uint32_t>(arr[mid]) == x){
        return mid;
    }
    else if (bp::extract<uint32_t>(arr[mid]) > x){
        return binary_search(arr, low, mid - 1, x);
    }
    else {
        return binary_search(arr, mid + 1, high, x);
    }
}


uint32_t simple_search(bpn::ndarray arr, uint32_t low, uint32_t high, uint32_t x){
    for(uint32_t i = low; i < high; i++){
        if( bp::extract<uint32_t>(arr[i]) == x ){
            return i;
        }
    }
    return 0;
}


void unidirectional_helper(std::vector<ReplaceEdge> & res, bpn::ndarray uni_source, bpn::ndarray uni_index, bpn::ndarray uni_counts, bpn::ndarray target){
    uint32_t n_uni_source = get_len(uni_source);
    #ifdef OPENMP
    #pragma omp parallel for if (n_uni_source < omp_get_num_threads()) shared(res) schedule(static) 
    #endif
    #ifdef _WIN32
    for (int i = 0; i < n_uni_source; i++)
    #else
    for (uint32_t i = 0; i < n_uni_source; i++)
    #endif
    {
        uint32_t start = bp::extract<uint32_t>(uni_index[i]);
        uint32_t stop = start + bp::extract<uint32_t>(uni_counts[i]);
        uint32_t u_s = bp::extract<uint32_t>(uni_source[i]);
        for(uint32_t j = start; j < stop; j++){
            uint32_t u_t = bp::extract<uint32_t>(target[j]);
            ReplaceEdge re;
            re.replace = 0;
            re.s = u_s;
            re.t = u_t;
            re.is = j;
            re.it = j;
            uint32_t t_idx = 0;
            if (u_t > u_s){
               t_idx = binary_search(uni_source, i, n_uni_source, u_t); 
               //t_idx = simple_search(uni_source, i, n_uni_source, u_t); 
            }
            else {
               t_idx = binary_search(uni_source, 0, i, u_t);  
               //t_idx = simple_search(uni_source, 0, i, u_t); 
            }
            uint32_t t_start = bp::extract<uint32_t>(uni_index[t_idx]);
            uint32_t t_stop = t_start + bp::extract<uint32_t>(uni_counts[t_idx]);
            bool found = false;
            uint32_t r_idx = 0;
            for(uint32_t k = t_start; k < t_stop; k++){
                if (bp::extract<uint32_t>(target[k]) == u_s){
                    r_idx = k;
                    found = true;
                    break;
                }
            }
            re.it = r_idx;
            if (!found){
                res.push_back(re);
                continue;
            }
            re.replace = 1;
            res.push_back(re);
        }
    }
}


void unidirectional_mask(std::vector<uint32_t> & mask, bpn::ndarray uni_source, bpn::ndarray uni_index, bpn::ndarray uni_counts, bpn::ndarray target){
    std::vector<ReplaceEdge> redges;
    unidirectional_helper(redges, uni_source, uni_index, uni_counts, target);
    uint32_t n_edges = get_len(target);
    std::vector<uint32_t> checked(n_edges, 0);
    for(int i = 0; i < redges.size(); i++){
        ReplaceEdge redge = redges[i];
        uint32_t s_idx = redge.is;
        uint32_t t_idx = redge.it;
        checked[s_idx] = 1;
        if(redge.replace == 0){
            continue;
        }
        if(checked[t_idx] == 1){
            continue;
        }
        checked[t_idx] = 1;
        mask[t_idx] = 0;
    }
}


void extract_neighbours(VertexDistTuple vdt, bpn::ndarray target, bpn::ndarray uni_index,
    bpn::ndarray uni_counts, bpn::ndarray distances, std::vector<VertexDistTuple> & nvdts)
{
    //printf("%i/%i\n", (int)vdt.v, (int)get_len(uni_index));
    uint32_t start = bp::extract<uint32_t>(uni_index[vdt.v]);
    uint32_t stop = start + bp::extract<uint32_t>(uni_counts[vdt.v]);
    uint32_t len = stop - start;
    nvdts.reserve(len);
    for(uint32_t i = start; i < stop; i++){
        uint32_t n_idx = bp::extract<uint32_t>(target[i]);
        float dist = bp::extract<float>(distances[i]);
        nvdts.push_back(VertexDistTuple(n_idx, dist + vdt.d));
    }
}


float search_bfs_source_target(uint32_t source_idx, uint32_t target_idx, bpn::ndarray uni_source, bpn::ndarray uni_index,
    bpn::ndarray uni_counts, bpn::ndarray target, bpn::ndarray distances)
{
    //printf("%i\n", (int)omp_get_num_threads());
    std::vector<VertexDistTuple> paths_to_check;
    std::vector<float> dists_found;
    float min_dist = std::numeric_limits<float>::max();
    uint32_t n_verts = get_len(uni_source);

    uint32_t v_s = bp::extract<uint32_t>(uni_source[source_idx]);
    uint32_t v_t = bp::extract<uint32_t>(uni_source[target_idx]);
    paths_to_check.push_back(VertexDistTuple(v_s, 0));
    std::map<uint32_t, float> all_p_lens;
    while(paths_to_check.size() > 0){
        std::map<uint32_t, float> tmp_paths_to_check;
        int i = 0;
        while(i < paths_to_check.size()){
            //target, path_distance = paths_to_check.pop(0)
            VertexDistTuple vdt = paths_to_check[i];
            i++;
            std::vector<VertexDistTuple> nvdts;
            extract_neighbours(vdt, target, uni_index, uni_counts, distances, nvdts);

            for(uint32_t j = 0; j < nvdts.size(); j++){
                VertexDistTuple nvdt = nvdts[j];
                if (nvdt.v >= n_verts)
                    continue;
                if (nvdt.d >= min_dist)
                    continue;
                if(nvdt.v == v_t){ // if we found the target vertex
                    min_dist = nvdt.d;
                    continue;
                }
                if(all_p_lens.count(nvdt.v)){
                    if(nvdt.d < all_p_lens[nvdt.v])
                        all_p_lens[nvdt.v] = nvdt.d;
                    else
                        continue;
                }
                else
                    all_p_lens[nvdt.v] = nvdt.d;
                tmp_paths_to_check[nvdt.v] = nvdt.d;
                //tmp_paths_to_check.insert ( std::pair<uint32_t, float>(nvdt.v, nvdt.d) );
            }
            //paths_to_check.pop_back();
        } // end inner while loop
        if(tmp_paths_to_check.size() == 0)
            break;
        paths_to_check.clear();
        paths_to_check.reserve(tmp_paths_to_check.size());
        //printf("%i, %i\n", (int) vi, (int)tmp_paths_to_check.size());
        for (std::map<uint32_t, float>::iterator it = tmp_paths_to_check.begin(); it != tmp_paths_to_check.end(); ++it)
        {
            paths_to_check.push_back(VertexDistTuple(it->first, it->second));
        }
        //printf("%i, %i\n", (int) vi, (int)paths_to_check.size());
    } // end outer while loop
    //printf("%f\n", (double) min_dist);
    return min_dist;
}


void search_bfs_radius_single(uint32_t i, boost::shared_ptr<bpn::ndarray> p_uni_source, boost::shared_ptr<bpn::ndarray> p_uni_index,
        boost::shared_ptr<bpn::ndarray> p_uni_counts, boost::shared_ptr<bpn::ndarray> p_target, boost::shared_ptr<bpn::ndarray> p_distances,
        float max_dist, std::vector<uint32_t> & out_source, std::vector<uint32_t> & out_target, std::vector<float> & out_distances,
        bool exclude_closest)
{
    
    bpn::ndarray uni_source = *p_uni_source;
    bpn::ndarray uni_index = *p_uni_index;
    bpn::ndarray uni_counts = *p_uni_counts;
    bpn::ndarray target = *p_target;
    bpn::ndarray distances = *p_distances;
    
    std::vector<VertexDistTuple> paths_to_check;
    float bound = max_dist;
    std::map<uint32_t, float> all_p_lens;
    all_p_lens[i] = 0.0f;
    uint32_t n_verts = get_len(uni_source);

    uint32_t vi = bp::extract<uint32_t>(uni_source[i]);
    paths_to_check.push_back(VertexDistTuple(vi, 0));
    while(paths_to_check.size() > 0){
        std::map<uint32_t, float> tmp_paths_to_check;
        int i = 0;
        while(i < paths_to_check.size()){
            //target, path_distance = paths_to_check.pop(0)
            VertexDistTuple vdt = paths_to_check[i];
            i++;
            std::vector<VertexDistTuple> nvdts;
            extract_neighbours(vdt, target, uni_index, uni_counts, distances, nvdts);

            for(uint32_t j = 0; j < nvdts.size(); j++){
                VertexDistTuple nvdt = nvdts[j];
                if (nvdt.v >= n_verts)
                    continue;
                if(nvdt.d >= bound)
                    continue;
                if(all_p_lens.count( nvdt.v )){ // contains
                    float p_d = all_p_lens[nvdt.v];
                    if(nvdt.d >= p_d)
                        continue;
                }
                all_p_lens[nvdt.v] = nvdt.d;
                tmp_paths_to_check[nvdt.v] = nvdt.d;
            }
        } // end inner while loop
        if (tmp_paths_to_check.size() == 0)
            break;
        paths_to_check.clear();
        paths_to_check.reserve(tmp_paths_to_check.size());

        for (std::map<uint32_t, float>::iterator it = tmp_paths_to_check.begin(); it != tmp_paths_to_check.end(); ++it)
        {
            paths_to_check.push_back(VertexDistTuple(it->first, it->second));
        }
    } // end outer while loop
    /*
    out_source.reserve(all_p_lens.size() - 1);
    out_target.reserve(all_p_lens.size() - 1);
    out_distances.reserve(all_p_lens.size() - 1);
    */
    for (std::map<uint32_t, float>::iterator it = all_p_lens.begin(); it != all_p_lens.end(); ++it){
        if (it->first == vi && exclude_closest)
            continue;
        out_source.push_back(vi);
        out_target.push_back(it->first);
        out_distances.push_back(it->second);
    }
}



bool search_bfs_radius_single(uint32_t i, bpn::ndarray uni_source, bpn::ndarray uni_index, bpn::ndarray uni_counts, 
        bpn::ndarray target, bpn::ndarray distances, float max_dist, std::vector<uint32_t> & out_source,
        std::vector<uint32_t> & out_target, std::vector<float> & out_distances, bool exclude_closest)
{
    std::vector<VertexDistTuple> paths_to_check;
    float bound = max_dist;
    std::map<uint32_t, float> all_p_lens;
    all_p_lens[i] = 0.0f;
    uint32_t n_verts = get_len(uni_source);

    uint32_t vi = bp::extract<uint32_t>(uni_source[i]);
    paths_to_check.push_back(VertexDistTuple(vi, 0));
    while(paths_to_check.size() > 0){
        std::map<uint32_t, float> tmp_paths_to_check;
        int i = 0;
        while(i < paths_to_check.size()){
            //target, path_distance = paths_to_check.pop(0)
            VertexDistTuple vdt = paths_to_check[i];
            i++;
            std::vector<VertexDistTuple> nvdts;
            extract_neighbours(vdt, target, uni_index, uni_counts, distances, nvdts);

            for(uint32_t j = 0; j < nvdts.size(); j++){
                VertexDistTuple nvdt = nvdts[j];
                if (nvdt.v >= n_verts)
                    continue;
                if(nvdt.d >= bound)
                    continue;
                if(all_p_lens.count( nvdt.v )){ // contains
                    float p_d = all_p_lens[nvdt.v];
                    if(nvdt.d >= p_d)
                        continue;
                }
                //printf("%f\n", (double)nvdt.d);
                all_p_lens[nvdt.v] = nvdt.d;
                tmp_paths_to_check[nvdt.v] = nvdt.d;
            }
        } // end inner while loop
        if (tmp_paths_to_check.size() == 0)
            break;
        paths_to_check.clear();
        paths_to_check.reserve(tmp_paths_to_check.size());
        for (std::map<uint32_t, float>::iterator it = tmp_paths_to_check.begin(); it != tmp_paths_to_check.end(); ++it)
        {
            paths_to_check.push_back(VertexDistTuple(it->first, it->second));
        }
    } // end outer while loop
    for (std::map<uint32_t, float>::iterator it = all_p_lens.begin(); it != all_p_lens.end(); ++it){
        //printf("dist: %f\n", (double) it->second);
        if (it->first == vi && exclude_closest)
            continue;
        out_source.push_back(vi);
        out_target.push_back(it->first);
        out_distances.push_back(it->second);
    }
    return true;
}


std::vector<bool> search_bfs_radius(bpn::ndarray uni_source, bpn::ndarray uni_index, bpn::ndarray uni_counts, bpn::ndarray target,
        bpn::ndarray distances, float max_dist, std::vector<uint32_t> & out_source, std::vector<uint32_t> & out_target,
        std::vector<float> & out_distances, bool exclude_closest)
{
    uint32_t n_uni_source = get_len(uni_source);
    uint32_t n_edges = get_len(target);
    std::vector<bool> okay;
    //#ifdef OPENMP
    //#pragma omp parallel for if (n_uni_source < omp_get_num_threads()) shared(out_source, out_target, out_distances) schedule(static)
    //#pragma omp parallel for shared(out_source, out_target, out_distances) // --> really slow
    //#endif
    #ifdef _WIN32
    for (int i = 0; i < n_uni_source; i++)
    #else
    for (uint32_t i = 0; i < n_uni_source; i++)
    #endif
    {
        bool ok = search_bfs_radius_single(i, uni_source, uni_index, uni_counts, target, distances,
            max_dist, out_source, out_target, out_distances, exclude_closest);
        if(!ok)
        {
            okay.push_back(false);
            return okay;
        }
    }
    //printf("size out_source: %i\n", (int)out_source.size());
    okay.push_back(true);
    return okay;
}


void get_neighbours_radius(uint32_t query, bpn::ndarray uni_index, bpn::ndarray uni_counts,
        bpn::ndarray target, bpn::ndarray distances, float radius, std::vector<uint32_t> & out_target,
        std::vector<float> & out_distances){
    uint32_t start = bp::extract<uint32_t>(uni_index[query]);
    uint32_t stop = start + bp::extract<uint32_t>(uni_counts[query]);
    for(uint32_t i = start; i < stop; i++){
        float dist = bp::extract<float>(distances[i]);
        if (dist>radius)
            continue;
        uint32_t idx = bp::extract<uint32_t>(target[i]);
        out_target.push_back(idx);
        out_distances.push_back(dist);
    }
}


bool search_bfs_single(uint32_t i, bpn::ndarray uni_source, bpn::ndarray uni_index, bpn::ndarray uni_counts, bpn::ndarray target,
        bpn::ndarray distances, uint32_t k, std::vector<uint32_t> & out_source, std::vector<uint32_t> & out_target,
        std::vector<float> & out_distances, bool exclude_closest)
{
    //printf("%i\n", i);
    std::vector<VertexDistTuple> paths_to_check;
    float bound = std::numeric_limits<float>::max();
    std::map<uint32_t, float> all_p_lens;
    uint32_t n_verts = get_len(uni_source);

    uint32_t vi = bp::extract<uint32_t>(uni_source[i]);
    all_p_lens[vi] = 0.0f;
    paths_to_check.push_back(VertexDistTuple(vi, 0));

    std::vector<std::pair<uint32_t, float>> tmp_v;

    while(paths_to_check.size() > 0){
        std::map<uint32_t, float> tmp_paths_to_check;
        int z = 0;
        while(z < paths_to_check.size()){
            //target, path_distance = paths_to_check.pop(0)
            VertexDistTuple vdt = paths_to_check[z];
            z++;
            std::vector<VertexDistTuple> nvdts;
            extract_neighbours(vdt, target, uni_index, uni_counts, distances, nvdts);
            for(uint32_t j = 0; j < nvdts.size(); j++){
                VertexDistTuple nvdt = nvdts[j];
                if (nvdt.v >= n_verts)
                    continue;
                if(all_p_lens.count( nvdt.v )){ // contains
                    float p_d = all_p_lens[nvdt.v];
                    if(nvdt.d >= p_d)
                        continue;
                }

                all_p_lens[nvdt.v] = nvdt.d;
                tmp_paths_to_check[nvdt.v] = nvdt.d;
            }
        } // end inner while loop
        paths_to_check.clear();
        paths_to_check.reserve(tmp_paths_to_check.size());
        
        tmp_v.clear();
        std::copy(all_p_lens.begin(), all_p_lens.end(), std::back_inserter<std::vector<std::pair<uint32_t, float>>>(tmp_v));
        std::sort(tmp_v.begin(), tmp_v.end(), 
            [](const std::pair<uint32_t, float> &lhs, const std::pair<uint32_t, float> &rhs){
                if (lhs.second != rhs.second)
                {
                    return lhs.second < rhs.second;
                }
 
                return lhs.first < rhs.first;
            }
        );

        all_p_lens.clear();
        for (uint32_t j = 0; j < tmp_v.size(); j++){
            all_p_lens.insert(tmp_v[j]);
        }

        //printf("%i, %f\n", k, bound);
        if (all_p_lens.size() >= k+1){
            std::map<uint32_t, float>::iterator it = all_p_lens.begin();
            std::advance(it, k);
            bound = it->second;
        }
        //printf("%i, %i\n", (int) vi, (int)tmp_paths_to_check.size());
        for (std::map<uint32_t, float>::iterator it = tmp_paths_to_check.begin(); it != tmp_paths_to_check.end(); ++it)
        {
            if (it->second >= bound)
                continue;
            paths_to_check.push_back(VertexDistTuple(it->first, it->second));
        }
        //printf("%i, %i\n", (int) vi, (int)paths_to_check.size());
    } // end outer while loop

    uint32_t added = 0;
    for (uint32_t j = 0; j < tmp_v.size(); j++){
        if (tmp_v[j].first == vi && exclude_closest)
            continue;
        if (added == k)
            break;
        out_source.push_back(vi);
        out_target.push_back(tmp_v[j].first);
        out_distances.push_back(tmp_v[j].second);
        added++;
    }

    if (added != k)
    {
        printf("Vertex %i: Only found %i/%i neighbours\n", vi, added, k);
        return false;
    }
    return true;
}



std::vector<bool> search_bfs(bpn::ndarray uni_source, bpn::ndarray uni_index, bpn::ndarray uni_counts, bpn::ndarray target,
        bpn::ndarray distances, uint32_t k, std::vector<uint32_t> & out_source, std::vector<uint32_t> & out_target,
        std::vector<float> & out_distances, bool exclude_closest)
{
    uint32_t n_uni_source = get_len(uni_source);
    uint32_t n_edges = get_len(target);
    out_source.reserve(n_uni_source * k);
    out_target.reserve(n_uni_source * k);
    out_distances.reserve(n_uni_source * k);
    std::vector<bool> ok;
    //#ifdef OPENMP
    //#pragma omp parallel for if (n_uni_source < omp_get_num_threads()) shared(out_source, out_target, out_distances) schedule(static) 
    //#endif
    #ifdef _WIN32
    for (int i = 0; i < n_uni_source; i++)
    #else
    for (uint32_t i = 0; i < n_uni_source; i++)
    #endif
    {
        bool okay = search_bfs_single(i, uni_source, uni_index, uni_counts, target,
            distances, k, out_source, out_target, out_distances, exclude_closest);
        if(!okay)
        {
            ok.push_back(false);
            return ok;
        }
    } // end outer for loop
    //printf("\nsearch bfs done\n");
    ok.push_back(true);
    return ok;
} // end function


bool search_bfs_depth_single(uint32_t idx, bpn::ndarray sources, bpn::ndarray uni_index, bpn::ndarray uni_counts, bpn::ndarray target,
        bpn::ndarray distances, std::vector<uint32_t> & out_source, std::vector<uint32_t> & out_target,
        std::vector<float> & out_distances, uint32_t depth, uint32_t n_verts, bool exclude_closest)
{
    std::vector<VertexDistTuple> paths_to_check;
    std::map<uint32_t, float> all_p_lens;
    all_p_lens[idx] = 0.0f;

    uint32_t vi = bp::extract<uint32_t>(sources[idx]);
    paths_to_check.push_back(VertexDistTuple(vi, 0));
    //while(paths_to_check.size() > 0){
    for(uint32_t k = 0; k < depth; k++){
        std::map<uint32_t, float> tmp_paths_to_check;
        int i = 0;
        while(i < paths_to_check.size()){
            //target, path_distance = paths_to_check.pop(0)
            VertexDistTuple vdt = paths_to_check[i];
            i++;
            std::vector<VertexDistTuple> nvdts;
            extract_neighbours(vdt, target, uni_index, uni_counts, distances, nvdts);

            for(uint32_t j = 0; j < nvdts.size(); j++){
                VertexDistTuple nvdt = nvdts[j];
                if (nvdt.v >= n_verts)
                    continue;
                if(all_p_lens.count( nvdt.v )){ // contains
                    float p_d = all_p_lens[nvdt.v];
                    if(nvdt.d >= p_d)
                        continue;
                }
                //printf("%f\n", (double)nvdt.d);
                all_p_lens[nvdt.v] = nvdt.d;
                tmp_paths_to_check[nvdt.v] = nvdt.d;
            }
        } // end inner while loop
        if (tmp_paths_to_check.size() == 0)
            break;
        paths_to_check.clear();
        paths_to_check.reserve(tmp_paths_to_check.size());
        for (std::map<uint32_t, float>::iterator it = tmp_paths_to_check.begin(); it != tmp_paths_to_check.end(); ++it)
        {
            paths_to_check.push_back(VertexDistTuple(it->first, it->second));
        }
    } // end outer while loop
    for (std::map<uint32_t, float>::iterator it = all_p_lens.begin(); it != all_p_lens.end(); ++it){
        //printf("dist: %f\n", (double) it->second);
        if (it->first == vi && exclude_closest)
            continue;
        out_source.push_back(vi);
        out_target.push_back(it->first);
        out_distances.push_back(it->second);
    }
    return true;
}


std::vector<bool> search_bfs_depth(bpn::ndarray sources, bpn::ndarray uni_index, bpn::ndarray uni_counts, bpn::ndarray target,
        bpn::ndarray distances, std::vector<uint32_t> & out_source, std::vector<uint32_t> & out_target,
        std::vector<float> & out_distances, uint32_t depth, uint32_t n_verts, bool exclude_closest)
{
    uint32_t n_sources = get_len(sources);
    std::vector<bool> okay;
    //#ifdef OPENMP
    //#pragma omp parallel for if (n_sources < omp_get_num_threads()) shared(out_source, out_target, out_distances) schedule(static) 
    //#endif
    #ifdef _WIN32
    for (int i = 0; i < n_sources; i++)
    #else
    for (uint32_t i = 0; i < n_sources; i++)
    #endif
    {
        bool ok = search_bfs_depth_single(i, sources, uni_index, uni_counts, target,
            distances, out_source, out_target, out_distances, depth, n_verts, exclude_closest);
        if(!ok)
        {
            okay.push_back(false);
            return okay;
        }
    } // end outer for loop
    //printf("\nsearch bfs done\n");
    okay.push_back(true);
    return okay;
} // end function


#endif