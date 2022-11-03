#include "../cpp_utils/python_utils.h"
#include "../cpp_utils/geodesic_knn.h"

namespace bp = boost::python;
namespace bpn = boost::python::numpy;


PyObject * components_graph(const bpn::ndarray & par_v, const bpn::ndarray & source, const bpn::ndarray & target){
    uint32_t n_edges = bp::len(source);
    std::vector<uint32_t> c_source;
    std::vector<uint32_t> c_target;
    for(uint32_t i=0; i<n_edges; i++){
        uint32_t s = bp::extract<uint32_t>(source[i]);
        uint32_t t = bp::extract<uint32_t>(target[i]);
        uint32_t label_s = bp::extract<uint32_t>(par_v[s]);
        uint32_t label_t = bp::extract<uint32_t>(par_v[t]);
        if (label_s == label_t){
            continue;
        }
        c_source.push_back(label_s);
        c_target.push_back(label_t);
    }
    return to_py_edges_tuple::convert_st(ST_tuple(c_source, c_target));
}


PyObject * all_to_all_k(const bpn::ndarray & source, const bpn::ndarray & target, const bpn::ndarray & distances, uint32_t n_verts, uint32_t k){
    std::vector<std::vector<float>> dists = all_to_all_k_shortest_paths(source, target, distances, n_verts, k);
    return VecvecToList<float>::convert(dists);
}


PyObject * all_to_all(const bpn::ndarray & source, const bpn::ndarray & target, const bpn::ndarray & distances, uint32_t n_verts){
    std::vector<std::vector<float>> dists = all_to_all_shortest_paths(source, target, distances, n_verts);
    return VecvecToList<float>::convert(dists);
}


PyObject * one_to_all(uint32_t s_idx, const bpn::ndarray & source, const bpn::ndarray & target, const bpn::ndarray & distances, uint32_t n_verts){
    std::vector<float> dists = one_to_all_shortest_paths(s_idx, source, target, distances, n_verts);
    return VecFloatToArray::convert(dists);
}


PyObject * unidirectional(const bpn::ndarray & uni_source, const bpn::ndarray & uni_index, 
        const bpn::ndarray & uni_counts, const bpn::ndarray & target){
    /*std::vector<uint32_t> mask;
    mask.reserve(bp::len(target));
    for (int i=0; i<bp::len(target); i++){
        mask.push_back(1);
    }*/
    std::vector<uint32_t> mask(bp::len(target), 1);
    //printf("%i\n", (int)mask[10]);
    unidirectional_mask(mask, uni_source, uni_index, uni_counts, target);
    return VecUIntToArray::convert(mask);
}


float geodesic_path(
    uint32_t source_idx,
    uint32_t target_idx,
    const bpn::ndarray & uni_source,
    const bpn::ndarray & uni_index, 
    const bpn::ndarray & uni_counts,
    const bpn::ndarray & target,
    const bpn::ndarray & distances)
{
    float dist = search_bfs_source_target(source_idx, target_idx, uni_source, uni_index, uni_counts, target, distances);
    //return to_py_tuple_mt::convert(Custom_tuple_mt(components, in_component, stats, (float)elapsed.count()));
    return dist;
}


PyObject * geodesic_knn_single(
    const uint32_t idx,
    const bpn::ndarray & uni_source,
    const bpn::ndarray & uni_index, 
    const bpn::ndarray & uni_counts,
    const bpn::ndarray & target,
    const bpn::ndarray & distances,
    const uint32_t knn,
    const bool exclude_closest)
{
    std::vector<uint32_t> out_source;
    std::vector<uint32_t> out_target;
    std::vector<float> out_distances;
    std::vector<bool> okay;
    bool ok = search_bfs_single(idx, uni_source, uni_index, uni_counts, target, distances, knn, out_source, out_target, out_distances, exclude_closest);
    okay.push_back(ok);
    //return to_py_tuple_mt::convert(Custom_tuple_mt(components, in_component, stats, (float)elapsed.count()));
    return to_py_edges_tuple::convert(Edges_tuple(out_source, out_target, out_distances, okay));
}


PyObject * geodesic_knn(
    const bpn::ndarray & uni_source,
    const bpn::ndarray & uni_index, 
    const bpn::ndarray & uni_counts,
    const bpn::ndarray & target,
    const bpn::ndarray & distances,
    const uint32_t knn,
    const bool exclude_closest)
{
    std::vector<uint32_t> out_source;
    std::vector<uint32_t> out_target;
    std::vector<float> out_distances;
    std::vector<bool> okay = search_bfs(uni_source, uni_index, uni_counts, target, distances, knn, out_source, out_target, out_distances, exclude_closest);
    //return to_py_tuple_mt::convert(Custom_tuple_mt(components, in_component, stats, (float)elapsed.count()));
    return to_py_edges_tuple::convert(Edges_tuple(out_source, out_target, out_distances, okay));
}


PyObject * geodesic_radiusnn_single(
    const uint32_t idx,
    const bpn::ndarray & uni_source,
    const bpn::ndarray & uni_index, 
    const bpn::ndarray & uni_counts,
    const bpn::ndarray & target,
    const bpn::ndarray & distances,
    const float radius, 
    const bool exclude_closest)
{
    std::vector<uint32_t> out_source;
    std::vector<uint32_t> out_target;
    std::vector<float> out_distances;
    std::vector<bool> okay;
    bool ok = search_bfs_radius_single(idx, uni_source, uni_index, uni_counts, target, distances, radius, out_source, out_target, out_distances, exclude_closest);
    okay.push_back(ok);
    //return to_py_tuple_mt::convert(Custom_tuple_mt(components, in_component, stats, (float)elapsed.count()));
    return to_py_edges_tuple::convert(Edges_tuple(out_source, out_target, out_distances, okay));
}


PyObject * geodesic_radiusnn(
    const bpn::ndarray & uni_source,
    const bpn::ndarray & uni_index, 
    const bpn::ndarray & uni_counts,
    const bpn::ndarray & target,
    const bpn::ndarray & distances,
    const float radius, 
    const bool exclude_closest)
{
    std::vector<uint32_t> out_source;
    std::vector<uint32_t> out_target;
    std::vector<float> out_distances;
    std::vector<bool> okay = search_bfs_radius(uni_source, uni_index, uni_counts, target, distances, radius, out_source, out_target, out_distances, exclude_closest);
    //return to_py_tuple_mt::convert(Custom_tuple_mt(components, in_component, stats, (float)elapsed.count()));
    return to_py_edges_tuple::convert(Edges_tuple(out_source, out_target, out_distances, okay));
}


PyObject * geodesic_neighbours(
    const bpn::ndarray & sources,
    const bpn::ndarray & uni_index, 
    const bpn::ndarray & uni_counts,
    const bpn::ndarray & target,
    const bpn::ndarray & distances,
    const uint32_t depth,
    const uint32_t n_verts,
    const bool exclude_closest)
{
    std::vector<uint32_t> out_source;
    std::vector<uint32_t> out_target;
    std::vector<float> out_distances;

    std::vector<bool> okay = search_bfs_depth(sources, uni_index, uni_counts, target, distances, out_source, out_target, out_distances,
        depth, n_verts, exclude_closest);

    //return to_py_tuple_mt::convert(Custom_tuple_mt(components, in_component, stats, (float)elapsed.count()));
    return to_py_edges_tuple::convert(Edges_tuple(out_source, out_target, out_distances, okay));
}


//using namespace boost::python;
BOOST_PYTHON_MODULE(libgeo)
{
    _import_array();
    Py_Initialize();
    bpn::initialize();
    //bp::to_python_converter<std::vector<std::vector<float>, std::allocator<std::vector<float> > >, VecvecToArray<float> >();
    //bp::to_python_converter< Components_tuple, to_py_tuple>();
    def("unidirectional", unidirectional);
    def("geodesic_knn", geodesic_knn);
    def("geodesic_knn_single", geodesic_knn_single);
    def("geodesic_radiusnn", geodesic_radiusnn);
    def("geodesic_radiusnn_single", geodesic_radiusnn_single);
    def("geodesic_path", geodesic_path);
    def("geodesic_neighbours", geodesic_neighbours);
    def("one_to_all", one_to_all);
    def("all_to_all", all_to_all);
    def("all_to_all_k", all_to_all_k);
    def("components_graph", components_graph);
}