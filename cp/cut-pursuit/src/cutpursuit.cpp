#include <iostream>
#include <cstdio>
#include <vector>
#include <chrono>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <numpy/ndarrayobject.h>
#include "boost/tuple/tuple.hpp"
#include "boost/python/object.hpp"
#include <../include/API.h>
//#include <../include/connected_components.h>

namespace bpn = boost::python::numpy;
namespace bp =  boost::python;

typedef boost::tuple< std::vector< std::vector<uint32_t> >, std::vector<uint32_t> > Custom_tuple;
typedef boost::tuple< std::vector< std::vector<uint32_t> >, std::vector<uint32_t>, std::vector<float>, float > Custom_tuple_mt;

struct VecToArray
{//converts a vector<uint32_t> to a numpy array
    static PyObject * convert(const std::vector<uint32_t> & vec)
    {
        npy_intp dims = vec.size();
        PyObject * obj = PyArray_SimpleNew(1, &dims, NPY_UINT32);
        void * arr_data = PyArray_DATA((PyArrayObject*)obj);
        memcpy(arr_data, &vec[0], dims * sizeof(uint32_t));
        return obj;
    }
};

struct VecToArray_float
{//converts a vector<float> to a numpy array
    static PyObject * convert(const std::vector<float> & vec)
    {
        npy_intp dims = vec.size();
        PyObject * obj = PyArray_SimpleNew(1, &dims, NPY_FLOAT32);
        void * arr_data = PyArray_DATA((PyArrayObject*)obj);
        memcpy(arr_data, &vec[0], dims * sizeof(float));
        return obj;
    }
};


template<class T>
struct VecvecToList
{//converts a vector< vector<T> > to a list
        static PyObject* convert(const std::vector< std::vector<T> > & vecvec)
    {
        boost::python::list* pylistlist = new boost::python::list();
        for(size_t i = 0; i < vecvec.size(); i++)
        {
            boost::python::list* pylist = new boost::python::list();
            for(size_t j = 0; j < vecvec[i].size(); j++)
            {
                pylist->append(vecvec[i][j]);
            }
            pylistlist->append((pylist, pylist[0]));
        }
        return pylistlist->ptr();
    }
};

struct to_py_tuple
{//converts output to a python tuple
    static PyObject* convert(const Custom_tuple& c_tuple){
        bp::list values;
        //add all c_tuple items to "values" list

        PyObject * vecvec_pyo = VecvecToList<uint32_t>::convert(c_tuple.get<0>());
        PyObject * vec_pyo = VecToArray::convert(c_tuple.get<1>());

        values.append(bp::handle<>(bp::borrowed(vecvec_pyo)));
        values.append(bp::handle<>(bp::borrowed(vec_pyo)));

        return bp::incref( bp::tuple( values ).ptr() );
    }
};

struct to_py_tuple_mt
{//converts output to a python tuple
    static PyObject* convert(const Custom_tuple_mt& c_tuple){
        bp::list values;
        //add all c_tuple items to "values" list

        PyObject * vecvec_pyo = VecvecToList<uint32_t>::convert(c_tuple.get<0>());
        PyObject * vec_pyo = VecToArray::convert(c_tuple.get<1>());
        PyObject * arr_pyo = VecToArray_float::convert(c_tuple.get<2>());
        float duration = c_tuple.get<3>();

        values.append(bp::handle<>(bp::borrowed(vecvec_pyo)));
        values.append(bp::handle<>(bp::borrowed(vec_pyo)));
        values.append(bp::handle<>(bp::borrowed(arr_pyo)));
        values.append(duration);

        return bp::incref( bp::tuple( values ).ptr() );
    }
};

struct to_py_tuple_list
{//converts output to a python list of tuples
    static PyObject* convert(const std::vector <Custom_tuple > & c_tuple_vec){
        int n_hierarchy = c_tuple_vec.size();
        bp::list values;
        //add all c_tuple items to "values" list
        for (int i_hierarchy = 0; i_hierarchy < n_hierarchy; i_hierarchy++)
        {
            PyObject * tuple_pyo = to_py_tuple::convert(c_tuple_vec[i_hierarchy]);
            values.append(bp::handle<>(bp::borrowed(tuple_pyo)));
        }
        return bp::incref(values.ptr());
    }
};

PyObject * cutpursuit(const bpn::ndarray & obs, const bpn::ndarray & source, const bpn::ndarray & target,const bpn::ndarray & edge_weight,
                      float lambda, const int cutoff, const int spatial, float weight_decay, float verbose, float speed, bool store_bin_labels)
{//read data and run the L0-cut pursuit partition algorithm
    srand(1);

    const uint32_t n_ver = bp::len(obs);
    const uint32_t n_edg = bp::len(source);
    const uint32_t n_obs = bp::len(obs[0]);
    const float * obs_data = reinterpret_cast<float*>(obs.get_data());
    const uint32_t * source_data = reinterpret_cast<uint32_t*>(source.get_data());
    const uint32_t * target_data = reinterpret_cast<uint32_t*>(target.get_data());
    const float * edge_weight_data = reinterpret_cast<float*>(edge_weight.get_data());
    std::vector<float> solution(n_ver *n_obs);
    //float solution [n_ver * n_obs];
    std::vector<float> node_weight(n_ver, 1.0f);
    std::vector<uint32_t> in_component(n_ver,0);
    std::vector< std::vector<uint32_t> > components(1,std::vector<uint32_t>(1,0.f));
    std::vector<float> stats;
    auto start = std::chrono::high_resolution_clock::now();

    if (spatial == 0)
    {
        CP::cut_pursuit<float>(n_ver, n_edg, n_obs, obs_data, source_data, target_data, edge_weight_data, &node_weight[0]
                 , solution.data(), in_component, components, lambda, (uint32_t)cutoff,  1.f, speed, weight_decay, verbose, stats, store_bin_labels);
    }
    else
    {
        CP::cut_pursuit<float>(n_ver, n_edg, n_obs, obs_data, source_data, target_data, edge_weight_data, &node_weight[0]
                 , solution.data(), in_component, components, lambda, (uint32_t)cutoff,  2.f, speed, weight_decay, verbose, stats, store_bin_labels);
    }
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;

    /*
    printf("stats:\n");
    for (int i = 0; i < stats.size(); i++)
    {
        printf("%f, ", stats[i]);
    }
    printf("\n");
    */
    return to_py_tuple_mt::convert(Custom_tuple_mt(components, in_component, stats, (float)elapsed.count()));
}

// maybe use BOOST_PYTHON_MODULE(cp_ext) for windows
BOOST_PYTHON_MODULE(libcp)
{
    _import_array();
    Py_Initialize();
    bpn::initialize();
    bp::to_python_converter< Custom_tuple, to_py_tuple>();
    
    def("cutpursuit", cutpursuit);
    def("cutpursuit", cutpursuit, (bp::args("cutoff")=0, bp::args("spatial")=0, bp::args("weight_decay")=1, bp::args("verbose")=0.f, bp::args("speed")=4.f, bp::args("store_bin_labels")=0));
}

