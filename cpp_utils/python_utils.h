#pragma once

#ifndef PYTHON_UTILS_H_
#define PYTHON_UTILS_H_

#include <iostream>
#include <cstdio>
#include <vector>
#define BOOST_PYTHON_MAX_ARITY 26
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <numpy/ndarrayobject.h>
#include "boost/tuple/tuple.hpp"
#include "boost/python/object.hpp"

namespace bp = boost::python;
namespace bpn = boost::python::numpy;

typedef boost::tuple< std::vector< std::vector<int> >, std::vector<int>, float > Components_tuple;
typedef boost::tuple< std::vector<uint32_t>, std::vector<uint32_t>, std::vector<float>, std::vector<bool> > Edges_tuple;
typedef boost::tuple< std::vector<uint32_t>, std::vector<uint32_t> > ST_tuple;


uint32_t get_len(bpn::ndarray & arr){
#ifdef _WIN32 || __linux__
    uint32_t len = bp::len(arr);
#else
    uint32_t len = arr.shape(0);
#endif
    return len;
}

uint32_t get_len(const bpn::ndarray& arr) {
#ifdef _WIN32 || __linux__
    uint32_t len = bp::len(arr);
#else
    uint32_t len = arr.shape(0);
#endif
    return len;
}

uint32_t get_len_dim2(bpn::ndarray & arr){
    #ifdef _WIN32 || __linux__
    const uint32_t len = bp::len(arr[0]);
    #else
    const uint32_t len = arr.shape(1);
    #endif
    return len;
}


void np_uni2vec(const bpn::ndarray & us, const bpn::ndarray & ui, const bpn::ndarray & uc, 
    std::vector<uint32_t> &uni_source, std::vector<uint32_t> &uni_index, std::vector<uint32_t> &uni_counts){
    uint32_t n = get_len(us);
    uni_source.reserve(n);
    uni_index.reserve(n);
    uni_counts.reserve(n);

    #ifdef _WIN32 || __linux__
    for(uint32_t i = 0; i < n; i++){
        uni_source.push_back(bp::extract<uint32_t>(us[i]));
        uni_index.push_back(bp::extract<uint32_t>(ui[i]));
        uni_counts.push_back(bp::extract<uint32_t>(uc[i]));
    }
    #else
    uint32_t* us_data = reinterpret_cast<uint32_t*>(us.get_data());
    uint32_t* ui_data = reinterpret_cast<uint32_t*>(ui.get_data());
    uint32_t* uc_data = reinterpret_cast<uint32_t*>(uc.get_data());
    for(uint32_t i = 0; i < n; i++)
    {
        uni_source.push_back(us_data[i]);
        uni_index.push_back(ui_data[i]);
        uni_counts.push_back(uc_data[i]);
    }
    #endif
}

void np2vec(const bpn::ndarray & arr, std::vector<uint32_t> & vec){
    uint32_t n = get_len(arr);
    vec.reserve(n);
    #ifdef _WIN32 || __linux__
    for(uint32_t i = 0; i < n; i++){
        vec.push_back(bp::extract<uint32_t>(arr[i]));
    }
    #else
    uint32_t* arr_data = reinterpret_cast<uint32_t*>(arr.get_data());
    for(uint32_t i = 0; i < n; i++){
        vec.push_back(arr_data[i]);
    }
    #endif
}


template<class T>
struct VecToList
{//converts a vector< vector<T> > to a list
    static PyObject* convert(const std::vector< T >& vec)
    {
        boost::python::list* pylist = new boost::python::list();
        for (size_t i = 0; i < vec.size(); i++)
        {
            pylist->append(vec[i]);
        }
        return pylist->ptr();
    }
};


struct VecUIntToArray
{//converts a vector<int> to a numpy array
    static PyObject * convert(const std::vector<uint32_t> & vec) {
        npy_intp dims = vec.size();
        PyObject * obj = PyArray_SimpleNew(1, &dims, NPY_UINT32);
        void * arr_data = PyArray_DATA((PyArrayObject*)obj);
        memcpy(arr_data, &vec[0], dims * sizeof(uint32_t));
        return obj;
    }
};


struct VecFloatToArray
{//converts a vector<int> to a numpy array
    static PyObject * convert(const std::vector<float> & vec) {
        npy_intp dims = vec.size();
        PyObject * obj = PyArray_SimpleNew(1, &dims, NPY_FLOAT32);
        void * arr_data = PyArray_DATA((PyArrayObject*)obj);
        memcpy(arr_data, &vec[0], dims * sizeof(float));
        return obj;
    }
};


struct VecToArray
{//converts a vector<int> to a numpy array
    static PyObject * convert(const std::vector<int> & vec) {
        npy_intp dims = vec.size();
        PyObject * obj = PyArray_SimpleNew(1, &dims, NPY_INT32);
        void * arr_data = PyArray_DATA((PyArrayObject*)obj);
        memcpy(arr_data, &vec[0], dims * sizeof(int));
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
    static PyObject* convert(const Components_tuple& c_tuple){
        bp::list values;
        //add all c_tuple items to "values" list

        PyObject * vecvec_pyo = VecvecToList<int>::convert(c_tuple.get<0>());
        PyObject * vec_pyo = VecToArray::convert(c_tuple.get<1>());
        float duration = c_tuple.get<2>();

        values.append(bp::handle<>(bp::borrowed(vecvec_pyo)));
        values.append(bp::handle<>(bp::borrowed(vec_pyo)));
        values.append(duration);

        return bp::incref( bp::tuple( values ).ptr() );
    }

    // apple
    static bp::list convert_lists(const Components_tuple& c_tuple){
        bp::list values;
        PyObject * vecvec_pyo = VecvecToList<int>::convert(c_tuple.get<0>());
        PyObject * vec_pyo = VecToList<int>::convert(c_tuple.get<1>());
        float duration = c_tuple.get<2>();
        values.append(bp::handle<>(bp::borrowed(vecvec_pyo)));
        values.append(bp::handle<>(bp::borrowed(vec_pyo)));
        values.append(duration);
        return values;
    }
};


struct to_py_edges_tuple
{//converts output to a python tuple
    static PyObject* convert(const Edges_tuple& c_tuple){
        bp::list values;
        //add all c_tuple items to "values" list

        PyObject * source = VecUIntToArray::convert(c_tuple.get<0>());
        PyObject * target = VecUIntToArray::convert(c_tuple.get<1>());
        PyObject * dists = VecFloatToArray::convert(c_tuple.get<2>());
        PyObject * okay = VecToList<bool>::convert(c_tuple.get<3>());

        values.append(bp::handle<>(bp::borrowed(source)));
        values.append(bp::handle<>(bp::borrowed(target)));
        values.append(bp::handle<>(bp::borrowed(dists)));
        values.append(bp::handle<>(bp::borrowed(okay)));

        return bp::incref( bp::tuple( values ).ptr() );
    }
    static PyObject* convert_st(const ST_tuple& c_tuple){
        bp::list values;
        //add all c_tuple items to "values" list

        PyObject * source = VecUIntToArray::convert(c_tuple.get<0>());
        PyObject * target = VecUIntToArray::convert(c_tuple.get<1>());

        values.append(bp::handle<>(bp::borrowed(source)));
        values.append(bp::handle<>(bp::borrowed(target)));

        return bp::incref( bp::tuple( values ).ptr() );
    }
};


#endif