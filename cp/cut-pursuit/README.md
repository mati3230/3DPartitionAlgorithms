# Build

You need boost:

```
conda install -c anaconda boost
```

## Linux

```
conda activate your_python_env
mkdir build && cd build
cmake . ..
make
```

Some quick-commands to build and evaluate the code:
```
cd build && make clean && cmake . .. && make && cd ../ && python test.py
cd build && make && cd ../ && python test.py
```

## Windows

```
conda activate your_python_env
mkdir build 
cd build
cmake . ..
```

Open the project in Visual Studio. Check the linker references. Build the project. 

Rename 'Release/cp.dll' to 'Release/libcp.pyd'

### Problems

#### boost_numpy-cmake is not found by cmake

Navigate to 'C:\Path\to\your\python\env\Library\lib\cmake\cmake' and copy the folder 'boost_numpy-1.73.0' and 'boost_python-1.73.0' into 'C:\Path\to\your\python\env\Library\lib\cmake'.

#### Could not find boost/python.hpp while compiling

Boost python has changed its folder structure for python 3.9. The file python.hpp can be found in boost/python/python.hpp in the boost include folder of your conda environment. Change in visual studio the include directive to '#include <boost/python/python.hpp>' and recompile. Repeat the compilation and renaming until this error disappears. 

### Could not find libboost_numpy.lib in the Linker Stage

Search for the file 'boost_python.lib' in your conda environment. Copy the file and rename it to 'libboost_python.lib'. Repeat this process with all libraries that start with boost_python and boost_numpy. After that, recompile the project. 

## Manual Specification of the Dependencies

It also possible to delete the find_package commands in the [./src/CMakeLists](./src/CMakeLists.txt). To do so, rename the [./src/CMakeLists_Manual](./src/CMakeLists_Manual.txt) to ./src/CMakeLists.txt. After that, libraries can be manually specified such as in the following example.
```
cmake . .. -DEIGEN3_INCLUDE_DIR=/usr/include/eigen3 -DPYTHON_LIBRARIES=/home/mati3230/anaconda3/envs/roy/lib/libpython3.9.so -DPYTHON_INCLUDE_DIRS=/home/mati3230/anaconda3/envs/roy/include/python3.9 -DPYTHON_EXECUTABLE=/home/mati3230/anaconda3/envs/roy/bin/python -DPYTHON_NUMPY_INCLUDE_DIR=/home/mati3230/anaconda3/envs/roy/lib/python3.9/site-packages/numpy/core/include -DBoost_INCLUDE_DIRS=/home/mati3230/anaconda3/envs/roy/include -DBoost_LIBRARY_DIRS=/home/mati3230/anaconda3/envs/roy/lib -DPYTHON_MAJOR_VER=3 -DPYTHON_MINOR_VER=9
``` 
Note that you must replace these paths and the python major and minor version according to yout system. 

# Original README of Landrieu et al.

# cut pursuit: a working-set strategy to compute piecewise constant functions on graphs
C/C++ implementation of the L0-cut pursuit algorithms with Matlab and Python interfaces.

![illustration](https://user-images.githubusercontent.com/1902679/34037816-738cf4ba-e18a-11e7-9343-7c27209b27e6.png)

Cut pursuit is a graph-cut-based working-set strategy to minimize functions regularized by graph-structured regularizers.
For _G_ = (_V_, _E_, _w_) a graph with edges weighted by _w_, the problem writes:  

    min<sub>_x_ ∈ _Ω_<sup>_V_</sup></sub>    _f_(_x_) + 
    ∑<sub>(_u_,_v_) ∈ _E_</sub> _w_<sub>(_u_,_v_)</sub>
    _φ_(_x_<sub>_u_</sub> - _x_<sub>_v_</sub>)  

where _Ω_ is the space in which lie the values associated with each node.  

We distinguish two different cases for _φ_, corresponding to different implementations:  
- _φ_: _t_ ↦ |_t_|: the convex case, the regularizer is the __graph total variation__.
Implemented for many different functionals _f_, such as quadratic, ℓ<sub>1</sub>-norm, box constraints, simplex constraints, linear, smoothed Kullback–Leibler.
See repository [CP_PFDR_graph_d1](https://github.com/1a7r0ch3/CP_PFDR_graph_d1), by Hugo Raguet. It is well-suited for regularization and inverse problems with a low total variation prior.
- _φ_: _t_ ↦ _δ_(_t_ ≠ 0) = 1 - _δ_<sub>0</sub>(t): the nonconvex case, the regularizer is the weight of the cut between the adjacent constant components. It is well-suited for segmentation/partitioning tasks. This repository corresponds to this problem.

 Current implementation supports the following fidelity functions:

- quadratic fidelity: _φ_: _x_ ↦ ∑<sub>_v_ in _V_</sub>||_x_<sub>_v_</sub> - _y_<sub>_v_</sub>||² with y an observed value associated with node _v_ (best for partitioning)
- linear fidelity: _φ_: _x_ ↦ - ∑<sub>_v_ in _V_</sub><_x_<sub>_v_</sub>, _y_<sub>_v_</sub>> with _y_<sub>_v_</sub> a weight associated with node _v_
- Kullback leibler fidelity _φ_: _x_ ↦ ∑<sub>_v_ in _V_</sub> KL(_x_<sub>_v_</sub>, _p_<sub>_v_</sub>) with _p_<sub>_v_</sub> a probability associated with node _v_. Only apply when _Ω_ is a simplex 

# Requirement

You need boost 1.58, or 1.65 if you want the python wrapper.

```conda install -c anaconda boost```

# Compilation

### C++
make sure that you use the following CPPFLAGS: 
```set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -pthread -fopenmp -O3 -Wall -std=c++11")```

add ```include<./cut-pursuit/include/API.h>``` and call any of the interface functions.

### MATLAB
To compile the MATLAB mex file type the following in MATLAB in the workspace containing the ```cut-pursuit``` folder:

```
mkdir ./cut-pursuit/bin
addpath('./cut-pursuit/bin/')
mex CXXFLAGS="\$CXXFLAGS -pthread -Wall -std=c++11 -fopenmp -O3"...
    LDFLAGS="\$LDFLAGS -fopenmp" cut-pursuit/mex/L0_cut_pursuit.cpp ...
    -output cut-pursuit/bin/L0_cut_pursuit
mex CXXFLAGS="\$CXXFLAGS -pthread -Wall -std=c++11 -fopenmp -O3"...
    LDFLAGS="\$LDFLAGS -fopenmp" cut-pursuit/mex/L0_cut_pursuit_segmentation.cpp ...
    -output cut-pursuit/bin/L0_cut_pursuit_segmentation
```

You can test the compilation with the following minimal example:

```
n_nodes = 100;
y = rand(3,n_nodes);
Eu = 0:(n_nodes-2);
Ev = 1:(n_nodes-1);
edge_weight = ones(numel(Eu),1);
node_weight = ones(n_nodes,1);
lambda = .1;
mode = 1;
cutoff = 0;
weigth_decay = 0;
speedmode = 2;
verbosity = 2;

[solution, in_component, components] = L0_cut_pursuit_segmentation(single(y),...
        uint32(Eu), uint32(Ev), single(lambda),...
        single(edge_weight), single(node_weight), mode, cutoff, speedmode,...
        weigth_decay, verbosity);

subplot(3,1,1)
imagesc(repmat(y, [1 1 1]))
title('input data')
subplot(3,1,2)
imagesc(solution)
title('piecewise constant approximation')
subplot(3,1,3)
imagesc(in_component')
title('components')

```

### Python
Compile the library from the ```cut-pursuit``` folder
```
mkdir build && cd build
cmake .. -DPYTHON_LIBRARY=$CONDAENV/lib/libpython3.6m.so -DPYTHON_INCLUDE_DIR=$CONDAENV/include/python3.6m -DBOOST_INCLUDEDIR=$CONDAENV/include  -DEIGEN3_INCLUDE_DIR=$CONDAENV/include/eigen3
make
```
This creates ```build/src/libcp.so``` which can be imported in python. see ```test.py``` to test it out.

# References:
Cut Pursuit: fast algorithms to learn piecewise constant functions on general weighted graphs,
L. Landrieu and G. Obozinski, SIAM Journal on Imaging Science 2017, Vol. 10, No. 4 : pp. 1724-1766
[[hal link]](https://hal.archives-ouvertes.fr/hal-01306779)

Cut-pursuit algorithm for nonsmooth functionals regularized by graph total variation, H. Raguet and L. Landrieu, in preparation. 

if using the L0-cut pursuit algorithm with \Omega other than R, one must also cite:

A structured regularization framework for spatially smoothing semantic labelings of 3D point clouds. Loic Landrieu, Hugo Raguet , Bruno Vallet , Clément Mallet, Martin Weinmann
