# Requirements

* cmake
* pcl

# Build

## Linux

Install the PCL: 

```
sudo apt-get install libpcl-dev
```

Build the Project: 

```
conda activate [your python environment]
mkdir build && cd build
cmake . .. -DEIGEN3_INCLUDE_DIR=/usr/include/eigen3 -DPCL_INCLUDE_DIRS=/usr/include/pcl-1.8 -DPCL_LIBRARY_DIRS=/usr/lib/x86_64-linux-gnu -DFLANN_INCLUDE_DIRS=/usr/include/flann
make
```

Quick-command to build and test:

```
cd build && make clean && cmake . .. -DEIGEN3_INCLUDE_DIR=/usr/include/eigen3 -DPCL_INCLUDE_DIRS=/usr/include/pcl-1.8 -DPCL_LIBRARY_DIRS=/usr/lib/x86_64-linux-gnu -DFLANN_INCLUDE_DIRS=/usr/include/flann && make && cd ../ && python test.py
```

## Windows

```
conda activate your_python_env
mkdir build 
cd build
cmake . .. -DEIGEN3_INCLUDE_DIR=D:\Libs\vcpkg\installed\x64-windows\include\eigen3 -DPCL_INCLUDE_DIRS=D:\Libs\vcpkg\installed\x64-windows\include -DPCL_LIBRARY_DIRS=D:\Libs\vcpkg\installed\x64-windows\lib -DFLANN_INCLUDE_DIRS=D:\Libs\vcpkg\installed\x64-windows\include
make
```

Open the project in Visual Studio. Change build version to *release*. Set the project *vccs* as start project. Change the target name of the library from 'vccs' to 'libvccs'. Add the following additional library paths: 
* C:/Users/MIREVI User/miniconda3/envs/blender/libs
* C:/Users/MIREVI User/miniconda3/envs/blender/Library/lib
* C:/Users/MIREVI User/miniconda3/envs/blender/Library/lib/$(Configuration)
* D:/Libs/vcpkg/installed/x64-windows/lib
* D:/Libs/vcpkg/installed/x64-windows/lib/$(Configuration)
Add the following additional libraries: 
* boost_numpy39.lib
* python39.lib
* boost_python39.lib
* pcl_common.lib
* pcl_segmentation.lib
* pcl_kdtree.lib
* pcl_search.lib

Check the linker references. Build the project.   

Make sure you add the path to 

* pcl_common.dll
* pcl_segmentation.dll
* pcl_kdtree.dll
* pcl_search.dll
* pcl_octree.dll
* lz4.dll

to your PATH variable.

Rename 'Release/libvccs.dll' to 'Release/libvccs.pyd'.

## Development Library Setup

* boost_numpy36.lib
* python36.lib
* python39.lib
* boost_python36.lib
* pcl_common.lib
* pcl_segmentation.lib
* pcl_kdtree.lib
* pcl_search.lib

### Problems

#### boost_numpy-cmake is not found by cmake

Navigate to 'C:\Path\to\your\python\env\Library\lib\cmake\cmake' and copy the folder 'boost_numpy-1.73.0' and 'boost_python-1.73.0' into 'C:\Path\to\your\python\env\Library\lib\cmake'.

#### Could not find boost/python.hpp while compiling

Boost python has changed its folder structure for python 3.9. The file python.hpp can be found in boost/python/python.hpp in the boost include folder of your conda environment. Change in visual studio the include directive to '#include <boost/python/python.hpp>' and recompile. Repeat the compilation and renaming until this error disappears. 

### Could not find libboost_numpy.lib in the Linker Stage

Search for the file 'boost_python.lib' in your conda environment. Copy the file and rename it to 'libboost_python.lib'. Repeat this process with all libraries that start with boost_python and boost_numpy. After that, recompile the project. 

## MACOS

Install gcc 7 with [brew](https://brew.sh/) (brew install gcc@7) and use the compiler as default. Replace 'username' appropriately. 

```
conda activate your_python_env
mkdir build 
cd build
cmake . .. -DEIGEN3_INCLUDE_DIR=/Users/username/include/eigen3 -DNUMPY_INCLUDE_DIR=/Users/username/opt/anaconda3/envs/blender/lib/python3.8/site-packages/numpy/core/include -DPCL_INCLUDE_DIRS=/path/to/pcl/include -DPCL_LIBRARY_DIRS=/path/to/pcl/lib -DFLANN_INCLUDE_DIRS=/path/to/flann/include
make
```

Rename 'vccs.dylib' to 'libvccs.so'.

# Original Reference

```
@inproceedings{Papon2013,
address = {Portland, Oregon, USA},
author = {Papon, Jeremie and Abramov, Alexey and Schoeler, Markus and Worgotter, Florentin},
booktitle = {Proceedings of the Conference on Computer Vision and Pattern Recognition - CVPR '13},
doi = {10.1109/CVPR.2013.264},
issn = {10636919},
keywords = {PCL,kinect,pointclouds,segmentation,superpixels},
publisher = {IEEE},
title = {{Voxel Cloud Connectivity Segmentation - Supervoxels for Point Clouds}},
url = {https://ieeexplore.ieee.org/document/6619108},
year = {2013}
}
```