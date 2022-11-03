# Requirements

* cmake
* eigen
* flann
* pcl

# Build

Note that you may have to change the paths to the libraries. 

## Linux

Install the Boost and the PCL: 

```
conda install -c jessemapel pcl
```

Build the Project: 

```
conda activate [your python environment]
mkdir build && cd build
cmake . .. -DEIGEN3_INCLUDE_DIR=/usr/include/eigen3 -DPCL_INCLUDE_DIRS=/usr/include/pcl-1.8 -DPCL_LIBRARY_DIRS=/usr/lib/x86_64-linux-gnu -DFLANN_INCLUDE_DIRS=/usr/include/flann
make
```

```
cd build && make clean && cmake . .. -DEIGEN3_INCLUDE_DIR=/usr/include/eigen3 -DPCL_INCLUDE_DIRS=/usr/include/pcl-1.8 -DPCL_LIBRARY_DIRS=/usr/lib/x86_64-linux-gnu -DFLANN_INCLUDE_DIRS=/usr/include/flann && make && cd ../ && python test.py
```

## Windows

Download Eigen3, FLANN and the PCL with [vcpkg](https://vcpkg.io/en/index.html). The path **Disk:\path\to\vcpkg\installed\x64-windows\include\eigen3** should be valid.

```
conda activate your_python_env
mkdir build 
cd build
cmake . .. -DEIGEN3_INCLUDE_DIR=D:\Libs\vcpkg\installed\x64-windows\include\eigen3 -DPCL_INCLUDE_DIRS=D:\Libs\vcpkg\installed\x64-windows\include -DPCL_LIBRARY_DIRS=D:\Libs\vcpkg\installed\x64-windows\lib -DFLANN_INCLUDE_DIRS=D:\Libs\vcpkg\installed\x64-windows\include
```

Open the project in Visual Studio. Change build version to *release*. Set the project *vgs* as start project. Change the target name of the library from 'vgs' to 'libvgs'. Add the following additional library paths:

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

Rename 'Release/libvgs.dll' to 'Release/libvgs.pyd'.

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

Rename 'vgs.dylib' to 'libvgs.so'.

# Original References

```
@article{Xu2017,  
  title={Geometric primitive extraction from point clouds of construction sites using VGS},  
  author={Xu, Yusheng and Tuttas, Sebastian and Hoegner, Ludwig and Stilla, Uwe},  
  journal={IEEE Geoscience and Remote Sensing Letters},    
  volume={14},  
  number={3},  
  pages={424--428},  
  year={2017},  
  publisher={IEEE}  
}
@article{Xu2018,
  title={A voxel- and graph-based strategy for segmenting man-made infrastructures using perceptual grouping laws: comparison and evaluation},
  author={Xu, Yusheng and Hoegner, Ludwig and Tuttas, Sebastian and Stilla, Uwe},
  journal={Photogrammetric Engineering \& Remote Sensing},
  volume={84},
  number={6},
  pages={377--391},
  year={2018},
  publisher={American Society for Photogrammetry and Remote Sensing}
}
```