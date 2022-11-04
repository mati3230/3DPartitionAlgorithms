# Requirements

* cmake
* boost python

# Build

```
conda install -c anaconda boost
```

## Linux

```
conda activate your_python_env
mkdir build && cd build
cmake . .. -DEIGEN3_INCLUDE_DIR=/usr/include/eigen3
make
```

```
cd build && make clean && cmake . .. -DEIGEN3_INCLUDE_DIR=/usr/include/eigen3 && make && cd ../ && python test.py
cd build && make && cd ../ && python test.py
```

## Windows

```
conda activate your_python_env
mkdir build 
cd build
cmake . .. -DEIGEN3_INCLUDE_DIR=D:\Libs\vcpkg\installed\x64-windows\include\eigen3
```

Open the project in Visual Studio. Check the linker references. Build the project. 

Rename 'Release/ply_c.dll' to 'Release/libply_c.pyd'

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
cmake . .. -DEIGEN3_INCLUDE_DIR=/Users/username/include/eigen3 -DNUMPY_INCLUDE_DIR=/Users/username/opt/anaconda3/envs/blender/lib/python3.8/site-packages/numpy/core/include
make
```

Rename 'geo.dylib' to 'libgeo.so'.

## Manual Specification of the Dependencies

It also possible to delete the find_package commands in the [CMakeLists](./CMakeLists.txt). To do so, rename the [CMakeLists_Manual](./CMakeLists_Manual.txt) to CMakeLists.txt. After that, libraries can be manually specified such as in the following example.
```
cmake . .. -DEIGEN3_INCLUDE_DIR=/usr/include/eigen3 -DPYTHON_LIBRARIES=/home/mati3230/anaconda3/envs/roy/lib/libpython3.9.so -DPYTHON_INCLUDE_DIRS=/home/mati3230/anaconda3/envs/roy/include/python3.9 -DPYTHON_EXECUTABLE=/home/mati3230/anaconda3/envs/roy/bin/python -DPYTHON_NUMPY_INCLUDE_DIR=/home/mati3230/anaconda3/envs/roy/lib/python3.9/site-packages/numpy/core/include -DBoost_INCLUDE_DIRS=/home/mati3230/anaconda3/envs/roy/include -DBoost_LIBRARY_DIRS=/home/mati3230/anaconda3/envs/roy/lib -DPYTHON_MAJOR_VER=3 -DPYTHON_MINOR_VER=9
``` 
Note that you must replace these paths and the python major and minor version according to yout system. 