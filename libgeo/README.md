# Build

## Linux

```
conda activate your_python_env
mkdir build && cd build
cmake . .. -DEIGEN3_INCLUDE_DIR=/usr/include/eigen3
make
```

Some quick-commands to build and evaluate the code:
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

Open the project in Visual Studio. Change build version to *release*. Set the project *geo* as start project. Change the target name of the library from 'geo' to 'libgeo'. Add the following additional library paths: 
* C:/Users/MIREVI User/miniconda3/envs/blender/libs
* C:/Users/MIREVI User/miniconda3/envs/blender/Library/lib
* C:/Users/MIREVI User/miniconda3/envs/blender/Library/lib/$(Configuration)
Add the following additional libraries: 
* boost_numpy39.lib
* python39.lib
* boost_python39.lib
Check the linker references. Build the project. 

Rename 'Release/libgeo.dll' to 'Release/libgeo.pyd'.

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

## Test

```
python test.py
```