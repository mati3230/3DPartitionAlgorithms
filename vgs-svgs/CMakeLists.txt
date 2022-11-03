cmake_minimum_required(VERSION 3.5)
project(VGS)

set (CMAKE_CXX_STANDARD 11)
if(APPLE)
    message("Using Apple Configuration")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall -std=c++11 -stdlib=libc++ -DBUILD_SHARED_LIBS=OFF")
else() # Win32 and Linux
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++11")
endif()

find_package(OpenMP)
if (${OpenMP_CXX_FOUND})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    add_definitions(-DOPENMP)
endif()

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_MODULE_PATH})

find_package(PythonLibs)
find_package(PythonInterp)
find_package(NumPy 1.5 REQUIRED)

# Find Boost
if(APPLE)
    find_package(Boost 1.73.0 REQUIRED) #system filesystem thread serialization
    if (${Boost_MINOR_VERSION} LESS 73 )
        find_package(Boost 1.73.0 COMPONENTS numpy${PYTHON_VERSION_MAJOR} REQUIRED) #system filesystem thread serialization
    else()
        set(PYTHONVERSION ${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR})
        find_package(Boost 1.73.0 COMPONENTS numpy${PYTHONVERSION} REQUIRED)
    endif()
else() # Win32 and Linux
    find_package(Boost 1.65.1 REQUIRED) #system filesystem thread serialization
    if (${Boost_MINOR_VERSION} LESS 67 )
        find_package(Boost 1.65.1 COMPONENTS numpy${PYTHON_VERSION_MAJOR} REQUIRED) #system filesystem thread serialization
    else()
        set(PYTHONVERSION ${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR})
        find_package(Boost 1.67.0 COMPONENTS numpy${PYTHONVERSION} REQUIRED)
    endif()
endif()

message("Python libraries are " ${PYTHON_LIBRARIES})
message("Python include dirs are " ${PYTHON_INCLUDE_DIRS})

message("Boost includes are " ${Boost_INCLUDE_DIRS})
message("Boost libraries dirs are " ${Boost_LIBRARY_DIRS})
#list(APPEND Boost_LIBRARIES ${Boost_LIBRARY_DIRS}/libboost_python3.so ${Boost_LIBRARY_DIRS}/libboost_numpy3.so)
#set(Boost_LIBRARIES ${Boost_LIBRARY_DIRS}/libboost_python3.so ${Boost_LIBRARY_DIRS}/libboost_numpy3.so ${Boost_LIBRARY_DIRS}/libboost_system.so ${Boost_LIBRARY_DIRS}/libboost_filesystem.so ${Boost_LIBRARY_DIRS}/libboost_thread.so ${Boost_LIBRARY_DIRS}/libboost_serialization.so)
message("Boost libraries are " ${Boost_LIBRARIES})

# Include and link Boost
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

# Include Eigen
include_directories(${EIGEN3_INCLUDE_DIR})
message("EIGEN3 libraries dir is " ${EIGEN3_INCLUDE_DIR})

# Include and link Python and NumPy
if(APPLE)
    ##############################################TODO#####################################################
    # replace username appropriately
    #set(NUMPY_INCLUDE_DIR /Users/username/opt/anaconda3/envs/blender/lib/python3.8/site-packages/numpy/core/include/)
    include_directories(${NUMPY_INCLUDE_DIR})
    include_directories(${PYTHON_INCLUDE_DIRS})
else() # Win32 and Linux
    include_directories(${PYTHON_INCLUDE_DIRS} ${PYTHON_NUMPY_INCLUDE_DIR})
endif()
link_directories(${PYTHON_LIBRARY_DIRS})

include_directories(${PCL_INCLUDE_DIRS} ${FLANN_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
set(PCL_LIBRARIES 
${PCL_LIBRARY_DIRS}/libpcl_common.so
#${PCL_LIBRARY_DIRS}/libflann.so
${PCL_LIBRARY_DIRS}/libflann_cpp.so
#${PCL_LIBRARY_DIRS}/libpcl_segmentation.so
#${PCL_LIBRARY_DIRS}/libpcl_kdtree.so
${PCL_LIBRARY_DIRS}/libpcl_search.so
#${PCL_LIBRARY_DIRS}/libpcl_io.so
#${PCL_LIBRARY_DIRS}/libpcl_octree.so
#${PCL_LIBRARY_DIRS}/libpcl_sample_consensus.so
#${PCL_LIBRARY_DIRS}/libpcl_filters.so
#${PCL_LIBRARY_DIRS}/libpcl_features.so
#${PCL_LIBRARY_DIRS}/libpcl_ml.so
#${PCL_LIBRARY_DIRS}/libpcl_visualization.so
#${PCL_LIBRARY_DIRS}/libpcl_surface.so
#${PCL_LIBRARY_DIRS}/libpcl_registration.so
#${PCL_LIBRARY_DIRS}/libpcl_keypoints.so
#${PCL_LIBRARY_DIRS}/libpcl_tracking.so
#${PCL_LIBRARY_DIRS}/libpcl_recognition.so
#${PCL_LIBRARY_DIRS}/libpcl_stereo.so
#${PCL_LIBRARY_DIRS}/libpcl_apps.so
#${PCL_LIBRARY_DIRS}/libpcl_outofcore.so
#${PCL_LIBRARY_DIRS}/libpcl_people.so
)
#add_definitions(${PCL_DEFINITIONS})
#add_definitions( -DPCL_NO_PRECOMPILE )

message("PCL includes ARE " ${PCL_INCLUDE_DIRS})
message("FLANN includes ARE " ${FLANN_INCLUDE_DIRS})
message("PCL LIBRARIES ARE " ${PCL_LIBRARY_DIRS})
message("PCL libraries ARE " ${PCL_LIBRARIES})

add_library(vgs SHARED vgs_svgs.cpp)
target_link_libraries(vgs
    ${Boost_LIBRARIES}
    ${PYTHON_LIBRARIES}
    ${PCL_LIBRARIES}
    )