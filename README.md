# Python Point Cloud Partition Algorithms

This repository contains the following point cloud partition algorithms:
* [Voxel Cloud Connectivity Segmentation (VCCS)](https://pcl.readthedocs.io/en/latest/supervoxel_clustering.html)
* [P-Linkage](https://github.com/xiaohulugo/PointCloudSegmentation)
* [Cut-Pursuit](https://github.com/loicland/cut-pursuit)
* [Voxel and Graph-based Segmentation (VGS)](https://github.com/Yusheng-Xu/VGS-SVGS-Segmentation)
* [Supervoxel and Graph-based Segmentation (SVGS)](https://github.com/Yusheng-Xu/VGS-SVGS-Segmentation)
* [Random Sample Consensus (RANSAC) Segmentation](https://pcl.readthedocs.io/projects/tutorials/en/latest/planar_segmentation.html)
* [Region Growing Algorithm](https://pcl-docs.readthedocs.io/en/latest/pcl/doc/tutorials/content/region_growing_segmentation.html)

Some of these algorithms are only available in C++. Therefore, we developed a [Boost.Python](https://www.boost.org/doc/libs/1_72_0/libs/python/doc/html/index.html) interface so that they all can be called from Python. Moreover, we extended the VCCS, P-Linkage, Cut-Pursuit, VGS/SVGS so that the edges of a mesh are used. The extensions produce more accurate eigenfeatures and, thus, sharper edges. All algorithms can be executed on Windows, Linux and MacOS. The algorithms can be seen in action on [YouTube](https://youtu.be/vklkLWeQSwg) where they are integrated in [OpenXtract](https://github.com/mati3230/openxtract).

## Compilation

It is recommended to create a [miniconda](https://docs.conda.io/en/latest/miniconda.html) environment for the compilation of the partition algorithms. Please install the required python packages:

```
pip install -r requirements.txt
```

You should also install Boost with 

```
conda install -c anaconda boost
```

as it is used for the C++/Python interface. Please compile the **libgeo** first (see [README](./libgeo/README.md)) so that the extended algorithms can be tested.

The exact compilation steps for each algorithm can be found in the corresponding subfolder. For instance, the [README](./vccs/README.md)) file in the vccs folder contains the compilation steps of the voxel cloud connectivity segmentation (VCCS) algorithm. The algorithms can be compiled with CMake. The RANSAC and Region Growing algorithm do not require any compilation. The extended algorithms use the libgeo library which approx. the geodesic distances between vertices by calculating the shortest paths. Optionally, it can be compiled and tested in isolation.

## Quickstart

Navigate to a folder of one of the partition algorithms. Edit the test.py script and load an arbitrary .ply file. To do so, search for the line **mesh = o3d.io.read_triangle_mesh** and provide a path to a .ply file on your disk. After that, execute the test.py script. 

## Comparison

Make sure, that the original and the extended partition algorithms can be executed. These algorithms can be compared against each other. To do so, download the complete [ScanNet version 2](http://www.scan-net.org/) dataset. Save it to an arbitrary place on disk. After that, create an environment variable called **SCANNET_DIR** that points to the dataset. The path **SCANNET_DIR/scans** should be a valid path. After that, you can call the script baseline.py:

```
python baseline.py
```

This will create a folder ./csvs_baseline where the accuracies and the partitition sizes will be stored in csv-files. These csv-files have to be concatenated. This can be done with the script concat_csvs.py: 

```
python concat_csvs.py --csv_dir ./csvs_baseline --file baseline.csv
```

After that, you can compute the partition results for the extended partition algorithms:

```
python baseline.py --mesh True
```

This will create a folder ./csvs_mesh. The csv-files in this folder have to be concatenated. This can be done with the script concat_csvs.py: 

```
python concat_csvs.py --csv_dir ./csvs_mesh --file mesh.csv
```

After that, you can start the jupyter notebook **Analyze.ipynb** and execute all cells.

## Citation

If you find this repository useful, please cite:
