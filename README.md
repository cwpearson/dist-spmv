# dist-spmv

**vortex**
```
module --force purge
module load StdEnv
module load xl/2021.03.11
module load cuda/10.1.243
module load spectrum-mpi/rolling-release
module load cmake/3.18.0
```

**ascicgpu030**

To get nsight, had to download the rpm and unpack into home directory with `cpio`.

```
module purge
module load sierra-devel/nvidia
module load cde/v2/cmake/3.19.2
```

```
mpirun -n 2 ~/software/nsight-systems-cli/2021.2.1/bin/nsys profile -c cudaProfilerApi -t cuda,mpi,nvtx -o dist-spmv_%q{OMPI_COMM_WORLD_RANK} -f true ./main
```

### To build with OpenMPI 4.1.1

```
module purge
module load sems-env
module load sems-cmake/3.19.1
module load sems-gcc/7.2.0
module load sems-cuda/10.1
cmake .. -DCMAKE_PREFIX_PATH=~cwpears/software/openmpi-4.1.1-cuda10.1-gcc7.2
```

```
~cwpears/software/openmpi-4.1.1-cuda10.1-gcc7.2/bin/mpirun -n 2 ./overlap

```

## Design Considerations

Minimize CUDA runtime calls