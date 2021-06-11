#include <mpi.h>

#include <nvToolsExt.h>
// #include <cuda_profiler_api.h>

#include <vector>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <map>

//#define VIEW_CHECK_BOUNDS

#include "at.hpp"
#include "cuda_runtime.hpp"
#include "csr_mat.hpp"
#include "row_part_spmv.cuh"


/* recv some amount of data, and put it in the right place
   in a full x
*/
std::vector<float> receive_x(const int n, const int dst, int src, MPI_Comm comm) {
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // which rows of x are local
    Range local = get_partition(n, rank, size);

    // probe for size
    MPI_Status stat;
    MPI_Probe(0, Tag::x, comm, &stat);
    int sz;
    MPI_Get_count(&stat, MPI_INT, &sz);
    if (sz != local.ub-local.lb) {
        throw std::logic_error(AT);
    }

    std::cerr << "recv " << sz << " x entries into offset " << local.lb << "\n";
    std::vector<float> x(n);
    MPI_Recv(x.data() + local.lb, sz, MPI_FLOAT, 0, Tag::x, comm, MPI_STATUS_IGNORE);

    return x;
}



// z += a
__global__ void vector_add(ArrayView<float> z, const ArrayView<float> a) {
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < z.size(); i += blockDim.x * gridDim.x) {
        z(i) += a(i);
    }
}

int main (int argc, char **argv) {

    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
  
    std::cerr << "get a gpu...\n";
    CUDA_RUNTIME(cudaSetDevice(rank % 4));
    CUDA_RUNTIME(cudaFree(0));
    std::cerr << "barrier...\n";
    MPI_Barrier(MPI_COMM_WORLD);

    // int64_t m = 150000;
    // int64_t n = 150000;
    // int64_t nnz = 11000000;
    // or
    int64_t m = 15000;
    int64_t n = m;
    int64_t bw = m/size; // ~50% local vs remote non-zeros for most ranks
    int64_t nnz = 1100000;

    CsrMat<Where::host> A; // "local A"

    // generate and distribute A
    if (0 == rank) {
        std::cerr << "generate matrix\n";
        A = random_band_matrix(m, bw, nnz);
    }

    
    RowPartSpmv spmv(A, 0, MPI_COMM_WORLD);

    if (0 == rank) {
        std::cerr << "A:        " << A.num_rows()         << "x" << A.num_cols() << " w/ " << A.nnz() << "\n";
    }
    std::cerr << "local A:  " << spmv.lA().num_rows() << "x" << spmv.lA().num_cols() << " w/ " << spmv.lA().nnz() << "\n";
    std::cerr << "remote A: " << spmv.rA().num_rows() << "x" << spmv.rA().num_cols() << " w/ " << spmv.rA().nnz() << "\n";

    const int nIters = 1;
    std::vector<double> times(nIters);

    nvtxRangePush("overlap");
    for (int i = 0; i < nIters; ++i) {
        MPI_Barrier(MPI_COMM_WORLD);
        double start = MPI_Wtime();
        spmv.pack_x_async();
        spmv.pack_x_wait();
        spmv.send_x_async();
        spmv.launch_local();
        spmv.recv_x_async();
        spmv.send_x_wait();
        spmv.recv_x_wait();
        spmv.launch_remote();
        spmv.finish();
        times[i] = MPI_Wtime() - start;
    }
    nvtxRangePop(); // one-shot
    MPI_Allreduce(MPI_IN_PLACE, times.data(), times.size(), MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (0 == rank) {
        std::sort(times.begin(), times.end());
        std::cerr << times[times.size() / 2] << "\n";
    }

    MPI_Finalize();

    return 0;
}