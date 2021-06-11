#include <mpi.h>

#include <nvToolsExt.h>
// #include <cuda_profiler_api.h>

#include <vector>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <map>

#include "cuda_runtime.hpp"
#include "csr_mat.hpp"
#include "row_part_spmv.cuh"

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
#define AT __FILE__ ":" TOSTRING(__LINE__) 

//#define VIEW_CHECK_BOUNDS


// mxn random matrix with nnz
CsrMat<Where::host> random_matrix(const int64_t m, const int64_t n, const int64_t nnz) {

    if (m * n < nnz) {
        throw std::logic_error(AT);
    }

    CooMat coo(m,n);
    while(coo.nnz() < nnz) {

        int64_t toPush = nnz - coo.nnz();
        std::cerr << "adding " << toPush << " non-zeros\n";
        for (int64_t _ = 0; _ < toPush; ++_) {
            int r = rand() % m;
            int c = rand() % n;
            float e = 1.0;
            coo.push_back(r, c, e);
        }
        std::cerr << "removing duplicate non-zeros\n";
        coo.remove_duplicates();
    }
    coo.sort();
    std::cerr << "coo: " << coo.num_rows() << "x" << coo.num_cols() << "\n";
    CsrMat<Where::host> csr(coo);
    std::cerr << "csr: " << csr.num_rows() << "x" << csr.num_cols() << " w/ " << csr.nnz() << "\n";
    return csr;
};

// nxn diagonal matrix with bandwidth b
CsrMat<Where::host> random_band_matrix(const int64_t n, const int64_t bw, const int64_t nnz) {

    CooMat coo(n,n);
    while(coo.nnz() < nnz) {

        int64_t toPush = nnz - coo.nnz();
        std::cerr << "adding " << toPush << " non-zeros\n";
        for (int64_t _ = 0; _ < toPush; ++_) {
            int r = rand() % n; // random row

            // column in the band
            int lb = r - bw;
            int ub = r + bw + 1;
            int64_t c = rand() % (ub - lb) + lb;
            if (c < 0 || c > n) {
                continue; // don't over-weight first or last column
            }
            
            float e = 1.0;
            coo.push_back(r, c, e);
        }
        std::cerr << "removing duplicate non-zeros\n";
        coo.remove_duplicates();
    }
    coo.sort();
    std::cerr << "coo: " << coo.num_rows() << "x" << coo.num_cols() << "\n";
    CsrMat<Where::host> csr(coo);
    std::cerr << "csr: " << csr.num_rows() << "x" << csr.num_cols() << " w/ " << csr.nnz() << "\n";
    return csr;
};

std::vector<float> random_vector(const int64_t n) {
    return std::vector<float>(n, 1.0);
}

Array<Where::host, float> random_array(const int64_t n) {
    return Array<Where::host, float>(n, 1.0);
}

#if 0
int send_x(int dst, int src, std::vector<float> &&v, MPI_Comm comm) {
    MPI_Send(v.data(), v.size(), MPI_FLOAT, dst, Tag::x, comm);
    return 0;
}
#endif

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
    int64_t m = 150000;
    int64_t n = m;
    int64_t bw = m/size; // ~50% local vs remote non-zeros for most ranks
    int64_t nnz = 11000000;

    CsrMat<Where::host> A; // "local A"

    // generate and distribute A
    if (0 == rank) {
        std::cerr << "generate matrix\n";
        A = random_band_matrix(m, bw, nnz);
    }

    
    RowPartSpmv spmv(A, 0, MPI_COMM_WORLD);


    std::cerr << "A:        " << A.num_rows()         << "x" << A.num_cols() << " w/ " << A.nnz() << "\n";
    std::cerr << "local A:  " << spmv.lA().num_rows() << "x" << spmv.lA().num_cols() << " w/ " << spmv.lA().nnz() << "\n";
    std::cerr << "remote A: " << spmv.rA().num_rows() << "x" << spmv.rA().num_cols() << " w/ " << spmv.rA().nnz() << "\n";


    int loPrio, hiPrio;
    CUDA_RUNTIME(cudaDeviceGetStreamPriorityRange (&loPrio, &hiPrio));

    cudaStream_t loS, hiS; // "lo/hi prio"
    CUDA_RUNTIME(cudaStreamCreateWithPriority(&loS, cudaStreamNonBlocking, hiPrio));
    CUDA_RUNTIME(cudaStreamCreateWithPriority(&hiS, cudaStreamNonBlocking, hiPrio));

    cudaEvent_t event;
    CUDA_RUNTIME(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));

    const int nIters = 30;
    std::vector<double> times(nIters);

    nvtxRangePush("overlap");
    for (int i = 0; i < nIters; ++i) {
        MPI_Barrier(MPI_COMM_WORLD);
        double start = MPI_Wtime();
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