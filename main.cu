#include <mpi.h>

#include <nvToolsExt.h>
// #include <cuda_profiler_api.h>

#include <vector>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <iostream>

#include "cuda_runtime.hpp"

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
#define AT __FILE__ ":" TOSTRING(__LINE__) 

//#define VIEW_CHECK_BOUNDS

template<typename ForwardIt>
void shift_left(ForwardIt first, ForwardIt last, size_t n) {
    while(first != last) {
        *(first-n) = *first;
        ++first;
    }
}

enum Tag : int {
    row_ptr,
    col_ind,
    val,
    x,
    num_cols
};

enum class Where {
    host,
    device
};

template <Where where, typename T>
class Array;


// A non-owning view of data
template <typename T>
struct ArrayView
{
    T *data_;
    int64_t size_;
    public:
    ArrayView() : data_(nullptr), size_(0){}
    ArrayView(const ArrayView &other) = default;
    ArrayView(ArrayView &&other) = default;
    ArrayView &operator=(const ArrayView &rhs) = default;

    __host__ __device__ int64_t size() const { return size_; }

    __host__ __device__ const T &operator()(int64_t i) const {
#ifdef VIEW_CHECK_BOUNDS
        if (i < 0) {
            printf("ERR: i < 0: %d\n", i);
        }
        if (i >= size_) {
            printf("ERR: i > size_: %d > %ld\n", i, size_);
        }
#endif
        return data_[i];
    }
    __host__ __device__ T &operator()(int64_t i) {
        return data_[i];
    }
};

/* device array
*/
template<typename T> class Array<Where::device, T>
{
public:

    // array owns the data in this view
    ArrayView<T> view_;
public:
    Array() = default;
    Array(const size_t n) {
        resize(n);
    }
    Array(const Array &other) = delete;
    Array(Array &&other) : view_(other.view_) {
        // view is non-owning, so have to clear other
        other.view_.data_ = nullptr;
        other.view_.size_ = 0;
    }

    Array(const std::vector<T> &v) {
        set_from(v);
    }

    ~Array() {
        CUDA_RUNTIME(cudaFree(view_.data_));
        view_.data_ = nullptr;
        view_.size_ = 0;
    }
    int64_t size() const { 
        return view_.size(); }

    ArrayView<T> view() const {
        return view_; // copy of internal view
    }

    operator std::vector<T>() const {
        std::vector<T> v(size());
        CUDA_RUNTIME(cudaMemcpy(v.data(), view_.data_, size() * sizeof(T), cudaMemcpyDeviceToHost));
        return v;
    }

    void set_from(const std::vector<T> &rhs, cudaStream_t stream = 0) {
        resize(rhs.size());
        CUDA_RUNTIME(cudaMemcpyAsync(view_.data_, rhs.data(), view_.size_ * sizeof(T), cudaMemcpyHostToDevice, stream));
    }

    void set_from(const Array<Where::host, T> &rhs, cudaStream_t stream = 0) {
        resize(rhs.size());
        CUDA_RUNTIME(cudaMemcpyAsync(view_.data_, rhs.data(), view_.size_ * sizeof(T), cudaMemcpyHostToDevice, stream));
    }

    // any change destroys all data
    void resize(size_t n) {
        if (size() != n) {
            view_.size_ = n;
            CUDA_RUNTIME(cudaFree(view_.data_));
            CUDA_RUNTIME(cudaMalloc(&view_.data_, view_.size_ * sizeof(T)));
        }
    }

};


/* host array
*/
template<typename T> class Array<Where::host, T>
{
public:

    // array owns the data in this view
    ArrayView<T> view_;
public:
    Array() = default;
    Array(const size_t n, const T &val) {
        resize(n);
        for (size_t i = 0; i < n; ++i) {
            view_(i) = val;
        }
    }
    Array(const Array &other) = delete;
    Array(Array &&other) : view_(other.view_) {
        // view is non-owning, so have to clear other
        other.view_.data_ = nullptr;
        other.view_.size_ = 0;
    }

    ~Array() {
        CUDA_RUNTIME(cudaFreeHost(view_.data_));
        view_.data_ = nullptr;
        view_.size_ = 0;
    }
    int64_t size() const { 
        return view_.size(); }

    ArrayView<T> view() const {
        return view_; // copy of internal view
    }

    // any change destroys all data
    void resize(size_t n) {
        if (size() != n) {
            view_.size_ = n;
            CUDA_RUNTIME(cudaFreeHost(view_.data_));
            CUDA_RUNTIME(cudaHostAlloc(&view_.data_, view_.size_ * sizeof(T), cudaHostAllocDefault));
        }
    }

    const T* data() const {
        return view_.data_;
    }
    T* data() {
        return view_.data_;
    }

};


class CooMat {
public:


    struct Entry {
        int i;
        int j;
        float e;

        Entry(int _i, int _j, int _e) : i(_i), j(_j), e(_e) {}

        static bool by_ij(const Entry &a, const Entry &b) {
            if (a.i < b.i) {
                return true;
            } else if (a.i > b.i) {
                return false;
            } else {
                return a.j < b.j;
            }
        }

        static bool same_ij(const Entry &a, const Entry &b) {
            return a.i == b.i && a.j == b.j;
        }
    };

private:

    // sorted during construction
    std::vector<Entry> data_;
    int64_t numRows_;
    int64_t numCols_;

public:
    CooMat(int m, int n) : numRows_(m), numCols_(n) {}
    const std::vector<Entry> &entries() const {return data_;}
    void push_back(int i, int j, int e) {
        data_.push_back(Entry(i,j,e));  
    }

    void sort() {
        std::sort(data_.begin(), data_.end(), Entry::by_ij);
    }

    void remove_duplicates() {
        std::sort(data_.begin(), data_.end(), Entry::by_ij);
        std::unique(data_.begin(), data_.end(), Entry::same_ij);
    }

    int64_t num_rows() const {return numRows_;}
    int64_t num_cols() const {return numCols_;}
    int64_t nnz() const {return data_.size();}
};

template <Where where>
class CsrMat {
public:
    CsrMat();
    int64_t nnz() const;
    int64_t num_rows() const;
};
template<> class CsrMat<Where::host>;
template<> class CsrMat<Where::device>;

/* host sparse matrix */
template<> class CsrMat<Where::host>
{
    friend class CsrMat<Where::device>; // device can see inside
    std::vector<int> rowPtr_;
    std::vector<int> colInd_;
    std::vector<float> val_;
    int64_t numCols_;

public:
    CsrMat() = default;
    CsrMat(int numRows, int numCols, int nnz) : rowPtr_(numRows+1), colInd_(nnz), val_(nnz), numCols_(numCols) {}

    CsrMat(const CooMat &coo) : numCols_(coo.num_cols()) {
        for (auto &e : coo.entries()) {
            while (rowPtr_.size() <= e.i) {
                rowPtr_.push_back(colInd_.size());
            }
            colInd_.push_back(e.j);
            val_.push_back(e.e);
        }
        while (rowPtr_.size() < coo.num_rows()+1){
            rowPtr_.push_back(colInd_.size());
        }
    }

    int64_t num_rows() const {
      if (rowPtr_.size() <= 1) {
        return 0; 
      } else { 
        return rowPtr_.size() - 1;
      }
    }

    int64_t num_cols() const {
        return numCols_;
      }

    int64_t nnz() const {
        if (colInd_.size() != val_.size()) {
            throw std::logic_error("bad invariant");
        }
        return colInd_.size();
    }

    const int &row_ptr(int64_t i) const {
        return rowPtr_[i];
    }
    const int &col_ind(int64_t i) const {
        return colInd_[i];
    }
    const float &val(int64_t i) const {
        return val_[i];
    }

    const int *row_ptr() const {return rowPtr_.data(); }
    int *row_ptr() {return rowPtr_.data(); }
    const int *col_ind() const {return colInd_.data(); }
    int *col_ind() {return colInd_.data(); }
    const float *val() const {return val_.data(); }
    float *val() {return val_.data(); }

    /* keep rows [rowStart, rowEnd)
    */
    void retain_rows(int rowStart, int rowEnd) {
        
        if (0 == rowEnd) {
            throw std::logic_error("unimplemented");
        }
        // erase rows after
        // dont want to keep rowEnd, so rowEnd points to end of rowEnd-1
        std::cerr << "rowPtr_ = rowPtr[:" << rowEnd+1 << "]\n";
        rowPtr_.resize(rowEnd+1);
        std::cerr << "resize entries to " << rowPtr_.back() << "\n";
        colInd_.resize(rowPtr_.back());
        val_.resize(rowPtr_.back());

        // erase early row pointers
        std::cerr << "rowPtr <<= " << rowStart << "\n";
        shift_left(rowPtr_.begin()+rowStart, rowPtr_.end(), rowStart);
        std::cerr << "resize rowPtr to " << rowEnd - rowStart+1 << "\n";
        rowPtr_.resize(rowEnd-rowStart+1);

        const int off = rowPtr_[0];
        // erase entries for first rows
        std::cerr << "entries <<= " << off << "\n";
        shift_left(colInd_.begin()+off, colInd_.end(), off);
        shift_left(val_.begin()+off, val_.end(), off);

        // adjust row pointer offset
        std::cerr << "subtract rowPtrs by " << off << "\n";
        for (auto &e : rowPtr_) {
            e -= off;
        }

        // resize entries
        std::cerr << "resize entries to " << rowPtr_.back() << "\n";
        colInd_.resize(rowPtr_.back());
        val_.resize(rowPtr_.back());
    }

};

/* device sparse matrix
*/
template<> class CsrMat<Where::device>
{
    Array<Where::device, int> rowPtr_;
    Array<Where::device, int> colInd_;
    Array<Where::device, float> val_;

public:

    struct View {
        ArrayView<int> rowPtr_;
        ArrayView<int> colInd_;
        ArrayView<float> val_;

        __device__ int num_rows() const {
            if (rowPtr_.size() > 0) {
                return rowPtr_.size() - 1;
            } else {
                return 0;
            }
        }

        __device__ const int &row_ptr(int64_t i) const {
            return rowPtr_(i);
        }

        __device__ const int &col_ind(int64_t i) const {
            return colInd_(i);
        }

        __device__ const float &val(int64_t i) const {
            return val_(i);
        }

    };

    CsrMat() = delete;
    CsrMat(CsrMat &&other) = delete;
    CsrMat(const CsrMat &other) = delete;

    // create device matrix from host
    CsrMat(const CsrMat<Where::host> &m) : 
        rowPtr_(m.rowPtr_), colInd_(m.colInd_), val_(m.val_) {
        if (colInd_.size() != val_.size()) {
            throw std::logic_error("bad invariant");
        }
    }
    ~CsrMat() {
    }
    int64_t num_rows() const {
        if (rowPtr_.size() <= 1) {
            return 0; 
          } else { 
            return rowPtr_.size() - 1;
          }
    }
  
    int64_t nnz() const {
        return colInd_.size();
    }

    View view() const {
        View v;
        v.rowPtr_ = rowPtr_.view();
        v.colInd_ = colInd_.view();
        v.val_ = val_.view();
        return v;
    }

};




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
            if (c < 0 || c >= n) {
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

struct Range {
    int lb;
    int ub;
};

/* get the ith part of splitting domain in to n pieces
*/
Range get_partition(const int domain, const int i, const int n) {
    int div = domain / n;
    int rem = domain % n;

    int lb, ub;

    if (i < rem) {
        lb = i * (div+1);
        ub = lb + (div+1);
    } else {
        lb = rem * (div+1) + (i-rem) * div;
        ub = lb + div;
    }
    return Range{.lb=lb, .ub=ub};
}

std::vector<CsrMat<Where::host>> part_by_rows(const CsrMat<Where::host> &m, const int parts) {

    std::vector<CsrMat<Where::host>> mats;

    for (int p = 0; p < parts; ++p) {
        Range range = get_partition(m.num_rows(), p, parts);
        std::cerr << "matrix part " << p << " has " << range.ub-range.lb << " rows\n";
        CsrMat<Where::host> part(m);
        part.retain_rows(range.lb, range.ub);
        mats.push_back(part);
    }

    return mats;
}

struct DistMat {
    CsrMat<Where::host> local;
    CsrMat<Where::host> remote;
};

DistMat split_local_remote(const CsrMat<Where::host> &m, MPI_Comm comm) {
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // which rows of x are local
    Range localRange = get_partition(m.num_cols(), rank, size);

    // build two matrices, local gets local non-zeros, remote gets remote non-zeros
    CooMat local(m.num_rows(), m.num_cols()), remote(m.num_rows(), m.num_cols());

    for (int r = 0; r < m.num_rows(); ++r) {
        for (int ci = m.row_ptr(r); ci < m.row_ptr(r+1); ++ci) {
            int c = m.col_ind(ci);
            float v = m.val(ci);

            if (c >= localRange.lb && c < localRange.ub) {
                local.push_back(r,c,v);
            } else {
                remote.push_back(r,c,v);
            }

        }
    }

    return DistMat {
        .local=local,
        .remote=remote
    };

}


std::vector<std::vector<float>> part_by_rows(const std::vector<float> &x, const int parts) {
    std::vector<std::vector<float>> xs;

    for (int p = 0; p < parts; ++p) {
        Range range = get_partition(x.size(), p, parts);
        std::cerr << "vector part " << p << " will have " << range.ub-range.lb << " rows\n";
        std::vector<float> part(x.begin()+range.lb, x.begin()+range.ub);
        xs.push_back(part);
    }

    if (xs.size() != parts) {
        throw std::logic_error("line " STRINGIFY(__LINE__));
    }
    return xs;
}

int send_matrix(int dst, int src, CsrMat<Where::host> &&m, MPI_Comm comm) {

    MPI_Request reqs[4];

    int numCols = m.num_cols();
    MPI_Isend(&numCols, 1, MPI_INT, dst, Tag::num_cols, comm, &reqs[0]);
    MPI_Isend(m.row_ptr(), m.num_rows()+1, MPI_INT, dst, Tag::row_ptr, comm, &reqs[1]);
    MPI_Isend(m.col_ind(), m.nnz(), MPI_INT, dst, Tag::col_ind, comm, &reqs[2]);
    MPI_Isend(m.val(), m.nnz(), MPI_FLOAT, dst, Tag::val, comm, &reqs[3]);
    MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

    return 0;
}

CsrMat<Where::host> receive_matrix(int dst, int src, MPI_Comm comm) {

    int numCols;
    MPI_Recv(&numCols, 1, MPI_INT, 0, Tag::num_cols, comm, MPI_STATUS_IGNORE);

    // probe for number of rows
    MPI_Status stat;
    MPI_Probe(0, Tag::row_ptr, comm, &stat);
    int numRows;
    MPI_Get_count(&stat, MPI_INT, &numRows);
    if (numRows > 0) {
        --numRows;
    }

    // probe for nnz
    MPI_Probe(0, Tag::col_ind, comm, &stat);
    int nnz;
    MPI_Get_count(&stat, MPI_INT, &nnz);

    std::cerr << "recv " << numRows << "x" << numCols << " w/ " << nnz << "\n";
    CsrMat<Where::host> csr(numRows, numCols, nnz);

    // receive actual data into matrix
    MPI_Recv(csr.row_ptr(), numRows+1, MPI_INT, 0, Tag::row_ptr, comm, MPI_STATUS_IGNORE);
    MPI_Recv(csr.col_ind(), nnz, MPI_INT, 0, Tag::col_ind, comm, MPI_STATUS_IGNORE);
    MPI_Recv(csr.val(), nnz, MPI_FLOAT, 0, Tag::val, comm, MPI_STATUS_IGNORE);

    return csr;
}

int send_x(int dst, int src, std::vector<float> &&v, MPI_Comm comm) {
    MPI_Send(v.data(), v.size(), MPI_FLOAT, dst, Tag::x, comm);
    return 0;
}

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

enum class ProductConfig {
    MODIFY, // b += 
    SET     // b =
};

/* Ax=b
*/
__global__ void spmv(ArrayView<float> b,
     const CsrMat<Where::device>::View A,
      const ArrayView<float> x,
      const ProductConfig pc
    ) {
    // one thread per row
    for (int r = blockDim.x * blockIdx.x + threadIdx.x; r < A.num_rows(); r += blockDim.x * gridDim.x) {
        float acc = 0;
        for (int ci = A.row_ptr(r); ci < A.row_ptr(r+1); ++ci) {
            int c = A.col_ind(ci);
            acc += A.val(ci) * x(c);
        }
        if (ProductConfig::SET == pc) {
            b(r) = acc;
        } else {
            b(r) += acc;
        }
    }
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

    CsrMat<Where::host> lA; // "local A"

    // generate and distribute A
    if (0 == rank) {
        std::cerr << "generate matrix\n";
        // lA = random_matrix(m, n, nnz);
        lA = random_band_matrix(m, bw, nnz);
        std::cerr << "partition matrix\n";
        std::vector<CsrMat<Where::host>> As = part_by_rows(lA, size);
        for (size_t dst = 1; dst < size; ++dst) {
            std::cerr << "send A to " << dst << "\n";
            send_matrix(dst, 0, std::move(As[dst]), MPI_COMM_WORLD);
        }
        lA = As[rank];
    } else {
        std::cerr << "recv A at " << rank << "\n";
        lA = receive_matrix(rank, 0, MPI_COMM_WORLD);
    }

    // each rank has a dense x. each rank owns part of it,
    // but it doesn't matter what the entries are
    Array<Where::host, float> lx = random_array(n); // "local x"
    std::cerr << "local X: " << lx.size() << "\n";
    std::cerr << "copy x to device\n";
    Array<Where::device, float> lxd(lx.size()), rxd(lx.size()); // "local/remote x device"

    // get a local and remote split of A
    std::cerr << "split local/remote A\n";
    CsrMat<Where::host> rA, A(lA);
    {
        DistMat d = split_local_remote(lA, MPI_COMM_WORLD);
        lA = d.local;
        rA = d.remote;
    }
    std::cerr << "A:        " << A.num_rows() << "x" << A.num_cols() << " w/ " << A.nnz() << "\n";
    std::cerr << "local A:  " << lA.num_rows() << "x" << lA.num_cols() << " w/ " << lA.nnz() << "\n";
    std::cerr << "remote A: " << rA.num_rows() << "x" << rA.num_cols() << " w/ " << rA.nnz() << "\n";

    std::cerr << "Copy A to GPU\n";
    CsrMat<Where::device> Ad(A), lAd(lA), rAd(rA);


    // Product vector size is same as local rows of A
    std::vector<float> b(lA.num_rows(), 0);
    std::cerr << "Copy b to GPU\n";
    Array<Where::device, float> lbd(b), rbd(b); // "local b device, remote b device"


    // plan allgather of remote x data
    std::cerr << "plan allgather xs\n";
    std::vector<int> recvcounts;
    std::vector<int> displs;
    for (int i = 0; i < size; ++i) {
        Range r = get_partition(lx.size(), i, size);
        recvcounts.push_back(r.ub-r.lb);
        if (displs.empty()) {
            displs.push_back(0);
        } else {
            displs.push_back(displs.back() + recvcounts.back());
        }
    }

    int loPrio, hiPrio;
    CUDA_RUNTIME(cudaDeviceGetStreamPriorityRange (&loPrio, &hiPrio));

    cudaStream_t loS, hiS; // "lo/hi prio"
    CUDA_RUNTIME(cudaStreamCreateWithPriority(&loS, cudaStreamNonBlocking, hiPrio));
    CUDA_RUNTIME(cudaStreamCreateWithPriority(&hiS, cudaStreamNonBlocking, hiPrio));

    cudaEvent_t event;
    CUDA_RUNTIME(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));

    const int nIters = 30;
    std::vector<double> times(nIters);

    /* ===== multiply in one shot
    */

    // do spmv
    dim3 dimBlock(256);
    dim3 dimGrid(100);

    nvtxRangePush("one-shot");
    for (int i = 0; i < nIters; ++i) {
        MPI_Barrier(MPI_COMM_WORLD);
        double start = MPI_Wtime();

        // distribute x to ranks
        MPI_Allgatherv(lx.data() + displs[rank], recvcounts[rank], MPI_FLOAT, lx.data(), recvcounts.data(), displs.data(), MPI_FLOAT, MPI_COMM_WORLD);

        // copy x to GPU
        lxd.set_from(lx, hiS);

        spmv<<<dimGrid, dimBlock, 0, hiS>>>(lbd.view(), Ad.view(), lxd.view(), ProductConfig::SET);
        CUDA_RUNTIME(cudaGetLastError());
        CUDA_RUNTIME(cudaStreamSynchronize(hiS));
        times[i] = MPI_Wtime() - start;
    }
    nvtxRangePop(); // one-shot
    MPI_Allreduce(MPI_IN_PLACE, times.data(), times.size(), MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (0 == rank) {
        std::sort(times.begin(), times.end());
        std::cerr << times[times.size() / 2] << "\n";
    }


    /* ===== split local and remote
       multiply local, gather & multiply remote
       TODO: the separate add launch can be removed if it is ensured
       that The remote happens strictly after the local.
       It's a small false serialization, but if we're in the case
       where that matters, the launch overhead dominates anyway.
    */
    nvtxRangePush("local/remote");
    for (int i = 0; i < nIters; ++i) {

        MPI_Barrier(MPI_COMM_WORLD);
        double start = MPI_Wtime();

        // overlap MPI with CUDA kernel launch
        MPI_Request req;
        MPI_Iallgatherv(lx.data() + displs[rank], recvcounts[rank], MPI_FLOAT, lx.data(), recvcounts.data(), displs.data(), MPI_FLOAT, MPI_COMM_WORLD, &req);

        spmv<<<dimGrid, dimBlock, 0, hiS>>>(lbd.view(), lAd.view(), lxd.view(), ProductConfig::SET);
        CUDA_RUNTIME(cudaGetLastError());

        MPI_Wait(&req, MPI_STATUS_IGNORE);

        rxd.set_from(lx, loS);
        
        // hiS blocks until transfer is done
        CUDA_RUNTIME(cudaEventRecord(event, loS));
        CUDA_RUNTIME(cudaStreamWaitEvent(hiS, event, 0));

        spmv<<<dimGrid, dimBlock, 0, hiS>>>(rbd.view(), rAd.view(), rxd.view(), ProductConfig::MODIFY);
        CUDA_RUNTIME(cudaGetLastError());

        // all is done when hiS is done
        CUDA_RUNTIME(cudaStreamSynchronize(hiS));
        times[i] = MPI_Wtime() - start;
    }
    nvtxRangePop(); // local/remote
    MPI_Allreduce(MPI_IN_PLACE, times.data(), times.size(), MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (0 == rank) {
        std::sort(times.begin(), times.end());
        std::cerr << times[times.size() / 2] << "\n";
    }

    // maybe better to atomic add into result than doing separate kernel launch?

    CUDA_RUNTIME(cudaStreamDestroy(loS));
    CUDA_RUNTIME(cudaStreamDestroy(hiS));

    MPI_Finalize();

    return 0;
}