#include <mpi.h>

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


/* device array
*/
template<typename T> class Array<Where::device, T>
{
public:

    // A non-owning view of data
    struct View
    {
        T *data_;
        int64_t size_;
        public:
        View() : data_(nullptr), size_(0){}
        View(const View &other) = default;
        View(View &&other) = default;
        View &operator=(const View &rhs) = default;

        __host__ __device__ int64_t size() const { return size_; }

        __device__ const T &operator()(int64_t i) const {
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
        __device__ T &operator()(int64_t i) {
            return data_[i];
        }
    };

    // array owns the data in this view
    View view_;
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

    View view() const {
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

    // any change destroys all data
    void resize(size_t n) {
        if (size() != n) {
            view_.size_ = n;
            CUDA_RUNTIME(cudaFree(view_.data_));
            CUDA_RUNTIME(cudaMalloc(&view_.data_, view_.size_ * sizeof(T)));
        }
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

        Entry entry(i,j,e);

        // first position not less than entry
        auto lb = std::lower_bound(data_.begin(), data_.end(), entry, Entry::by_ij);

        // overwrite if already exists
        if (lb != data_.end() && lb->i == entry.i && lb->j == entry.j) {
            *lb = entry;
        } else {
            data_.insert(lb, entry);
        }
    }

    void sort() {
        std::sort(data_.begin(), data_.end(), Entry::by_ij);
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
        Array<Where::device, int>::View rowPtr_;
        Array<Where::device, int>::View colInd_;
        Array<Where::device, float>::View val_;

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
        int r = rand() % m;
        int c = rand() % n;
        float e = 1.0;
        coo.push_back(r, c, e);
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

    int numCols = m.num_cols();
    MPI_Send(&numCols, 1, MPI_INT, dst, Tag::num_cols, comm);
    MPI_Send(m.row_ptr(), m.num_rows()+1, MPI_INT, dst, Tag::row_ptr, comm);
    MPI_Send(m.col_ind(), m.nnz(), MPI_INT, dst, Tag::col_ind, comm);
    MPI_Send(m.val(), m.nnz(), MPI_FLOAT, dst, Tag::val, comm);

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
    // receive actual data into matrix
    MPI_Recv(csr.col_ind(), nnz, MPI_INT, 0, Tag::col_ind, comm, MPI_STATUS_IGNORE);
    // receive actual data into matrix
    MPI_Recv(csr.val(), nnz, MPI_FLOAT, 0, Tag::val, comm, MPI_STATUS_IGNORE);

    return csr;
}

int send_vector(int dst, int src, std::vector<float> &&v, MPI_Comm comm) {
    MPI_Send(v.data(), v.size(), MPI_FLOAT, dst, Tag::x, comm);
    return 0;
}

std::vector<float> receive_vector(int dst, int src, MPI_Comm comm) {

    // probe for size
    MPI_Status stat;
    MPI_Probe(0, Tag::x, comm, &stat);
    int sz;
    MPI_Get_count(&stat, MPI_INT, &sz);
    std::vector<float> x(sz);

    std::cerr << "recv " << sz << " x entries\n";

    // receive actual data into matrix
    MPI_Recv(x.data(), x.size(), MPI_FLOAT, 0, Tag::x, comm, MPI_STATUS_IGNORE);

    return x;
}

/* Ax=b
*/
__global__ void spmv(Array<Where::device, float>::View b, const CsrMat<Where::device>::View A, const Array<Where::device, float>::View x) {

    // one thread per row
    for (int r = blockDim.x * blockIdx.x + threadIdx.x; r < A.num_rows(); r += blockDim.x * gridDim.x) {
        float acc = 0;
        for (int ci = A.row_ptr(r); ci < A.row_ptr(r+1); ++ci) {
            int c = A.col_ind(ci);
            acc += A.val(ci) * x(c);
        }
        b(r) = acc;
    }

}

int main (int argc, char **argv) {

    MPI_Init(&argc, &argv);

    std::cerr << "get a gpu...\n";
    CUDA_RUNTIME(cudaFree(0));
    std::cerr << "barrier...\n";
    MPI_Barrier(MPI_COMM_WORLD);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
  
    int64_t m = 100;
    int64_t n = 50;
    int64_t nnz = 100;

    CsrMat<Where::host> A;
    std::vector<float> x;

    // generate and distribute A and x
    if (0 == rank) {
        A = random_matrix(m, n, nnz);
        x = random_vector(n);
        std::vector<CsrMat<Where::host>> As = part_by_rows(A, size);
        std::vector<std::vector<float>> xs = part_by_rows(x, size);
        for (size_t dst = 1; dst < size; ++dst) {
            std::cerr << "send A to " << dst << "\n";
            send_matrix(dst, 0, std::move(As[dst]), MPI_COMM_WORLD);
            std::cerr << "send x to " << dst << "\n";
            send_vector(dst, 0, std::move(xs[dst]), MPI_COMM_WORLD);
        }
        A = As[rank];
        x = xs[rank];
    } else {
        std::cerr << "recv A at " << rank << "\n";
        A = receive_matrix(rank, 0, MPI_COMM_WORLD);
        std::cerr << "recv x at " << rank << "\n";
        x = receive_vector(rank, 0, MPI_COMM_WORLD);
    }

    std::cerr << "A[" << A.num_rows() << "," << A.num_cols() << "] w/ " << A.nnz() << "\n";
    std::cerr << "Copy A to GPU\n";
    CsrMat<Where::device> Ad(A);


    // Product vector size is same as local rows of A
    std::vector<float> b(A.num_rows(), 0);
    std::cerr << "Copy b to GPU\n";
    Array<Where::device, float> bd(b);


    // plan allgather of complete x
    std::cerr << "plan allgather xs\n";
    std::vector<float> lx(A.num_cols());
    Array<Where::device, float> lxd(A.num_cols());
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

    

    cudaStream_t stream;
    CUDA_RUNTIME(cudaStreamCreate(&stream));
    // do spmv
    dim3 dimBlock(256);
    dim3 dimGrid(100);

    for (int i = 0; i < 10; ++i) {

        // distribute x to ranks
        std::cerr << "Allgather xs\n";
        MPI_Allgatherv(x.data(), x.size(), MPI_FLOAT, lx.data(), recvcounts.data(), displs.data(), MPI_FLOAT, MPI_COMM_WORLD);

        std::cerr << "Copy x to GPU\n";
        lxd.set_from(lx, stream);

        std::cerr << "launch spmv\n";
        spmv<<<dimGrid, dimBlock, 0, stream>>>(bd.view(), Ad.view(), lxd.view());
        CUDA_RUNTIME(cudaGetLastError());
        CUDA_RUNTIME(cudaStreamSynchronize(stream));
        std::cerr << "done spmv\n";

        // copy product back to host
        // b = bd;

        // for (int r = 0; r < size; ++r) {
        //     MPI_Barrier(MPI_COMM_WORLD);
        //     if (rank == r) {
        //         for (auto &e : b) {
        //             std::cout << e << "\n";
        //         }
        //     }
        // }

    }

    MPI_Finalize();

    return 0;
}