#include <mpi.h>

#include <vector>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <iostream>

#include "cuda_runtime.hpp"

template<typename ForwardIt>
void shift_left(ForwardIt first, ForwardIt last, size_t n) {
    for (size_t i = 0; i < last-first; ++i) {
        *(first-n+i) = *(first+i);
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
class Array {
public:
    Array();
    int64_t size() const;
};


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

        // create view from array
        View(const Array &a) {
            size_ = a.size();
            data_ = a.data_;
        }
        __device__ int64_t size() const { return size_; }
    };

    // array owns the data in this view
    View view_;
public:
    Array() = default;
    Array(const Array &other) = delete;

    Array(const std::vector<T> &v) {
        view_.size_ = v.size();
        CUDA_RUNTIME(cudaMalloc(&view_.data_, view_.size_ * sizeof(T)));
        CUDA_RUNTIME(cudaMemcpy(view_.data_, v.data(), view_.size_ * sizeof(T), cudaMemcpyHostToDevice));
    }
    ~Array() {
        CUDA_RUNTIME(cudaFree(view_.data_));
        view_.data_ = nullptr;
        view_.size_ = 0;
    }
    int64_t size() const { return view_.size(); }

    View view() const {
        return view_; // copy of internal view
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
    std::vector<Entry> data_;
    int64_t numRows_;
    int64_t numCols_;

public:
    CooMat(int m, int n) : numRows_(m), numCols_(n) {}
    const std::vector<Entry> &entries() const {return data_;}
    void push_back(int i, int j, int e) {
        data_.push_back(Entry(i, j, e));
    }

    void sort() {
        std::sort(data_.begin(), data_.end(), Entry::by_ij);
    }

    int64_t num_rows() const {return numRows_;}
    int64_t num_cols() const {return numRows_;}
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
    CsrMat(int numRows, int numCols, int nnz) : rowPtr_(numRows+1), colInd_(nnz), val_(nnz) {}
    
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
        std::cerr << "resize rowPtr_ to " << rowEnd+1 << "\n";
        rowPtr_.resize(rowEnd+1);
        std::cerr << "resize entries to " << rowPtr_.back() << "\n";
        colInd_.resize(rowPtr_.back());
        val_.resize(rowPtr_.back());

        // erase early row pointers
        std::cerr << "shl rowPtr by " << rowStart << "\n";
        shift_left(rowPtr_.begin()+rowStart, rowPtr_.end(), rowStart);
        std::cerr << "resize rowPtr to " << rowEnd - rowStart+1 << "\n";
        rowPtr_.resize(rowEnd-rowStart+1);

        const int off = rowPtr_[0];
        // erase entries for first rows
        std::cerr << "shl entries by " << off << "\n";
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
    };

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
    CooMat coo(m,n);
    for (int i = 0; i < nnz; ++i) {
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

std::vector<CsrMat<Where::host>> part_by_rows(const CsrMat<Where::host> &m, const int parts) {

    std::vector<CsrMat<Where::host>> mats;

    int rowStart = 0;
    for (int p = 0; p < parts; ++p) {
        int partSize = m.num_rows() / parts;
        if (p < m.num_rows() % parts) {
            ++partSize;
        }
        std::cerr << "matrix part " << p << " has " << partSize << " rows\n";
        const int rowEnd = rowStart + partSize;
        CsrMat<Where::host> part(m);
        part.retain_rows(rowStart, rowEnd);
        rowStart = rowEnd;
        mats.push_back(part);
    }

    return mats;
}

std::vector<std::vector<float>> part_by_rows(const std::vector<float> &x, const int parts) {
    std::vector<std::vector<float>> xs;

    int rowStart = 0;
    for (int p = 0; p < parts; ++p) {
        int partSize = x.size() / parts;
        if (p < x.size() % parts) {
            ++partSize;
        }
        std::cerr << "vector part " << p << " has " << partSize << " rows\n";
        const int rowEnd = rowStart + partSize;
        std::vector<float> part(x.begin()+rowStart, x.begin()+rowEnd);
        xs.push_back(part);
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

__global__ void spmv(Array<Where::device, float>::View b, const CsrMat<Where::device>::View A, const Array<Where::device, float>::View x) {

    // one block per row
    for (int r = blockIdx.x; r < A.num_rows(); r += gridDim.x) {

    }

}

int main (int argc, char **argv) {

MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
  
    int64_t m = 100;
    int64_t n = 50;
    int64_t nnz = 5000;

    CsrMat<Where::host> A;
    std::vector<float> x;

    // generate and send or recv A
    if (0 == rank) {
        A = random_matrix(m, n, nnz);
        std::vector<float> x = random_vector(n);
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

    // Product vector size is same as local rows of A
    std::vector<float> b(A.num_rows());

    // get GPU versions
    CsrMat<Where::device> Ad(A);
    Array<Where::device, float> xd(x);
    Array<Where::device, float> bd(b);

    // do spmv
    dim3 dimBlock(32,8,1);
    dim3 dimGrid(100);
    spmv<<<dimGrid, dimBlock>>>(bd.view(), Ad.view(), xd.view());
    CUDA_RUNTIME(cudaDeviceSynchronize());

MPI_Finalize();

}