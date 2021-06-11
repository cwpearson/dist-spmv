#pragma once

#include <cuda_runtime.h>

#include "array.hpp"
#include "coo_mat.hpp"
#include "algorithm.hpp"

#include <cassert>

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
    int64_t numCols_;

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

    CsrMat() = default;
    CsrMat(CsrMat &&other) = delete;
    CsrMat(const CsrMat &other) = delete;

    CsrMat &operator=(CsrMat &&rhs) {
        if (this != &rhs) {
            rowPtr_ = std::move(rhs.rowPtr_);
            colInd_ = std::move(rhs.colInd_);
            val_ = std::move(rhs.val_);
            numCols_ = std::move(rhs.numCols_);
        }
        return *this;
    }

    // create device matrix from host
    CsrMat(const CsrMat<Where::host> &m) : 
        rowPtr_(m.rowPtr_), colInd_(m.colInd_), val_(m.val_), numCols_(m.numCols_) {
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
    int64_t num_cols() const {
        return numCols_;
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
                // retry, don't over-weight first or last column
                continue;
            }
            float e = 1.0;
            
            assert(c < n);
            assert(r < n);
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