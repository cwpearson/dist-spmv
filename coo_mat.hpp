#pragma once

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

    std::vector<Entry>::iterator begin() {return data_.begin();}
    std::vector<Entry>::iterator end() {return data_.end();}
};