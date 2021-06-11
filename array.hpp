#pragma once

#include <vector>

#include "cuda_runtime.hpp"
#include "where.hpp"

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

    const T* data() const {
        return data_;
    }
    T* data() {
        return data_;
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
    Array &operator=(Array &&other) {
        view_ = std::move(other.view_);
        // view is non-owning, so have to clear other
        other.view_.data_ = nullptr;
        other.view_.size_ = 0;
        return *this;
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

    const T* data() const {
        return view_.data();
    }
    T* data() {
        return view_.data();
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