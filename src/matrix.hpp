#ifndef MATRIX_HPP
#define MATRIX_HPP
#include <assert.h>
#include <stddef.h>
#include "blas.hpp"
#include "alloc.hpp"
#include "irange.hpp"

/// A non-owning matrix view.  "Non-owning" means that the memory associated
/// with the matrix is not freed upon the destruction of `Matrix`.
///
/// Usually, to create a `Matrix` from scratch, you would do:
///
///     Matrix<double> mat;
///     std::unique_ptr<double[]> buf = alloc(mat.alloc_req(100, 100));
///
/// The `buf` variable holds the actual memory buffer.  When `buf` is
/// destroyed, so is the memory associated with the matrix.  It is the user's
/// responsibility to make sure that `mat` does not outlive `buf`.
///
/// It is also possible to use `AllocReqBatch` for allocations.  This allows
/// multiple `Matrix` and similar objects to be stored in a single contiguous
/// block of memory.  For example, this example allocates two 100-by-100
/// matrices into a single array of length 20000:
///
///     Matrix<double> mat1, mat2;
///     std::unique_ptr<double[]> buf;
///     {
///         AllocReqBatch<double> batch;
///         batch.push(mat1.alloc_req(100, 100));
///         batch.push(mat2.alloc_req(100, 100));
///         assert(batch.size() == 20000);
///         buf = alloc(std::move(alloc_req_batch));
///     }
///
template<typename T>
class Matrix {

    T *_data = nullptr;

    size_t _num_rows = 0;

    size_t _num_cols = 0;

    size_t _stride = 0;

    // assertions to check the sanity of the object
    void _validate() const
    {
        // num_rows * num_cols must not overflow
        assert(this->num_rows() == 0 ||
               this->num_cols() < (size_t)(-1) / this->num_rows());
        // num_rows * stride must not overflow
        assert(this->num_rows() == 0 ||
               this->stride() < (size_t)(-1) / this->num_rows());
    }

public:

    Matrix()
    {
        this->_validate();
    }

    Matrix(T *data, size_t num_rows, size_t num_cols)
        : _data(data)
        , _num_rows(num_rows)
        , _num_cols(num_cols)
        , _stride(num_cols)
    {
        this->_validate();
    }

    Matrix(T *data, size_t num_rows, size_t num_cols, size_t stride)
        : _data(data)
        , _num_rows(num_rows)
        , _num_cols(num_cols)
        , _stride(stride)
    {
        this->_validate();
    }

    PtrAllocReq<T> alloc_req(size_t num_rows, size_t num_cols)
    {
        *this = Matrix(nullptr, num_rows, num_cols);
        return PtrAllocReq<T>(&this->_data, num_rows * num_cols);
    }

    operator Matrix<const T>() const
    {
        return Matrix<const T>(this->data(),
                               this->num_rows(),
                               this->num_cols(),
                               this->stride());
    }

    const T *data() const
    {
        return this->_data;
    }

    T *data()
    {
        return this->_data;
    }

    size_t num_rows() const
    {
        return this->_num_rows;
    }

    size_t num_cols() const
    {
        return this->_num_cols;
    }

    size_t stride() const
    {
        return this->_stride;
    }

    const T &operator()(size_t row_index, size_t col_index) const
    {
        // this is indeed safe :P
        return const_cast<Matrix &>(*this)(row_index, col_index);
    }

    T &operator()(size_t row_index, size_t col_index)
    {
        assert(this->data());
        assert(row_index < this->num_rows());
        assert(col_index < this->num_cols());
        return this->data()[row_index * this->stride() + col_index];
    }

    Matrix<const T> slice(const IndexRange &row_index_range,
                          const IndexRange &col_index_range) const
    {
        // this is fine :)
        return const_cast<Matrix *>(this)->slice(row_index_range,
                                                 col_index_range);
    }

    Matrix slice(const IndexRange &row_index_range,
                 const IndexRange &col_index_range)
    {
        return Matrix(&(*this)(row_index_range.start, col_index_range.start),
                      row_index_range.size(),
                      col_index_range.size(),
                      this->stride());
    }

    Matrix &operator=(double value)
    {
        for (size_t i = 0; i < this->num_rows(); ++i) {
            for (size_t j = 0; j < this->num_cols(); ++j) {
                (*this)(i, j) = value;
            }
        }
        return *this;
    }

};

inline void transpose_indices(CBLAS_TRANSPOSE trans, size_t &m, size_t &n)
{
    switch (trans) {
    case CblasTrans:
    case CblasConjTrans:
        std::swap(m, n);
        break;
    case CblasNoTrans:
        break;
    default:
        assert(0);
    }
}

inline void gemm(CBLAS_TRANSPOSE transa,
                 CBLAS_TRANSPOSE transb,
                 double alpha,
                 const Matrix<const double> a,
                 const Matrix<const double> b,
                 double beta,
                 Matrix<double> c)
{
    size_t m_a = a.num_rows();
    size_t k_a = a.num_cols();
    size_t k_b = b.num_rows();
    size_t n_b = b.num_cols();
    size_t m_c = c.num_rows();
    size_t n_c = c.num_cols();
    transpose_indices(transa, m_a, k_a);
    transpose_indices(transb, k_b, n_b);
    assert(m_a == m_c);
    assert(n_b == n_c);
    assert(k_a == k_b);
    cblas_dgemm(CblasRowMajor,
                transa,
                transb,
                (CBLAS_INT)m_c,
                (CBLAS_INT)n_c,
                (CBLAS_INT)k_a,
                alpha,
                a.data(),
                (CBLAS_INT)a.stride(),
                b.data(),
                (CBLAS_INT)b.stride(),
                beta,
                c.data(),
                (CBLAS_INT)c.stride());
}

#endif
