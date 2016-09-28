#ifndef MATRIX_HPP
#define MATRIX_HPP
#include <assert.h>
#include <stddef.h>
#include <array>                        // for ManyBodyOperator
#include <vector>                       // for Operator
#include "blas.hpp"
#include "allocation.hpp"
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
/// It is also possible to use `Stage` for allocations.  This allows multiple
/// `Matrix` and similar objects to be stored in a single contiguous block of
/// memory.  For example, this example allocates two 100-by-100 matrices into
/// a single array of length 200000:
///
///     Matrix<double> mat;
///     Stage<double> stage;
///     arena.prepare(mat.alloc_req(100, 100));
///     arena.prepare(mat.alloc_req(100, 100));
///     arena.execute();
///     assert(arena.size() == 200000);
///
template<typename T>
class Matrix {

    T *_data = nullptr;

    size_t _num_rows = 0;

    size_t _num_cols = 0;

    size_t _stride = 0;

public:

    struct AllocReq : public GenericAllocReq<T> {

        Matrix &matrix;

        size_t num_rows;

        size_t num_cols;

        AllocReq(Matrix &matrix, size_t num_rows, size_t num_cols)
            : matrix(matrix)
            , num_rows(num_rows)
            , num_cols(num_cols)
        {
        }

        size_t size() const override
        {
            return this->num_rows * this->num_cols;
        }

        void fulfill(T *data) const override
        {
            matrix = Matrix(data, this->num_rows, this->num_cols);
        }

    };

    Matrix()
    {
    }

    Matrix(T *data, size_t num_rows, size_t num_cols)
        : _data(data)
        , _num_rows(num_rows)
        , _num_cols(num_cols)
        , _stride(num_cols)
    {
    }

    Matrix(T *data, size_t num_rows, size_t num_cols, size_t stride)
        : _data(data)
        , _num_rows(num_rows)
        , _num_cols(num_cols)
        , _stride(stride)
    {
    }

    AllocReq alloc_req(size_t num_rows, size_t num_cols)
    {
        return AllocReq(*this, num_rows, num_cols);
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
        // this is totally safe
        return const_cast<Matrix &>(*this)(row_index, col_index);
    }

    T &operator()(size_t row_index, size_t col_index)
    {
        assert(row_index < this->num_rows());
        assert(col_index < this->num_cols());
        return this->data[row_index * this->stride() + col_index];
    }

    Matrix<const T> slice(const IRange<size_t> &row_index_range,
                          const IRange<size_t> &col_index_range) const
    {
        return const_cast<Matrix *>(this)->slice(row_index_range,
                                                 col_index_range);
    }

    Matrix slice(const IRange<size_t> &row_index_range,
                 const IRange<size_t> &col_index_range)
    {
        return Matrix(&(*this)(row_index_range.start, col_index_range.start),
                      row_index_range.size(),
                      col_index_range.size(),
                      this->stride());
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

typedef std::vector<Matrix<double>> Operator;

/// A many-body operator contains three operators in standard form:
///
///   - Zero-body operator (constant term) in 000 form.  This is always has a
///     single block containing one element.
///
///   - One-body operator in 100 form.
///
///   - Two-body operator in 200 form.
///
typedef std::array<Operator, 3> ManyBodyOperator;

#endif
