#ifndef MATRIX_HPP
#define MATRIX_HPP
#include <assert.h>
#include <stddef.h>
#include <algorithm>
#include <ostream>
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
/// The `buf` variable owns the block of memory belonging to `Matrix`.  When
/// `buf` is destroyed, so the memory associated with the matrix is freed and
/// thus `mat` will contain a dangling pointer.  Therefore, it is thus
/// critical that `mat` does not outlive `buf`.
///
/// It is also possible to use `AllocReqBatch` to allocate multiple objects
/// all at once into a single, contiguous block of memory.  For example, this
/// code allocates two 100-by-100 matrices into a single array of length
/// 20000:
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
/// As before, the lifetime of `mat1` and `mat2` is tied to that of `buf`.
/// When `buf` is destroyed, `mat1` and `mat2` would hold dangling pointers.
///
template<typename T>
class Matrix {

    T *_data = nullptr;

    size_t _num_rows = 0;

    size_t _num_cols = 0;

    size_t _stride = 0;

    // assertions to check invariants:
    void _validate() const
    {
        // num_rows * num_cols must not overflow
        assert(this->num_rows() == 0 ||
               this->num_cols() < (size_t)(-1) / this->num_rows());
        // num_rows * stride must not overflow
        assert(this->num_rows() == 0 ||
               this->stride() < (size_t)(-1) / this->num_rows());

        // we don't require data to be a valid pointer; it is needed for the
        // two-stage initialization via alloc_req

        // we also don't require stride to be greater than or equal to the
        // num_cols to allow for "creative" representations of matrices
        // (though BLAS wouldn't be happy with this)
    }

    size_t _index(size_t row_index, size_t col_index) const
    {
        assert(row_index < this->num_rows());
        assert(col_index <= this->num_cols()); // intentional
        return row_index * this->stride() + col_index;
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
        return {&this->_data, this->size()};
    }

    operator Matrix<const T>() const
    {
        return Matrix<const T>(this->data(),
                               this->num_rows(),
                               this->num_cols(),
                               this->stride());
    }

    T *data() const
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

    size_t size() const
    {
        return this->num_rows() * this->stride();
    }

    size_t index(size_t row_index, size_t col_index) const
    {
        assert(col_index < this->num_cols());
        return this->_index(row_index, col_index);
    }

    T &operator()(size_t row_index, size_t col_index) const
    {
        assert(this->data() != nullptr);
        assert(row_index < this->num_rows());
        assert(col_index < this->num_cols());
        return this->data()[this->index(row_index, col_index)];
    }

    Matrix slice(const IndexRange &row_index_range,
                 const IndexRange &col_index_range) const
    {
        assert(this->data() != nullptr);
        assert(row_index_range.stop <= this->num_rows());
        assert(col_index_range.stop <= this->num_cols());
        // avoid undefined behavior due to out-of-bounds pointer arithmetic
        T *data = this->data();
        if (row_index_range.start != this->num_rows()) {
            data += this->_index(row_index_range.start, col_index_range.start);
        }
        return Matrix(data,
                      row_index_range.size(),
                      col_index_range.size(),
                      this->stride());
    }

    const Matrix &operator=(double value) const
    {
        for (size_t i = 0; i < this->num_rows(); ++i) {
            for (size_t j = 0; j < this->num_cols(); ++j) {
                (*this)(i, j) = value;
            }
        }
        return *this;
    }

    const Matrix &operator+=(const Matrix &other) const
    {
        assert(this->num_rows() == other.num_rows());
        assert(this->num_cols() == other.num_cols());
        for (size_t i = 0; i < this->num_rows(); ++i) {
            cblas_daxpy((CBLAS_INT)this->num_cols(),
                        1.0,
                        other.data() + other._index(i, 0),
                        1,
                        this->data() + this->_index(i, 0),
                        1);
        }
        return *this;
    }

    const Matrix &operator*=(double alpha) const
    {
        for (size_t i = 0; i < this->num_rows(); ++i) {
            cblas_dscal((CBLAS_INT)this->num_cols(),
                        alpha,
                        this->data() + this->_index(i, 0),
                        1);
        }
        return *this;
    }

};

template<typename T>
std::ostream &operator<<(std::ostream &stream, const Matrix<T> &self)
{
    stream << "{\"num_cols\": " << self.num_cols()
           << ", \"num_rows\": " << self.num_rows()
           << ", \"data\": {";
    bool first = true;
    for (size_t i = 0; i < self.num_rows(); ++i) {
        for (size_t j = 0; j < self.num_cols(); ++j) {
            if (!first) {
                stream << ", ";
            }
            T value = self(i, j);
            if ((bool)value) {
                stream << "\"(" << i << ", " << j << ")\": " << value;
                first = false;
            }
        }
    }
    stream << "}}";
    return stream;
}

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
                (CBLAS_INT)std::max<size_t>(a.stride(), 1),
                b.data(),
                (CBLAS_INT)std::max<size_t>(b.stride(), 1),
                beta,
                c.data(),
                (CBLAS_INT)std::max<size_t>(c.stride(), 1));
}

#endif
