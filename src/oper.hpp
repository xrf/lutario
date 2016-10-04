#ifndef OPER_HPP
#define OPER_HPP
#include <vector>
#include "matrix.hpp"
#include "basis.hpp"

class Oper {

    const ManyBodyBasis *_many_body_basis = nullptr;

    std::vector<Matrix<double>> _blocks;

public:

    AllocReqBatch<double> alloc_req(const ManyBodyBasis &mbasis, OperKind kk);

    const ManyBodyBasis &many_body_basis() const
    {
        return *this->_many_body_basis;
    }

    const Matrix<double> &operator[](size_t l) const
    {
        assert(l < this->_blocks.size());
        return this->_blocks[l];
    }

    Matrix<double> &operator[](size_t l)
    {
        assert(l < this->_blocks.size());
        return this->_blocks[l];
    }

};

/// A many-body operator contains three operators in standard form:
///
///   - Zero-body operator (constant term) in 000 form.  This is always has a
///     single block containing one element.
///
///   - One-body operator in 100 form.
///
///   - Two-body operator in 200 form.
///
class ManyBodyOper {

    const ManyBodyBasis *_many_body_basis = nullptr;

    Oper _opers[3];

public:

    AllocReqBatch<double> alloc_req(const ManyBodyBasis &mbasis);

    const ManyBodyBasis &many_body_basis() const
    {
        return *this->_many_body_basis;
    }

    const Oper &operator[](size_t r) const
    {
        assert(r < 3);
        return this->_opers[r];
    }

    Oper &operator[](size_t r)
    {
        assert(r < 3);
        return this->_opers[r];
    }

};

#endif
