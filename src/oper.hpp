#ifndef OPER_HPP
#define OPER_HPP
#include <vector>
#include "matrix.hpp"
#include "basis.hpp"

class Oper {

    double *_data = nullptr;

    const ManyBodyBasis *_basis = nullptr;

    OperKind _kind = OPER_KIND_000;

public:

    Oper()
        : _data()
        , _basis()
        , _kind()
    {
    }

    Oper(double *data, const ManyBodyBasis &basis, OperKind kind)
        : _data(data)
        , _basis(&basis)
        , _kind(kind)
    {
    }

    PtrAllocReq<double> alloc_req(const ManyBodyBasis &, OperKind);

    double *data() const
    {
        return this->_data;
    }

    const ManyBodyBasis &basis() const
    {
        assert(this->_basis != nullptr);
        return *this->_basis;
    }

    OperKind kind() const
    {
        return this->_kind;
    }

    size_t size() const
    {
        return this->basis().oper_size(this->kind());
    }

    size_t num_blocks() const
    {
        Rank r = oper_kind_to_rank(this->kind());
        return this->basis().num_channels(r);
    }

    Matrix<double> operator[](size_t l) const
    {
        assert(l < this->num_blocks());
        size_t i = this->basis().block_offset(this->kind(), l);
        size_t nu1, nu2;
        this->basis().block_size(this->kind(), l, &nu1, &nu2);
        return {this->data() + i, nu1, nu2};
    }

    double &operator()() const
    {
        assert(this->kind() == OPER_KIND_000);
        return (*this)[0](0, 0);
    }

    double &operator()(const Orbital &lu1, const Orbital &lu2) const
    {
        assert(this->kind() == OPER_KIND_100);
        assert(this->basis().is_conserved_1(lu1, lu2));
        size_t l1 = lu1.channel_index();
        size_t u1 = lu1.auxiliary_index();
        size_t u2 = lu2.auxiliary_index();
        return (*this)[l1](u1, u2);
    }

    double &operator()(const Orbital &lu1, const Orbital &lu2,
                       const Orbital &lu3, const Orbital &lu4) const
    {
        assert(this->kind() == OPER_KIND_200);
        assert(this->basis().is_conserved_2(lu1, lu2, lu3, lu4));
        // note: we are assuming l1 + l2 exists!
        //       (this holds if is_conserved_2 returns true)
        Orbital lu12 = *this->basis().combine_20(lu1, lu2);
        Orbital lu34 = *this->basis().combine_20(lu3, lu4);
        size_t l12 = lu12.channel_index();
        size_t u12 = lu12.auxiliary_index();
        size_t u34 = lu34.auxiliary_index();
        return (*this)[l12](u12, u34);
    }

    const Oper &operator=(double value) const
    {
        for (size_t l = 0; l < this->num_blocks(); ++l) {
            (*this)[l] = value;
        }
        return *this;
    }

    const Oper &operator+=(const Oper &other) const
    {
        assert(this->basis() == other.basis());
        assert(this->kind() == other.kind());
        assert(this->num_blocks() == other.num_blocks());
        for (size_t l = 0; l < this->num_blocks(); ++l) {
            (*this)[l] += other[l];
        }
        return *this;
    }

    const Oper &operator*=(double alpha) const
    {
        for (size_t l = 0; l < this->num_blocks(); ++l) {
            (*this)[l] *= alpha;
        }
        return *this;
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

    double *_data;

    const ManyBodyBasis *_basis;

public:

    ManyBodyOper()
        : _data()
        , _basis()
    {
    }

    ManyBodyOper(double *data, const ManyBodyBasis &basis)
        : _data(data)
        , _basis(&basis)
    {
    }

    PtrAllocReq<double> alloc_req(const ManyBodyBasis &);

    double *data() const
    {
        return this->_data;
    }

    const ManyBodyBasis &basis() const
    {
        assert(this->_basis != nullptr);
        return *this->_basis;
    }

    size_t size() const
    {
        return this->basis().many_body_oper_size();
    }

    Oper oper(size_t rank) const
    {
        assert(rank < RANK_COUNT);
        Rank r = (Rank)rank;
        size_t i = this->basis().oper_offset(r);
        OperKind kk = standard_oper_kind(r);
        return {this->data() + i, this->basis(), kk};
    }

    double &operator()() const
    {
        return this->oper(RANK_0)();
    }

    double &operator()(const Orbital &lu1, const Orbital &lu2) const
    {
        return this->oper(RANK_1)(lu1, lu2);
    }

    double &operator()(const Orbital &lu1, const Orbital &lu2,
                       const Orbital &lu3, const Orbital &lu4) const
    {
        return this->oper(RANK_2)(lu1, lu2, lu3, lu4);
    }

    const ManyBodyOper &operator=(double value) const
    {
        for (size_t r = 0; r < RANK_COUNT; ++r) {
            this->oper(r) = value;
        }
        return *this;
    }

    const ManyBodyOper &operator+=(const ManyBodyOper &other) const
    {
        assert(this->basis() == other.basis());
        for (size_t r = 0; r < RANK_COUNT; ++r) {
            this->oper(r) += other.oper(r);
        }
        return *this;
    }

    const ManyBodyOper &operator*=(double value) const
    {
        for (size_t r = 0; r < RANK_COUNT; ++r) {
            this->oper(r) *= value;
        }
        return *this;
    }

};

#endif
