#ifndef OPER_HPP
#define OPER_HPP
#include <vector>
#include "matrix.hpp"
#include "basis.hpp"

class Oper {

    const ManyBodyBasis *_basis = nullptr;

    std::vector<Matrix<double>> _blocks;

    OperKind _kind;

public:

    AllocReqBatch<double> alloc_req(const ManyBodyBasis &, OperKind);

    const ManyBodyBasis &basis() const
    {
        return *this->_basis;
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

    OperKind kind() const
    {
        return this->_kind;
    }

    double operator()() const
    {
        return const_cast<Oper &>(*this)();
    }

    double operator()(const Orbital &p1, const Orbital &p2) const
    {
        return const_cast<Oper &>(*this)(p1, p2);
    }

    double operator()(const Orbital &p1, const Orbital &p2,
                      const Orbital &p3, const Orbital &p4) const
    {
        return const_cast<Oper &>(*this)(p1, p2, p3, p4);
    }

    double &operator()()
    {
        assert(this->kind() == OPER_KIND_000);
        return (*this)[0](0, 0);
    }

    double &operator()(const Orbital &lu1, const Orbital &lu2)
    {
        assert(this->kind() == OPER_KIND_100);
        assert(this->basis().is_conserved_1(lu1, lu2));
        size_t l1 = lu1.channel_index();
        size_t u1 = lu1.auxiliary_index();
        size_t u2 = lu2.auxiliary_index();
        return (*this)[l1](u1, u2);
    }

    double &operator()(const Orbital &lu1, const Orbital &lu2,
                       const Orbital &lu3, const Orbital &lu4)
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

    Oper &operator=(double value)
    {
        for (Matrix<double> &block : this->_blocks) {
            block = value;
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

    const ManyBodyBasis *_basis = nullptr;

public:

    Oper opers[3];

    AllocReqBatch<double> alloc_req(const ManyBodyBasis &);

    const ManyBodyBasis &basis() const
    {
        return *this->_basis;
    }

    /// Forward any `operator()` calls with `2 * r` arguments to `operator()`
    /// calls on the rank-`r` operator.
    template<typename... Ts>
    double operator()(Ts &&... args) const
    {
        return const_cast<ManyBodyOper &>(*this)(std::forward<Ts>(args)...);
    }

    /// Forward any `operator()` calls with `2 * r` arguments to `operator()`
    /// calls on the rank-`r` operator.
    template<typename... Ts>
    double &operator()(Ts &&... args)
    {
        constexpr size_t r = sizeof...(Ts) / 2;
        static_assert(sizeof...(Ts) % 2 == 0,
                      "must call operator() with even number of arguments");
        static_assert(r < 3, "too many arguments to call of operator()");
        return this->opers[r](std::forward<Ts>(args)...);
    }

    ManyBodyOper &operator=(double value)
    {
        for (Oper &oper : this->opers) {
            oper = value;
        }
        return *this;
    }

};

#endif
