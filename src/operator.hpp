#ifndef OPERATOR_HPP
#define OPERATOR_HPP
#include <vector>
#include "matrix.hpp"

class Operator {

    std::vector<Matrix<double>> blocks;

    void prepare(OperatorKind kk,
                 Stage<double> &stage) const
    {
        Rank r = operator_kind_to_rank(kk);
        size_t nl = this->table().num_channels(r);
        q_out.resize(nl);
        for (size_t l = 0; l < nl; ++l) {
            size_t nu1, nu2;
            this->block_size(kk, l, &nu1, &nu2);
            stage.prepare(q_out[l].alloc_req(nu1, nu2));
        }
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
class ManyBodyOperator {

    Operator operators[3];

};

#endif
