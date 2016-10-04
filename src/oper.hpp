#ifndef OPER_HPP
#define OPER_HPP
#include <vector>
#include "matrix.hpp"
#include "basis.hpp"

struct Oper {

    std::vector<Matrix<double>> blocks;

    AllocReqBatch<double> alloc_req(const ManyBodyBasis &mbasis, OperKind kk);

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
struct ManyBodyOper {

    Oper opers[3];

    AllocReqBatch<double> alloc_req(const ManyBodyBasis &mbasis);

};

#endif
