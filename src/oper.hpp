#ifndef OPER_HPP
#define OPER_HPP
#include <vector>
#include "matrix.hpp"
#include "basis.hpp"

struct Oper {

    std::vector<Matrix<double>> blocks;

    AllocReqBatch<double> alloc_req(const ManyBodyBasis &mbasis, OperKind kk)
    {
        Rank r = oper_kind_to_rank(kk);
        size_t nl = mbasis.table().num_channels(r);
        this->blocks.resize(nl);
        AllocReqBatch<double> reqs;
        for (size_t l = 0; l < nl; ++l) {
            size_t nu1, nu2;
            mbasis.block_size(kk, l, &nu1, &nu2);
            reqs.emplace_back(this->blocks[l].alloc_req(nu1, nu2));
        }
        return reqs;
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
struct ManyBodyOper {

    Oper opers[3];

    AllocReqBatch<double> alloc_req(const ManyBodyBasis &mbasis)
    {
        AllocReqBatch<double> reqs;
        for (OperKind kk : {OPER_KIND_000, OPER_KIND_100, OPER_KIND_200}) {
            Rank r = oper_kind_to_rank(kk);
            reqs.emplace_back(this->opers[r].alloc_req(mbasis, kk));
        }
        return reqs;
    }

};

#endif
