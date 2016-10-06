#include "alloc.hpp"
#include "basis.hpp"
#include "oper.hpp"

AllocReqBatch<double> Oper::alloc_req(const ManyBodyBasis &mbasis, OperKind kk)
{
    this->_many_body_basis = &mbasis;
    Rank r = oper_kind_to_rank(kk);
    size_t nl = mbasis.table().num_channels(r);
    this->_blocks.resize(nl);
    AllocReqBatch<double> reqs;
    for (size_t l = 0; l < nl; ++l) {
        size_t nu1, nu2;
        mbasis.block_size(kk, l, &nu1, &nu2);
        reqs.push((*this)[l].alloc_req(nu1, nu2));
    }
    return reqs;
}

AllocReqBatch<double> ManyBodyOper::alloc_req(const ManyBodyBasis &mbasis)
{
    this->_many_body_basis = &mbasis;
    AllocReqBatch<double> reqs;
    for (OperKind kk : {OPER_KIND_000, OPER_KIND_100, OPER_KIND_200}) {
        Rank r = oper_kind_to_rank(kk);
        reqs.push((*this)[r].alloc_req(mbasis, kk));
    }
    return reqs;
}