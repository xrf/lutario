#include <assert.h>
#include <math.h>
#include "alloc.hpp"
#include "basis.hpp"
#include "math.hpp"
#include "oper.hpp"

PtrAllocReq<double> Oper::alloc_req(const ManyBodyBasis &basis, OperKind kk)
{
    *this = Oper(nullptr, basis, kk);
    return {&this->_data, this->size()};
}

PtrAllocReq<double> ManyBodyOper::alloc_req(const ManyBodyBasis &basis)
{
    *this = ManyBodyOper(nullptr, basis);
    return {&this->_data, this->size()};
}

double hermitivity(const ManyBodyOper &q)
{
    double h = 0.0;
    for (size_t r = 0; r < RANK_COUNT; ++r) {
        Oper q_r = q.oper(r);
        for (size_t l = 0; l < q_r.num_blocks(); ++l) {
            Matrix<const double> q_r_l = q_r[l];
            size_t q_r_l_dim = q_r_l.num_rows();
            assert(q_r_l_dim == q_r_l.num_cols());
            for (size_t u1 = 0; u1 < q_r_l_dim; ++u1) {
                for (size_t u2 = 0; u2 < q_r_l_dim; ++u2) {
                    h += pow(q_r_l(u1, u2) - conj(q_r_l(u2, u1)), 2.0);
                }
            }
        }
    }
    return h;
}
