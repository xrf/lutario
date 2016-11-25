#include <assert.h>
#include <math.h>
#include <ostream>
#include <stdexcept>
#include "alloc.hpp"
#include "basis.hpp"
#include "math.hpp"
#include "oper.hpp"

#include <iostream>

PtrAllocReq<double> Oper::alloc_req(const ManyBodyBasis &basis, OperKind kk)
{
    *this = Oper(nullptr, basis, kk);
    return {&this->_data, this->size()};
}

std::ostream &operator<<(std::ostream &stream, const Oper &self)
{
    stream << "{";
    for (size_t l = 0; l < self.num_blocks(); ++l) {
        if (l != 0) {
            stream << ", ";
        }
        stream << "\"block_" << l << "\": " << self[l];
    }
    stream << "}";
    return stream;
}

PtrAllocReq<double> ManyBodyOper::alloc_req(const ManyBodyBasis &basis)
{
    *this = ManyBodyOper(nullptr, basis);
    return {&this->_data, this->size()};
}

std::ostream &operator<<(std::ostream &stream, const ManyBodyOper &self)
{
    stream << "{";
    for (size_t r = 0; r < RANK_COUNT; ++r) {
        if (r != 0) {
            stream << ", ";
        }
        stream << "\"rank_" << r << "\": " << self.oper(r);
    }
    stream << "}";
    return stream;
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
                    if (u1 > u2) {
                        continue;
                    }
                    h += normsq(q_r_l(u1, u2) - conj(q_r_l(u2, u1)));
                }
            }
        }
    }
    return sqrt(h);
}

double exch_antisymmetry(const Oper &q)
{
    const ManyBodyBasis &basis = q.basis();
    double z = 0.0;
    switch (q.kind()) {
    case OPER_KIND_000:
        break;
    case OPER_KIND_100:
        break;
    case OPER_KIND_211:
        throw std::logic_error("not implemented");
    case OPER_KIND_200:
        for (size_t l12 : basis.channels(RANK_2)) {
            basis.for_u20(l12, UNOCC_PP, [&](Orbital o1, Orbital o2) {
                if (o1.to_tuple() > o2.to_tuple()) {
                    return;
                }
                basis.for_u20(l12, UNOCC_PP, [&](Orbital o3, Orbital o4) {
                    if (o3.to_tuple() > o4.to_tuple()) {
                        return;
                    }
                    z += normsq(q(o1, o2, o3, o4) + q(o1, o2, o4, o3))
                       + normsq(q(o1, o2, o3, o4) + q(o2, o1, o3, o4));
                });
            });
        }
    }
    return sqrt(z);
}

double exch_antisymmetry(const ManyBodyOper &q)
{
    double z = 0.0;
    for (size_t r = 0; r < RANK_COUNT; ++r) {
        z += normsq(exch_antisymmetry(q.oper(r)));
    }
    return sqrt(z);
}
