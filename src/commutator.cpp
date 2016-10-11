#include <stddef.h>
#include "matrix.hpp"
#include "oper.hpp"
#include "commutator.hpp"

#define UNOCC_P {0, 2}
#define UNOCC_I {0, 1}
#define UNOCC_A {1, 2}

#define UNOCC_PP {0, 4}
#define UNOCC_II {0, 1}
#define UNOCC_AA {3, 4}

void term_2220(double alpha,
               const Oper &a,
               const Oper &b,
               double beta,
               Oper &c)
{
    const ManyBodyBasis &basis = a.basis();
    assert(&basis == &b.basis());
    assert(&basis == &c.basis());
    assert(a.kind() == OPER_KIND_200);
    assert(b.kind() == OPER_KIND_200);
    assert(c.kind() == OPER_KIND_200);
    size_t nl2 = basis.table().num_channels(RANK_2);
    for (size_t l : IndexRange(0, nl2)) {
        gemm(CblasNoTrans,
             CblasNoTrans,
             0.5 * alpha,
             basis.slice_by_unoccupancy_200(b, l, UNOCC_PP, UNOCC_II),
             basis.slice_by_unoccupancy_200(a, l, UNOCC_II, UNOCC_PP),
             beta,
             c[l]);
    }
}

void term_2222(double alpha,
               const Oper &a,
               const Oper &b,
               double beta,
               Oper &c)
{
    const ManyBodyBasis &basis = a.basis();
    assert(&basis == &b.basis());
    assert(&basis == &c.basis());
    assert(a.kind() == OPER_KIND_200);
    assert(b.kind() == OPER_KIND_200);
    assert(c.kind() == OPER_KIND_200);
    size_t nl2 = basis.table().num_channels(RANK_2);
    for (size_t l : IndexRange(0, nl2)) {
        gemm(CblasNoTrans,
             CblasNoTrans,
             0.5 * alpha,
             basis.slice_by_unoccupancy_200(a, l, UNOCC_PP, UNOCC_AA),
             basis.slice_by_unoccupancy_200(b, l, UNOCC_AA, UNOCC_PP),
             beta,
             c[l]);
    }
}
