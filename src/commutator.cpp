#include <assert.h>
#include <stddef.h>
#include "matrix.hpp"
#include "oper.hpp"
#include "commutator.hpp"

#define UNOCC_P {0, 2}
#define UNOCC_I {0, 1}
#define UNOCC_A {1, 2}

#define UNOCC_PP {0, 4}
#define UNOCC_II {0, 1}
#define UNOCC_IA {1, 2}
#define UNOCC_AI {2, 3}
#define UNOCC_AA {3, 4}

void term_22ai(double alpha, const Oper &a, const Oper &b, Oper &c)
{
    const ManyBodyBasis &basis = a.basis();
    assert(&basis == &b.basis());
    assert(&basis == &c.basis());
    assert(a.kind() == OPER_KIND_200);
    assert(b.kind() == OPER_KIND_200);
    assert(c.kind() == OPER_KIND_200);

    for (size_t l12 : basis.channels(RANK_2)) {
        basis.for_u20(l12, UNOCC_PP, [&](Orbital o1, Orbital o2) {
            basis.for_u20(l12, UNOCC_PP, [&](Orbital o3, Orbital o4) {
                size_t l1 = o1.channel_index();
                size_t l3 = o3.channel_index();
                // in our current system, subtracting two rank-1 channels
                // always yields a valid rank-2 channel
                size_t l13 = *basis.table().subtract_channels(l1, l3);
                basis.for_u21(l13, UNOCC_AI, [&](Orbital o5, Orbital o6) {
                     c(o1, o2, o3, o4) += 4.0 * alpha *
                         a(o1, o6, o5, o3) *
                         b(o5, o2, o4, o6);
                });
            });
        });
    }
    for (size_t l12 : basis.channels(RANK_2)) {
        basis.for_u20(l12, UNOCC_PP, [&](Orbital o1, Orbital o2) {
            if (o1.to_tuple() > o2.to_tuple()) {
                return;
            }
            basis.for_u20(l12, UNOCC_PP, [&](Orbital o3, Orbital o4) {
                if (o3.to_tuple() > o4.to_tuple()) {
                    return;
                }
                double z = (
                    c(o1, o2, o3, o4) -
                    c(o1, o2, o4, o3) +
                    c(o2, o1, o4, o3) -
                    c(o2, o1, o3, o4)) / 4.0;
                c(o1, o2, o3, o4) = z;
                c(o1, o2, o4, o3) = -z;
                c(o2, o1, o4, o3) = z;
                c(o2, o1, o3, o4) = -z;
            });
        });
    }
}

void term_22ii(double alpha, const Oper &a, const Oper &b, Oper &c)
{
    const ManyBodyBasis &basis = a.basis();
    assert(&basis == &b.basis());
    assert(&basis == &c.basis());
    assert(a.kind() == OPER_KIND_200);
    assert(b.kind() == OPER_KIND_200);
    assert(c.kind() == OPER_KIND_200);

    for (size_t l : basis.channels(RANK_2)) {
        gemm(CblasNoTrans,
             CblasNoTrans,
             0.5 * alpha,
             basis.slice_by_unoccupancy_200(b, l, UNOCC_PP, UNOCC_II),
             basis.slice_by_unoccupancy_200(a, l, UNOCC_II, UNOCC_PP),
             1.0,
             c[l]);
    }
}

void term_22aa(double alpha, const Oper &a, const Oper &b, Oper &c)
{
    const ManyBodyBasis &basis = a.basis();
    assert(&basis == &b.basis());
    assert(&basis == &c.basis());
    assert(a.kind() == OPER_KIND_200);
    assert(b.kind() == OPER_KIND_200);
    assert(c.kind() == OPER_KIND_200);

    for (size_t l : basis.channels(RANK_2)) {
        gemm(CblasNoTrans,
             CblasNoTrans,
             0.5 * alpha,
             basis.slice_by_unoccupancy_200(a, l, UNOCC_PP, UNOCC_AA),
             basis.slice_by_unoccupancy_200(b, l, UNOCC_AA, UNOCC_PP),
             1.0,
             c[l]);
    }
}
