#include <assert.h>
#include <stddef.h>
#include "basis.hpp"
#include "math.hpp"
#include "matrix.hpp"
#include "oper.hpp"
#include "commutator.hpp"

void exch_antisymmetrize(Oper &a)
{
    const ManyBodyBasis &basis = a.basis();
    switch (a.kind()) {
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
                    double z = (
                        a(o1, o2, o3, o4) -
                        a(o1, o2, o4, o3) +
                        a(o2, o1, o4, o3) -
                        a(o2, o1, o3, o4)) / 4.0;
                    a(o1, o2, o3, o4) = z;
                    a(o1, o2, o4, o3) = -z;
                    a(o2, o1, o4, o3) = z;
                    a(o2, o1, o3, o4) = -z;
                });
            });
        }
    }
}

void trace_1(const IndexRange &ys, double alpha, const Oper &a, Oper &r)
{
    const ManyBodyBasis &basis = r.basis();
    assert(basis == a.basis());
    assert(basis == r.basis());
    assert(a.kind() == OPER_KIND_100);
    assert(r.kind() == OPER_KIND_000);

    for (size_t l1 : basis.channels(RANK_1)) {
        basis.for_u10(l1, ys, [&](Orbital o1) {
            r() += alpha * a(o1, o1);
        });
    }
}

void trace_2(const IndexRange &ys, double alpha, const Oper &a, Oper &r)
{
    const ManyBodyBasis &basis = r.basis();
    assert(basis == a.basis());
    assert(basis == r.basis());
    assert(a.kind() == OPER_KIND_200);
    assert(r.kind() == OPER_KIND_100);

    for (size_t l1 : basis.channels(RANK_1)) {
        basis.for_u10(l1, UNOCC_P, [&](Orbital o1) {
            basis.for_u10(l1, UNOCC_P, [&](Orbital o2) {
                for (size_t l3 : basis.channels(RANK_1)) {
                    basis.for_u10(l3, ys, [&](Orbital o3) {
                        r(o1, o2) += alpha * a(o1, o3, o2, o3);
                    });
                }
            });
        });
    }
}

void term_11i(double alpha, const Oper &a, const Oper &b, Oper &r)
{
    const ManyBodyBasis &basis = a.basis();
    assert(basis == b.basis());
    assert(basis == r.basis());
    assert(a.kind() == OPER_KIND_100);
    assert(b.kind() == OPER_KIND_100);
    assert(r.kind() == OPER_KIND_100);

    for (size_t l : basis.channels(RANK_1)) {
        gemm(CblasNoTrans,
             CblasNoTrans,
             -alpha,
             basis.slice_by_unoccupancy_100(b, l, UNOCC_P, UNOCC_I),
             basis.slice_by_unoccupancy_100(a, l, UNOCC_I, UNOCC_P),
             1.0,
             r[l]);
    }
}

void term_11a(double alpha, const Oper &a, const Oper &b, Oper &r)
{
    const ManyBodyBasis &basis = a.basis();
    assert(basis == b.basis());
    assert(basis == r.basis());
    assert(a.kind() == OPER_KIND_100);
    assert(b.kind() == OPER_KIND_100);
    assert(r.kind() == OPER_KIND_100);

    for (size_t l : basis.channels(RANK_1)) {
        gemm(CblasNoTrans,
             CblasNoTrans,
             alpha,
             basis.slice_by_unoccupancy_100(a, l, UNOCC_P, UNOCC_A),
             basis.slice_by_unoccupancy_100(b, l, UNOCC_A, UNOCC_P),
             1.0,
             r[l]);
    }
}

void term_12i_raw(double alpha, const Oper &a, const Oper &b, Oper &r)
{
    const ManyBodyBasis &basis = a.basis();
    assert(basis == b.basis());
    assert(basis == r.basis());
    assert(a.kind() == OPER_KIND_100);
    assert(b.kind() == OPER_KIND_200);
    assert(r.kind() == OPER_KIND_200);

    for (size_t l12 : basis.channels(RANK_2)) {
        basis.for_u20(l12, UNOCC_PP, [&](Orbital o1, Orbital o2) {
            basis.for_u20(l12, UNOCC_PP, [&](Orbital o3, Orbital o4) {
                size_t l5 = o4.channel_index();
                basis.for_u10(l5, UNOCC_I, [&](Orbital o5) {
                    r(o1, o2, o3, o4) -=
                        2.0 * alpha * a(o5, o4) * b(o1, o2, o3, o5);
                });
            });
        });
    }
}

void term_12a_raw(double alpha, const Oper &a, const Oper &b, Oper &r)
{
    const ManyBodyBasis &basis = a.basis();
    assert(basis == b.basis());
    assert(basis == r.basis());
    assert(a.kind() == OPER_KIND_100);
    assert(b.kind() == OPER_KIND_200);
    assert(r.kind() == OPER_KIND_200);

    for (size_t l12 : basis.channels(RANK_2)) {
        basis.for_u20(l12, UNOCC_PP, [&](Orbital o1, Orbital o2) {
            basis.for_u20(l12, UNOCC_PP, [&](Orbital o3, Orbital o4) {
                size_t l5 = o2.channel_index();
                basis.for_u10(l5, UNOCC_A, [&](Orbital o5) {
                    r(o1, o2, o3, o4) +=
                        2.0 * alpha * a(o2, o5) * b(o1, o5, o3, o4);
                });
            });
        });
    }
}

void term_21i_raw(double alpha, const Oper &a, const Oper &b, Oper &r)
{
    const ManyBodyBasis &basis = a.basis();
    assert(basis == b.basis());
    assert(basis == r.basis());
    assert(a.kind() == OPER_KIND_200);
    assert(b.kind() == OPER_KIND_100);
    assert(r.kind() == OPER_KIND_200);

    for (size_t l12 : basis.channels(RANK_2)) {
        basis.for_u20(l12, UNOCC_PP, [&](Orbital o1, Orbital o2) {
            basis.for_u20(l12, UNOCC_PP, [&](Orbital o3, Orbital o4) {
                size_t l5 = o2.channel_index();
                basis.for_u10(l5, UNOCC_I, [&](Orbital o5) {
                    r(o1, o2, o3, o4) -=
                        2.0 * alpha * a(o1, o5, o3, o4) * b(o2, o5);
                });
            });
        });
    }
}

void term_21a_raw(double alpha, const Oper &a, const Oper &b, Oper &r)
{
    const ManyBodyBasis &basis = a.basis();
    assert(basis == b.basis());
    assert(basis == r.basis());
    assert(a.kind() == OPER_KIND_200);
    assert(b.kind() == OPER_KIND_100);
    assert(r.kind() == OPER_KIND_200);

    for (size_t l12 : basis.channels(RANK_2)) {
        basis.for_u20(l12, UNOCC_PP, [&](Orbital o1, Orbital o2) {
            basis.for_u20(l12, UNOCC_PP, [&](Orbital o3, Orbital o4) {
                size_t l5 = o4.channel_index();
                basis.for_u10(l5, UNOCC_A, [&](Orbital o5) {
                    r(o1, o2, o3, o4) +=
                        2.0 * alpha * a(o1, o2, o3, o5) * b(o5, o4);
                });
            });
        });
    }
}

void term_22ai(double alpha, const Oper &a, const Oper &b, Oper &c)
{
    const ManyBodyBasis &basis = a.basis();
    assert(basis == b.basis());
    assert(basis == c.basis());
    assert(a.kind() == OPER_KIND_200);
    assert(b.kind() == OPER_KIND_200);
    assert(c.kind() == OPER_KIND_200);

    for (size_t l12 : basis.channels(RANK_2)) {
        basis.for_u20(l12, UNOCC_PP, [&](Orbital o1, Orbital o2) {
            basis.for_u20(l12, UNOCC_PP, [&](Orbital o3, Orbital o4) {
                size_t l1 = o1.channel_index();
                size_t l4 = o4.channel_index();
                // in our current system, subtracting two rank-1 channels
                // always yields a valid rank-2 channel
                size_t l14 = *basis.table().subtract_channels(l1, l4);
                basis.for_u21(l14, UNOCC_AI, [&](Orbital o5, Orbital o6) {
                    c(o1, o2, o3, o4) += -4.0 * alpha *
                        a(o1, o6, o5, o4) *
                        b(o5, o2, o3, o6);
                });
            });
        });
    }

    exch_antisymmetrize(c);
}

void term_22ii(double alpha, const Oper &a, const Oper &b, Oper &c)
{
    const ManyBodyBasis &basis = a.basis();
    assert(basis == b.basis());
    assert(basis == c.basis());
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
    assert(basis == b.basis());
    assert(basis == c.basis());
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

void linked_product(ManyBodyOper &tmp,
                    double alpha,
                    const ManyBodyOper &a,
                    const ManyBodyOper &b,
                    ManyBodyOper &r)
{
    /*

         11ai                                       22aaii
          / \                                         / \
         /   \                                       /   \
        /     \                                     /     \
      11i    11a     12ai          21ai          22aii   22aai
                      / \           / \           / \     / \
                     /   \         /   \         /   \   /   \
                    /     \       /     \       /     \ /     \
                  12i    12a    21i    21a    22ii   22ai    22aa

    */

    // 11i
    term_11i(alpha, a.opers[1], b.opers[1], r.opers[1]);

    // 11a
    tmp.opers[1] = 0.0;
    term_11a(alpha, a.opers[1], b.opers[1], tmp.opers[1]);
    r.opers[1] += tmp.opers[1];

    // 11ai
    trace_1(UNOCC_I, 1.0, tmp.opers[1], r.opers[0]);

    // 12i
    term_12i_raw(alpha, a.opers[1], b.opers[2], r.opers[2]);

    // 12a
    tmp.opers[2] = 0.0;
    term_12a_raw(alpha, a.opers[1], b.opers[2], tmp.opers[2]);
    r.opers[2] += tmp.opers[2];

    // 12ai
    trace_2(UNOCC_I, 0.5, tmp.opers[2], r.opers[1]);

    // 21i
    term_21i_raw(alpha, a.opers[2], b.opers[1], r.opers[2]);

    // 21a
    tmp.opers[2] = 0.0;
    term_21a_raw(alpha, a.opers[2], b.opers[1], tmp.opers[2]);
    r.opers[2] += tmp.opers[2];

    // 21ai
    trace_2(UNOCC_I, 0.5, tmp.opers[2], r.opers[1]);

    // 22ai
    term_22ai(alpha, a.opers[2], b.opers[2], r.opers[2]);

    // 22ii
    tmp.opers[2] = 0.0;
    term_22ii(alpha, a.opers[2], b.opers[2], tmp.opers[2]);
    r.opers[2] += tmp.opers[2];

    // 22aii
    trace_2(UNOCC_A, -1.0, tmp.opers[2], r.opers[1]);

    // 22aa
    tmp.opers[2] = 0.0;
    term_22aa(alpha, a.opers[2], b.opers[2], tmp.opers[2]);
    r.opers[2] += tmp.opers[2];

    // 22aai
    tmp.opers[1] = 0.0;
    trace_2(UNOCC_I, 1.0, tmp.opers[2], tmp.opers[1]);
    r.opers[1] += tmp.opers[1];

    // 22aaii
    trace_1(UNOCC_I, 0.5, tmp.opers[1], r.opers[0]);

    // magically fix everything :D
    exch_antisymmetrize(r.opers[2]);
}

void commutator(ManyBodyOper &tmp,
                double alpha,
                const ManyBodyOper &a,
                const ManyBodyOper &b,
                ManyBodyOper &r)
{
    // the commutator of two operators consists of only the *linked* diagrams;
    // the unlinked diagrams always cancel out
    linked_product(tmp, alpha, a, b, r);
    linked_product(tmp, -alpha, b, a, r);
}

void diagonal_part(const ManyBodyOper &a, ManyBodyOper &r)
{
    const ManyBodyBasis &basis = r.basis();
    assert(basis == a.basis());

    r = 0.0;

    for (size_t l1 : basis.channels(RANK_1)) {
        basis.for_u10(l1, UNOCC_P, [&](Orbital o1) {
            r(o1, o1) = a(o1, o1);
        });
    }

    for (size_t l12 : basis.channels(RANK_2)) {
        basis.for_u20(l12, UNOCC_PP, [&](Orbital o1, Orbital o2) {
            r(o1, o2, o1, o2) = 2.0 * a(o1, o2, o1, o2);
        });
    }

    exch_antisymmetrize(r.opers[2]);
}

void wegner_generator(const ManyBodyOper &a, ManyBodyOper &r)
{
    const ManyBodyBasis &basis = r.basis();
    assert(basis == a.basis());

    r = 0.0;

    for (size_t l1 : basis.channels(RANK_1)) {
        basis.for_u10(l1, UNOCC_P, [&](Orbital o1) {
            basis.for_u10(l1, UNOCC_P, [&](Orbital o2) {
                bool x1 = basis.get_unocc(o1);
                bool x2 = basis.get_unocc(o2);
                r(o1, o2) = a(o1, o2) * (
                    a(o1, o1) - a(o2, o2) +
                    (x2 - x1) * a(o1, o2, o1, o2));
            });
        });
    }

    for (size_t l12 : basis.channels(RANK_2)) {
        basis.for_u20(l12, UNOCC_PP, [&](Orbital o1, Orbital o2) {
            basis.for_u20(l12, UNOCC_PP, [&](Orbital o3, Orbital o4) {
                bool x1 = basis.get_unocc(o1);
                bool x3 = basis.get_unocc(o3);
                r(o1, o2, o3, o4) =
                    + a(o1, o2, o3, o4) * (
                        + 2.0 * (a(o1, o1) - a(o3, o3))
                        + 4.0 * (x3 - x1) * a(o1, o3, o1, o3)
                        + (2.0 * x1 - 1.0) * a(o1, o2, o1, o2)
                        - (2.0 * x3 - 1.0) * a(o3, o4, o3, o4))
                    + (o1 == o3 ? 4.0 * a(o2, o4) * (
                           + a(o1, o2, o1, o2)
                           - a(o3, o4, o3, o4)
                       ) : 0.0);
            });
        });
    }

    exch_antisymmetrize(r.opers[2]);
}

void white_generator(const ManyBodyOper &a, ManyBodyOper &r)
{
    const ManyBodyBasis &basis = r.basis();
    assert(basis == a.basis());

    for (size_t l1 : basis.channels(RANK_1)) {
        basis.for_u10(l1, UNOCC_I, [&](Orbital o1) {
            basis.for_u10(l1, UNOCC_A, [&](Orbital o2) {
                double z = a(o1, o2) / (
                    + a(o1, o1)
                    - a(o2, o2)
                    + a(o1, o2, o1, o2));
                r(o1, o2) = z;
                r(o2, o1) = -conj(z);
            });
        });
    }

    for (size_t l12 : basis.channels(RANK_2)) {
        basis.for_u20(l12, UNOCC_II, [&](Orbital o1, Orbital o2) {
            basis.for_u20(l12, UNOCC_AA, [&](Orbital o3, Orbital o4) {
                double z = 4.0 * a(o1, o2, o3, o4) / (
                    + a(o1, o1)
                    + a(o2, o2)
                    - a(o3, o3)
                    - a(o4, o4)
                    + a(o1, o3, o1, o3)
                    + a(o1, o4, o1, o4)
                    + a(o2, o3, o2, o3)
                    + a(o2, o4, o2, o4)
                    - a(o1, o2, o1, o2)
                    - a(o3, o4, o3, o4));
                r(o1, o2, o3, o4) = z;
                r(o3, o4, o1, o2) = -conj(z);
            });
        });
    }
}
