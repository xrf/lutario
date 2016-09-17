#ifndef COMMUTATOR_HPP
#define COMMUTATOR_HPP
#include <stddef.h>
#include "matrix.hpp"

// (FUTURE) IDEA: maybe we can avoid storing
// all the states; we could just store the
// channels (l12) and subchannels (l1) and use that information
// to iterate over the elements?  from (l12, l1), we can get l2,
// and with (l1, l2, x1, x2) we know exactly how many u's are possible;
// it is just: n_u_1 * n_u_2
//
// TODO: figure out if this actually improves efficiency
//
// (FUTURE) IDEA #2: we can probably type-erase the channels
// and have an addition table and a negation table; this
// isn't gonna cost us much memory considering we already
// store something like [(l12, l1, X)] where l12 is of even bigger dim than l1!
// would this make things faster or slower? this will probably
// speed up pairing model since the channels are rather complex;
// but for a simple one like circular QD would it be a pessimization?
// tho, it would certainly make the code more pleasant to deal with!

struct Op000 {

};

struct Op100 {

};

struct Op200 {

};

void term_2220(double alpha,
               const Op200 &a,
               const Op200 &b,
               double beta,
               Op200 &c)
{
    for (size_t l : IRange<size_t>(0, a.size())) {
        gemm(CblasNoTrans,
             CblasNoTrans,
             0.5 * alpha,
             b[l].slice_by_part(PART_2_PP, PART_2_II),
             a[l].slice_by_part(PART_2_II, PART_2_PP),
             beta,
             c);
    }
}

void term_2222(double alpha,
               const Op200 &a,
               const Op200 &b,
               double beta,
               Op200 &c)
{
    for (size_t l : IRange<size_t>(0, a.size())) {
        gemm(CblasNoTrans,
             CblasNoTrans,
             0.5 * alpha,
             a[l].slice_by_part(PART_2_PP, PART_2_AA),
             b[l].slice_by_part(PART_2_AA, PART_2_PP),
             beta,
             c);
    }
}

#endif
