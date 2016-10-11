#ifndef COMMUTATOR_HPP
#define COMMUTATOR_HPP
#include <stddef.h>
#include "matrix.hpp"

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
    for (size_t l : IndexRange(0, a.size())) {
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
    for (size_t l : IndexRange(0, a.size())) {
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
