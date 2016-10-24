#ifndef COMMUTATOR_HPP
#define COMMUTATOR_HPP
#include "irange.hpp"
#include "oper.hpp"

void exch_antisymmetrize(Oper &a_inout);

void trace_1(const IndexRange &ys, double alpha, const Oper &a, Oper &r_inout);

// note: the trace is performed over the *second* pair

void trace_2(const IndexRange &ys, double alpha, const Oper &a, Oper &r_inout);

void term_11i(double alpha, const Oper &a, const Oper &b, Oper &r_inout);

void term_11a(double alpha, const Oper &a, const Oper &b, Oper &r_inout);

// note: raw versions are not antisymmetrized and always contract on last pair

void term_12i_raw(double alpha, const Oper &a, const Oper &b, Oper &r_inout);

void term_12a_raw(double alpha, const Oper &a, const Oper &b, Oper &r_inout);

void term_21i_raw(double alpha, const Oper &a, const Oper &b, Oper &r_inout);

void term_21a_raw(double alpha, const Oper &a, const Oper &b, Oper &r_inout);

void term_22ai(double alpha, const Oper &a, const Oper &b, Oper &r_inout);

void term_22aa(double alpha, const Oper &a, const Oper &b, Oper &r_inout);

void term_22ii(double alpha, const Oper &a, const Oper &b, Oper &r_inout);

/// Compute the linked product of two `ManyBodyOper`.  Temporary space must be
/// allocated for this operation in `tmp`.  Note that `tmp` need to be cleared.
void linked_product(ManyBodyOper &tmp,
                    double alpha,
                    const ManyBodyOper &a,
                    const ManyBodyOper &b,
                    ManyBodyOper &r_inout);

void commutator(ManyBodyOper &tmp,
                double alpha,
                const ManyBodyOper &a,
                const ManyBodyOper &b,
                ManyBodyOper &r_inout);

/// Precondition: `r_out` is allocated.
void diagonal_part(const ManyBodyOper &a, ManyBodyOper &r_out);

/// Precondition: `r_out` is allocated.
void wegner_generator(const ManyBodyOper &a, ManyBodyOper &r_out);

/// Precondition: `r_out` is allocated.
void white_generator(const ManyBodyOper &a, ManyBodyOper &r_out);

#endif
