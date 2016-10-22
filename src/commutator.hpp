#ifndef COMMUTATOR_HPP
#define COMMUTATOR_HPP
#include "irange.hpp"
#include "oper.hpp"

void exch_antisymmetrize(Oper &a_inout);

void trace_1(const IndexRange &ys, double alpha, const Oper &a, Oper &r);

// note: the trace is performed over the *second* pair

void trace_2(const IndexRange &ys, double alpha, const Oper &a, Oper &r);

void term_11i(double alpha, const Oper &a, const Oper &b, Oper &r_out);

void term_11a(double alpha, const Oper &a, const Oper &b, Oper &r_out);

// note: raw versions are not antisymmetrized and always contract on last pair

void term_12i_raw(double alpha, const Oper &a, const Oper &b, Oper &r_out);

void term_12a_raw(double alpha, const Oper &a, const Oper &b, Oper &r_out);

void term_21i_raw(double alpha, const Oper &a, const Oper &b, Oper &r_out);

void term_21a_raw(double alpha, const Oper &a, const Oper &b, Oper &r_out);

void term_22ai(double alpha, const Oper &a, const Oper &b, Oper &r_out);

void term_22aa(double alpha, const Oper &a, const Oper &b, Oper &r_out);

void term_22ii(double alpha, const Oper &a, const Oper &b, Oper &r_out);

void linked_product(ManyBodyOper &tmp,
                    double alpha,
                    const ManyBodyOper &a,
                    const ManyBodyOper &b,
                    ManyBodyOper &r_out);

void commutator(ManyBodyOper &tmp,
                double alpha,
                const ManyBodyOper &a,
                const ManyBodyOper &b,
                ManyBodyOper &r_out);

#endif
