#ifndef COMMUTATOR_HPP
#define COMMUTATOR_HPP
#include "irange.hpp"
#include "oper.hpp"

void exch_antisymmetrize_2(Oper &a_inout);

void trace_1(const IndexRange &ys, double alpha, const Oper &a, Oper &r);

void trace_2(const IndexRange &ys, double alpha, const Oper &a, Oper &r);

void term_11i(double alpha, const Oper &a, const Oper &b, Oper &r_out);

void term_11a(double alpha, const Oper &a, const Oper &b, Oper &r_out);

void term_12i_raw(double alpha, const Oper &a, const Oper &b, Oper &r_out);

void term_12a_raw(double alpha, const Oper &a, const Oper &b, Oper &r_out);

void term_21i_raw(double alpha, const Oper &a, const Oper &b, Oper &r_out);

void term_21a_raw(double alpha, const Oper &a, const Oper &b, Oper &r_out);

void term_22ai(double alpha, const Oper &a, const Oper &b, Oper &r_out);

void term_22aa(double alpha, const Oper &a, const Oper &b, Oper &r_out);

void term_22ii(double alpha, const Oper &a, const Oper &b, Oper &r_out);

#endif
