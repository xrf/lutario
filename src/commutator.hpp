#ifndef COMMUTATOR_HPP
#define COMMUTATOR_HPP
#include "oper.hpp"

void term_22ai(double alpha, const Oper &a, const Oper &b, Oper &c_out);

void term_22aa(double alpha, const Oper &a, const Oper &b, Oper &c_out);

void term_22ii(double alpha, const Oper &a, const Oper &b, Oper &c_out);

#endif
