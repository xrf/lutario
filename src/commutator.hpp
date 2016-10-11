#ifndef COMMUTATOR_HPP
#define COMMUTATOR_HPP
#include "oper.hpp"

void term_2220(double alpha,
               const Oper &a,
               const Oper &b,
               double beta,
               Oper &c);

void term_2222(double alpha,
               const Oper &a,
               const Oper &b,
               double beta,
               Oper &c);

#endif
