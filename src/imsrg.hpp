#ifndef IMSRG_HPP
#define IMSRG_HPP
#include <functional>
#include <memory>
#include "ode.hpp"
#include "oper.hpp"

typedef std::function<void (const ManyBodyOper, ManyBodyOper)> GeneratorFunction;

class Imsrg {

public:

    Imsrg(ManyBodyOper hamiltonian, GeneratorFunction generator);

    const ManyBodyOper &hamiltonian() const;

    const GeneratorFunction &generator() const;

    void deriv(double flow, const double *hamil, double *deriv_out);

    Ode ode();

private:

    ManyBodyOper _hamiltonian, _eta, _tmp;

    GeneratorFunction _generator;

    std::unique_ptr<double []> _buf;

};

#endif
