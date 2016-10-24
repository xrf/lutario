#ifndef IMSRG_HPP
#define IMSRG_HPP
#include <functional>
#include <utility>
#include "oper.hpp"

typedef std::function<void (const ManyBodyOper &, ManyBodyOper &)> GeneratorFunction;

class Imsrg {

public:

    Imsrg(const ManyBodyOper &Hamiltonian, GeneratorFunction generator);

    const GeneratorFunction &generator() const;

    const ManyBodyOper &hamiltonian() const;

    double ground_state_energy() const;

    void step();

private:

    GeneratorFunction _generator;

    ManyBodyOper _hamiltonian;

};

#endif
