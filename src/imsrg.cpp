#include "imsrg.hpp"

Imsrg::Imsrg(const ManyBodyOper &hamiltonian, GeneratorFunction generator)
    : _generator(std::move(generator))
    , _hamiltonian(hamiltonian)
{
}

const GeneratorFunction &Imsrg::generator() const
{
    return this->_generator;
}

const ManyBodyOper &Imsrg::hamiltonian() const
{
    return this->_hamiltonian;
}

double Imsrg::ground_state_energy() const
{
    return this->hamiltonian().opers[0]();
}
