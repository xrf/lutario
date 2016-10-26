#include <assert.h>
#include <functional>
#include <utility>
#include "alloc.hpp"
#include "commutator.hpp"
#include "oper.hpp"
#include "imsrg.hpp"

Imsrg::Imsrg(ManyBodyOper hamiltonian, GeneratorFunction generator)
    : _hamiltonian(std::move(hamiltonian))
    , _generator(std::move(generator))
{
    AllocReqBatch<double> batch;
    batch.push(this->_eta.alloc_req(this->hamiltonian().basis()));
    batch.push(this->_tmp.alloc_req(this->hamiltonian().basis()));
    this->_buf = alloc(std::move(batch));
}

const ManyBodyOper &Imsrg::hamiltonian() const
{
    return this->_hamiltonian;
}

const GeneratorFunction &Imsrg::generator() const
{
    return this->_generator;
}

void Imsrg::deriv(double flow,
                  const double *hamil_data,
                  double *deriv_data)
{
    (void)flow; // flow parameter is ignored
    const ManyBodyBasis &basis = this->hamiltonian().basis();
    ManyBodyOper hamil = {const_cast<double *>(hamil_data), basis};
    ManyBodyOper deriv = {deriv_data, basis};
    this->generator()(hamil, /*mut*/this->_eta);
    deriv = 0.0;
    commutator(/*mut*/this->_tmp, 1.0, this->_eta, hamil, /*mut*/deriv);
}

Ode Imsrg::ode()
{
    using namespace std::placeholders;
    return {
        this->hamiltonian().size(),
        0.0,
        this->hamiltonian().data(),
        std::bind(&Imsrg::deriv, this, _1, _2, _3)
    };
}
