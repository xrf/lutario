#include <assert.h>
#include <functional>
#include <iostream>
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

bool run_imsrg(const ManyBodyOper &h)
{
    const ManyBodyBasis &basis = h.basis();
    ManyBodyOper hn;
    std::unique_ptr<double[]> hn_buf = alloc(hn.alloc_req(basis));

    std::cout << "{\"|H - H†|\": " << hermitivity(h)
              << ", \"ATSY(H)\": " << exch_antisymmetry(h)
              << "}" << std::endl;

    normal_order(h, hn);

    Imsrg imsrg = {hn, &white_generator_mp};
    ShampineGordon sg = {imsrg.ode()};
    double e = hn.oper(RANK_0)();
    double s = 0.0;
    std::cout << "{\"s\": " << s
              << ", \"E\": " << e
              << ", \"|H - H†|\": " << hermitivity(hn)
              << ", \"ATSY(H)\": " << exch_antisymmetry(hn)
              << "}" << std::endl;
    while (true) {
        s += 1.0;
        ShampineGordon::Status status = sg.step(s, {1e-8, 1e-8});
        if (status != ShampineGordon::Status::Ok) {
            std::cout << status << std::endl;
            return false;
        }
        double e_new = hn.oper(RANK_0)();
        double herm = hermitivity(hn);
        double atsy = exch_antisymmetry(hn);
        std::cout << "{\"s\": " << s
                  << ", \"E\": " << e_new
                  << ", \"|H - H†|\": " << herm
                  << ", \"ATSY(H)\": " << atsy
                  << "}" << std::endl;
        if (Tolerance{1e-8, 1e-8}.check(e_new, e)) {
            break;
        }
        e = e_new;
    }
    return true;
}
