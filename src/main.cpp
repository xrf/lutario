#include <stddef.h>
#include <iostream>
#include <memory>
#include "basis.hpp"
#include "oper.hpp"

#include "pairing_model.hpp"
#include "quantum_dot.hpp"

//#define pairing_model quantum_dot

int main()
{
    using pairing_model::Basis;
    using pairing_model::Orbital;
    using pairing_model::Channel;

    Basis basis = pairing_model::get_basis(3, 3);
    std::cout << basis << std::endl;

    OrbitalTranslationTable<Orbital, Channel> trans(basis);
    ManyBodyBasis mbasis{StateIndexTable(trans)};

    ManyBodyOper h, eta;
    std::unique_ptr<double[]> h_buf = alloc(h.alloc_req(mbasis));
    std::unique_ptr<double[]> eta_buf = alloc(eta.alloc_req(mbasis));
#if 0
    calc_white_generator(mbasis, h.get(), eta.get());
#endif

    return 0;
}
