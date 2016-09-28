#include <stddef.h>
#include <iostream>
#include <memory>
#include "many_body_basis.hpp"
#include "pairing_model.hpp"

int main()
{
    using pairing_model::Basis;
    using pairing_model::Orbital;
    using pairing_model::Channel;

    Basis basis = pairing_model::get_basis(3, 3);
    std::cout << basis << std::endl;

    OrbitalTranslationTable<Orbital, Channel> trans(basis);
    ManyBodyBasis mbasis = StateIndexTable(trans);

    ManyBodyOperator h, eta;
    std::unique_ptr<double[]> h_buf = mbasis.alloc_many_body_operator(h);
    std::unique_ptr<double[]> eta_buf = mbasis.alloc_many_body_operator(eta);
#if 0
    calc_white_generator(mbasis, h.get(), eta.get());
#endif

    return 0;
}
