#include <stdlib.h>
#include <iostream>
#include <memory>
#include "basis.hpp"
#include "imsrg.hpp"
#include "oper.hpp"

#include "pairing_model.hpp"

int main(int argc, char **argv)
{
    std::cout.precision(8);

    using pairing_model::Basis;
    using pairing_model::Orbital;
    using pairing_model::Channel;
    using pairing_model::Hamiltonian;

    if (argc != 4) {
        fprintf(stderr, "usage: main <num_occ> <num_unocc> <g>\n");
        fflush(stderr);
        return EXIT_FAILURE;
    }
    Basis basis_states = pairing_model::get_basis((unsigned)atoi(argv[1]),
                                                  (unsigned)atoi(argv[2]));

    double g = atof(argv[3]);
    Hamiltonian hamil = {g};
    std::cout << hamil << std::endl;

    OrbitalTranslationTable<Orbital, Channel> table(basis_states);
    ManyBodyBasis basis{table};

    ManyBodyOper h;
    std::unique_ptr<double[]> h_buf = alloc(h.alloc_req(basis));

    fill_many_body_oper(table, basis, hamil, h);

    if (run_imsrg(h) == false) {
        return EXIT_FAILURE;
    }
    return 0;
}
