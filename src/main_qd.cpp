#include <stdlib.h>
#include <iostream>
#include <memory>
#include "basis.hpp"
#include "commutator.hpp"
#include "imsrg.hpp"
#include "oper.hpp"
#include "ode.hpp"
#include "quantum_dot.hpp"

int main(int argc, char **argv)
{
    std::cout.precision(8);

    using quantum_dot::Basis;
    using quantum_dot::Orbital;
    using quantum_dot::Channel;

    if (argc != 4) {
        fprintf(stderr, "usage: main_qd <num_occ> <num_unocc> <g>\n");
        fflush(stderr);
        return EXIT_FAILURE;
    }
    Basis basis_states = quantum_dot::get_basis((unsigned)atoi(argv[1]),
                                                (unsigned)atoi(argv[2]));

    OrbitalTranslationTable<Orbital, Channel> table(basis_states);
    ManyBodyBasis basis{table};

    ManyBodyOper h, hn;
    std::unique_ptr<double[]> h_buf = alloc(h.alloc_req(basis));
    std::unique_ptr<double[]> hn_buf = alloc(hn.alloc_req(basis));

    double omega = atof(argv[3]);
    quantum_dot::init_harm_osc(table, omega, h.oper(1));
    std::cout << "loading interaction file ..." << std::flush;
    quantum_dot::load_interaction_file(table, omega, "clh2k_shells=20.dat",
                                       h.oper(2));
    std::cout << "done." << std::endl;

    if (run_imsrg(h) == false) {
        return EXIT_FAILURE;
    }
    return 0;
}
