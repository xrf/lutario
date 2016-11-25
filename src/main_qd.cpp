#include <stddef.h>
#include <iostream>
#include <memory>
#include "basis.hpp"
#include "commutator.hpp"
#include "imsrg.hpp"
#include "oper.hpp"
#include "ode.hpp"
#include "quantum_dot.hpp"

int main()
{
    std::cout.precision(8);

    using quantum_dot::Basis;
    using quantum_dot::Orbital;
    using quantum_dot::Channel;

    Basis basis_states = quantum_dot::get_basis(2, 1);

    OrbitalTranslationTable<Orbital, Channel> table(basis_states);
    ManyBodyBasis basis{table};

    ManyBodyOper h, hn;
    std::unique_ptr<double[]> h_buf = alloc(h.alloc_req(basis));
    std::unique_ptr<double[]> hn_buf = alloc(hn.alloc_req(basis));

    double omega = 1.0;
    quantum_dot::init_harm_osc(table, omega, h.oper(1));
    std::cout << "loading interaction file ..." << std::flush;
    quantum_dot::load_interaction_file(table, omega, "clh2k_shells=20.dat",
                                       h.oper(2));
    std::cout << "done." << std::endl;

    normal_order(h, hn);

    Imsrg imsrg = {hn, &white_generator};
    ShampineGordon sg = {imsrg.ode()};
    double e = hn.oper(RANK_0)();
    double s = 0.0;
    std::cout << "{\"s\": " << s << ", \"E\": " << e << "}" << std::endl;
    while (true) {
        s += 2.0;
        ShampineGordon::Status status = sg.step(s, {1e-6, 1e-6});
        if (status != ShampineGordon::Status::Ok) {
            std::cout << status << std::endl;
            return EXIT_FAILURE;
        }
        double e_new = hn.oper(RANK_0)();
        double herm = hermitivity(hn);
        std::cout << "{\"s\": " << s
                  << ", \"E\": " << e_new
                  << ", \"|H - H†|\": " << herm << "}" << std::endl;
        if (Tolerance{1e-8, 1e-8}.check(e_new, e)) {
            break;
        }
        e = e_new;
    }

    return 0;
}
