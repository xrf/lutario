#include <stddef.h>
#include <iostream>
#include <memory>
#include "basis.hpp"
#include "commutator.hpp"
#include "imsrg.hpp"
#include "oper.hpp"
#include "ode.hpp"

#include "pairing_model.hpp"
#include "quantum_dot.hpp"

//#define pairing_model quantum_dot

int main()
{
    using pairing_model::Basis;
    using pairing_model::Orbital;
    using pairing_model::Channel;
    using pairing_model::Hamiltonian;

    Basis basis_states = pairing_model::get_basis(4, 2);

    double g = 1.0;
    Hamiltonian hamil = {g};
    std::cout << "g = " << g << std::endl;

    OrbitalTranslationTable<Orbital, Channel> table(basis_states);
    ManyBodyBasis basis{table};

    ManyBodyOper h, hn;
    std::unique_ptr<double[]> h_buf = alloc(h.alloc_req(basis));
    std::unique_ptr<double[]> hn_buf = alloc(hn.alloc_req(basis));

    fill_many_body_oper(table, basis, hamil, h);

    normal_order(h, hn);

    Imsrg imsrg = {hn, &wegner_generator};
    ShampineGordon sg = {imsrg.ode()};
    double e = hn.oper(RANK_0)();
    double s = 0.0;
    std::cout << "(s, E) = (" << s << ", " << e << ")\n";
    std::cout.flush();
    while (true) {
        s += 1.0;
        ShampineGordon::Status status = sg.step(s, {1e-8, 1e-8});
        if (status != ShampineGordon::Status::Ok) {
            std::cout << status << std::endl;
            return EXIT_FAILURE;
        }
        double e_new = hn.oper(RANK_0)();
        std::cout << "(s, E) = (" << s << ", " << e_new << ")\n";
        std::cout.flush();
        if (Tolerance{1e-8, 1e-8}.check(e_new, e)) {
            break;
        }
        e = e_new;
    }

    return 0;
}
