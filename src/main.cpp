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

    Basis basis = pairing_model::get_basis(3, 3);
    std::cout << basis << std::endl;

    double g = 1.0;
    Hamiltonian hamil = {g};
    std::cout << "g = " << g << std::endl;

    OrbitalTranslationTable<Orbital, Channel> trans(basis);
    ManyBodyBasis mbasis{trans};

    ManyBodyOper h;
    std::unique_ptr<double[]> h_buf = alloc(h.alloc_req(mbasis));

    Imsrg imsrg = {h, &wegner_generator};
    ShampineGordon sg = {imsrg.ode()};
    double e = h.oper(RANK_0)();
    double s = 0.0;
    while (true) {
        s += 1.0;
        sg.step(s, {1e-8, 1e-8});
        double e_new = h.oper(RANK_0)();
        if (Tolerance{1e-8, 1e-8}.check(e_new, e)) {
            break;
        }
        e = e_new;
    }

    return 0;
}
