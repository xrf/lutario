#include <assert.h>
#include <stddef.h>
#include <array>
#include <ostream>
#include <vector>
#include "pairing_model.hpp"

namespace pairing_model {

Orbital::Orbital(unsigned n, TwiceSpin s)
    : n(n)
    , s(s)
{
}

Channel Orbital::channel() const
{
    return Channel({{this->n, this->s}});
}

std::ostream &operator<<(std::ostream &stream, const Orbital &self)
{
    stream << "Orbital(" << self.n << ", " << self.s << ")";
    return stream;
}

Basis::Basis(unsigned num_occupied_shells, unsigned num_unoccupied_shells)
{
    unsigned num_total_shells = num_occupied_shells + num_unoccupied_shells;
    for (unsigned n = 0; n < num_occupied_shells; ++n) {
        this->_orbitals[0].push_back(Orbital(n, -1));
        this->_orbitals[0].push_back(Orbital(n, 1));
    }
    for (unsigned n = num_occupied_shells; n < num_total_shells; ++n) {
        this->_orbitals[1].push_back(Orbital(n, -1));
        this->_orbitals[1].push_back(Orbital(n, 1));
    }
}

const std::vector<Orbital> &Basis::orbitals(size_t unoccupied) const
{
    return this->_orbitals[unoccupied];
}

std::array<std::vector<Channel>, 2> Basis::orbital_channels() const
{
    std::array<std::vector<Channel>, 2> orbital_channels;
    for (size_t x = 0; x < 2; ++x) {
        for (Orbital p : this->_orbitals[x]) {
            orbital_channels[x].push_back(p.channel());
        }
    }
    return orbital_channels;
}

Hamiltonian::Hamiltonian(double g)
    : g(g)
{
}

double Hamiltonian::one_body(Orbital p1, Orbital p2) const
{
    if (p1.channel() != p2.channel()) {
        return 0.0;
    }
    return this->one_body_conserv(p1, p2);
}

double Hamiltonian::one_body_conserv(Orbital p1, Orbital p2) const
{
    assert(p1.channel() == p2.channel());
    return p1.n - 1;
}

double Hamiltonian::two_body(Orbital p1, Orbital p2, Orbital p3,
                             Orbital p4) const
{
    if (p1.channel() + p2.channel() != p3.channel() + p4.channel()) {
        return 0.0;
    }
    return this->two_body_conserv(p1, p2, p3, p4);
}

double Hamiltonian::two_body_conserv(Orbital p1, Orbital p2, Orbital p3,
                                     Orbital p4) const
{
    assert(p1.channel() + p2.channel() == p3.channel() + p4.channel());
    if (p1.channel() != Channel()) {
        return 0.0;
    }
    double sign;
    if (p1.s == p3.s) {
        sign = -1.0;
    } else {
        sign = 1.0;
    }
    return sign * this->g / 2.0;
}

}
