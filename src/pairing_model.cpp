#include <assert.h>
#include <stddef.h>
#include <array>
#include <ostream>
#include <vector>
#include "pairing_model.hpp"

namespace {

void write_orbital_vector(std::ostream &stream,
                          const std::vector<pairing_model::Orbital> &self)
{
    bool first = true;
    stream << "{";
    for (const pairing_model::Orbital &p : self) {
        if (first) {
            first = false;
        } else {
            stream << ", ";
        }
        stream << "{" << p.n << ", " << p.s << "}";
    }
    stream << "}";
}

}

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
    stream << "pairing_model::Orbital(" << self.n << ", " << self.s << ")";
    return stream;
}

Basis get_basis(unsigned num_occupied_shells, unsigned num_unoccupied_shells)
{
    Basis orbitals;
    unsigned num_total_shells = num_occupied_shells + num_unoccupied_shells;
    for (unsigned n = 0; n < num_occupied_shells; ++n) {
        orbitals[0].push_back(Orbital(n, -1));
        orbitals[0].push_back(Orbital(n, 1));
    }
    for (unsigned n = num_occupied_shells; n < num_total_shells; ++n) {
        orbitals[1].push_back(Orbital(n, -1));
        orbitals[1].push_back(Orbital(n, 1));
    }
    return orbitals;
}

std::ostream &operator<<(std::ostream &stream, const Basis &self)
{
    stream << "pairing_model::Basis(";
    write_orbital_vector(stream, self[0]);
    stream << ", ";
    write_orbital_vector(stream, self[1]);
    stream << ")";
    return stream;
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

std::ostream &operator<<(std::ostream &stream, const Hamiltonian &self)
{
    stream << "pairing_model::Hamiltonian(" << self.g << ")";
    return stream;
}

}
