#include <assert.h>
#include <stddef.h>
#include <functional>
#include <ostream>
#include <tuple>
#include <vector>
#include "sparse_vector.hpp"
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

bool Orbital::operator==(const Orbital &other) const
{
    return this->n == other.n && this->s == other.s;
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
    for (unsigned n = 0; n < num_total_shells; ++n) {
        for (TwiceSpin s = -1; s <= 1; s += 2) {
            Orbital p = {n, s};
            Channel c = p.channel();
            bool x = n >= num_occupied_shells;
            orbitals.emplace_back(std::move(p), std::move(c), x);
        }
    }
    return orbitals;
}

std::ostream &operator<<(std::ostream &stream, const Basis &self)
{
    stream << "pairing_model::Basis({";
    bool first = true;
    for (const std::tuple<Orbital, Channel, bool> &pcx : self) {
        if (first) {
            first = false;
        } else {
            stream << ", ";
        }
        stream << "{" << std::get<0>(pcx)
               << ", " << std::get<1>(pcx)
               << ", " << std::get<2>(pcx)
               << "}";
    }
    stream << "})";
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

namespace std {

size_t hash<pairing_model::Orbital>::
operator()(const pairing_model::Orbital &p) const
{
    // not sure how good this hash function is tbh
    size_t hn = this->_hash_n(p.n);
    size_t hs = this->_hash_s(p.s);
    // a magical hash combining algorithm from Boost
    return hn ^ (hs + 0x9e3779b9 + (hn << 6) + (hn >> 2));
}

}
