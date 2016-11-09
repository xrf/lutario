#include <assert.h>
#include <stddef.h>
#include <map>
#include <ostream>
#include <tuple>
#include <vector>
#include "pairing_model.hpp"
#include "utility.hpp"

namespace pairing_model {

Channel::Channel(const std::map<unsigned, int> &entries)
    : _entries(entries)
{
    // eliminate zero entries to ensure invariant holds
    for (auto ikv = this->_entries.begin(); ikv != this->_entries.end();) {
        if (ikv->second) {
            ++ikv;
        } else {
            ikv = this->_entries.erase(ikv);
        }
    }
}

/// Get the nonzero entries.
const std::map<unsigned, int> &Channel::entries() const
{
    return this->_entries;
}

Channel Channel::operator+(const Channel &other) const
{
    Channel r = *this;
    for (const std::pair<const unsigned, int> &kv : other.entries()) {
        if (!(r._entries[kv.first] += kv.second)) {
            r._entries.erase(kv.first);
        }
    }
    return r;
}

Channel Channel::operator-() const
{
    Channel r = *this;
    for (std::pair<const unsigned, int> &kv : r._entries) {
        kv.second = -kv.second;
    }
    return r;
}

bool Channel::operator==(const Channel &other) const
{
    return this->entries() == other.entries();
}

bool Channel::operator!=(const Channel &other) const
{
    return this->entries() != other.entries();
}

bool Channel::operator<(const Channel &other) const
{
    return this->entries() < other.entries();
}

std::ostream &operator<<(std::ostream &stream, const Channel &vec)
{
    stream << "pairing_model::Channel({";
    bool first = true;
    for (const auto &kv : vec.entries()) {
        if (first) {
            first = false;
        } else {
            stream << ", ";
        }
        stream << "{" << kv.first << ", " << kv.second << "}";
    }
    stream << "})";
    return stream;
}

Channel Orbital::channel() const
{
    return Channel({{this->n, this->tms}});
}

std::tuple<unsigned, int> Orbital::to_tuple() const
{
    return std::make_tuple(this->n, this->tms);
}

bool Orbital::operator==(const Orbital &other) const
{
    return this->to_tuple() == other.to_tuple();
}

bool Orbital::operator!=(const Orbital &other) const
{
    return this->to_tuple() != other.to_tuple();
}

bool Orbital::operator<(const Orbital &other) const
{
    return this->to_tuple() < other.to_tuple();
}

std::ostream &operator<<(std::ostream &stream, const Orbital &self)
{
    stream << "pairing_model::Orbital(" << self.n << ", " << self.tms << ")";
    return stream;
}

Basis get_basis(unsigned num_occupied_shells, unsigned num_unoccupied_shells)
{
    Basis orbitals;
    unsigned num_total_shells = num_occupied_shells + num_unoccupied_shells;
    for (unsigned n = 0; n < num_total_shells; ++n) {
        for (int s = -1; s <= 1; s += 2) {
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
    stream << "pairing_model::Basis";
    write_basis(stream, self);
    return stream;
}

Hamiltonian::Hamiltonian(double g)
    : g(g)
{
}

double Hamiltonian::zero_body() const
{
    return this->zero_body_conserv();
}

double Hamiltonian::zero_body_conserv() const
{
    return 0.0;
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
    return p1.n;
}

double Hamiltonian::two_body(Orbital p1, Orbital p2,
                             Orbital p3, Orbital p4) const
{
    if (p1.channel() + p2.channel() != p3.channel() + p4.channel()) {
        return 0.0;
    }
    return this->two_body_conserv(p1, p2, p3, p4);
}

double Hamiltonian::two_body_conserv(Orbital p1, Orbital p2,
                                     Orbital p3, Orbital p4) const
{
    assert(p1.channel() + p2.channel() == p3.channel() + p4.channel());
    if (p1.channel() + p2.channel() != Channel()) {
        return 0.0;
    }
    double sign;
    if (p1.tms == p3.tms) {
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
