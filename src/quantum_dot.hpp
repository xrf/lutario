#ifndef QUANTUM_DOT_HPP
#define QUANTUM_DOT_HPP
#include <stddef.h>
#include <functional>
#include <iosfwd>
#include <tuple>
#include <vector>
#include "utility.hpp" // for combine_hash

namespace quantum_dot {

/// Type for the conserved quantum number(s).
struct Channel {

    int ml;

    int tms;

    Channel()
        : ml()
        , tms()
    {
    }

    Channel(int ml, int tms)
        : ml(ml)
        , tms(tms)
    {
    }

    Channel operator-() const
    {
        Channel r = *this;
        r.ml = -r.ml;
        r.tms = -r.tms;
        return r;
    }

    Channel operator+(const Channel &other) const
    {
        Channel r = *this;
        r.ml += other.ml;
        r.tms += other.tms;
        return r;
    }

    bool operator==(const Channel &other) const
    {
        return this->ml == other.ml && this->tms == other.tms;
    }

};

/// Write a `Channel` to a stream.
std::ostream &operator<<(std::ostream &, const Channel &);

inline
std::ostream &operator<<(std::ostream &stream, const Channel &self)
{
    stream << "pairing_model::Channel("
           << self.ml << ", "
           << self.tms << ")";
    return stream;
}

/// A single-particle state in the pairing model basis.
///
/// Values of this type are constructed via aggregate initialization, e.g.:
///
///     Orbital{2, -1, 1}
///
struct Orbital {

    /// Principal quantum number.
    unsigned n;

    /// Angular momentum projection quantum number.
    int ml;

    /// Spin quantum number.
    int tms;

    Orbital() = delete;

    /// Return the set of conserved quantum numbers.
    Channel channel() const
    {
        return Channel(this->ml, this->tms);
    }

    bool operator==(const Orbital &other) const
    {
        return
            this->n == other.n &&
            this->ml == other.ml &&
            this->tms == other.tms;
    }

};

/// Write an `Orbital` to a stream.
std::ostream &operator<<(std::ostream &, const Orbital &);

inline
std::ostream &operator<<(std::ostream &stream, const Orbital &self)
{
    stream << "pairing_model::Orbital("
           << self.n << ", "
           << self.ml << ", "
           << self.tms << ")";
    return stream;
}

/// Single-particle basis for the pairing model.
typedef std::vector<std::tuple<Orbital, Channel, bool>> Basis;

/// Construct the list of orbitals.
Basis get_basis(unsigned num_occupied_shells, unsigned num_unoccupied_shells)
{
    Basis basis;
    unsigned num_shells = num_occupied_shells + num_unoccupied_shells;
    for (int k = 0; k < (int)num_shells; ++k) {
        bool unocc = k >= (int)num_occupied_shells;
        for (int ml = -k; ml <= k; ml += 2) {
            unsigned n = (unsigned)(k - abs(ml)) / 2;
            for (int tms = -1; tms <= 1; tms += 2) {
                Orbital p = {n, ml, tms};
                Channel c = p.channel();
                basis.emplace_back(std::move(p), std::move(c), unocc);
            }
        }
    }
    return basis;
}

/// Write a `Basis` to a stream.
std::ostream &operator<<(std::ostream &, const Basis &);

inline
std::ostream &operator<<(std::ostream &stream, const Basis &self)
{
    stream << "quantum_dot::Basis({";
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

}

namespace std {

template<>
struct hash<quantum_dot::Channel> {

    size_t operator()(const quantum_dot::Channel &x) const
    {
        size_t h = 0;
        h = combine_hash(h, this->_hash_int(x.ml));
        h = combine_hash(h, this->_hash_int(x.tms));
        return h;
    }

private:

    hash<int> _hash_int;

};

template<>
struct hash<quantum_dot::Orbital> {

    size_t operator()(const quantum_dot::Orbital &x) const
    {
        size_t h = 0;
        h = combine_hash(h, this->_hash_uint(x.n));
        h = combine_hash(h, this->_hash_channel(x.channel()));
        return h;
    }

private:

    hash<unsigned> _hash_uint;
    hash<quantum_dot::Channel> _hash_channel;

};

}

#endif
