#ifndef PAIRING_MODEL_HPP
#define PAIRING_MODEL_HPP
#include <cassert>
#include <cstddef>
#include <array>
#include <functional>
#include <iosfwd>
#include <unordered_map>
#include <vector>

namespace pairing_model {

/// Spin is stored as twice its normal value.  This way we can represent spins
/// as exact integers (no need to resort to floating-point arithmetic).
typedef int TwiceSpin;

/// Type for the conserved quantum number(s).
class Channel {

    // Invariant: all entries must have nonzero value.
    std::unordered_map<unsigned, TwiceSpin> _entries;

public:

    Channel(unsigned n = 0, TwiceSpin tms = 0);

    const std::unordered_map<unsigned, TwiceSpin> &entries() const;

    Channel operator+(const Channel &other) const;

    Channel operator-(const Channel &other) const;

    bool operator==(const Channel &other) const;

    bool operator!=(const Channel &other) const;

};

std::ostream &operator<<(std::ostream &stream, const Channel &channel);

/// A single-particle state in the pairing model basis.
struct Orbital {

    /// Principal quantum number.
    unsigned n;

    /// Spin quantum number.
    TwiceSpin tms;

    /// Construct an Orbital with the given quantum numbers.
    Orbital(unsigned n, TwiceSpin tms)
        : n(n)
        , tms(tms)
    {
    }

    /// Return the set of conserved quantum numbers, namely the spin quantum
    /// number.
    Channel channel() const
    {
        return Channel(this->n, this->tms);
    }

};

/// Single-particle basis for the pairing model.
class Basis {

    std::vector<Orbital> _orbitals[2];

public:

    /// Construct the list of orbitals.
    Basis(unsigned num_occupied_shells, unsigned num_unoccupied_shells);

    /// Get the list of orbitals.
    ///
    /// @param unoccupied
    /// `0` for the list of occupied orbitals.
    /// `1` for the list of unoccupied orbitals.
    const std::vector<Orbital> &orbitals(std::size_t unoccupied) const;

    /// Construct two lists containing channels for each orbital in the exact
    /// same order (including possibly duplicates).  The first list contains
    /// the occupied channels, while the second list contains the unoccupied
    /// channels.
    std::array<std::vector<Channel>, 2> orbital_channels() const;

};

struct Hamiltonian {

    double strength;

    Hamiltonian(double strength)
        : strength(strength)
    {
    }

    /// Calculate the one-body matrix element.
    ///
    double one_body(Orbital p1, Orbital p2) const
    {
        if (p1.channel() != p2.channel()) {
            return 0.0;
        }
        return this->one_body_conserv(p1, p2);
    }

    /// Calculate the two-body matrix element.
    ///
    /// Pre-condition: the conservation law must hold (i.e. `p1` and `p2` must
    /// reside in the same channel.
    double one_body_conserv(Orbital p1, Orbital p2) const
    {
        assert(p1.channel() == p2.channel());
        if (p1.n != p2.n) {
            return 0.0;
        }
        return p1.n - 1;
    }

    /// Calculate the two-body interaction matrix element.
    double two_body(Orbital p1, Orbital p2, Orbital p3, Orbital p4) const
    {
        if (p1.channel() + p2.channel() != p3.channel() + p4.channel()) {
            return 0.0;
        }
        return this->two_body_conserv(p1, p2, p3, p4);
    }

    /// Calculate the two-body interaction matrix element.
    ///
    /// Pre-condition: the conservation law must hold (i.e. `(p1, p2)` and
    /// `(p3, p4)` must reside in the same channel.
    double two_body_conserv(Orbital p1, Orbital p2, Orbital p3,
                            Orbital p4) const
    {
        assert(p1.channel() + p2.channel() == p3.channel() + p4.channel());
        if (p1.n != p2.n || p3.n != p4.n) {
            return 0.0;
        }
        double sign;
        if (p1.tms == p3.tms) {
            sign = 1.0;
        } else {
            sign = -1.0;
        }
        return sign * this->strength / 2.0;
    }

};

}

namespace std {

template<>
struct hash<pairing_model::Channel> {

    size_t operator()(const pairing_model::Channel &channel) const;

private:

    hash<unsigned> _hash_unsigned;

    hash<pairing_model::TwiceSpin> _hash_TwiceSpin;

};

}

#endif
