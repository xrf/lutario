#ifndef PAIRING_MODEL_HPP
#define PAIRING_MODEL_HPP
#include <stddef.h>
#include <array>
#include <iosfwd>
#include <vector>
#include "sparse_vector.hpp"

namespace pairing_model {

/// Spin is stored as twice its normal value.  This way we can represent spins
/// as exact integers without having to resort to floating-point arithmetic.
typedef int TwiceSpin;

/// Type for the conserved quantum number(s).
typedef SparseVector<unsigned, TwiceSpin> Channel;

/// A single-particle state in the pairing model basis.
struct Orbital {

    /// Principal quantum number.
    unsigned n;

    /// Spin quantum number.
    TwiceSpin s;

    /// Construct an Orbital with the given quantum numbers.
    Orbital(unsigned n, TwiceSpin s);

    /// Return the set of conserved quantum numbers.
    Channel channel() const;

};

/// Write an `Orbital` to a stream.
std::ostream &operator<<(std::ostream &, const Orbital &);

/// Single-particle basis for the pairing model.
typedef std::array<std::vector<Orbital>, 2> Basis;

/// Construct the list of orbitals.
Basis get_basis(unsigned num_occupied_shells, unsigned num_unoccupied_shells);

/// Write an `Basis` to a stream.
std::ostream &operator<<(std::ostream &, const Basis &);

typedef std::array<std::vector<Channel>, 2> OrbitalChannels;

/// Construct two lists containing channels for each orbital in the exact
/// same order (including possibly duplicates).  The first list contains
/// the occupied channels, while the second list contains the unoccupied
/// channels.
OrbitalChannels get_orbital_channels(const Basis &);

struct Hamiltonian {

    /// Strength of the two-body interaction.
    double g;

    /// Construct a Hamiltonian with the given two-body interaction strength.
    Hamiltonian(double g);

    /// Calculate the one-body matrix element.
    double one_body(Orbital p1, Orbital p2) const;

    /// Calculate the two-body matrix element.
    ///
    /// Pre-condition: the conservation law must hold (i.e. `p1` and `p2` must
    /// reside in the same channel.
    double one_body_conserv(Orbital p1, Orbital p2) const;

    /// Calculate the two-body interaction matrix element.
    double two_body(Orbital p1, Orbital p2, Orbital p3, Orbital p4) const;

    /// Calculate the two-body interaction matrix element.
    ///
    /// Pre-condition: the conservation law must hold (i.e. `(p1, p2)` and
    /// `(p3, p4)` must reside in the same channel.
    double two_body_conserv(Orbital p1, Orbital p2, Orbital p3,
                            Orbital p4) const;

};

/// Write an `Basis` to a stream.
std::ostream &operator<<(std::ostream &, const Hamiltonian &);

}

std::ostream &operator<<(std::ostream &, const pairing_model::OrbitalChannels &);

#endif
