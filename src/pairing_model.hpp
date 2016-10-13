#ifndef PAIRING_MODEL_HPP
#define PAIRING_MODEL_HPP
#include <stddef.h>
#include <functional>
#include <iosfwd>
#include <tuple>
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

    bool operator==(const Orbital &) const;

};

/// Write an `Orbital` to a stream.
std::ostream &operator<<(std::ostream &, const Orbital &);

/// Single-particle basis for the pairing model.
typedef std::vector<std::tuple<Orbital, Channel, bool>> Basis;

/// Construct the list of orbitals.
Basis get_basis(unsigned num_occupied_shells, unsigned num_unoccupied_shells);

/// Write a `Basis` to a stream.
std::ostream &operator<<(std::ostream &, const Basis &);

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
    double two_body_conserv(Orbital p1, Orbital p2,
                            Orbital p3, Orbital p4) const;

};

/// Write a `Hamiltonian` to a stream.
std::ostream &operator<<(std::ostream &, const Hamiltonian &);

}

namespace std {

template<>
struct hash<pairing_model::Orbital> {

    size_t operator()(const pairing_model::Orbital &orbital) const;

private:

    hash<unsigned> _hash_n;

    hash<pairing_model::TwiceSpin> _hash_s;

};

}

#endif
