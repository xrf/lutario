#ifndef PAIRING_MODEL_HPP
#define PAIRING_MODEL_HPP
#include <stddef.h>
#include <iosfwd>
#include <map>
#include <tuple>
#include <vector>

namespace pairing_model {

/// Type for the conserved quantum number(s).
class Channel {

public:

    explicit Channel(const std::map<unsigned, int> &entries = {});

    /// Get the nonzero entries.
    const std::map<unsigned, int> &entries() const;

    Channel operator+(const Channel &other) const;

    Channel operator-() const;

    bool operator==(const Channel &other) const;

    bool operator!=(const Channel &other) const;

    bool operator<(const Channel &other) const;

private:

    // Invariant: all entries must have nonzero value.
    std::map<unsigned, int> _entries;

};

std::ostream &operator<<(std::ostream &, const Channel &);

/// A single-particle state in the pairing model basis.
struct Orbital {

    /// Principal quantum number.
    unsigned n;

    /// Spin quantum number.
    ///
    /// It is stored as twice its normal value, so we can represent spins as
    /// exact integers without having to resort to floating-point arithmetic.
    int tms;

    Orbital() = delete;

    /// Return the set of conserved quantum numbers.
    Channel channel() const;

    std::tuple<unsigned, int> to_tuple() const;

    bool operator==(const Orbital &) const;

    bool operator!=(const Orbital &) const;

    bool operator<(const Orbital &) const;

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

#endif
