#ifndef QUANTUM_DOT_HPP
#define QUANTUM_DOT_HPP
#include <iosfwd>
#include <tuple>
#include <vector>

namespace quantum_dot {

/// Type for the conserved quantum number(s).
struct Channel {

    int ml;

    int tms;

    Channel();

    Channel(int ml, int tms);

    std::tuple<int, int> to_tuple() const;

    Channel operator-() const;

    Channel operator+(const Channel &) const;

    bool operator<(const Channel &) const;

};

/// Write a `Channel` to a stream.
std::ostream &operator<<(std::ostream &, const Channel &);

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
    Channel channel() const;

    std::tuple<unsigned, int, int> to_tuple() const;

    bool operator<(const Orbital &) const;

    /// Mostly for compatibility with old code.
    static Orbital from_index(size_t p);

};

/// Write an `Orbital` to a stream.
std::ostream &operator<<(std::ostream &, const Orbital &);

/// Single-particle basis for the pairing model.
typedef std::vector<std::tuple<Orbital, Channel, bool>> Basis;

/// Construct the list of orbitals.
Basis get_basis(unsigned num_occupied_shells, unsigned num_unoccupied_shells);

/// Write a `Basis` to a stream.
std::ostream &operator<<(std::ostream &, const Basis &);

}

#endif
