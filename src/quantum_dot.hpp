#ifndef QUANTUM_DOT_HPP
#define QUANTUM_DOT_HPP
#include <iosfwd>
#include <tuple>
#include <vector>
#include "basis.hpp"
#include "oper.hpp"

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

// The "Simple" matrix element table format
// https://github.com/xrf/clh2-openfci/blob/master/clh2of-simple-format.md
struct Entry {
    uint8_t n1;
    int8_t ml1;
    uint8_t n2;
    int8_t ml2;
    uint8_t n3;
    int8_t ml3;
    uint8_t n4;
    int8_t ml4;
    double value;
};

void init_harm_osc(
    const OrbitalTranslationTable<Orbital, Channel> &table,
    double omega,
    Oper op_out);

void load_interaction_file(
    const OrbitalTranslationTable<Orbital, Channel> &table,
    double omega,
    const char *filename,
    Oper op_out);

}

#endif
