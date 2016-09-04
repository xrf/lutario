#ifndef PAIRING_MODEL_HPP
#define PAIRING_MODEL_HPP
#include <array>
#include <vector>

namespace pairing_model {

/// Spin is stored as twice its normal value.  This way we can represent spins
/// as exact integers (no need to resort to floating-point arithmetic).
typedef int TwiceSpin;

/// The type used to store the set of conserved quantum numbers.
typedef TwiceSpin Channel;

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
        return tms;
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
    const std::vector<Orbital> &orbitals(size_t unoccupied) const;

    /// Construct two lists containing channels for each orbital in the exact
    /// same order (including possibly duplicates).  The first list contains
    /// the occupied channels, while the second list contains the unoccupied
    /// channels.
    std::array<std::vector<Channel>, 2> orbital_channels() const;

};

}

#endif
