#include "pairing_model.hpp"

namespace pairing_model {

Basis::Basis(unsigned num_occupied_shells, unsigned num_unoccupied_shells)
{
    unsigned num_total_shells = num_occupied_shells + num_unoccupied_shells;
    for (unsigned n = 0; n < num_occupied_shells; ++n) {
        this->_orbitals[0].push_back(Orbital(n, -1));
        this->_orbitals[0].push_back(Orbital(n, 1));
    }
    for (unsigned n = num_occupied_shells; n < num_total_shells; ++n) {
        this->_orbitals[1].push_back(Orbital(n, -1));
        this->_orbitals[1].push_back(Orbital(n, 1));
    }
}

const std::vector<Orbital> &Basis::orbitals(size_t unoccupied) const
{
    return this->_orbitals[unoccupied];
}

std::array<std::vector<Channel>, 2> Basis::orbital_channels() const
{
    std::array<std::vector<Channel>, 2> orbital_channels;
    for (size_t x = 0; x < 2; ++x) {
        for (Orbital p : this->_orbitals[x]) {
            orbital_channels[x].push_back(p.channel());
        }
    }
    return orbital_channels;
}

}
