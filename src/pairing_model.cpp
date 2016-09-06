#include <ostream>
#include "pairing_model.hpp"

namespace pairing_model {

Channel::Channel(unsigned n, TwiceSpin tms)
    : _entries(tms ? std::unordered_map<unsigned, TwiceSpin>{{n, tms}}
                   : std::unordered_map<unsigned, TwiceSpin>{})
{
}

const std::unordered_map<unsigned, TwiceSpin> &Channel::entries() const
{
    return this->_entries;
}

Channel Channel::operator+(const Channel &other) const
{
    Channel r = *this;
    for (const auto &kv : other.entries()) {
        unsigned k = kv.first;
        TwiceSpin v = kv.second;
        if (!(r._entries[k] += v)) {
            r._entries.erase(k);
        }
    }
    return r;
}

Channel Channel::operator-(const Channel &other) const
{
    Channel r = *this;
    for (const auto &kv : other.entries()) {
        unsigned k = kv.first;
        TwiceSpin v = kv.second;
        if (!(r._entries[k] -= v)) {
            r._entries.erase(k);
        }
    }
    return r;
}

bool Channel::operator==(const Channel &other) const
{
    return this->entries() == other.entries();
}

bool Channel::operator!=(const Channel &other) const
{
    return !(*this == other);
}

std::ostream &operator<<(std::ostream &stream, const Channel &channel)
{
    if (channel.entries().empty()) {
        return stream << "0";
    }
    bool first = true;
    for (const auto &kv : channel.entries()) {
        unsigned k = kv.first;
        TwiceSpin v = kv.second;
        if (first) {
            first = false;
        } else if (v >= 0) {
            stream << "+";
        }
        stream << v << "e[" << k << "]";
    }
    return stream;
}

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

namespace std {

size_t hash<pairing_model::Channel>::
operator()(const pairing_model::Channel &channel) const
{
    // not sure how good this hash function is tbh
    size_t h = 0;
    for (const auto &kv : channel.entries()) {
        size_t hk = this->_hash_unsigned(kv.first);
        size_t hv = this->_hash_TwiceSpin(kv.second);
        // a magical hash combining algorithm from Boost
        h ^= hk ^ (hv + 0x9e3779b9 + (hk << 6) + (hk >> 2));
    }
    return h;
}

}
