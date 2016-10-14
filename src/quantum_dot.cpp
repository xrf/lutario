#include <ostream>
#include <tuple>
#include "quantum_dot.hpp"
#include "utility.hpp"

namespace quantum_dot {

Channel::Channel()
    : ml()
    , tms()
{
}

Channel::Channel(int ml, int tms)
    : ml(ml)
    , tms(tms)
{
}

std::tuple<int, int> Channel::to_tuple() const
{
    return {this->ml, this->tms};
}

Channel Channel::operator-() const
{
    Channel r = *this;
    r.ml = -r.ml;
    r.tms = -r.tms;
    return r;
}

Channel Channel::operator+(const Channel &other) const
{
    Channel r = *this;
    r.ml += other.ml;
    r.tms += other.tms;
    return r;
}

bool Channel::operator<(const Channel &other) const
{
    return this->to_tuple() < other.to_tuple();
}

std::ostream &operator<<(std::ostream &stream, const Channel &self)
{
    stream << "quantum_dot::Channel("
           << self.ml << ", "
           << self.tms << ")";
    return stream;
}

Channel Orbital::channel() const
{
    return {this->ml, this->tms};
}

std::tuple<unsigned, int, int> Orbital::to_tuple() const
{
    return {this->n, this->ml, this->tms};
}

bool Orbital::operator<(const Orbital &other) const
{
    return this->to_tuple() < other.to_tuple();
}

std::ostream &operator<<(std::ostream &stream, const Orbital &self)
{
    stream << "quantum_dot::Orbital("
           << self.n << ", "
           << self.ml << ", "
           << self.tms << ")";
    return stream;
}

Basis get_basis(unsigned num_occupied_shells, unsigned num_unoccupied_shells)
{
    Basis basis;
    unsigned num_shells = num_occupied_shells + num_unoccupied_shells;
    for (unsigned k = 0; k < num_shells; ++k) {
        bool unocc = k >= num_occupied_shells;
        for (int ml = -(int)k; ml <= (int)k; ml += 2) {
            unsigned n = (k - (unsigned)abs(ml)) / 2;
            for (int tms = -1; tms <= 1; tms += 2) {
                Orbital p = {n, ml, tms};
                Channel c = p.channel();
                basis.emplace_back(std::move(p), std::move(c), unocc);
            }
        }
    }
    return basis;
}

std::ostream &operator<<(std::ostream &stream, const Basis &self)
{
    stream << "quantum_dot::Basis";
    write_basis(stream, self);
    return stream;
}

}
