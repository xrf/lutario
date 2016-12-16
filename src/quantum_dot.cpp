#include <math.h>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include "commutator.hpp"
#include "math.hpp"
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
    return std::make_tuple(this->ml, this->tms);
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
    stream << "{\"ml\": " << self.ml
           << ", \"ms\": " << (self.tms / 2.0)
           << "}";
    return stream;
}

Channel Orbital::channel() const
{
    return {this->ml, this->tms};
}

std::tuple<unsigned, int, int> Orbital::to_tuple() const
{
    return std::make_tuple(this->n, this->ml, this->tms);
}

bool Orbital::operator<(const Orbital &other) const
{
    return this->to_tuple() < other.to_tuple();
}

Orbital Orbital::from_index(size_t p)
{
    int tms = (int)(p % 2 * 2) - 1;
    unsigned k = (isqrt((unsigned)(4 * p + 1)) - 1) / 2;
    int ml = (int)(p - p % 2) - (int)(k * (k + 2));
    unsigned n = (k - (unsigned)abs(ml)) / 2;
    return {n, ml, tms};
}

std::ostream &operator<<(std::ostream &stream, const Orbital &self)
{
    stream << "{\"n\": " << self.n
           << ", \"ml\": " << self.ml
           << ", \"ms\": " << (self.tms / 2.0)
           << "}";
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
    write_basis(stream, self);
    return stream;
}

void init_harm_osc(
    const OrbitalTranslationTable<Orbital, Channel> &table,
    double omega,
    Oper op_out)
{
    for (unsigned k = 0; ; ++k) {
         for (int ml = -(int)k; ml <= (int)k; ml += 2) {
            unsigned n = (k - (unsigned)abs(ml)) / 2;
            for (int tms = -1; tms <= 1; tms += 2) {
                ::Orbital lu;
                if (!encode_orbital(table, {n, ml, tms}, &lu)) {
                    return;
                }
                op_out(lu, lu) = omega * (k + 1);
            }
        }
    }
}

void load_interaction_file(
    const OrbitalTranslationTable<Orbital, Channel> &table,
    double omega,
    const char *filename,
    Oper op_out)
{
    File file{fopen(filename, "rb")};
    if (!file) {
        std::ostringstream s;
        s << "can't open file: " << filename;
        throw std::runtime_error(s.str());
    }
    size_t num_entries = BUFSIZ / sizeof(Entry);
    std::vector<Entry> entries(num_entries);
    Entry *entries_data = entries.data();
    double sqrt_omega = sqrt(omega);
    size_t num_read;
    do {
        num_read = fread(entries_data, sizeof(*entries_data),
                         num_entries, file.get());
        for (size_t i = 0; i < num_read; ++i) {
            const Entry &entry = entries_data[i];
            double value = 2 * entry.value * sqrt_omega;
            for (int tms1 = -1; tms1 <= 1; tms1 += 2) {
                ::Orbital lu1, lu2, lu3, lu4;
                if (!encode_orbital(table, {entry.n1, entry.ml1, tms1},
                                    &lu1)) {
                    goto done;
                }
                if (!encode_orbital(table, {entry.n3, entry.ml3, tms1},
                                    &lu3)) {
                    goto done;
                }
                for (int tms2 = -1; tms2 <= 1; tms2 += 2) {
                    if (!encode_orbital(table, {entry.n2, entry.ml2, tms2},
                                        &lu2)) {
                        goto done;
                    }
                    if (!encode_orbital(table, {entry.n4, entry.ml4, tms2},
                                        &lu4)) {
                        goto done;
                    }
                    op_out(lu1, lu2, lu3, lu4) = value;
                    op_out(lu2, lu1, lu4, lu3) = value;
                    op_out(lu3, lu4, lu1, lu2) = value;
                    op_out(lu4, lu3, lu2, lu1) = value;
                }
            }
        }
    } while (num_read == num_entries);
done:
    exch_antisymmetrize(op_out);
}

}
