#undef NDEBUG
#include <assert.h>
#include <math.h>
#include <fstream>
#include <ios>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <sstream>
#include "alloc.hpp"
#include "basis.hpp"
#include "commutator.hpp"
#include "oper.hpp"
#include "quantum_dot.hpp"
#include "str.hpp"

namespace {

bool within_tolerance(double abserr, double relerr, double x, double y)
{
    return fabs(x - y) < abserr + relerr * 0.5 * fabs(x + y);
}

class QDTest {

public:

    QDTest()
        : _basis(quantum_dot::get_basis(2, 1))
        , _table(this->_basis)
        , _mbasis(StateIndexTable(this->_table))
    {
    }

    void term_22ai_test()
    {
        ManyBodyOper a, b, c, c_old;

        AllocReqBatch<double> batch;
        batch.push(a.alloc_req(this->_mbasis));
        batch.push(b.alloc_req(this->_mbasis));
        batch.push(c.alloc_req(this->_mbasis));
        batch.push(c_old.alloc_req(this->_mbasis));
        std::unique_ptr<double[]> buf = alloc(std::move(batch));

        this->_load_mbo("src/commutator_test_qd_a.txt", a);
        this->_load_mbo("src/commutator_test_qd_b.txt", b);
        this->_load_mbo("src/commutator_test_qd_c_22ai.txt", c_old);

        term_22ai(1.0, a.opers[2], b.opers[2], c.opers[2]);
        term_22ai(-1.0, b.opers[2], a.opers[2], c.opers[2]);

        this->_save_mbo("out_commutator_test_qd_a.txt", a);
        this->_save_mbo("out_commutator_test_qd_b.txt", b);
        this->_save_mbo("out_commutator_test_qd_c.txt", c);

        this->_assert_eq_mbo(1e-15, 1e-15, c, c_old);
    }

private:

    quantum_dot::Basis _basis;

    const OrbitalTranslationTable<
        quantum_dot::Orbital,
        quantum_dot::Channel> _table;

    ManyBodyBasis _mbasis;

    Orbital _orbital_from_index(size_t p) const
    {
        if (p >= this->_basis.size()) {
            std::ostringstream s;
            s << "orbital index out of range: " << p;
            throw std::runtime_error(s.str());
        }
        const quantum_dot::Orbital &orbital = std::get<0>(this->_basis.at(p));
        size_t l, u;
        if (!try_get(this->_table.encode_channel(orbital.channel()), &l)) {
            std::ostringstream s;
            s << "cannot encode channel: " << orbital.channel();
            throw std::runtime_error(s.str());
        }
        if (!try_get(this->_table.encode_orbital(orbital), &u)) {
            std::ostringstream s;
            s << "cannot encode orbital: " << orbital;
            throw std::runtime_error(s.str());
        }
        return {l, u};
    }

    template<typename F>
    void _for_oper_1(F callback) const
    {
        size_t np = this->_basis.size();
        for (size_t p1 = 0; p1 < np; ++p1) {
            Orbital lu1 = this->_orbital_from_index(p1);
            for (size_t p2 = 0; p2 < np; ++p2) {
                Orbital lu2 = this->_orbital_from_index(p2);
                if (!this->_mbasis.is_conserved_1(lu1, lu2)) {
                    continue;
                }
                callback(p1, p2, lu1, lu2);
            }
        }
    }

    template<typename F>
    void _for_oper_2(F callback) const
    {
        size_t np = this->_basis.size();
        for (size_t p1 = 0; p1 < np; ++p1) {
            Orbital lu1 = this->_orbital_from_index(p1);
            for (size_t p2 = 0; p2 < np; ++p2) {
                Orbital lu2 = this->_orbital_from_index(p2);
                for (size_t p3 = 0; p3 < np; ++p3) {
                    Orbital lu3 = this->_orbital_from_index(p3);
                    for (size_t p4 = 0; p4 < np; ++p4) {
                        Orbital lu4 = this->_orbital_from_index(p4);
                        if (!this->_mbasis.is_conserved_2(lu1, lu2,
                                                          lu3, lu4)) {
                            continue;
                        }
                        callback(p1, p2, p3, p4, lu1, lu2, lu3, lu4);
                    }
                }
            }
        }
    }

    void _assert_eq_mbo(double relerr, double abserr,
                        const ManyBodyOper &a, const ManyBodyOper &b) const
    {
        double va = a(), vb = b();
        if (!within_tolerance(relerr, abserr, va, vb)) {
             std::cerr
                 << "error: discrepancy in 0-body matrix element:\n"
                 << "  LHS = " << va << "\n"
                 << "  RHS = " << vb << "\n";
             abort();
        }
        this->_for_oper_1([&](size_t p1, size_t p2,
                              Orbital lu1, Orbital lu2) {
            double va = a(lu1, lu2), vb = b(lu1, lu2);
            if (!within_tolerance(relerr, abserr, va, vb)) {
                std::cerr
                    << "error: discrepancy in 1-body matrix element ("
                    << p1 << ", " << p2 << "):\n"
                    << "  LHS = " << va << "\n"
                    << "  RHS = " << vb << "\n";
                abort();
            }
        });
        this->_for_oper_2([&](size_t p1, size_t p2,
                              size_t p3, size_t p4,
                              Orbital lu1, Orbital lu2,
                              Orbital lu3, Orbital lu4) {
            double va = a(lu1, lu2, lu3, lu4), vb = b(lu1, lu2, lu3, lu4);
            if (!within_tolerance(relerr, abserr, va, vb)) {
                std::cerr
                    << "error: discrepancy in 2-body matrix element ("
                    << p1 << ", " << p2 << ", " << p3 << ", " << p4 << "):\n"
                    << "  LHS = " << va << "\n"
                    << "  RHS = " << vb << "\n";
                abort();
            }
        });
    }

    /// Note: the output operator must be already preallocated.
    void _load_mbo(const char *filename, ManyBodyOper &out) const
    {
        std::fstream file(filename, std::ios_base::in);
        if (!file.good()) {
            std::ostringstream s;
            s << filename << ": cannot open file for reading";
            throw std::runtime_error(s.str());
        }
        file.precision(std::numeric_limits<double>::max_digits10);

        out = 0.0; // clear everything

        std::string line;
        while (std::getline(file, line)) {
            std::string trimmed_line = trim(line);
            // skip empty or commented lines
            if (trimmed_line.size() == 0 || trimmed_line[0] == '#') {
                continue;
            }
            std::istringstream line_stream(line);
            std::vector<double> xs;
            double x;
            while (line_stream >> x) {
                xs.emplace_back(x);
            }
            if (xs.size() == 0) {
                std::ostringstream s;
                s << filename << ": expected numbers: " << line;
                throw std::runtime_error(s.str());
            }
            std::vector<Orbital> ps;
            for (size_t i = 0; i < xs.size() - 1; ++i) {
                ps.emplace_back(this->_orbital_from_index((size_t)xs[i]));
            }
            switch (ps.size()) {
            case 0:
                out() = xs.back();
                break;
            case 2:
                out(ps[0], ps[1]) = xs.back();
                break;
            case 4:
                out(ps[0], ps[1], ps[2], ps[3]) = xs.back();
                break;
            default:
                std::ostringstream s;
                s << filename << ": wrong number of entries: " << line;
                throw std::runtime_error(s.str());
            }
        }
    }

    void _save_mbo(const char *filename, const ManyBodyOper &in) const
    {
        std::fstream file(filename, std::ios_base::out);
        if (!file.good()) {
            std::ostringstream s;
            s << filename << ": cannot open file for writing";
            throw std::runtime_error(s.str());
        }
        file.precision(std::numeric_limits<double>::max_digits10);
        double value = in();
        if (!(value == 0.0)) {
             file << value << "\n";
        }
        this->_for_oper_1([&](size_t p1, size_t p2,
                              Orbital lu1, Orbital lu2) {
            double value = in(lu1, lu2);
            if (!(value == 0.0)) {
                file << p1 << " " << p2 << " "
                     << value << "\n";
            }
        });
        this->_for_oper_2([&](size_t p1, size_t p2,
                              size_t p3, size_t p4,
                              Orbital lu1, Orbital lu2,
                              Orbital lu3, Orbital lu4) {
                double value = in(lu1, lu2, lu3, lu4);
                if (!(value == 0.0)) {
                    file << p1 << " " << p2 << " "
                         << p3 << " " << p4 << " "
                         << value << "\n";
                }
            }
        );
    }

};

}

int main(void)
{
    QDTest qdtest;
    qdtest.term_22ai_test();
    return 0;
}
