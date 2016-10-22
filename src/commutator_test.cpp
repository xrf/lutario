#undef NDEBUG
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
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

unsigned fails;

void fail()
{
    ++fails;
}

struct Location {
    const char *file;
    unsigned long line;
    const char *func;
};

bool within_tolerance(double abserr, double relerr, double x, double y)
{
    return fabs(x - y) < abserr + relerr * 0.5 * fabs(x + y);
}

void overwrite_file(const char *src, const char *dest, const char *tmp_suffix)
{
    if (rename(src, dest) != 0) {
        std::string tmp_fn = dest;
        tmp_fn += tmp_suffix;
        if (rename(dest, tmp_fn.c_str()) != 0) {
            std::ostringstream s;
            s << "can't rename: " << dest << " -> " << tmp_fn;
            throw std::runtime_error(s.str());
        }
        if (rename(src, dest) != 0) {
            std::ostringstream s;
            s << "can't rename: " << src << " -> " << dest;
            throw std::runtime_error(s.str());
        }
        if (remove(tmp_fn.c_str()) != 0) {
            std::cerr << "[warning] can't remove temporary file: "
                      << tmp_fn << std::endl;
        }
    }
}

#define D(f, ...) f({__FILE__, __LINE__, __func__}, __VA_ARGS__)

class QDTest {

public:

    QDTest()
        : _basis(quantum_dot::get_basis(2, 1))
        , _table(this->_basis)
        , _mbasis(StateIndexTable(this->_table))
    {
        // allocate and initialize the operators to zero
        AllocReqBatch<double> batch;
        batch.push(this->_a.alloc_req(this->_mbasis));
        batch.push(this->_b.alloc_req(this->_mbasis));
        batch.push(this->_c.alloc_req(this->_mbasis));
        batch.push(this->_d.alloc_req(this->_mbasis));
        this->_buf = alloc(std::move(batch));

        // load the mock operators (random matrix elements)
        this->_load_mbo("src/commutator_test_qd_a.txt", this->_a);
        this->_load_mbo("src/commutator_test_qd_b.txt", this->_b);

        std::cout << "Orbitals:\n";
        for (size_t p = 0; p < this->_basis.size(); ++p) {
            const quantum_dot::Orbital &o = std::get<0>(this->_basis.at(p));
            Orbital lu = this->_orbital_from_index(p);
            std::cout << p << "\t" << o << "\t" << lu << "\n";
        }

        const StateIndexTable &table = this->_mbasis.table();
        size_t nl1 = table.num_channels(RANK_1);
        size_t nl2 = table.num_channels(RANK_2);
        std::cout << "2-particle state table (20):\n";
        std::cout << "l12" << "\t" << "y1,y2" << "\t"
                  << "l1" << "\t" << "l2" << "\t" << "[uo1,uo2)" << "\n";
        for (size_t l12 = 0; l12 < nl2; ++l12) {
            for (size_t y1 = 0; y1 < 2; ++y1) {
                for (size_t y2 = 0; y2 < 2; ++y2) {
                    for (size_t l1 = 0; l1 < nl1; ++l1) {
                        size_t l2;
                        if (!try_get(table.subtract_channels(l12, l1)
                                    .within({0, nl1}), &l2)) {
                            continue;
                        }
                        size_t x = (y1 * 2 + y2) * nl1 + l1;
                        size_t uoa = table.state_offset(STATE_KIND_20, l12, x);
                        size_t uob = table.state_offset(STATE_KIND_20, l12,
                                                        x + 1);
                        if (uoa == uob) {
                            continue;
                        }
                        std::cout << l12 << "\t" << y1 << "," << y2 << "\t"
                                  << l1 << "\t" << l2
                                  << "\t[" << uoa << ", " << uob << ")\n";
                    }
                }
            }
        }
        std::cout << "2-particle state table (21):\n";
        std::cout << "l14" << "\t" << "y1,y4" << "\t"
                  << "l1" << "\t" << "l4" << "\t" << "[uo1,uo2)" << "\n";
        for (size_t l12 = 0; l12 < nl2; ++l12) {
            for (size_t y1 = 0; y1 < 2; ++y1) {
                for (size_t y2 = 0; y2 < 2; ++y2) {
                    for (size_t l1 = 0; l1 < nl1; ++l1) {
                        size_t l2;
                        if (!try_get(table.subtract_channels(l1, l12)
                                    .within({0, nl1}), &l2)) {
                            continue;
                        }
                        size_t x = (y1 * 2 + y2) * nl1 + l1;
                        size_t uoa = table.state_offset(STATE_KIND_21, l12, x);
                        size_t uob = table.state_offset(STATE_KIND_21, l12,
                                                        x + 1);
                        if (uoa == uob) {
                            continue;
                        }
                        std::cout << l12 << "\t" << y1 << "," << y2 << "\t"
                                  << l1 << "\t" << l2
                                  << "\t[" << uoa << ", " << uob << ")\n";
                    }
                }
            }
        }

        std::cout << std::flush;
    }

    void test()
    {
        this->_term_11ai_test();
        this->_term_11i_11a_test();
        this->_term_12ai_21ai();
        this->_term_12i_12a_21i_21a_test();
        this->_term_22aaii_test();
        this->_term_22aai_test();
        this->_term_22aii_test();
        this->_term_22ai_test();
        this->_term_22ii_test();
        this->_term_22aa_test();
        this->_commutator_test();
    }

private:

    quantum_dot::Basis _basis;

    OrbitalTranslationTable<
        quantum_dot::Orbital,
        quantum_dot::Channel> _table;

    ManyBodyBasis _mbasis;

    ManyBodyOper _a, _b, _c, _d;

    std::unique_ptr<double[]> _buf;

    void _term_11ai_test()
    {
        this->_c = 0.0;

        this->_d.opers[1] = 0.0;
        term_11a(1.0, this->_a.opers[1], this->_b.opers[1], this->_d.opers[1]);
        trace_1(UNOCC_I, 1.0, this->_d.opers[1], this->_c.opers[0]);

        this->_d.opers[1] = 0.0;
        term_11a(1.0, this->_b.opers[1], this->_a.opers[1], this->_d.opers[1]);
        trace_1(UNOCC_I, -1.0, this->_d.opers[1], this->_c.opers[0]);

        std::string fn = "commutator_test_qd_c_11ai.txt";
        this->_save_mbo(("out_" + fn).c_str(), this->_c);
        this->_load_save_mbo(("src/" + fn).c_str(), this->_d);
        D(this->_assert_eq_mbo, 1e-13, 1e-13, this->_c, this->_d);
    }

    void _term_11i_11a_test()
    {
        this->_c = 0.0;

        term_11i(1.0,
                 this->_a.opers[1],
                 this->_b.opers[1],
                 this->_c.opers[1]);
        term_11a(1.0,
                 this->_a.opers[1],
                 this->_b.opers[1],
                 this->_c.opers[1]);
        term_11i(-1.0,
                 this->_b.opers[1],
                 this->_a.opers[1],
                 this->_c.opers[1]);
        term_11a(-1.0,
                 this->_b.opers[1],
                 this->_a.opers[1],
                 this->_c.opers[1]);

        std::string fn = "commutator_test_qd_c_11i_11a.txt";
        this->_save_mbo(("out_" + fn).c_str(), this->_c);
        this->_load_save_mbo(("src/" + fn).c_str(), this->_d);
        D(this->_assert_eq_mbo, 1e-13, 1e-13, this->_c, this->_d);
    }

    void _term_12ai_21ai()
    {
        this->_c = 0.0;

        this->_d.opers[2] = 0.0;
        term_12a_raw(0.5,
                     this->_a.opers[1],
                     this->_b.opers[2],
                     this->_d.opers[2]);
        trace_2(UNOCC_I, 1.0, this->_d.opers[2], this->_c.opers[1]);
        this->_d.opers[2] = 0.0;
        term_21a_raw(0.5,
                     this->_a.opers[2],
                     this->_b.opers[1],
                     this->_d.opers[2]);
        trace_2(UNOCC_I, 1.0, this->_d.opers[2], this->_c.opers[1]);

        this->_d.opers[2] = 0.0;
        term_12a_raw(0.5,
                     this->_b.opers[1],
                     this->_a.opers[2],
                     this->_d.opers[2]);
        trace_2(UNOCC_I, -1.0, this->_d.opers[2], this->_c.opers[1]);
        this->_d.opers[2] = 0.0;
        term_21a_raw(0.5,
                     this->_b.opers[2],
                     this->_a.opers[1],
                     this->_d.opers[2]);
        trace_2(UNOCC_I, -1.0, this->_d.opers[2], this->_c.opers[1]);

        std::string fn = "commutator_test_qd_c_12ai_21ai.txt";
        this->_save_mbo(("out_" + fn).c_str(), this->_c);
        this->_load_save_mbo(("src/" + fn).c_str(), this->_d);
        D(this->_assert_eq_mbo, 1e-13, 1e-13, this->_c, this->_d);
    }

    void _term_12i_12a_21i_21a_test()
    {
        this->_c = 0.0;

        term_12i_raw(1.0,
                     this->_a.opers[1],
                     this->_b.opers[2],
                     this->_c.opers[2]);
        term_12a_raw(1.0,
                     this->_a.opers[1],
                     this->_b.opers[2],
                     this->_c.opers[2]);
        term_21i_raw(1.0,
                     this->_a.opers[2],
                     this->_b.opers[1],
                     this->_c.opers[2]);
        term_21a_raw(1.0,
                     this->_a.opers[2],
                     this->_b.opers[1],
                     this->_c.opers[2]);
        term_12i_raw(-1.0,
                     this->_b.opers[1],
                     this->_a.opers[2],
                     this->_c.opers[2]);
        term_12a_raw(-1.0,
                     this->_b.opers[1],
                     this->_a.opers[2],
                     this->_c.opers[2]);
        term_21i_raw(-1.0,
                     this->_b.opers[2],
                     this->_a.opers[1],
                     this->_c.opers[2]);
        term_21a_raw(-1.0,
                     this->_b.opers[2],
                     this->_a.opers[1],
                     this->_c.opers[2]);
        exch_antisymmetrize(this->_c.opers[2]);

        std::string fn = "commutator_test_qd_c_12i_12a_21i_21a.txt";
        this->_save_mbo(("out_" + fn).c_str(), this->_c);
        this->_load_save_mbo(("src/" + fn).c_str(), this->_d);
        D(this->_assert_eq_mbo, 1e-13, 1e-13, this->_c, this->_d);
    }

    void _term_22aaii_test()
    {
        this->_c = 0.0;

        this->_d.opers[2] = 0.0;
        term_22aa(1.0,
                  this->_a.opers[2],
                  this->_b.opers[2],
                  this->_d.opers[2]);
        this->_d.opers[1] = 0.0;
        trace_2(UNOCC_I, 1.0, this->_d.opers[2], this->_d.opers[1]);
        trace_1(UNOCC_I, 0.5, this->_d.opers[1], this->_c.opers[0]);

        this->_d.opers[2] = 0.0;
        term_22aa(1.0,
                  this->_b.opers[2],
                  this->_a.opers[2],
                  this->_d.opers[2]);
        this->_d.opers[1] = 0.0;
        trace_2(UNOCC_I, 1.0, this->_d.opers[2], this->_d.opers[1]);
        trace_1(UNOCC_I, -0.5, this->_d.opers[1], this->_c.opers[0]);

        std::string fn = "commutator_test_qd_c_22aaii.txt";
        this->_save_mbo(("out_" + fn).c_str(), this->_c);
        this->_load_save_mbo(("src/" + fn).c_str(), this->_d);
        D(this->_assert_eq_mbo, 1e-13, 1e-13, this->_c, this->_d);
    }

    void _term_22aai_test()
    {
        this->_c = 0.0;

        this->_d.opers[2] = 0.0;
        term_22aa(1.0,
                  this->_a.opers[2],
                  this->_b.opers[2],
                  this->_d.opers[2]);
        trace_2(UNOCC_I, 1.0, this->_d.opers[2], this->_c.opers[1]);

        this->_d.opers[2] = 0.0;
        term_22aa(1.0,
                  this->_b.opers[2],
                  this->_a.opers[2],
                  this->_d.opers[2]);
        trace_2(UNOCC_I, -1.0, this->_d.opers[2], this->_c.opers[1]);

        std::string fn = "commutator_test_qd_c_22aai.txt";
        this->_save_mbo(("out_" + fn).c_str(), this->_c);
        this->_load_save_mbo(("src/" + fn).c_str(), this->_d);
        D(this->_assert_eq_mbo, 1e-13, 1e-13, this->_c, this->_d);
    }

    void _term_22aii_test()
    {
        this->_c = 0.0;

        this->_d.opers[2] = 0.0;
        term_22ii(1.0,
                  this->_a.opers[2],
                  this->_b.opers[2],
                  this->_d.opers[2]);
        trace_2(UNOCC_A, -1.0, this->_d.opers[2], this->_c.opers[1]);

        this->_d.opers[2] = 0.0;
        term_22ii(1.0,
                  this->_b.opers[2],
                  this->_a.opers[2],
                  this->_d.opers[2]);
        trace_2(UNOCC_A, 1.0, this->_d.opers[2], this->_c.opers[1]);

        std::string fn = "commutator_test_qd_c_22aii.txt";
        this->_save_mbo(("out_" + fn).c_str(), this->_c);
        this->_load_save_mbo(("src/" + fn).c_str(), this->_d);
        D(this->_assert_eq_mbo, 1e-13, 1e-13, this->_c, this->_d);
    }

    void _term_22ai_test()
    {
        this->_c = 0.0;

        term_22ai(1.0,
                  this->_a.opers[2],
                  this->_b.opers[2],
                  this->_c.opers[2]);
        term_22ai(-1.0,
                  this->_b.opers[2],
                  this->_a.opers[2],
                  this->_c.opers[2]);

        std::string fn = "commutator_test_qd_c_22ai.txt";
        this->_save_mbo(("out_" + fn).c_str(), this->_c);
        this->_load_save_mbo(("src/" + fn).c_str(), this->_d);
        D(this->_assert_eq_mbo, 1e-13, 1e-13, this->_c, this->_d);
    }

    void _term_22ii_test()
    {
        this->_c = 0.0;

        term_22ii(1.0,
                  this->_a.opers[2],
                  this->_b.opers[2],
                  this->_c.opers[2]);
        term_22ii(-1.0,
                  this->_b.opers[2],
                  this->_a.opers[2],
                  this->_c.opers[2]);

        std::string fn = "commutator_test_qd_c_22ii.txt";
        this->_save_mbo(("out_" + fn).c_str(), this->_c);
        this->_load_save_mbo(("src/" + fn).c_str(), this->_d);
        D(this->_assert_eq_mbo, 1e-13, 1e-13, this->_c, this->_d);
    }

    void _term_22aa_test()
    {
        this->_c = 0.0;

        term_22aa(1.0,
                  this->_a.opers[2],
                  this->_b.opers[2],
                  this->_c.opers[2]);
        term_22aa(-1.0,
                  this->_b.opers[2],
                  this->_a.opers[2],
                  this->_c.opers[2]);

        std::string fn = "commutator_test_qd_c_22aa.txt";
        this->_save_mbo(("out_" + fn).c_str(), this->_c);
        this->_load_save_mbo(("src/" + fn).c_str(), this->_d);
        D(this->_assert_eq_mbo, 1e-13, 1e-13, this->_c, this->_d);
    }

    void _commutator_test()
    {
        this->_c = 0.0;

        commutator(this->_d, 1.0, this->_a, this->_b, this->_c);

        std::string fn = "commutator_test_qd_c.txt";
        this->_save_mbo(("out_" + fn).c_str(), this->_c);
        this->_load_save_mbo(("src/" + fn).c_str(), this->_d);
        D(this->_assert_eq_mbo, 1e-13, 1e-13, this->_c, this->_d);
    }

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

    void _assert_eq_mbo(const Location &loc,
                        double relerr, double abserr,
                        const ManyBodyOper &a, const ManyBodyOper &b) const
    {
        std::cerr.precision(std::numeric_limits<double>::max_digits10);
        double va = a(), vb = b();
        if (!within_tolerance(relerr, abserr, va, vb)) {
             std::cerr
                 << loc.file << ":" << loc.line << ":" << loc.func << ":"
                 << " *** discrepancy in ():\n"
                 << "  LHS = " << va << "\n"
                 << "  RHS = " << vb << "\n";
             fail();
        }
        this->_for_oper_1([&](size_t p1, size_t p2,
                              Orbital lu1, Orbital lu2) {
            double va = a(lu1, lu2), vb = b(lu1, lu2);
            if (!within_tolerance(relerr, abserr, va, vb)) {
                std::cerr
                    << loc.file << ":" << loc.line << ":" << loc.func << ":"
                    << " *** discrepancy in ("
                    << p1 << ", " << p2 << "):\n"
                    << "  LHS = " << va << "\n"
                    << "  RHS = " << vb << "\n";
                fail();
            }
        });
        this->_for_oper_2([&](size_t p1, size_t p2,
                              size_t p3, size_t p4,
                              Orbital lu1, Orbital lu2,
                              Orbital lu3, Orbital lu4) {
            double va = a(lu1, lu2, lu3, lu4), vb = b(lu1, lu2, lu3, lu4);
            if (!within_tolerance(relerr, abserr, va, vb)) {
                std::cerr
                    << loc.file << ":" << loc.line << ":" << loc.func << ":"
                    << " *** discrepancy in ("
                    << p1 << ", " << p2 << ", " << p3 << ", " << p4 << "):\n"
                    << "  LHS = " << va << "\n"
                    << "  RHS = " << vb << "\n";
                fail();
            }
        });
    }

    /// Note: the output operator must be already preallocated.
    void _load_mbo(const char *filename, ManyBodyOper &out) const
    {
        std::fstream file(filename, std::ios_base::in);
        if (!file.good()) {
            std::ostringstream s;
            s << "can't open file for reading: " << filename;
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

    /// Similar to `_load_mbo`, but also re-saves the file to
    /// normalize the file format.
    void _load_save_mbo(const char *filename, ManyBodyOper &out) const
    {
        this->_load_mbo(filename, out);
        this->_save_mbo(filename, out);
    }

    void _save_mbo(const char *filename, const ManyBodyOper &in) const
    {
        std::string tmp_fn = filename;
        tmp_fn += ".save.tmp";
        std::fstream file(tmp_fn, std::ios_base::out);
        if (!file.good()) {
            std::ostringstream s;
            s << "can't open temporary file for writing: " << tmp_fn;
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
        overwrite_file(tmp_fn.c_str(), filename, ".save2.tmp");
    }

};

}

int main(void)
{
    QDTest qdtest;
    qdtest.test();
    return fails ? EXIT_FAILURE : 0;
}
