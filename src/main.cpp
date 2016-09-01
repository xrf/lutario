#include <assert.h>
#include <stdint.h>
#include <memory>
#include <unordered_map>
#include <vector>

template<typename T>
struct MatView {
    T *data;
    size_t stride;
    MatView(T *data, size_t stride)
        : data(data)
        , stride(stride)
    {
    }
    T &operator()(size_t i, size_t j) const
    {
        return data[i * stride + j];
    }
};

namespace pairing_model {

typedef int Spin;

struct Orbital {
    unsigned n;
    Spin ms;
};

class PairingModelBasis {
public:
    const size_t num_shells;

    const size_t num_filled_shells;

    const std::vector<Orbital> orbitals;

    PairingModelBasis(size_t num_shells, size_t num_filled_shells)
        : num_shells(num_shells), num_filled_shells(num_filled_shells), orbitals(generate_orbitals())
    {

    }
};

}

/// A many-body operator contains three operators:
///
///   - Zero-body operator (constant term).  Always has a single block
///     containing one element.
///
///   - One-body operator.
///
///   - Two-body operator.
///
typedef double *ManyBodyOperator;

template<typename C>
class ManyBodyBasis {
public:
    ManyBodyBasis()
    {
        this->_operator_size = 42424242424242;

        for (size_t l = 0; l < this->_num_blocks_2; ++l) {
            this->_channel_map.emplace(this->_channels[l], l);
        }
    }

    //////////////////////////////////////////////////////////////////////////

    /// Return the number of elements required to store the underlying array
    /// of a many-body operator.
    size_t operator_size() const
    {
        return this->_operator_size;
    }

    /// Convenience function for getting an element from a many-body operator.
    double &get(ManyBodyOperator op, size_t rank, size_t block_index, size_t i, size_t j) const
    {
        return op[this->block_offset(rank, block_index) +
                  i * this->block_stride(rank, block_index) + j];
    }

    size_t block_offset(size_t rank, size_t block_index) const
    {
        return 424242424242;
    }

    size_t block_stride(size_t rank, size_t block_index) const
    {
        return 424242424242;
    }

    //////////////////////////////////////////////////////////////////////////

    size_t num_blocks(size_t rank) const
    {
        return this->_num_blocks(rank);
    }

    bool pack_channel(size_t rank, C channel, size_t &block_index_out) const
    {
        auto it = _channel_map.find(channel);
        if (it == _channel_map.end()) {
            return false;
        }
        size_t l = *it;
        if (l >= this->num_blocks[rank]) {
            return false;
        }
        block_index_out = l;
        return true;
    }

    C unpack_channel(size_t rank, size_t block_index) const
    {
        (void)rank; // rank is not actually being used
        assert(block_index < this->num_blocks(rank));
        return _channels[block_index];
    }

private:
    size_t _operator_size;

    //////////////////////////////////////////////////////////////////////////

    size_t _num_blocks[3];

    /// The channels are stored here in one single array, with the the lower
    /// portions of it containing the one-body channels
    std::vector<C> _channels;

    /// Inverse of _channels
    std::unordered_map<C, size_t> _channel_map;
};

#define ITER_BLOCKS(var, basis, rank)                                          \
    size_t var = 0;                                                            \
    var < basis.num_blocks_##rank();                                           \
    ++var

#define ITER_SUBINDICES(var, block_index, group_begin, group_end, basis, rank) \
    size_t var = basis.block_dim_##rank(block_index, group_begin);             \
    var < basis.block_dim_##rank(block_index, group_end);                      \
    ++var

/// This function is for decorative purposes.  Taking the complex conjugate of
/// a real number has no effect.
inline double conj(double x)
{
    return x;
}

/*\(\b\w\.o\w\)\[\(\w*\)\](*/
/*b.get(\1, \2, */

/// Allocate a many-body operator for the given many-body basis.
template<typename C>
std::unique_ptr<double[]>
alloc_many_body_operator(const ManyBodyBasis<C> &mbasis)
{
    return std::unique_ptr<double[]>(new double[mbasis.operator_size()]());
}

template<typename C>
void calc_white_generator(const ManyBodyBasis<C> &b, ManyBodyOperator h,
                          ManyBodyOperator eta_out)
{
    for (ITER_BLOCKS(li, b, 1)) {
        for (ITER_SUBINDICES(ua, li, 1, 2, b, 1)) {
            for (ITER_SUBINDICES(ui, li, 0, 1, b, 1)) {
                size_t lii = b.add_11(li, li);
                size_t uia = b.combine_11(li, ui, li, ua);
                double z = b.get(h, 1, li, ui, ua) /
                           (b.get(h, 2, lii, uia, uia) +
                            b.get(h, 1, li, ui, ui) - b.get(h, 1, li, ua, ua));
                b.get(eta_out, 1, li, ui, ua) = z;
                b.get(eta_out, 1, li, ua, ui) = -conj(z);
            }
        }
    }
    for (ITER_BLOCKS(lij, b, 2)) {
        for (ITER_SUBINDICES(uab, lij, 3, 4, b, 2)) {
            for (ITER_SUBINDICES(uij, lij, 0, 1, b, 2)) {
                size_t li, ui, lj, uj;
                b.split_2(lij, uij, li, ui, lj, uj);
                size_t la, ua, lb, ub;
                b.split_2(lij, uab, la, ua, lb, ub);

                size_t lia = b.add_11(li, la);
                size_t lib = b.add_11(li, lb);
                size_t lja = b.add_11(lj, la);
                size_t ljb = b.add_11(lj, lb);

                size_t uia = b.combine_11(li, ui, la, ua);
                size_t uib = b.combine_11(li, ui, lb, ub);
                size_t uja = b.combine_11(lj, uj, la, ua);
                size_t ujb = b.combine_11(lj, uj, lb, ub);
                double z =
                    b.get(h, 2, lij, uij, uab) /
                    (b.get(h, 1, li, ui, ui) + b.get(h, 1, lj, uj, uj) -
                     b.get(h, 1, la, ua, ua) - b.get(h, 1, lb, ub, ub) +
                     b.get(h, 2, lia, uia, uia) + b.get(h, 2, lib, uib, uib) +
                     b.get(h, 2, lja, uja, uja) + b.get(h, 2, ljb, ujb, ujb) -
                     b.get(h, 2, lij, uij, uij) - b.get(h, 2, lij, uab, uab));
                b.get(eta_out, 2, lij, uij, uab) = z;
                b.get(eta_out, 2, lij, uab, uij) = -conj(z);
            }
        }
    }
}

/*

/// The White generator.
template<class T>
struct WhiteGenerator {

    // use the reduced form of the commutator that uses only hole-particle
    // matrix elements (hp or hhpp)
    hermitian_commutator<adjoint_symmetry::antihermitian,
                         adjoint_symmetry::hermitian, T, BasisM, true>
        _commutator;

    /// Internal buffer used to store the generator matrix.
    std::unique_ptr<T[]> _gen_data;

    /// Constructor.
    ///
    /// @param basis_m  The many-particle basis.
    WhiteGenerator(const BasisM &basis_m)
        : _basis_m(basis_m), _commutator(_basis_m)
    {
        _gen_data = alloc<T>(matrix_m_size(_basis_m));
        gen = make_matrix_m_view(_basis_m, unmove(_gen_data.get()));
    }

    /// Calculates the generator.
    ///
    /// @param      in      The hermitian operator being evolved.
    template<class MatrixM_in>
    void calc_generator(const MatrixM_in &in)
    {
        auto &&basis_1 = get<1>(_basis_m);
        auto &&basis_2 = get<2>(_basis_m);
        auto &&h1 = get<1>(in);
        auto &&h2 = get<2>(in);
        auto &&eta[1] = get<1>(gen);
        auto &&eta[2] = get<2>(gen);
        for (auto &&li : basis_1.channels())
            for (auto &&ua : basis_1.subindices(li, 1))
                for (auto &&ui : basis_1.subindices(li, 0)) {
                    auto &&lii = l_add_11(_basis_m, li, li);
                    // no need to account for sign when fusing since hole states
                    // always occur before excited states
                    auto &&uia = to_unsigned(u_fuse_11(_basis_m, li, ui, li,
ua));
                    auto &&z = b.get(h, 1, li, ui, ua) /
                               (b.get(h, 2, lii, uia, uia) + b.get(h, 1, li, ui, ui) - b.get(h, 1, li, ua, ua));
                    eta[1][li](ui, ua) = z;
                    eta[1][li](ua, ui) = -conj(z);
                }
        for (auto &&lij : basis_2.channels())
            for (auto &&uab : basis_2.subindices(lij, 2))
                for (auto &&uij : basis_2.subindices(lij, 0)) {
                    auto &&li_ui_lj_uj = lu_split_2(_basis_m, lij, uij);
                    auto &&li = get<0>(li_ui_lj_uj);
                    auto &&ui = get<1>(li_ui_lj_uj);
                    auto &&lj = get<2>(li_ui_lj_uj);
                    auto &&uj = get<3>(li_ui_lj_uj);
                    auto &&la_ua_lb_ub = lu_split_2(_basis_m, lij, uab);
                    auto &&la = get<0>(la_ua_lb_ub);
                    auto &&ua = get<1>(la_ua_lb_ub);
                    auto &&lb = get<2>(la_ua_lb_ub);
                    auto &&ub = get<3>(la_ua_lb_ub);
                    auto &&lia = l_add_11(_basis_m, li, la);
                    auto &&lib = l_add_11(_basis_m, li, lb);
                    auto &&lja = l_add_11(_basis_m, lj, la);
                    auto &&ljb = l_add_11(_basis_m, lj, lb);
                    // no need to account for sign when fusing since hole states
                    // always occur before excited states
                    auto &&uia = to_unsigned(u_fuse_11(_basis_m, li, ui, la,
ua));
                    auto &&uib = to_unsigned(u_fuse_11(_basis_m, li, ui, lb,
ub));
                    auto &&uja = to_unsigned(u_fuse_11(_basis_m, lj, uj, la,
ua));
                    auto &&ujb = to_unsigned(u_fuse_11(_basis_m, lj, uj, lb,
ub));
                    auto &&z = b.get(h, 2, lij, uij, uab) /
                               (b.get(h, 1, li, ui, ui) + b.get(h, 1, lj, uj, uj) - b.get(h, 1, la, ua, ua)
-
                                b.get(h, 1, lb, ub, ub) + b.get(h, 2, lia, uia, uia) +
b.get(h, 2, lib, uib, uib) +
                                b.get(h, 2, lja, uja, uja) + b.get(h, 2, ljb, ujb, ujb) -
                                b.get(h, 2, lij, uij, uij) - b.get(h, 2, lij, uab, uab));
                    b.get(eta, 2, lij, uij, uab) = z;
                    b.get(eta, 2, lij, uab, uij) = -conj(z);
                }
    }

    /// Calculates the generator and evolves the hermitian operator using the
    /// generator.
    ///
    /// @param      in      The hermitian operator being evolved.
    /// @param[out] out     The result.
    template<class MatrixM_in, class MatrixM_out>
    void operator()(const MatrixM_in &in, MatrixM_out &out)
    {
        calc_generator(in);
        _commutator(gen, in, out);
    }
    ntuple_t<std::vector<matrix_view<T>>, BasisM::max_arity> gen;
};

// */

int main()
{
}
// TODO: look up pairing model
