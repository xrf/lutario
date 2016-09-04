#include <assert.h>
#include <stdint.h>
#include <memory>
#include <unordered_map>
#include <vector>
#include "pairing_model.hpp"

/// This function is for decorative purposes.  Taking the complex conjugate of
/// a real number has no effect.
inline double conj(double x)
{
    return x;
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

/// The `C` type must be an abelian group and support the following binary
/// operators:
///
///   - `+` ("addition")
///   - `-` ("subtraction", inverse of "addition")
///
/// The default constructor of `C` must construct the additive identity
/// ("zero").
///
template<typename C>
class ManyBodyBasis {

    size_t _operator_size;

    std::vector<size_t> _block_offsets[3];

    std::vector<size_t> _block_strides[3];

    //////////////////////////////////////////////////////////////////////////

    /// Number of channels by operator rank
    size_t _num_channels[3];

    /// The channels are stored here in one single array, with the the lower
    /// portions of it containing the one-body channels
    std::vector<C> _channels;

    /// Inverse of _channels
    std::unordered_map<C, size_t> _channel_map;

    /// Stores the orbital indices grouped by block index
    std::vector<std::vector<size_t>> _orbitals_by_channel;

    // This function must be called in the correct order, starting with
    // channels of rank 0, then rank 1, then rank 2.  Otherwise, it will
    // hopelessly corrupt the data structures.
    //
    // Pre-condition: the channel must not already exist.
    size_t _add_channel(size_t rank, const C &channel)
    {
        size_t l = this->_channels.size();
        this->_channel_map.emplace(channel, l);
        this->_channels.push_back(channel);
        for (size_t &n : this->_num_channels) {
            ++n;
        }
        return l;
    }

    // Note: for use during initialization phase only.
    void _add_orbital(size_t orbital_index, const C &channel)
    {
        size_t l;
        if (!this->pack_channel(1, channel, l)) {
            l = this->_add_channel(1, channel);
            this->_orbitals_by_channel.push_back(std::vector<size_t>());
        }
        this->_orbitals_by_channel[l].push_back(orbital_index);
    }

public:
    ManyBodyBasis(const std::array<std::vector<C>, 2> &orbital_channels)
    {
        // add the zero-body channel
        this->_add_channel(0, C());

        // add the one-body channels
        size_t p = 0;
        for (size_t x = 0; x < 2; ++x) {
            for (const C &c : orbital_channels[x]) {
                this->_add_orbital(p, c);
                ++p;
            }
        }

        // add the two-body channels (sometimes we need subtraction too...)
        // TODO: perhaps we can rewrite this in a different way, by reusing
        // the (TODO) thing that generates all the 2-particle states.
        size_t num_channels_1 = this->num_channels(1);
        for (size_t l1 = 0; l1 < num_channels_1; ++l1) {
            for (size_t l2 = 0; l2 < num_channels_1; ++l2) {
                C c1 = this->unpack_channel(1, l1);
                C c2 = this->unpack_channel(1, l2);
                C c12 = c1 + c2;
                if (!this->channel_exists(2, c12)) {
                    this->_add_channel(2, c12);
                }
            }
        }
        for (size_t l1 = 0; l1 < num_channels_1; ++l1) {
            for (size_t l2 = 0; l2 < num_channels_1; ++l2) {
                C c1 = this->unpack_channel(1, l1);
                C c2 = this->unpack_channel(1, l2);
                C c12 = c1 - c2;
                if (!this->channel_exists(2, c12)) {
                    this->_add_channel(2, c12);
                }
            }
        }

        this->_operator_size = 42424242424242;

        this->_block_offsets[0].push_back(0);
        this->_block_offsets[1].push_back(1);
        //this->_block_offsets[3];

        this->_block_strides[0].push_back(1);
        //this->_block_strides[3];
    }

    //////////////////////////////////////////////////////////////////////////

    /// Return the number of elements required to store the underlying array
    /// of a many-body operator.
    size_t operator_size() const
    {
        return this->_operator_size;
    }

    /// Convenience function for getting an element from a many-body operator.
    double &get(double *op, size_t rank, size_t block_index,
                size_t i, size_t j) const
    {
        return op[this->block_offset(rank, block_index) +
                  i * this->block_stride(rank, block_index) + j];
    }

    size_t block_offset(size_t rank, size_t block_index) const
    {
        assert(rank < 3);
        return this->_block_offsets[rank][block_index];
    }

    size_t block_stride(size_t rank, size_t block_index) const
    {
        assert(rank < 3);
        return this->_block_strides[rank][block_index];
    }

    size_t add(size_t r1, size_t r2, size_t r12, size_t l1, size_t l2) const
    {
        C c1 = this->unpack_channel(r1, l1);
        C c2 = this->unpack_channel(r2, l2);
        C c12 = c1 + c2;
        size_t l12;
        this->pack_channel(r12, c12, l12);
        return l12;
    }

    size_t sub(size_t r1, size_t r2, size_t r12, size_t l1, size_t l2) const
    {
        C c1 = this->unpack_channel(r1, l1);
        C c2 = this->unpack_channel(r2, l2);
        C c12 = c1 - c2;
        size_t l12;
        this->pack_channel(r12, c12, l12);
        return l12;
    }

    //////////////////////////////////////////////////////////////////////////

    size_t num_channels(size_t rank) const
    {
        assert(rank < 3);
        return this->_num_channels[rank];
    }

    bool channel_exists(size_t rank, C channel) const
    {
        size_t l;
        return pack_channel(rank, channel, l);
    }

    bool pack_channel(size_t rank, C channel, size_t &block_index_out) const
    {
        auto it = _channel_map.find(channel);
        if (it == _channel_map.end()) {
            return false;
        }
        size_t l = it->second;
        if (l >= this->num_channels(rank)) {
            return false;
        }
        block_index_out = l;
        return true;
    }

    C unpack_channel(size_t rank, size_t block_index) const
    {
        (void)rank; // rank is not actually being used
        assert(block_index < this->num_channels(rank));
        return _channels[block_index];
    }
};

#define ITER_BLOCKS(var, basis, rank)                                          \
    size_t var = 0;                                                            \
    var < basis.num_channels_##rank();                                         \
    ++var

#define ITER_SUBINDICES(var, block_index, group_begin, group_end, basis, rank) \
    size_t var = basis.block_dim_##rank(block_index, group_begin);             \
    var < basis.block_dim_##rank(block_index, group_end);                      \
    ++var

/// Allocate a many-body operator for the given many-body basis.
template<typename C>
std::unique_ptr<double[]>
alloc_many_body_operator(const ManyBodyBasis<C> &mbasis)
{
    return std::unique_ptr<double[]>(new double[mbasis.operator_size()]());
}

template<typename C>
void calc_white_generator(const ManyBodyBasis<C> &b, const ManyBodyOperator &h,
                          ManyBodyOperator &eta_out)
{
    /* SELECT i, a WHERE x(i) = 0 AND x(a) = 1 AND g(i, a) AND g(i, i) AND g(a, a) AND G(i, a, i, a) */
    for (ITER_BLOCKS(li, b, 1)) {
        for (ITER_SUBINDICES(ua, li, 1, 2, b, 1)) {
            for (ITER_SUBINDICES(ui, li, 0, 1, b, 1)) {
                size_t lii = b.add(1, 1, 2, li, li);
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

                size_t lia = b.add(1, 1, 2, li, la);
                size_t lib = b.add(1, 1, 2, li, lb);
                size_t lja = b.add(1, 1, 2, lj, la);
                size_t ljb = b.add(1, 1, 2, lj, lb);

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
    pairing_model::Basis basis(3, 3);
    ManyBodyBasis<pairing_model::Channel> mbasis(basis.orbital_channels());
}
// TODO: look up pairing model
