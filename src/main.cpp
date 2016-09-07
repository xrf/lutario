#include <cassert>
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

    std::size_t _operator_size;

    std::vector<std::size_t> _block_offsets[3];

    std::vector<std::size_t> _block_strides[3];

    //////////////////////////////////////////////////////////////////////////

    // rank -> num_channels
    std::size_t _num_channels[3];

    // ChannelIndex -> Channel
    //
    // The channels are stored here in one single array:
    //
    //   - Element `0` contains the zero channel.
    //
    //   - Elements `[1, num_channels(1))` contain the nonzero one-body
    //     channels.
    //
    //   - Elements `[num_channels(1), num_channels(2))` contain the
    //     two-body channels that aren't also a one-body channel.
    //
    std::vector<C> _channels;

    // channel -> channel_index
    std::unordered_map<C, std::size_t> _channel_map;

    // channel_index -> auxiliary_index -> orbital_index
    typedef std::vector<std::vector<std::size_t>> StatesByChannel1;

    // channel_index -> auxiliary_index -> (orbital_index, orbital_index)
    typedef std::vector<std::vector<std::array<std::size_t, 2>>>
        StatesByChannel2;

    // (orbital_index, ...) -> (channel_index, auxiliary_index)
    typedef std::vector<std::array<std::size_t, 2>> ChannelsByState;

    StatesByChannel1 _states_by_channel_1;

    StatesByChannel2 _states_by_channel_2[2];

    ChannelsByState _channels_by_state_1;

    ChannelsByState _channels_by_state_2[2];

    // This function must be called in the correct order, starting with
    // channels of rank 0, then rank 1, then rank 2.  Otherwise, it will
    // hopelessly corrupt the data structures.
    bool _get_or_add_channel_index(std::size_t rank, const C &channel,
                                   std::size_t *channel_index_out)
    {
        bool exists = this->pack_channel(rank, channel, channel_index_out);
        if (!exists) {
            std::size_t l = this->_channels.size();
            this->_channel_map.emplace(channel, l);
            this->_channels.push_back(channel);
            for (std::size_t &n : this->_num_channels) {
                ++n;
            }
            if (channel_index_out) {
                *channel_index_out = l;
            }
        }
        return exists;
    }

    // Note: for use during initialization phase only.
    void _add_orbital(const C &channel, std::size_t p)
    {
        std::size_t l;
        if (!this->_get_or_add_channel_index(1, channel, &l)) {
            this->_states_by_channel_1.push_back({});
        }
        std::size_t u = this->_states_by_channel_1[l].size();
        this->_channels_by_state_1.push_back({{l, u}});
        this->_states_by_channel_1[l].push_back(p);
    }

    // Note: for use during initialization phase only.
    void _add_state_2(size_t m, const C &channel, std::size_t p1,
                      std::size_t p2)
    {
        std::size_t l;
        if (!this->_get_or_add_channel_index(2, channel, &l)) {
            this->_states_by_channel_2[m].push_back({});
        }
        std::size_t u = this->_states_by_channel_2[m][l].size();
        this->_channels_by_state_2[m]
                                  [p1 * this->num_orbitals() + p2] = {{l, u}};
        this->_states_by_channel_2[m][l].push_back({{p1, p2}});
    }

public:

    ManyBodyBasis(const std::array<std::vector<C>, 2> &orbital_channels)
    {
        // add the zero-particle channel
        this->_get_or_add_channel_index(0, C(), nullptr);

        // add the one-particle channels
        for (std::size_t x = 0; x < 2; ++x) {
            for (const C &c : orbital_channels[x]) {
                this->_add_orbital(c, this->num_orbitals());
            }
        }

        // add the two-particle channels
        std::size_t np = this->num_orbitals();
        this->_channels_by_state_2[0].resize(np * np);
        this->_channels_by_state_2[1].resize(np * np);
        for (std::size_t p1 = 0; p1 < np; ++p1) {
            for (std::size_t p2 = 0; p2 < np; ++p2) {
                std::size_t l1 = this->_channels_by_state_1[p1][0];
                std::size_t l2 = this->_channels_by_state_1[p2][0];
                const C &c1 = this->unpack_channel(1, l1);
                const C &c2 = this->unpack_channel(1, l2);
                this->_add_state_2(0, c1 + c2, p1, p2);
                this->_add_state_2(1, c1 - c2, p1, p2);
            }
        }

        this->_operator_size = 42424242424242;

        this->_block_offsets[0].push_back(0);
        this->_block_offsets[1].push_back(1);
        //this->_block_offsets[3];

        this->_block_strides[0].push_back(1);
        //this->_block_strides[3];
    }

    std::size_t num_orbitals() const
    {
        return this->_channels_by_state_1.size();
    }

    //////////////////////////////////////////////////////////////////////////

    /// Return the number of elements required to store the underlying array
    /// of a many-body operator.
    std::size_t operator_size() const
    {
        return this->_operator_size;
    }

    /// Convenience function for getting an element from a many-body operator.
    double &get(ManyBodyOperator op, std::size_t rank, std::size_t block_index,
                std::size_t i, std::size_t j) const
    {
        return op[this->block_offset(rank, block_index) +
                  i * this->block_stride(rank, block_index) + j];
    }

    std::size_t block_offset(std::size_t rank, std::size_t block_index) const
    {
        assert(rank < 3);
        return this->_block_offsets[rank][block_index];
    }

    std::size_t block_stride(std::size_t rank, std::size_t block_index) const
    {
        assert(rank < 3);
        return this->_block_strides[rank][block_index];
    }

    std::size_t add(std::size_t r1, std::size_t r2, std::size_t r12,
                    std::size_t l1, std::size_t l2) const
    {
        C c1 = this->unpack_channel(r1, l1);
        C c2 = this->unpack_channel(r2, l2);
        C c12 = c1 + c2;
        std::size_t l12;
        this->pack_channel(r12, c12, l12);
        return l12;
    }

    std::size_t sub(std::size_t r1, std::size_t r2, std::size_t r12,
                    std::size_t l1, std::size_t l2) const
    {
        C c1 = this->unpack_channel(r1, l1);
        C c2 = this->unpack_channel(r2, l2);
        C c12 = c1 - c2;
        std::size_t l12;
        this->pack_channel(r12, c12, l12);
        return l12;
    }

    //////////////////////////////////////////////////////////////////////////

    std::size_t num_channels(std::size_t rank) const
    {
        assert(rank < 3);
        return this->_num_channels[rank];
    }

    bool channel_exists(std::size_t rank, C channel) const
    {
        std::size_t l;
        return this->pack_channel(rank, channel, l);
    }

    bool pack_channel(std::size_t rank, C channel,
                      std::size_t *channel_index_out) const
    {
        auto it = this->_channel_map.find(channel);
        if (it == this->_channel_map.end()) {
            return false;
        }
        std::size_t l = it->second;
        if (l >= this->num_channels(rank)) {
            return false;
        }
        if (channel_index_out) {
            *channel_index_out = l;
        }
        return true;
    }

    C unpack_channel(std::size_t rank, std::size_t block_index) const
    {
        (void)rank; // avoid warning about unused `rank` when asserts are off
        assert(block_index < this->num_channels(rank));
        return this->_channels[block_index];
    }

};

#define ITER_BLOCKS(var, basis, rank)                                          \
    std::size_t var = 0;                                                       \
    var < basis.num_channels_##rank();                                         \
    ++var

#define ITER_SUBINDICES(var, block_index, group_begin, group_end, basis, rank) \
    std::size_t var = basis.block_dim_##rank(block_index, group_begin);        \
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
                std::size_t lii = b.add(1, 1, 2, li, li);
                std::size_t uia = b.combine_11(li, ui, li, ua);
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
                std::size_t li, ui, lj, uj;
                b.split_2(lij, uij, li, ui, lj, uj);
                std::size_t la, ua, lb, ub;
                b.split_2(lij, uab, la, ua, lb, ub);

                std::size_t lia = b.add(1, 1, 2, li, la);
                std::size_t lib = b.add(1, 1, 2, li, lb);
                std::size_t lja = b.add(1, 1, 2, lj, la);
                std::size_t ljb = b.add(1, 1, 2, lj, lb);

                std::size_t uia = b.combine_11(li, ui, la, ua);
                std::size_t uib = b.combine_11(li, ui, lb, ub);
                std::size_t uja = b.combine_11(lj, uj, la, ua);
                std::size_t ujb = b.combine_11(lj, uj, lb, ub);
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
