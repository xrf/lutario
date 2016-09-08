#include <assert.h>
#include <stdlib.h>
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

enum Rank { RANK_0, RANK_1, RANK_2, RANK_COUNT };

enum StateKind {
    STATE_KIND_00,
    STATE_KIND_10,
    STATE_KIND_20,
    STATE_KIND_21,
    STATE_KIND_COUNT
};

Rank state_kind_to_rank(StateKind state_kind)
{
    assert(state_kind < STATE_KIND_COUNT);
    if (state_kind >= STATE_KIND_20) {
        return RANK_2;
    }
    if (state_kind >= STATE_KIND_10) {
        return RANK_1;
    }
    return RANK_0;
}

StateKind standard_state_kind(Rank rank)
{
    switch (rank) {
    case RANK_0:
        return STATE_KIND_00;
    case RANK_1:
        return STATE_KIND_10;
    case RANK_2:
        return STATE_KIND_20;
    default:
        abort(); // unreachable
    }
}

enum OperatorKind {
    OPERATOR_KIND_000,
    OPERATOR_KIND_100,
    OPERATOR_KIND_200,
    OPERATOR_KIND_211,
    OPERATOR_KIND_COUNT
};

Rank operator_kind_to_rank(OperatorKind operator_kind)
{
    assert(operator_kind < OPERATOR_KIND_COUNT);
    if (operator_kind >= OPERATOR_KIND_200) {
        return RANK_2;
    }
    if (operator_kind >= OPERATOR_KIND_100) {
        return RANK_1;
    }
    return RANK_0;
}

OperatorKind standard_operator_kind(Rank rank)
{
    switch (rank) {
    case RANK_0:
        return OPERATOR_KIND_000;
    case RANK_1:
        return OPERATOR_KIND_100;
    case RANK_2:
        return OPERATOR_KIND_200;
    default:
        abort(); // unreachable
    }
}

void split_operator_kind(OperatorKind operator_kind,
                         StateKind *state_kind_1_out,
                         StateKind *state_kind_2_out)
{
    StateKind k1, k2;
    switch (operator_kind) {
    case OPERATOR_KIND_000:
        k1 = STATE_KIND_00;
        k2 = STATE_KIND_00;
        break;
    case OPERATOR_KIND_100:
        k1 = STATE_KIND_10;
        k2 = STATE_KIND_10;
        break;
    case OPERATOR_KIND_200:
        k1 = STATE_KIND_20;
        k2 = STATE_KIND_20;
        break;
    case OPERATOR_KIND_211:
        k1 = STATE_KIND_21;
        k2 = STATE_KIND_21;
        break;
    default:
        abort(); // unreachable
    }
    if (state_kind_1_out) {
        *state_kind_1_out = k1;
    }
    if (state_kind_2_out) {
        *state_kind_2_out = k2;
    }
}

struct ChannelizedOrbital {

    size_t channel_index;

    size_t auxiliary_index;

    ChannelizedOrbital(size_t channel_index, size_t auxiliary_index)
        : channel_index(channel_index)
        , auxiliary_index(auxiliary_index)
    {
    }

};

class ChannelStates {

    Rank _rank;

    size_t _size;

    std::vector<size_t> _orbital_indices;

public:

    ChannelStates(Rank rank)
        : _rank(rank)
        , _size()
    {
    }

    Rank rank() const
    {
        return this->_rank;
    }

    size_t size() const
    {
        return this->_size;
    }

    const size_t *operator[](size_t auxiliary_index) const
    {
        assert(auxiliary_index < this->size());
        return this->_orbital_indices.data() + auxiliary_index * this->rank();
    }

    void emplace_back(std::initializer_list<size_t> orbital_indices)
    {
        assert(this->rank() == orbital_indices.size());
        for (size_t p : orbital_indices) {
            this->_orbital_indices.emplace_back(p);
        }
        ++this->_size;
    }

};

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

    // notation:
    //
    //   - k = state_kind
    //   - c = channel
    //   - l = channel_index
    //   - u = auxiliary_index
    //   - r = rank
    //   - i = particle_index
    //   - p = orbital_index
    //   - n_p = num_orbitals

    // offsets in a matrix that contains a full many-body operator in standard
    // form (000, 100, 200)
    size_t _operator_offsets[RANK_COUNT + 1];

    std::vector<size_t> _block_offsets[OPERATOR_KIND_COUNT];

    // _num_channels[r] gives the number of channels for rank r
    size_t _num_channels[RANK_COUNT];

    // _channels[l] = c
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

    // _channel_map[c] = l
    std::unordered_map<C, size_t> _channel_map;

    // std::get<0>(_num_states_by_channel[k][l]) = n_p
    // std::get<1>(_states_by_channel[k][l])[u * r + i] = p[i]
    std::vector<ChannelStates> _states_by_channel[STATE_KIND_COUNT];

    // _channels_by_state[k][combine(p, n_p)] = (l, u)
    //
    // combine(p, n_p) = ((p[0] * n_p + p[1]) * n_p + p[2]) * n_p + p[3] ...
    std::vector<ChannelizedOrbital> _channels_by_state[STATE_KIND_COUNT];

    // This function must be called in the correct order, starting with
    // channels of rank 0, then rank 1, then rank 2.  Otherwise, it will
    // hopelessly corrupt the data structures.
    bool _get_or_add_channel_index(Rank rank, const C &channel,
                                   size_t *channel_index_out)
    {
        bool exists = this->pack_channel(rank, channel, channel_index_out);
        if (!exists) {
            size_t l = this->_channels.size();
            this->_channel_map.emplace(channel, l);
            this->_channels.emplace_back(channel);
            for (size_t &n : this->_num_channels) {
                ++n;
            }
            if (channel_index_out) {
                *channel_index_out = l;
            }
        }
        return exists;
    }

    // This function must be called in the correct order, starting with
    // channels of rank 0, then rank 1, then rank 2.  Otherwise, it will
    // hopelessly corrupt the data structures.
    //
    // Note: for use during initialization phase only.
    //
    // p[i + 1] must increment faster than p[i]
    void _add_state(StateKind k, const C &channel,
                    std::initializer_list<size_t> ps)
    {
        Rank r = state_kind_to_rank(k);
        size_t l;
        this->_get_or_add_channel_index(r, channel, &l);
        if (l >= this->_states_by_channel[k].size()) {
            this->_states_by_channel[k].resize(l + 1, ChannelStates(r));
        }
        size_t u = this->_states_by_channel[k][l].size();
        this->_states_by_channel[k][l].emplace_back(std::move(ps));
        this->_channels_by_state[k].emplace_back(l, u);
    }

public:

    /// Create a generic `ManyBodyBasis` from a single-particle basis defined
    /// by the provided `orbital_channels`, which contains two vectors: one
    /// for occupied orbitals and one for unoccupied orbitals.  The vectors
    /// contain a sequence of channels for each orbital in some arbitrary
    /// order defined by the single-particle basis itself.
    ///
    /// For example, consider an ordered sequence of 5 orbitals `{v, w, x, y,
    /// z}` that form a single-particle basis, with states `{v, w, x}`
    /// occupied, and `{y, z}` unoccupied.  If we denote the channel of an
    /// orbital `x` by `x_channel`, then one should pass the following as an
    /// argument:
    ///
    ///     {
    ///         {v_channel, w_channel, x_channel},
    ///         {y_channel, z_channel}
    ///     }
    ///
    ManyBodyBasis(const std::array<std::vector<C>, 2> &orbital_channels)
    {
        // add the zero-particle states
        this->_add_state(STATE_KIND_00, C(), {});

        // add the one-particle states
        for (size_t x = 0; x < 2; ++x) {
            for (const C &c : orbital_channels[x]) {
                this->_add_state(STATE_KIND_10, c, {this->num_orbitals()});
            }
        }

        // add the two-particle states
        size_t n_p = this->num_orbitals();
        const auto &lu_by_p = this->_channels_by_state[STATE_KIND_10];
        for (size_t p1 = 0; p1 < n_p; ++p1) {
            for (size_t p2 = 0; p2 < n_p; ++p2) {
                size_t l1 = lu_by_p[p1].channel_index;
                size_t l2 = lu_by_p[p2].channel_index;
                const C &c1 = this->unpack_channel(RANK_1, l1);
                const C &c2 = this->unpack_channel(RANK_1, l2);
                C c12_20 = c1 + c2;
                C c12_21 = c1 - c2;
                // warning: c1 and c2 are references and may expire after next
                // line due to internal data structures being modified
                this->_add_state(STATE_KIND_20, c12_20, {p1, p2});
                this->_add_state(STATE_KIND_21, c12_21, {p1, p2});
            }
        }

        for (OperatorKind k = OperatorKind(); k < OPERATOR_KIND_COUNT;
             k = (OperatorKind)(k + 1)) {
            StateKind k1, k2;
            split_operator_kind(k, &k1, &k2); // assuming row-major
            size_t i = 0;
            size_t n_l = std::min(this->_states_by_channel[k1].size(),
                                  this->_states_by_channel[k2].size());
            for (size_t l = 0; l < n_l; ++l) {
                this->_block_offsets[k].emplace_back(i);
                size_t n_u1, n_u2;
                this->block_size(k, l, &n_u1, &n_u2);
                i += n_u1 * n_u2;
            }
            this->_block_offsets[k].emplace_back(i);
        }
        size_t i = 0;
        for (Rank r = Rank(); r < RANK_COUNT; r = (Rank)(r + 1)) {
            this->_operator_offsets[r] = i;
            i += this->_block_offsets[standard_operator_kind(r)].back();
        }
        this->_operator_offsets[RANK_COUNT] = i;
    }

    size_t num_orbitals() const
    {
        return this->_channels_by_state[STATE_KIND_10].size();
    }

    /// Return the number of elements required to store the underlying array
    /// of a many-body operator.
    size_t full_operator_size() const
    {
        return this->operator_offset(RANK_COUNT);
    }

    /// Offset of an operator inside a full operator.
    size_t operator_offset(Rank rank) const
    {
        assert(rank <= RANK_COUNT);
        return this->_operator_offsets[rank];
    }

    size_t block_offset(OperatorKind operator_kind, size_t channel_index) const
    {
        assert(operator_kind < OPERATOR_KIND_COUNT);
        assert(channel_index < this->_block_offsets[operator_kind].size());
        return this->_block_offsets[operator_kind][channel_index];
    }

    size_t block_stride(OperatorKind operator_kind, size_t channel_index) const
    {
        size_t n;
        // assuming row-major
        this->block_size(operator_kind, channel_index, NULL, &n);
        return n;
    }

    void block_size(OperatorKind operator_kind, size_t channel_index,
                    size_t *block_size_1_out, size_t *block_size_2_out) const
    {
        StateKind k1, k2;
        split_operator_kind(operator_kind, &k1, &k2); // assuming row-major
        size_t n_u1 = this->_states_by_channel[k1][channel_index].size();
        size_t n_u2 = this->_states_by_channel[k2][channel_index].size();
        if (block_size_1_out) {
            *block_size_1_out = n_u1;
        }
        if (block_size_2_out) {
            *block_size_2_out = n_u2;
        }
    }

    /// Convenience function for getting an element from a many-body operator.
    double &get(ManyBodyOperator op, Rank rank, size_t channel_index, size_t i,
                size_t j) const
    {
        OperatorKind k = standard_operator_kind(rank);
        return op[this->operator_offset(rank) +
                  this->block_offset(k, channel_index) +
                  i * this->block_stride(k, channel_index) + j];
    }

    size_t add(size_t r1, size_t r2, size_t r12, size_t l1, size_t l2) const
    {
        const C &c1 = this->unpack_channel(r1, l1);
        const C &c2 = this->unpack_channel(r2, l2);
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

    size_t num_channels(Rank rank) const
    {
        assert(rank < RANK_COUNT);
        return this->_num_channels[rank];
    }

    bool pack_channel(Rank rank, const C &channel,
                      size_t *channel_index_out) const
    {
        auto it = this->_channel_map.find(channel);
        if (it == this->_channel_map.end()) {
            return false;
        }
        size_t l = it->second;
        if (l >= this->num_channels(rank)) {
            return false;
        }
        if (channel_index_out) {
            *channel_index_out = l;
        }
        return true;
    }

    const C &unpack_channel(Rank rank, size_t block_index) const
    {
        (void)rank; // avoid warning about unused `rank` when asserts are off
        assert(block_index < this->num_channels(rank));
        return this->_channels[block_index];
    }

};

#define ITER_BLOCKS(var, basis, rank)                                          \
    size_t var = 0;                                                            \
    var < basis.num_channels(rank);                                            \
    ++var

#define ITER_SUBINDICES(var, block_index, group_begin, group_end, basis, rank) \
    size_t var = basis.block_offset(rank, block_index, group_begin);           \
    var < basis.block_offset_end(rank, block_index, group_end);                \
    ++var

/// Allocate a many-body operator for the given many-body basis.
template<typename C>
std::unique_ptr<double[]>
alloc_many_body_operator(const ManyBodyBasis<C> &mbasis)
{
    return std::unique_ptr<double[]>(new double[mbasis.full_operator_size()]());
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
