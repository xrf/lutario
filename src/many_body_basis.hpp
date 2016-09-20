#ifndef MANY_BODY_BASIS_HPP
#define MANY_BODY_BASIS_HPP
#include <assert.h>
#include <stdlib.h>
#include <memory>
#include <unordered_map>
#include <vector>

/// A many-body operator contains three operators in standard form:
///
///   - Zero-body operator (constant term) in 000 form.  This is always has a
///     single block containing one element.
///
///   - One-body operator in 100 form.
///
///   - Two-body operator in 200 form.
///
typedef double *ManyBodyOperator;

/// Determines the number of particles that an operator couples.
///
///   - A zero-body operator has `RANK_0`.
///   - A one-body operator has `RANK_1`.
///   - A two-body operator has `RANK_2`.
///
enum Rank { RANK_0, RANK_1, RANK_2, RANK_COUNT };

/// Determines the number of particles in a state and how their channels are
/// to be combined.
enum StateKind {
    STATE_KIND_00,
    STATE_KIND_10,
    STATE_KIND_20,
    STATE_KIND_21,
    STATE_KIND_COUNT
};

/// Get the number of particles in a state of the given kind.
inline Rank state_kind_to_rank(StateKind state_kind)
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

/// Determines the rank of an operator as well as how the matrix element
/// blocks of the operator are to be organized.
enum OperatorKind {
    OPERATOR_KIND_000,
    OPERATOR_KIND_100,
    OPERATOR_KIND_200,
    OPERATOR_KIND_211,
    OPERATOR_KIND_COUNT
};

/// Get the rank of an operator with the given kind.
inline Rank operator_kind_to_rank(OperatorKind operator_kind)
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

/// Get the canonical operator representation for a given operator rank.
inline OperatorKind standard_operator_kind(Rank rank)
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

/// Get the kind of the left states and the kind of right states that the
/// operator couples.
inline void split_operator_kind(OperatorKind operator_kind,
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

/// A channelized representation of orbitals.  Each orbital is determined by
///
///   - a channel index (associated with a channel in some `ManyBodyBasis`) and
///   - an auxiliary index to identify the orbital within the said channel.
///
struct Orbital {

    size_t channel_index;

    size_t auxiliary_index;

    Orbital(size_t channel_index, size_t auxiliary_index)
        : channel_index(channel_index)
        , auxiliary_index(auxiliary_index)
    {
    }

};

/// A sequential data structure used to store orbital indices of some rank.
/// The `rank` is set dynamically, allowing `States` of different rank to be
/// stored together using the same type.
///
/// The data type is conceptually isomorphic to the following type:
///
///     struct {
///         Rank rank;
///         vector<size_t> part_offsets;
///         vector<vector<size_t>> orbital_indices;
///     };
///
/// with the invariant that for all `u`, `orbital_indices[u].size() == rank`.
///
class States {

    Rank _rank;

    size_t _size;

    std::vector<size_t> _part_offsets;

    std::vector<size_t> _orbital_indices;

public:

    States(Rank rank, size_t num_parts)
        : _rank(rank)
        , _size()
        , _part_offsets(num_parts + 1)
    {
    }

    Rank rank() const
    {
        return this->_rank;
    }

    size_t num_parts() const
    {
        assert(this->_part_offsets.size() > 0);
        return this->_part_offsets.size() - 1;
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

    size_t part_offset(size_t part) const {
        assert(part < this->_part_offsets.size());
        return this->_part_offsets[part];
    }

    /// Note: parts must be added in ascending order to avoid data corruption.
    void add(size_t part, std::initializer_list<size_t> orbital_indices)
    {
        assert(part < this->num_parts());
        assert(orbital_indices.size() == this->rank());
        for (size_t p : orbital_indices) {
            this->_orbital_indices.emplace_back(p);
        }
        ++this->_size;
        for (size_t x = part; x < this->num_parts(); ++x) {
            this->_part_offsets[part + 1] = this->_size;
        }
    }

};

inline size_t pack_part(std::initializer_list<size_t> part)
{
    size_t xs = 0;
    for (size_t x : part) {
        xs = xs * 2 + x;
    }
    return xs;
}

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
struct ChannelTable {

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

};

template<typename C>
class ChannelTranslationTable {

    // Contains all the channel sets, partitioned by rank.
    //
    // [ rank-0 ] [ == rank-1-exclusive == ] [ ==== rank-2-exclusive ==== ]
    //
    // There's only one rank-0 channel, namely the group identity.  The number
    // of rank-1 channels is given by `_num_channels_1`.
    //
    // The channel sets are cumulative: rank-2 channels include rank-1
    // channels, which include rank-0 channels.  This may not coincide with
    // the physical channel sets.  In fact, it usually does not because in
    // most fermionic systems the physical rank-1 channel set is disjoint from
    // the physical rank-0 or rank-2 channel sets due to the spin projection
    // being half-integers.  Nonetheless, it is useful to have a universal
    // translation table is not rank-dependent to avoid implementation
    // complexity.
    //
    std::vector<C> _decode_table;

    // Inverse mapping of `_decode_table`.  Must satisfy:
    //
    //   - for all `l`, `_encode_table[_decode_table[l]] == l`
    //   - for all `c`, `_decode_table[_encode_table[c]] == c`
    //
    std::unordered_map<C, size_t> _encode_table;

    // Must store the number of single-particle channels here because nothing
    // else knows about it.
    size_t _num_channels_1;

    void _insert(const C &c)
    {
        if (this->_encode_table.find(c) != this->_encode_table.end()) {
            return;
        }
        this->_encode_table.emplace(c, this->_decode_table.size());
        this->_decode_table.emplace_back(c);
    }

public:

    ChannelTranslationTable(const std::vector<C> &orbital_channels)
    {
        this->_insert(C());

        for (const C &c : orbital_channels) {
            this->_insert(c);
            // make sure set is closed under negation
            this->_insert(-c);
        }

        this->_num_channels_1 = this->_decode_table.size();

        for (const C &c1 : orbital_channels) {
            for (const C &c2 : orbital_channels) {
                this->_insert(c1 + c2);
            }
        }
    }

    size_t num_channels(Rank rank) const
    {
        switch (rank) {
        case RANK_0:
            return 1;
        case RANK_1:
            return this->_num_channels_1;
        case RANK_2:
            return this->_decode_table.size();
        }
    }

    bool encode(const C &c, size_t *l_out) const
    {
        auto it = this->_encode_table.find(c);
        if (it == this->_encode_table.end()) {
            return false;
        }
        *l_out = it->second;
        return true;
    }

    const C &decode(size_t l) const
    {
        return this->_decode_table.at(l);
    }

    bool add(size_t l1, size_t l2, size_t *l_out) const
    {
        const C &c1 = this->decode(l1);
        const C &c2 = this->decode(l1);
        return this->encode(c1 + c2, l_out);
    }

    size_t negate(size_t l) const
    {
        size_t l_out;
        if (!this->encode(-this->decode(l), &l_out)) {
            throw std::logic_error("channel sets not closed under negation");
        }
        return l_out;
    }

};

class ChannelIndexGroup {

    // Since L1 is a subset of L2, this table suffices for all our purposes.
    //
    // (L2, L1) -> L1
    //
    std::vector<size_t> _addition_table;

    // Negation table
    //
    // L2 -> L2
    //
    std::vector<size_t> _negation_table;

    size_t _num_channels_1;

public:

    ChannelIndexGroup(const ChannelTranslationTable &table)
        : _addition_table(table.num_channels(RANK_2) *
                          table.num_channels(RANK_1))
        , _negation_table(table.num_channels(RANK_1))
    {
        nl1 = table.num_channels(RANK_1);
        nl2 = table.num_channels(RANK_2);
        for (size_t l1 = 0; l < nl2; ++l) {
            for (size_t l2 = 0; l < nl1; ++l) {
                this->_addition_table[l1 * nl1 + l2] = table.add(l1 + l2);
            }
        }
        for (size_t l = 0; l < nl1; ++l) {
            this->_negation_table[l] = table.negate(l);
        }
    }

    size_t num_channels(Rank rank) const
    {
        switch (rank) {
        case RANK_0:
            return 1;
        case RANK_1:
            return this->_num_channels_1;
        case RANK_2:
            return this->_negation_table.size();
        }
    }

    size_t negate(size_t l) const
    {
        return this->_negation_table[l];
    }

    bool add(size_t l1, size_t l2, size_t *l_out) const
    {
        size_t l = this->_addition_table[l1 * nl1 + l2];
        if (l >= this->num_channels(RANK_2)) {
            return false;
        }
        *l_out = l;
        return true;
    }

    bool sub_12(size_t l1, size_t l2, size_t *l_out) const
    {
        size_t l;
        if (!this->sub_21(l2, l1, &l)) {
            return false;
        }
        *l_out = this->negate(l);
        return true;
    }

    bool sub_21(size_t l1, size_t l2, size_t *l_out) const
    {
        return this->add(l1, this->negate(l2), l_out);
    }

};

struct {
    // L1 -> num_U1
    std::vector<size_t> lu;
    // Note: L2 includes BOTH L1+L1 and L1-L1
    // Note: L1 is closed under negation

    // (L1, L1) -> L2
    std::vector<size_t> addition_table_112;

    // (L2, L1) -> L1
    std::vector<size_t> addition_table_211;

    std::vector<size_t> negation_table;

    size_t num_channels(size_t rank) const
    {
        switch (rank) {
        case 0:
            return 1;
        case 1:
            return this->lu.size();
        case 2:
            return this->num_channels_2;
        }
    }
}

/// Defines the layout of many-body operator matrices in memory.  The
/// `ManyBodyBasis` contains information about the many-body states, the
/// channel arithmetics, as well as the offsets of diagonal blocks in memory.
///
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

    // offsets in a matrix that contains a many-body operator in standard form
    // (000, 100, 200)
    size_t _operator_offsets[RANK_COUNT + 1];

    std::vector<size_t> _block_offsets[OPERATOR_KIND_COUNT];

    size_t _orbital_index_offsets[3];

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

    // _states_by_channel[k][l].size() = n_ps
    // _states_by_channel[k][l][u][i] = p[i]
    std::vector<States> _states_by_channel[STATE_KIND_COUNT];

    // _channels_by_state[k][combine(p, n_p)] = {l, u}
    //
    // combine(p, n_p) = ((p[0] * n_p + p[1]) * n_p + p[2]) * n_p + p[3] ...
    std::vector<Orbital> _channels_by_state[STATE_KIND_COUNT];

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
                    std::initializer_list<size_t> xs,
                    std::initializer_list<size_t> ps)
    {
        Rank r = state_kind_to_rank(k);
        size_t l;
        this->_get_or_add_channel_index(r, channel, &l);
        if (l >= this->_states_by_channel[k].size()) {
            this->_states_by_channel[k].resize(l + 1, States(r, 1 << r));
        }
        size_t u = this->_states_by_channel[k][l].size();
        this->_states_by_channel[k][l].add(pack_part(std::move(xs)),
                                           std::move(ps));
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
    template<typename O>
    ManyBodyBasis(const std::vector<O> &orbital_channels)
    {
        // add the zero-particle states
        this->_add_state(STATE_KIND_00, C(), {}, {});

        // add the one-particle states
        size_t p = 0;
        for (size_t x = 0; x < 2; ++x) {
            this->_orbital_index_offsets[x] = p;
            for (const C &c : orbital_channels[x]) {
                this->_add_state(STATE_KIND_10, c, {x}, {p});
                ++p;
            }
        }
        this->_orbital_index_offsets[2] = p;

        // add the two-particle states
        const auto &lu_by_p = this->_channels_by_state[STATE_KIND_10];
        for (size_t x1 = 0; x1 < 2; ++x1) {
            for (size_t x2 = 0; x2 < 2; ++x2) {
                for (size_t p1 = this->orbital_index_offset(x1);
                     p1 < this->orbital_index_offset(x1 + 1); ++p1) {
                    for (size_t p2 = this->orbital_index_offset(x2);
                         p2 < this->orbital_index_offset(x2 + 1); ++p2) {
                        size_t l1 = lu_by_p[p1].channel_index;
                        size_t l2 = lu_by_p[p2].channel_index;
                        const C &c1 = this->unpack_channel(RANK_1, l1);
                        const C &c2 = this->unpack_channel(RANK_1, l2);
                        C c12_20 = c1 + c2;
                        C c12_21 = c1 - c2;
                        // warning: c1 and c2 are references and may expire after next
                        // line due to internal data structures being modified
                        this->_add_state(STATE_KIND_20, c12_20, {x1, x2}, {p1, p2});
                        this->_add_state(STATE_KIND_21, c12_21, {x1, x2}, {p1, p2});
                    }
                }
            }
        }

        // get the offset of blocks within each operator
        for (OperatorKind kk = OperatorKind(); kk < OPERATOR_KIND_COUNT;
             kk = (OperatorKind)(kk + 1)) {
            StateKind k1, k2;
            split_operator_kind(kk, &k1, &k2);
            size_t i = 0;
            size_t n_l = std::min(this->_states_by_channel[k1].size(),
                                  this->_states_by_channel[k2].size());
            for (size_t l = 0; l < n_l; ++l) {
                this->_block_offsets[kk].emplace_back(i);
                size_t n_u1, n_u2;
                this->block_size(kk, l, &n_u1, &n_u2);
                i += n_u1 * n_u2;
            }
            this->_block_offsets[kk].emplace_back(i);
        }

        // get the offset of operators within a many-body operator
        size_t i = 0;
        for (Rank r = Rank(); r < RANK_COUNT; r = (Rank)(r + 1)) {
            this->_operator_offsets[r] = i;
            i += this->_block_offsets[standard_operator_kind(r)].back();
        }
        this->_operator_offsets[RANK_COUNT] = i;
    }

    size_t orbital_index_offset(size_t part) const
    {
        assert(part < 3);
        return this->_orbital_index_offsets[part];
    }

    size_t num_orbitals(size_t part) const
    {
        return this->orbital_index_offset(part + 1);
    }

    /// Return the number of elements required to store the underlying array
    /// of a many-body operator.
    size_t many_body_operator_size() const
    {
        return this->operator_offset(RANK_COUNT);
    }

    /// Offset of an operator inside a many-body operator.
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
        split_operator_kind(operator_kind, &k1, &k2);
        size_t n_u1 = this->_states_by_channel[k1][channel_index].size();
        size_t n_u2 = this->_states_by_channel[k2][channel_index].size();
        if (block_size_1_out) {
            *block_size_1_out = n_u1;
        }
        if (block_size_2_out) {
            *block_size_2_out = n_u2;
        }
    }

    size_t auxiliary_index_offset(StateKind state_kind, size_t channel_index, size_t part) const
    {
        assert(state_kind < STATE_KIND_COUNT);
        assert(channel_index < this->_states_by_channel[state_kind].size());
        return this->_states_by_channel[state_kind][channel_index].auxiliary_index_offset(part);
    }

    /// Convenience function for getting an element from a many-body operator.
    double &get(ManyBodyOperator op, Rank rank, size_t channel_index, size_t i,
                size_t j) const
    {
        OperatorKind kk = standard_operator_kind(rank);
        return op[this->operator_offset(rank) +
                  this->block_offset(kk, channel_index) +
                  i * this->block_stride(kk, channel_index) + j];
    }

    size_t add(size_t r1, size_t l1, size_t r2, size_t l2, size_t r12) const
    {
        const C &c1 = this->unpack_channel((Rank)r1, l1);
        const C &c2 = this->unpack_channel((Rank)r2, l2);
        C c12 = c1 + c2;
        size_t l12;
        this->pack_channel(r12, c12, l12);
        return l12;
    }

    size_t sub(size_t r1, size_t l1, size_t r2, size_t l2, size_t r12) const
    {
        const C &c1 = this->unpack_channel((Rank)r1, l1);
        const C &c2 = this->unpack_channel((Rank)r2, l2);
        C c12 = c1 - c2;
        size_t l12;
        this->pack_channel(r12, c12, l12);
        return l12;
    }

#if 0
    size_t combine(size_t r1, size_t l1, size_t u1, size_t r2, size_t l2, size_t u2, size_t r12) const
    {
    }
#endif

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

    /// Allocate a many-body operator for the given many-body basis.
    std::unique_ptr<double[]> alloc_many_body_operator() const
    {
        return std::unique_ptr<double[]>(
            new double[this->many_body_operator_size()]());
    }

};

#if 0
#define ITER_CHANNELS(var, basis, rank)                                      \
    size_t var = 0;                                                          \
    var < basis.num_channels(rank);                                          \
    ++var

#define ITER_AUXILIARY(var, channel_index, part_begin, part_end, basis, rank) \
    size_t var = basis.auxiliary_index_offset(rank, channel_index, part_begin); \
    var < basis.auxiliary_index_offset(rank, channel_index, part_end);  \
    ++var

#define CHANNEL_1(index, ) \
    size_t l##index :
#endif

#endif
