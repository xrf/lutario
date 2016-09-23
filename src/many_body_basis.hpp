#ifndef MANY_BODY_BASIS_HPP
#define MANY_BODY_BASIS_HPP
#include <assert.h>
#include <stdlib.h>
#include <memory>
#include <unordered_map>
#include <vector>

/*

There's several pieces of information we need to use a type-erased channelized
basis effectively:

  (e) number of l in rank 1 (this comes directly from the translation table)
  (a) number of l in rank r (this comes directly from the translation table)
  (b) number of u in rank r and channel l (this is implied by (c))
  (c) offsets of u groupings in rank r and channel l (this requires n_u(r=1, l))
  (d) addition table of l

There's a couple different ways to get (d): we can do this separately by
building up the addition table from the translation table, or we can do it
while building up the translation table itself.

There's also other things that are useful for relating back to the physics and
also debugging, but these are <C> and <P> dependent and therefore must be
put outside ManyBodyBasis:

  (0) translation between l1 and c
  (1) translation between l and c (requires (0))
  (2) translation between (l, u1) and p
  (3) addition helpers for l

Do we want to store everything in separate pieces?  Keep the <C/P> dependent
parts completely in separate variables from the ManyBodyBasis?  Or do we wanna
keep them together but use virtual stuff?  (arguably less safe since you still
need dynamic_cast to do things like say printing <C/P> objects)

*/

/// A channel index used to denote an invalid channel.
/// The value is greater than all valid channels.
static const size_t INVALID_CHANNEL = (size_t)(-1);

struct Operator {
};

/// A many-body operator contains three operators in standard form:
///
///   - Zero-body operator (constant term) in 000 form.  This is always has a
///     single block containing one element.
///
///   - One-body operator in 100 form.
///
///   - Two-body operator in 200 form.
///
struct ManyBodyOperator {
    Operator o0;
    Operator o1;
    Operator o2;
};

/// Determines the number of particles that an operator couples.
///
///   - A zero-body operator has `RANK_0`.
///   - A one-body operator has `RANK_1`.
///   - A two-body operator has `RANK_2`.
///
enum Rank {
    RANK_0,
    RANK_1,
    RANK_2
};

static const size_t RANK_COUNT = 3;

/// Determines the number of particles in a state and how their channels are
/// to be combined.
enum StateKind {
    STATE_KIND_00,
    STATE_KIND_10,
    STATE_KIND_20,
    STATE_KIND_21
};
static const size_t STATE_KIND_COUNT = 4;

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
    OPERATOR_KIND_211
};
static const size_t OPERATOR_KIND_COUNT = 4;

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
    void insert(size_t part, std::initializer_list<size_t> orbital_indices)
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

class ChannelGroup {

public:

    virtual ~ChannelGroup()
    {
    }

    // Number of channels in a given rank.
    virtual size_t num_channels(Rank rank) const = 0;

    // Negate a channel.  The result is always valid.
    virtual size_t negate(size_t l) const = 0;

    // Add two channels.  The result might be `INVALID_CHANNEL`.
    virtual size_t add(size_t l1, size_t l2) const = 0;

    // Subtract two channels.  The result might be `INVALID_CHANNEL`.
    size_t sub(size_t l1, size_t l2) const
    {
        return this->add(l1, this->negate(l2));
    }

};

/// A data structure used for converting channels into abstract indices and
/// back.  For each channel `l` (which can be the channel of a 0-particle,
/// 1-particle, or 2-particle state) we associate with a unique abstract index
/// `l` called the channel index.
///
/// This, in conjunction with the `ChannelIndexGroup`, allows us to abstract
/// over the operations on channels in a way that does not care about the
/// internal details of what a channel means and can improve efficiency by
/// reducing channel operations that may be moderately expensive to compute to
/// simple table lookups.
///
/// The indices are contiguous, so if two indices `l1` and `l2` correspond to
/// two channels and `l3` is sandwiched between `l1` and `l2`, then `l3` must
/// also correspond to another channel.
///
/// The `C` type must be an abelian group and support the following binary
/// operators:
///
///   - `x + y` ("addition", the binary group operation)
///   - `-x` ("negation", returns the inverse element)
///
/// The default constructor of `C` must construct the group identity ("zero").
///
template<typename C>
class ChannelTranslationTable final : public ChannelGroup {

    // Contains all the channel sets, partitioned by rank.
    //
    // [ rank-0 ] [ == rank-1-exclusive == ] [ ==== rank-2-exclusive ==== ]
    //
    // The channels are stored here in one single array:
    //
    //   - Elements `[0, num_channels(RANK_0))` contain the zero channel
    //     (there is only one, so `num_channels(RANK_0)` is trivially `1`.
    //
    //   - Elements `[num_channels(RANK_0), num_channels(RANK_1))` contain the
    //     nonzero one-body channels.
    //
    //   - Elements `[num_channels(RANK_1), num_channels(RANK_2))` contain the
    //     two-body channels that aren't also a one-body channel.
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

    // Add a channel to the table if it doesn't already exist.
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

    size_t num_channels(Rank rank) const override
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

    size_t negate(size_t l) const override
    {
        size_t l_out = this->encode(-this->decode(l));
        assert(l_out != INVALID_CHANNEL);
        return l_out;
    }

    size_t add(size_t l1, size_t l2) const override
    {
        return this->encode(this->decode(l1) + this->decode(l2));
    }

    /// Returns either INVALID_CHANNEL or a valid channel index.
    size_t encode(const C &c) const
    {
        auto it = this->_encode_table.find(c);
        if (it == this->_encode_table.end()) {
            return INVALID_CHANNEL;
        }
        return it->second;
    }

    const C &decode(size_t l) const
    {
        return this->_decode_table.at(l);
    }

};

template<typename P>
std::vector<typename P::Channel>
get_orbital_channels(const std::array<std::vector<P>, 2> &orbitals)
{
    std::vector<typename P::Channel> orbital_channels;
    for (size_t x = 0; x < 2; ++x) {
        for (const P &p : orbitals[x]) {
            orbital_channels.emplace_back(p.channel());
        }
    }
    return orbital_channels;
}

/// An orbital translation table is a more powerful version of the
/// `ChannelTranslationTable` that maps orbitals (single-particle states) of a
/// basis to index pairs of the form `(l, u)` and vice versa.
///
/// Here `l` is the channel index, and `u` is the auxiliary index.  The `l` is
/// unique for each channel, with `u` serving as the discriminator for
/// orbitals that share the same channel.  Similar to the
/// `ChannelTranslationTable`, the indices are contiguous, so if two indices
/// `u1` and `u2` correspond to orbitals of the same channel `l` and `u3` is
/// sandwiched between `u1` and `u2`, then `u3` must also correspond to
/// another orbital of the same channel.
///
template<typename P>
class OrbitalTranslationTable {

    ChannelTranslationTable<typename P::Channel> _channel_translation_table;

    std::vector<size_t> _auxiliary_offsets;

    std::vector<std::vector<P>> _decode_table;

    std::unordered_map<P, size_t> _encode_table;

    // note: this must be run for all the x = 0 cases before doing it for the
    // x = 1 cases
    void _insert(const P &p, size_t x)
    {
        if (this->_encode_table.find(p) != this->_encode_table.end()) {
            return;
        }
        size_t l = this->channel_translation_table().encode(p.channel());
        assert(l != INVALID_CHANNEL);
        size_t u = this->_decode_table.size();
        this->_encode_table.emplace(p, u);
        this->_decode_table.at(l).emplace_back(p);
        if (x == 0) {
            ++this->_auxiliary_offsets.at(l);
        }
    }

public:

    OrbitalTranslationTable(const std::array<std::vector<P>, 2> &orbitals)
        : _channel_translation_table(get_orbital_channels(orbitals))
    {
        size_t nl1 = this->channel_translation_table().num_channels(RANK_1);
        this->_auxiliary_offsets.resize(nl1);
        this->_encode_table.resize(nl1);
        for (size_t x = 0; x < 2; ++x) {
            for (const P &p : orbitals[x]) {
                this->_insert(p, x);
            }
        }
    }

    ChannelTranslationTable<typename P::Channel>
    channel_translation_table() const
    {
        return this->_channel_translation_table;
    }

    size_t num_orbitals() const
    {
        return this->_encode_table.size();
    }

    size_t num_orbitals_in_channel(size_t l) const
    {
        return this->_decode_table.at(l).size();
    }

    size_t auxiliary_offset_in_channel_part(size_t l, size_t x) const
    {
        switch (x) {
        case 0:
            return 0;
        case 1:
            return this->_auxiliary_offsets.at(l).size();
        case 2:
            return this->num_orbitals_in_channel(l);
        default:
            abort();
        }
    }

    bool encode(const P &p, size_t *l_out, size_t *u_out) const
    {
        auto it = this->_encode_table.find(p);
        if (it == this->_encode_table.end()) {
            return false;
        }
        if (l_out) {
            *l_out = this->channel_translation_table().encode(p.channel());
            assert(*l_out != INVALID_CHANNEL);
        }
        if (u_out) {
            *u_out = it->second;
        }
        return true;
    }

    const P &decode(size_t l, size_t u) const
    {
        return this->_decode_table.at(l).at(u);
    }

};

// TODO: make this class inherit from ChannelGroup
//       but make sure it does not impact performance for GCC+Clang!
class ChannelIndexGroup /* final : public ChannelGroup */ {

    // Since L1 is a subset of L2, this table suffices for all our purposes.
    //
    // (L2, L1) -> L2
    //
    std::vector<size_t> _addition_table;

    // Negation table
    //
    // L2 -> L2
    //
    std::vector<size_t> _negation_table;

    size_t _num_channels_1;

    // Add two channels.  The result might be `INVALID_CHANNEL`.
    size_t _add(size_t l1, size_t l2) const
    {
        // we only support adding a rank-2 channel to a rank-1 channel (or
        // vice versa, if the wrapper function `add` is used); we don't allow
        // adding rank-2 to rank-2, and we shouldn't ever need that
        assert(l1 < this->num_channels(RANK_2));
        assert(l2 < this->num_channels(RANK_1));
        return this->_addition_table[l1 * this->num_channels(RANK_1) + l2];
    }

public:

    ChannelIndexGroup(const ChannelGroup &table)
        : _addition_table(table.num_channels(RANK_2) *
                          table.num_channels(RANK_1))
        , _negation_table(table.num_channels(RANK_1))
    {
        size_t nl1 = table.num_channels(RANK_1);
        size_t nl2 = table.num_channels(RANK_2);
        for (size_t l1 = 0; l1 < nl2; ++l1) {
            for (size_t l2 = 0; l2 < nl1; ++l2) {
                this->_addition_table[l1 * nl1 + l2] = table.add(l1, l2);
            }
        }
        for (size_t l = 0; l < nl1; ++l) {
            this->_negation_table[l] = table.negate(l);
        }
    }

    size_t num_channels(Rank rank) const /* override */
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

    // Negate a channel.
    size_t negate(size_t l) const /* override */
    {
        return this->_negation_table[l];
    }

    // Add two channels.  The result might be `INVALID_CHANNEL`.
    size_t add(size_t l1, size_t l2) const /* override */
    {
        if (l1 < l2) {
            std::swap(l1, l2);
        }
        return this->_add(l1, l2);
    }

    // Subtract two channels.  The result might be `INVALID_CHANNEL`.
    size_t sub(size_t l1, size_t l2) const /* override */
    {
        return this->add(l1, this->negate(l2));
    }

};

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

    std::vector<size_t> _auxiliary_offsets_20;
    std::vector<size_t> _auxiliary_offsets_21;

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
    ManyBodyBasis(ChannelIndexGroup table)
        : _table(std::move(table))
        , _auxiliary_offsets_20(table.num_channels(RANK_2) * 4 *
                                table.num_channels(RANK_1) + 1)
        , _auxiliary_offsets_21(table.num_channels(RANK_2) * 4 *
                                table.num_channels(RANK_1) + 1)
    {
        size_t nl1 = this->_table.num_channels(RANK_1);
        size_t nl2 = this->_table.num_channels(RANK_2);
        // _auxiliary_offsets_20[(l12 * 4 + x12) * nl1 + l1]
        {
            size_t i = 0;
            for (size_t l12 = 0; l12 < nl2; ++l12) {
                for (size_t x1 = 0; x1 < 2; ++x1) {
                    for (size_t x2 = 0; x2 < 2; ++x2) {
                        for (size_t l1 = 0; l1 < nl1; ++l1) {
                            size_t l2 = table.sub(l12, l1);
                            if (l2 >= nl1) {
                                continue;
                            }
                            this->_auxiliary_offsets_20.emplace_back(i);
                            i += this->_table.num_orbitals_in_channel_part(l1, x1) *
                                 this->_table.num_orbitals_in_channel_part(l2, x2);
                        }
                    }
                }
            }
            this->_auxiliary_offsets_20.emplace_back(i);
        }
        // _auxiliary_offsets_21[(l14 * 4 + x14) * nl1 + l1]
        {
            size_t i = 0;
            for (size_t l14 = 0; l14 < nl2; ++l14) {
                for (size_t x1 = 0; x1 < 2; ++x1) {
                    for (size_t x2 = 0; x2 < 2; ++x2) {
                        for (size_t l1 = 0; l1 < nl1; ++l1) {
                            size_t l2 = table.sub(l1, l14);
                            if (l2 >= nl1) {
                                continue;
                            }
                            this->_auxiliary_offsets_21.emplace_back(i);
                            i += this->_table.num_orbitals_in_channel_part(l1, x1) *
                                 this->_table.num_orbitals_in_channel_part(l2, x2);
                        }
                    }
                }
            }
            this->_auxiliary_offsets_21.emplace_back(i);
        }
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

    size_t combine(size_t l1, size_t u1, size_t l2, size_t u2, size_t *u12_out) const
    {
        l12 = add(l1, l2);
        u12 = auxiliary_offset[l12, l1] +
              u1 * num_orbitals_in_channel(l2) + u2;
        if (u12_out) {
            *u12_out = u12;
        }
        return l12;
    }

    size_t num_channels(Rank rank) const
    {
        assert(rank < RANK_COUNT);
        return this->_num_channels[rank];
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
