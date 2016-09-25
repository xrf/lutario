#ifndef MANY_BODY_BASIS_HPP
#define MANY_BODY_BASIS_HPP
#include <assert.h>
#include <stdlib.h>
#include <iterator>
#include <memory>
#include <unordered_map>
#include <vector>

// TODO: 'part' could mean just the occupation number, or it could refer to
// the combination of occupation number and secondary channel indices;
// figure out a better nomenclature

/// A value used to denote an invalid index (such as channel index or
/// auxiliary orbital index).  It is defined to be `SIZE_MAX`.  Hence, it is
/// larger than all valid indices.
static const size_t INVALID_INDEX = (size_t)(-1);

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

inline size_t pack_part(std::initializer_list<size_t> part)
{
    size_t xs = 0;
    for (size_t x : part) {
        xs = xs * 2 + x;
    }
    return xs;
}

class GenericOrbitalTable {

public:

    virtual ~GenericOrbitalTable()
    {
    }

    /// Number of channels in a given rank.
    virtual size_t num_channels(Rank rank) const = 0;

    /// Negate a valid channel.  The result is always valid.
    virtual size_t negate_channel(size_t channel) const = 0;

    /// Add two valid channels.  The result might be `INVALID_INDEX`.
    virtual size_t add_channels(size_t channel_1, size_t channel_2) const = 0;

    /// Subtract two valid channels.  The result might be `INVALID_INDEX`.
    size_t subtract_channel(size_t channel_1, size_t channel_2) const
    {
        return this->add_channels(channel_1, this->negate_channel(channel_2));
    }

    virtual size_t orbital_offset(size_t channel, size_t part) const = 0;

    size_t num_orbitals_in_channel(size_t channel) const
    {
        assert(channel < this->num_channels(RANK_1));
        return this->orbital_offset(channel, 2) -
               this->orbital_offset(channel, 0);
    }

    size_t num_orbitals_in_channel_part(size_t channel, size_t part) const
    {
        assert(channel < this->num_channels(RANK_1));
        assert(part < 2);
        return this->orbital_offset(channel, part + 1) -
               this->orbital_offset(channel, part);
    }

};

/// An orbital translation table that maps orbitals (single-particle states)
/// of a basis to index pairs of the form `(l, u)` and vice versa.
/// Additionally, the translation table allows channels to be converted into
/// abstract indices and back.
///
/// For each channel `c` (which can be the channel of a 0-particle,
/// 1-particle, or 2-particle state) we associate with a unique abstract index
/// `l` called the channel index.  The channel index is independent of rank,
/// so the same `l` may denote the channel of a 1-particle or 2-particle
/// state.  In particular, the set of valid `l` for 1-particle states is a
/// subset of the set of valid `l` for 2-particle states.
///
/// The auxiliary orbital index `u` serves as the discriminator for orbitals
/// that share the same channel.  The indices are contiguous, so if two
/// indices `u1` and `u2` correspond to orbitals of the same channel `l` and
/// `u3` is sandwiched between `u1` and `u2`, then `u3` must also correspond
/// to another orbital of the same channel.  (By definition, the auxiliary
/// orbital index is for rank-1 states.)
///
/// In conjunction with the `OrbitalIndexTable`, the data structure allows us
/// to abstract over the operations on channels in a way that does not care
/// about the internal details of what a channel means and can improve
/// efficiency by reducing channel operations that may be moderately expensive
/// to compute to simple table lookups.
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
template<typename P, typename C>
class OrbitalTranslationTable final : public GenericOrbitalTable {

    // Channels are stored here in one single array, subdivided into three
    // channel sets based on rank:
    //
    //   - [0, num_channels(RANK_0)) contains the one and only rank-0 channel.
    //
    //   - [0, num_channels(RANK_1)) contains the rank-1 channels.
    //
    //   - [0, num_channels(RANK_2)) contains the rank-2 channels.
    //
    // The channel sets here aren't necessarily the same as the physical
    // channel sets, but they are always supersets of the physical sets.
    // Usually, they are *strict* supersets because in most fermionic systems
    // the physical rank-1 channel set is disjoint from the physical rank-0 or
    // rank-2 channel sets due to the spin projection being half-integers.
    // Nonetheless, we choose to enforce cumulativeness on the channel sets to
    // simplify the implementation.
    //
    // L2 -> C
    std::vector<C> _channel_decoder;

    // Inverse mapping of _channel_decoder.  Must satisfy:
    //
    //   - for all l, _channel_encoder[_channel_decoder[l]] == l
    //   - for all c, _channel_decoder[_channel_encoder[c]] == c
    //
    // C -> L2
    std::unordered_map<C, size_t> _channel_encoder;

    // L1 -> U1
    std::vector<size_t> _orbital_offsets;

    // L1 -> U1 -> P
    std::vector<std::vector<P>> _orbital_decoders;

    // P -> U1
    std::unordered_map<P, size_t> _orbital_encoder;

    // Add a channel to the table if it doesn't already exist.
    void _insert_channel(const C &c)
    {
        if (this->encode_channel(c) != INVALID_INDEX) {
            return;
        }
        size_t l = this->_channel_decoder.size();
        this->_channel_encoder.emplace(c, l);
        this->_channel_decoder.emplace_back(c);
    }

    // Add an orbital to the table.
    void _insert_orbital(const P &p, const C &c)
    {
        size_t l = this->encode_channel(c);
        if (l == INVALID_INDEX) {
            throw std::logic_error("channel not found");
        }
        if (this->encode_orbital(p) != INVALID_INDEX) {
            throw std::logic_error("orbitals are not unique");
        }
        size_t u = this->_orbital_decoders.at(l).size();
        this->_orbital_encoder.emplace(p, u);
        this->_orbital_decoders.at(l).emplace_back(p);
    }

public:

    /// Construct an `OrbitalTranslationTable` from a list of orbitals for a
    /// basis.  The list contains triples of the form `(p, c, x)` where `p` is
    /// the orbital, `c` is its corresponding channel, and `x` is a boolean
    /// indicating whether the orbital is unoccupied.  The list must contain
    /// unique orbitals.
    explicit OrbitalTranslationTable(
        const std::vector<std::tuple<P, C, bool>> &orbitals)
    {
        // construct channel translation table
        this->_insert_channel(C());
        for (bool x_filter : {false, true}) {
            for (const auto &pcx : orbitals) {
                const C &c = std::get<1>(pcx);
                size_t x = std::get<2>(pcx);
                if (x == x_filter) {
                    this->_insert_channel(c);
                    this->_insert_channel(-c); // ensure closure under negation
                }
            }
        }
        size_t nl1 = this->_channel_decoder.size();
        for (size_t l1 = 0; l1 < nl1; ++l1) {
            for (size_t l2 = 0; l2 <= l1; ++l2) {
                const C &c1 = this->decode_channel(l1);
                const C &c2 = this->decode_channel(l2);
                this->_insert_channel(c1 + c2);
            }
        }

        // construct orbital translation table
        this->_orbital_offsets.resize(nl1);
        this->_orbital_decoders.resize(nl1);
        for (bool x_filter : {false, true}) {
            for (const auto &pcx : orbitals) {
                const P &p = std::get<0>(pcx);
                const C &c = std::get<1>(pcx);
                size_t x = std::get<2>(pcx);
                if (x == x_filter) {
                    this->_insert_orbital(p, c);
                }
            }
            if (x_filter == 0) {
                for (size_t l = 0; l < nl1; ++l) {
                    this->_orbital_offsets.at(l) =
                        this->_orbital_decoders.at(l).size();
                }
            }
        }
    }

    size_t num_channels(Rank r) const override
    {
        switch (r) {
        case RANK_0:
            return 1;
        case RANK_1:
            return this->_orbital_decoders.size();
        case RANK_2:
            return this->_channel_decoder.size();
        }
    }

    size_t num_orbitals() const
    {
        return this->_orbital_encoder.size();
    }

    size_t orbital_offset(size_t l, size_t x) const override
    {
        switch (x) {
        case 0:
            return 0;
        case 1:
            return this->_orbital_offsets.at(l);
        case 2:
            return this->_orbital_decoders.at(l).size();
        default:
            throw std::logic_error("invalid part");
        }
    }

    size_t negate_channel(size_t l) const override
    {
        size_t l_out = this->encode_channel(-this->decode_channel(l));
        assert(l_out != INVALID_INDEX);
        return l_out;
    }

    size_t add_channels(size_t l1, size_t l2) const override
    {
        return this->encode_channel(this->decode_channel(l1) +
                                    this->decode_channel(l2));
    }

    /// Returns either INVALID_INDEX or a valid channel index.
    size_t encode_channel(const C &c) const
    {
        auto it = this->_channel_encoder.find(c);
        if (it == this->_channel_encoder.end()) {
            return INVALID_INDEX;
        }
        return it->second;
    }

    const C &decode_channel(size_t l) const
    {
        return this->_channel_decoder.at(l);
    }

    /// Returns either INVALID_INDEX or a valid auxiliary orbital index.
    size_t encode_orbital(const P &p) const
    {
        auto it = this->_orbital_encoder.find(p);
        if (it == this->_orbital_encoder.end()) {
            return INVALID_INDEX;
        }
        return it->second;
    }

    const P &decode_orbital(size_t l, size_t u) const
    {
        return this->_orbital_decoders.at(l).at(u);
    }

};

// TODO: make this class inherit from ChannelIndexGroup
//       but make sure it does not impact performance for GCC+Clang!
class StateIndexTable /* final : public GenericOrbitalTable */ {

    // Since L1 is a subset of L2, this table suffices for all our purposes.
    //
    // (L2, L1) -> L2
    std::vector<size_t> _addition_table;

    // Negation table
    //
    // L2 -> L2
    std::vector<size_t> _negation_table;

    // StateKind -> L -> U
    std::vector<size_t> _state_offsets[STATE_KIND_COUNT];

    size_t _num_channels_1;

    // Adds a rank-2 channel to a rank-1 channel.  The result might be
    // `INVALID_INDEX`.
    size_t _add_channels(size_t l1, size_t l2) const
    {
        assert(l1 < this->num_channels(RANK_2));
        assert(l2 < this->num_channels(RANK_1));
        return this->_addition_table[l1 * this->num_channels(RANK_1) + l2];
    }

public:

    explicit StateIndexTable(const GenericOrbitalTable &table);

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

    size_t orbital_offset(size_t l, size_t x) const /* override */
    {
        return this->state_offset(STATE_KIND_10, l, x);
    }

    size_t num_orbitals_in_channel(size_t channel) const /* delete me later */
    {
        assert(channel < this->num_channels(RANK_1));
        return this->orbital_offset(channel, 2) -
               this->orbital_offset(channel, 0);
    }

    size_t num_orbitals_in_channel_part(size_t channel,
                                        size_t part) const /* delete me later */
    {
        assert(channel < this->num_channels(RANK_1));
        assert(part < 2);
        return this->orbital_offset(channel, part + 1) -
               this->orbital_offset(channel, part);
    }

    size_t state_offset(StateKind k, size_t l, size_t x) const
    {
        size_t nl1 = this->num_channels(RANK_1);
        size_t nx;
        switch (k) {
        case STATE_KIND_00:
            nx = 1;
            break;
        case STATE_KIND_10:
            nx = 2;
            break;
        case STATE_KIND_20:
            nx = 4 * nl1;
            break;
        case STATE_KIND_21:
            nx = 4 * nl1;
            break;
        }
        assert(x <= nx);
        assert(l < this->num_channels(state_kind_to_rank(k)));
        return this->_state_offsets[k][l * (nx + 1) + x];
    }

    size_t num_states_in_channel_part(StateKind k, size_t l, size_t x) const
    {
        return this->state_offset(k, l, x + 1) - this->state_offset(k, l, x);
    }

    // Negate a channel.
    size_t negate_channel(size_t l) const /* override */
    {
        return this->_negation_table[l];
    }

    // Adds a rank-1 channel to a rank-2 channel or vice versa.  The result
    // might be `INVALID_INDEX`.
    size_t add_channels(size_t l1, size_t l2) const /* override */
    {
        if (l1 < l2) {
            std::swap(l1, l2);
        }
        return this->_add_channels(l1, l2);
    }

    // Subtract two channels.  The result might be `INVALID_INDEX`.
    size_t subtract_channels(size_t l1, size_t l2) const /* delete me later */
    {
        return this->add_channels(l1, this->negate_channel(l2));
    }

};

#if 0
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
    ManyBodyBasis(OrbitalIndexTable table)
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
                            i += this->_num_orbitals_in_channel_part(l1, x1) *
                                 this->_num_orbitals_in_channel_part(l2, x2);
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
                            i += this->_num_orbitals_in_channel_part(l1, x1) *
                                 this->_num_orbitals_in_channel_part(l2, x2);
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
    double &get(ManyBodyOperator &op, Rank r, size_t l, size_t i, size_t j) const
    {
        return op[this->get_index(r, l, i, j)];
    }

    double get(const ManyBodyOperator &op, Rank r, size_t l, size_t i, size_t j) const
    {
        return op[this->get_index(r, l, i, j)];
    }

    /// Convenience function for getting an element from a many-body operator.
    size_t get_index(Rank rank, size_t channel_index, size_t i, size_t j) const
    {
        OperatorKind kk = standard_operator_kind(rank);
        return this->operator_offset(rank) +
            this->block_offset(kk, channel_index) +
            i * this->block_stride(kk, channel_index) + j;
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

#endif
