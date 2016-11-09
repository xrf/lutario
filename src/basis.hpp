#ifndef MANY_BODY_BASIS_HPP
#define MANY_BODY_BASIS_HPP
#include <assert.h>
#include <stdlib.h>
#include <iterator>
#include <map>
#include <memory>
#include <vector>
#include "irange.hpp"
#include "optional.hpp"

// TODO: 'part' could mean just the occupation number, or it could refer to
// the combination of occupation number and secondary channel indices;
// figure out a better nomenclature

// Notation:
//
//   - k = state_kind
//   - c = channel
//   - l = channel_index
//   - u = auxiliary_index
//   - r = rank
//   - i = particle_index
//   - p = orbital_index

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

#define UNOCC_P {0, 2}
#define UNOCC_I {0, 1}
#define UNOCC_A {1, 2}

#define UNOCC_PP {0, 4}
#define UNOCC_II {0, 1}
#define UNOCC_IA {1, 2}
#define UNOCC_AI {2, 3}
#define UNOCC_AA {3, 4}

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
enum OperKind {
    OPER_KIND_000,
    OPER_KIND_100,
    OPER_KIND_200,
    OPER_KIND_211
};
static const size_t OPER_KIND_COUNT = 4;

/// Get the rank of an operator with the given kind, equal to the lower of the
/// ranks of the respective state kinds.
inline Rank oper_kind_to_rank(OperKind oper_kind)
{
    if (oper_kind >= OPER_KIND_200) {
        return RANK_2;
    }
    if (oper_kind >= OPER_KIND_100) {
        return RANK_1;
    }
    return RANK_0;
}

/// Get the canonical operator representation for a given operator rank.
inline OperKind standard_oper_kind(Rank rank)
{
    switch (rank) {
    case RANK_0:
        return OPER_KIND_000;
    case RANK_1:
        return OPER_KIND_100;
    case RANK_2:
        return OPER_KIND_200;
    }
}

/// Get the kind of the left states and the kind of right states that the
/// operator couples.
inline void split_oper_kind(OperKind oper_kind,
                            StateKind *state_kind_1_out,
                            StateKind *state_kind_2_out)
{
    StateKind k1, k2;
    switch (oper_kind) {
    case OPER_KIND_000:
        k1 = STATE_KIND_00;
        k2 = STATE_KIND_00;
        break;
    case OPER_KIND_100:
        k1 = STATE_KIND_10;
        k2 = STATE_KIND_10;
        break;
    case OPER_KIND_200:
        k1 = STATE_KIND_20;
        k2 = STATE_KIND_20;
        break;
    case OPER_KIND_211:
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

/// Call the given unary function on the OperKind.  It is semantically
/// equivalent to simply calling `unary_func(oper_kind)` directly, but the
/// function body is duplciated for every possible value of `oper_kind`.  This
/// can sometimes expose additional optimizations.
template<typename F>
auto dispatch_on_oper_kind(OperKind oper_kind, F unary_func) ->
    decltype(unary_func(OPER_KIND_000))
{
    switch (oper_kind) {
    case OPER_KIND_000:
        return unary_func(OPER_KIND_000);
    case OPER_KIND_100:
        return unary_func(OPER_KIND_100);
    case OPER_KIND_200:
        return unary_func(OPER_KIND_200);
    case OPER_KIND_211:
        return unary_func(OPER_KIND_211);
    }
}

/// An index that may or may not be valid.  This is typically used to indicate
/// failure in some operation.
class OptionalIndex {

    size_t _value;

public:

    // A value used to denote an invalid index (such as channel index or
    // auxiliary orbital index).  It is defined to be `SIZE_MAX`.  Hence, it
    // is larger than all valid indices.
    static const size_t INVALID_VALUE = (size_t)(-1);

    /// Construct an optional that does not contain a value.
    OptionalIndex()
        : _value(INVALID_VALUE)
    {
    }

    /// Construct an optional that contains the given value.
    OptionalIndex(size_t value)
        : _value(value)
    {
        // INVALID_VALUE is reserved
        assert(value != INVALID_VALUE);
    }

    /// Check if a value is present.
    explicit operator bool() const
    {
        return this->_value != INVALID_VALUE;
    }

    /// Extract the value.  The value must exist.
    size_t operator*() const
    {
        assert(static_cast<bool>(*this));
        return this->_value;
    }

    /// Check if the index is within the given half-open range.
    /// If there is no value, the result is always `false`.
    Optional<size_t> within(const IndexRange &range) const
    {
        // we take advantage of the fact that INVALID_VALUE is SIZE_MAX to
        // eliminate the unnecessary check that "value != INVALID_VALUE"
        if (!(this->_value >= range.start && this->_value < range.stop)) {
            return {};
        }
        return **this;
    }

};

/// A channelized representation of orbitals.  Each orbital is determined by
///
///   - a channel index (associated with a channel in some `ManyBodyBasis`) and
///   - an auxiliary index to identify the orbital within the said channel.
///
class Orbital {

    size_t _channel_index;

    size_t _auxiliary_index;

public:

    Orbital()
        : _channel_index()
        , _auxiliary_index()
    {
    }

    Orbital(size_t channel_index, size_t auxiliary_index)
        : _channel_index(channel_index)
        , _auxiliary_index(auxiliary_index)
    {
        assert(this->_channel_index != OptionalIndex::INVALID_VALUE);
        assert(this->_auxiliary_index != OptionalIndex::INVALID_VALUE);
    }

    size_t channel_index() const
    {
        return this->_channel_index;
    }

    size_t auxiliary_index() const
    {
        return this->_auxiliary_index;
    }

    std::tuple<size_t, size_t> to_tuple() const
    {
        return std::make_tuple(this->channel_index(), this->auxiliary_index());
    }

    bool operator==(const Orbital &other) const
    {
        return this->to_tuple() == other.to_tuple();
    }

};

std::ostream &operator<<(std::ostream &, const Orbital &);

/// An `Orbital` that may or may not be valid.  This is typically used to
/// indicate failure in some operation.
class OptionalOrbital {

    size_t _channel_index;

    size_t _auxiliary_index;

public:

    /// Construct an optional that does not contain a value.
    OptionalOrbital()
        : _channel_index(OptionalIndex::INVALID_VALUE)
        , _auxiliary_index(OptionalIndex::INVALID_VALUE)
    {
    }

    /// Construct an optional that contains the given value.
    OptionalOrbital(const Orbital &orbital)
        : _channel_index(orbital.channel_index())
        , _auxiliary_index(orbital.auxiliary_index())
    {
        assert(static_cast<bool>(*this));
    }

    /// Check if a value is present.
    explicit operator bool() const
    {
        assert((this->_channel_index != OptionalIndex::INVALID_VALUE) ==
               (this->_auxiliary_index != OptionalIndex::INVALID_VALUE));
        return this->_channel_index != OptionalIndex::INVALID_VALUE;
    }

    /// Extract the value.  It is an error to do this if there is no value.
    Orbital operator*() const
    {
        assert(static_cast<bool>(*this));
        return {this->_channel_index, this->_auxiliary_index};
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

    /// Negate a channel.
    virtual size_t negate_channel(size_t channel) const = 0;

    /// Add two channels.  The result may not exist.
    virtual OptionalIndex add_channels(size_t channel_1,
                                       size_t channel_2) const = 0;

    /// Subtract two channels.  The result may not exist.
    OptionalIndex subtract_channels(size_t channel_1, size_t channel_2) const
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
    std::map<C, size_t> _channel_encoder;

    // L1 -> U1
    std::vector<size_t> _orbital_offsets;

    // L1 -> U1 -> P
    std::vector<std::vector<P>> _orbital_decoders;

    // P -> U1
    std::map<P, size_t> _orbital_encoder;

    // Add a channel to the table if it doesn't already exist.
    void _insert_channel(const C &c)
    {
        if (this->encode_channel(c)) {
            return;
        }
        size_t l = this->_channel_decoder.size();
        this->_channel_encoder.emplace(c, l);
        this->_channel_decoder.emplace_back(c);
    }

    // Add an orbital to the table.
    void _insert_orbital(const P &p, const C &c)
    {
        size_t l;
        if (!try_get(this->encode_channel(c), &l)) {
            throw std::logic_error("channel not found");
        }
        if (this->encode_orbital(p)) {
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
        return *this->encode_channel(-this->decode_channel(l));
    }

    OptionalIndex add_channels(size_t l1, size_t l2) const override
    {
        return this->encode_channel(this->decode_channel(l1) +
                                    this->decode_channel(l2));
    }

    OptionalIndex encode_channel(const C &c) const
    {
        auto it = this->_channel_encoder.find(c);
        if (it == this->_channel_encoder.end()) {
            return OptionalIndex();
        }
        return OptionalIndex(it->second);
    }

    const C &decode_channel(size_t l) const
    {
        return this->_channel_decoder.at(l);
    }

    OptionalIndex encode_orbital(const P &p) const
    {
        auto it = this->_orbital_encoder.find(p);
        if (it == this->_orbital_encoder.end()) {
            return OptionalIndex();
        }
        return OptionalIndex(it->second);
    }

    const P &decode_orbital(Orbital lu) const
    {
        size_t l = lu.channel_index();
        size_t u = lu.auxiliary_index();
        return this->_orbital_decoders.at(l).at(u);
    }

};

/// Defines the layout of many-body operator matrices in memory.  The
/// `ManyBodyBasis` contains information about the many-body states, the
/// channel arithmetics, as well as the offsets of diagonal blocks in memory.
class ManyBodyBasis {

public:

    explicit ManyBodyBasis(const GenericOrbitalTable &table);

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

    size_t orbital_offset(size_t l, size_t x) const
    {
        return this->state_offset(STATE_KIND_10, l, x);
    }

    size_t num_orbitals_in_channel(size_t channel) const
    {
        assert(channel < this->num_channels(RANK_1));
        return this->orbital_offset(channel, 2) -
               this->orbital_offset(channel, 0);
    }

    size_t num_orbitals_in_channel_part(size_t channel,
                                        size_t part) const
    {
        assert(channel < this->num_channels(RANK_1));
        assert(part < 2);
        return this->orbital_offset(channel, part + 1) -
               this->orbital_offset(channel, part);
    }

    size_t num_parts(StateKind k) const
    {
        size_t nl1 = this->num_channels(RANK_1);
        switch (k) {
        case STATE_KIND_00:
            return 1;
        case STATE_KIND_10:
            return 2;
        case STATE_KIND_20:
            return 4 * nl1;
        case STATE_KIND_21:
            return 4 * nl1;
        }
    }

    size_t state_offset(StateKind k, size_t l, size_t x) const
    {
        size_t nx = this->num_parts(k);
        assert(x <= nx);
        assert(l < this->num_channels(state_kind_to_rank(k)));
        return this->_state_offsets[k][l * (nx + 1) + x];
    }

    size_t num_states_in_channel(StateKind k, size_t l) const
    {
        return this->state_offset(k, l, this->num_parts(k)) -
               this->state_offset(k, l, 0);
    }

    size_t num_states_in_channel_part(StateKind k, size_t l, size_t x) const
    {
        return this->state_offset(k, l, x + 1) - this->state_offset(k, l, x);
    }

    // Negate a channel.
    size_t negate_channel(size_t l) const
    {
        return this->_negation_table[l];
    }

    // Adds a rank-1 channel to a rank-2 channel or vice versa.
    OptionalIndex add_channels(size_t l1, size_t l2) const
    {
        if (l1 < l2) {
            std::swap(l1, l2);
        }
        return this->_add_channels(l1, l2);
    }

    // Subtract two channels.
    OptionalIndex subtract_channels(size_t l1, size_t l2) const
    {
        return this->add_channels(l1, this->negate_channel(l2));
    }

    size_t many_body_oper_size() const
    {
        return this->oper_offset(RANK_COUNT);
    }

    size_t oper_offset(size_t rank) const
    {
        return this->_oper_offsets[rank];
    }

    size_t oper_size(OperKind oper_kind) const
    {
        Rank r = oper_kind_to_rank(oper_kind);
        size_t nl = this->num_channels(r);
        return this->block_offset(oper_kind, nl);
    }

    size_t block_offset(OperKind oper_kind, size_t channel_index) const
    {
        return this->_block_offsets[oper_kind][channel_index];
    }

    size_t block_stride(OperKind oper_kind, size_t channel_index) const
    {
        size_t n;
        // assuming row-major
        this->block_size(oper_kind, channel_index, NULL, &n);
        return n;
    }

    void block_size(OperKind kk, size_t l,
                    size_t *nu1_out, size_t *nu2_out) const
    {
        StateKind k1, k2;
        split_oper_kind(kk, &k1, &k2);
        size_t nu1 = this->num_states_in_channel(k1, l);
        size_t nu2 = this->num_states_in_channel(k2, l);
        if (nu1_out) {
            *nu1_out = nu1;
        }
        if (nu2_out) {
            *nu2_out = nu2;
        }
    }

    IndexRange channels(Rank r) const
    {
        size_t nl = this->num_channels(r);
        return {0, nl};
    }

    bool is_conserved_1(Orbital lu1, Orbital lu2) const
    {
        size_t l1 = lu1.channel_index();
        size_t l2 = lu2.channel_index();
        return l1 == l2;
    }

    bool is_conserved_2(Orbital lu1, Orbital lu2,
                        Orbital lu3, Orbital lu4) const
    {
        Orbital lu12, lu34;
        if (!try_get(this->combine_20(lu1, lu2), &lu12)) {
            return false;
        }
        if (!try_get(this->combine_20(lu3, lu4), &lu34)) {
            return false;
        }
        size_t l12 = lu12.channel_index();
        size_t l34 = lu34.channel_index();
        return l12 == l34;
    }

    /// The function is expected to be like `({l1, u1}) -> void`.
    template<typename F>
    void for_u10(size_t l1, IndexRange y1s, F func) const
    {
        for (size_t x1 : y1s) {
            size_t nu1a = this->state_offset(STATE_KIND_10, l1, x1);
            size_t nu1b = this->state_offset(STATE_KIND_10, l1, x1 + 1);
            for (size_t u1 : IndexRange(nu1a, nu1b)) {
                func({l1, u1});
            }
        }
    }

    /// The function is expected to be like `({l1, u1}, {l2, u2}) -> void`.
    template<typename F>
    void for_u20(size_t l12, IndexRange y12s, F func) const
    {
        size_t nl1 = this->num_channels(RANK_1);
        for (size_t y12 : y12s) {
            size_t x1 = y12 / 2;
            size_t x2 = y12 % 2;
            for (size_t l1 : IndexRange(0, nl1)) {
                size_t l2;
                if (!try_get(this->subtract_channels(l12, l1)
                             .within({0, nl1}), &l2)) {
                    continue;
                }
                size_t nu1a = this->state_offset(STATE_KIND_10, l1, x1);
                size_t nu1b = this->state_offset(STATE_KIND_10, l1, x1 + 1);
                size_t nu2a = this->state_offset(STATE_KIND_10, l2, x2);
                size_t nu2b = this->state_offset(STATE_KIND_10, l2, x2 + 1);
                for (size_t u1 : IndexRange(nu1a, nu1b)) {
                    for (size_t u2 : IndexRange(nu2a, nu2b)) {
                        func({l1, u1}, {l2, u2});
                    }
                }
            }
        }
    }

    /// The function is expected to be like `({l1, u1}, {l4, u4}) -> void`.
    template<typename F>
    void for_u21(size_t l14, IndexRange y14s, F func) const
    {
        size_t nl1 = this->num_channels(RANK_1);
        for (size_t y14 : y14s) {
            size_t x1 = y14 / 2;
            size_t x4 = y14 % 2;
            for (size_t l1 : IndexRange(0, nl1)) {
                size_t l4;
                if (!try_get(this->subtract_channels(l1, l14)
                             .within({0, nl1}), &l4)) {
                    continue;
                }
                size_t nu1a = this->state_offset(STATE_KIND_10, l1, x1);
                size_t nu1b = this->state_offset(STATE_KIND_10, l1, x1 + 1);
                size_t nu4a = this->state_offset(STATE_KIND_10, l4, x4);
                size_t nu4b = this->state_offset(STATE_KIND_10, l4, x4 + 1);
                for (size_t u1 : IndexRange(nu1a, nu1b)) {
                    for (size_t u4 : IndexRange(nu4a, nu4b)) {
                        func({l1, u1}, {l4, u4});
                    }
                }
            }
        }
    }

    bool get_unocc(Orbital lu1) const
    {
        size_t l1 = lu1.channel_index();
        size_t u1 = lu1.auxiliary_index();
        return u1 >= this->orbital_offset(l1, 1);
    }

    OptionalOrbital combine_20(Orbital lu1, Orbital lu2) const
    {
        size_t l1 = lu1.channel_index();
        size_t u1 = lu1.auxiliary_index();
        size_t l2 = lu2.channel_index();
        size_t u2 = lu2.auxiliary_index();
        size_t l12;
        if (!try_get(this->add_channels(l1, l2), &l12)) {
            return OptionalOrbital();
        }
        bool x1 = this->get_unocc(lu1);
        bool x2 = this->get_unocc(lu2);
        size_t uo1 = this->orbital_offset(l1, x1);
        size_t uo2 = this->orbital_offset(l2, x2);
        size_t nl1 = this->num_channels(RANK_1);
        size_t nu = this->num_orbitals_in_channel_part(l2, x2);
        size_t ub = this->state_offset(STATE_KIND_20, l12,
                                       (x1 * 2 + x2) * nl1 + l1);
        size_t u12 = ub + (u1 - uo1) * nu + (u2 - uo2);
        return OptionalOrbital({l12, u12});
    }

    OptionalOrbital combine_21(Orbital lu1, Orbital lu4) const
    {
        size_t l1 = lu1.channel_index();
        size_t u1 = lu1.auxiliary_index();
        size_t l4 = lu4.channel_index();
        size_t u4 = lu4.auxiliary_index();
        size_t l14;
        if (!try_get(this->subtract_channels(l1, l4), &l14)) {
            return OptionalOrbital();
        }
        bool x1 = this->get_unocc(lu1);
        bool x4 = this->get_unocc(lu4);
        size_t uo1 = this->orbital_offset(l1, x1);
        size_t uo4 = this->orbital_offset(l4, x4);
        size_t nl1 = this->num_channels(RANK_1);
        size_t nu = this->num_orbitals_in_channel_part(l4, x4);
        size_t ub = this->state_offset(STATE_KIND_21, l14,
                                       (x1 * 2 + x4) * nl1 + l1);
        size_t u14 = ub + (u1 - uo1) * nu + (u4 - uo4);
        return OptionalOrbital({l14, u14});
    }

    template<typename B>
    auto slice_by_unoccupancy_100(const B &blocks,
                                  size_t l1,
                                  IndexRange y1s,
                                  IndexRange y2s) const ->
        decltype(blocks[l1].slice({0, 0}, {0, 0}))
    {
        IndexRange u1s(
            this->state_offset(STATE_KIND_10, l1, y1s.start),
            this->state_offset(STATE_KIND_10, l1, y1s.stop)
        );
        IndexRange u2s(
            this->state_offset(STATE_KIND_10, l1, y2s.start),
            this->state_offset(STATE_KIND_10, l1, y2s.stop)
        );
        return blocks[l1].slice(u1s, u2s);
    }

    template<typename B>
    auto slice_by_unoccupancy_200(const B &blocks,
                                  size_t l12,
                                  IndexRange y12s,
                                  IndexRange y34s) const ->
        decltype(blocks[l12].slice({0, 0}, {0, 0}))
    {
        size_t nl1 = this->num_channels(RANK_1);
        IndexRange u12s(
            this->state_offset(STATE_KIND_20, l12, y12s.start * nl1),
            this->state_offset(STATE_KIND_20, l12, y12s.stop * nl1)
        );
        IndexRange u34s(
            this->state_offset(STATE_KIND_20, l12, y34s.start * nl1),
            this->state_offset(STATE_KIND_20, l12, y34s.stop * nl1)
        );
        return blocks[l12].slice(u12s, u34s);
    }

    /// Check if the objects are equal.
    ///
    /// Note that we don't use a "deep equality" comparison, since that would
    /// be both slow and pointless, since the only reason one should ever
    /// compare these objects is to assert the user has used a consistent
    /// basis for each operator.
    ///
    bool operator==(const ManyBodyBasis &other) const
    {
        return this == &other;
    }

private:

    size_t _oper_offsets[RANK_COUNT + 1];

    std::vector<size_t> _block_offsets[OPER_KIND_COUNT + 1];

    // Since L1 is a subset of L2, this table suffices for all our purposes.
    //
    // (L2, L1) -> L2
    std::vector<OptionalIndex> _addition_table;

    // Negation table
    //
    // L2 -> L2
    std::vector<size_t> _negation_table;

    // StateKind -> L -> U
    std::vector<size_t> _state_offsets[STATE_KIND_COUNT];

    size_t _num_channels_1;

    // Adds a rank-2 channel to a rank-1 channel.
    OptionalIndex _add_channels(size_t l1, size_t l2) const
    {
        assert(l1 < this->num_channels(RANK_2));
        assert(l2 < this->num_channels(RANK_1));
        return this->_addition_table[l1 * this->num_channels(RANK_1) + l2];
    }

};

#endif
