#include "basis.hpp"

std::ostream &operator<<(std::ostream &stream, const Orbital &self)
{
    return stream
        << "{"
        << self.channel_index() << ", "
        << self.auxiliary_index() << "}";
}

ManyBodyBasis::ManyBodyBasis(const GenericOrbitalTable &table)
{
    size_t nl1 = table.num_channels(RANK_1);
    size_t nl2 = table.num_channels(RANK_2);

    // construct the addition and negation tables for channels
    this->_num_channels_1 = nl1;
    for (size_t l1 = 0; l1 < nl2; ++l1) {
        for (size_t l2 = 0; l2 < nl1; ++l2) {
            this->_addition_table.emplace_back(table.add_channels(l1, l2));
        }
    }
    for (size_t l = 0; l < nl2; ++l) {
        this->_negation_table.emplace_back(table.negate_channel(l));
    }

    // construct the rank-0 state offset table
    this->_state_offsets[STATE_KIND_00].emplace_back(0);
    this->_state_offsets[STATE_KIND_00].emplace_back(1);

    // construct the rank-1 state offset table
    for (size_t l = 0; l < nl1; ++l) {
        for (size_t x = 0; x <= 2; ++x) {
            size_t u = table.orbital_offset(l, x);
            this->_state_offsets[STATE_KIND_10].emplace_back(u);
        }
    }

    // construct the rank-2 state offset tables
    for (size_t l12 = 0; l12 < nl2; ++l12) {
        size_t u = 0;
        for (size_t x1 = 0; x1 < 2; ++x1) {
            for (size_t x2 = 0; x2 < 2; ++x2) {
                for (size_t l1 = 0; l1 < nl1; ++l1) {
                    this->_state_offsets[STATE_KIND_20].emplace_back(u);
                    size_t l2;
                    if (try_get(this->subtract_channels(l12, l1)
                                .within({0, nl1}), &l2)) {
                        u += this->num_orbitals_in_channel_part(l1, x1) *
                             this->num_orbitals_in_channel_part(l2, x2);
                    }
                }
            }
        }
        this->_state_offsets[STATE_KIND_20].emplace_back(u);
    }
    for (size_t l14 = 0; l14 < nl2; ++l14) {
        size_t u = 0;
        for (size_t x1 = 0; x1 < 2; ++x1) {
            for (size_t x4 = 0; x4 < 2; ++x4) {
                for (size_t l1 = 0; l1 < nl1; ++l1) {
                    this->_state_offsets[STATE_KIND_21].emplace_back(u);
                    size_t l4;
                    if (try_get(this->subtract_channels(l1, l14)
                                .within({0, nl1}), &l4)) {
                        u += this->num_orbitals_in_channel_part(l1, x1) *
                             this->num_orbitals_in_channel_part(l4, x4);
                    }
                }
            }
        }
        this->_state_offsets[STATE_KIND_21].emplace_back(u);
    }

    for (size_t ikk = 0; ikk < OPER_KIND_COUNT; ++ikk) {
        OperKind kk = (OperKind)ikk;
        Rank r = oper_kind_to_rank(kk);
        size_t nl = this->num_channels(r);
        size_t i = 0;
        for (size_t l = 0; l < nl; ++l) {
            this->_block_offsets[kk].emplace_back(i);
            size_t nu1, nu2;
            this->block_size(kk, l, &nu1, &nu2);
            i += nu1 * nu2;
        }
        this->_block_offsets[kk].emplace_back(i);
    }

    size_t i = 0;
    Rank r;
    for (r = RANK_0; r < RANK_COUNT; r = (Rank)(r + 1)) {
        OperKind kk = standard_oper_kind(r);
        this->_oper_offsets[r] = i;
        size_t nl = this->num_channels(r);
        i += this->block_offset(kk, nl);
    }
    this->_oper_offsets[r] = i;
}
