#ifndef INDEXED_SET_HPP
#define INDEXED_SET_HPP
#include <assert.h>
#include <stddef.h>
#include <functional>
#include <unordered_map>
#include <vector>

/// An `IndexedSet` is a set where each element is also associated with a
/// unique index drawn from the half-open range [0, N) where is the number of
/// elements in the set.  This forms a bijective map between an arbitrary set
/// of values with a set of ordinary integers.  The container provides
/// efficient lookup by either the value (`find`) or the index (`operator[]`).
/// The container does not support deletion of items.
///
/// The container guarantees the following properties:
///
///     // for any IndexedSet S and any Key K
///     size_t i;
///     if (S.find(K, &i)) {
///         assert(i < S.size());
///         assert(S[i] == K);
///     }
///
///     // for any IndexedSet S and any integer I
///     if (I >= 0 && I < S.size()) {
///         size_t i;
///         assert(S.find(S[I], &i) == true);
///         assert(I == i);
///     }
///
template<typename Key,
          typename Hash = std::hash<Key>,
          typename KeyEqual = std::equal_to<Key>>
class IndexedSet {

    std::vector<Key> _from_index;

    // We could, in principle, store this as an unordered_set<size_t> where
    // the elements are the indices to the _from_index vector, but there's no
    // way to do this efficiently through the interface of unordered_set due
    // to the lack of a means to look up using a type different from size_t
    // (we could store the index in the vector temporarily, but that would
    // require making an unnecessary copy, not to mention violating const-ness
    // rules).  We could also store "const Key &" but that would become
    // dangling if the vector gets resized.
    std::unordered_map<Key, size_t, Hash, KeyEqual> _to_index;

public:

    /// Construct an `IndexedSet` with no elements.
    explicit IndexedSet(size_t bucket_count = 0,
                        const Hash &hash = Hash(),
                        const KeyEqual &equal = KeyEqual());

    /// Returns the number of (unique) items in the set.
    size_t size() const;

    /// Gets index of an item in the set.  Returns whether the item was found.
    bool find(const Key &key, size_t *index_out = nullptr) const;

    /// Same as `operator[]` but fails with `std::out_of_range` if the index
    /// is out of range.
    const Key &at(size_t index) const;

    /// Get the item at the given index, which must be less than `size()`.
    const Key &operator[](size_t index) const;

    /// Insert an item if it does not exist already.  Returns whether the
    /// insertion was performed.
    bool insert(const Key &key, size_t *index_out = nullptr);

};

#include "indexed_set_details.hpp"
#endif
