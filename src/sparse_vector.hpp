#ifndef SPARSE_VECTOR_HPP
#define SPARSE_VECTOR_HPP
#include <stddef.h>
#include <functional>
#include <ostream>
#include <unordered_map>

template<typename K, typename T>
class SparseVector {

    // Invariant: all entries must have nonzero value.
    std::unordered_map<K, T> _entries;

public:

    explicit SparseVector(const std::unordered_map<K, T> &entries = {})
        : _entries(entries)
    {
        // eliminate zero entries to ensure invariant holds
        for (auto ikv = this->_entries.begin(); ikv != this->_entries.end();) {
            if (ikv->second) {
                ++ikv;
            } else {
                ikv = this->_entries.erase(ikv);
            }
        }
    }

    /// Get the nonzero entries.
    const std::unordered_map<K, T> &entries() const
    {
        return this->_entries;
    }

    SparseVector operator+(const SparseVector &other) const
    {
        SparseVector r = *this;
        for (const std::pair<const K, T> &kv : other.entries()) {
            if (!(r._entries[kv.first] += kv.second)) {
                r._entries.erase(kv.first);
            }
        }
        return r;
    }

    SparseVector operator-(const SparseVector &other) const
    {
        SparseVector r = *this;
        for (const std::pair<const K, T> &kv : other.entries()) {
            if (!(r._entries[kv.first] -= kv.second)) {
                r._entries.erase(kv.first);
            }
        }
        return r;
    }

    SparseVector operator-() const
    {
        SparseVector r = *this;
        for (std::pair<const K, T> &kv : r._entries) {
            kv.second = -kv.second;
        }
        return r;
    }

    bool operator==(const SparseVector &other) const
    {
        return this->entries() == other.entries();
    }

    bool operator!=(const SparseVector &other) const
    {
        return !(*this == other);
    }

};

template<typename K, typename T>
inline std::ostream &operator<<(std::ostream &stream,
                                const SparseVector<K, T> &vec)
{
    stream << "SparseVector({";
    bool first = true;
    for (const auto &kv : vec.entries()) {
        if (first) {
            first = false;
        } else {
            stream << ", ";
        }
        stream << "{" << kv.first << ", " << kv.second << "}";
    }
    stream << "})";
    return stream;
}

namespace std {

template<typename K, typename T>
struct hash<SparseVector<K, T>> {

    size_t operator()(const SparseVector<K, T> &channel) const
    {
        // not sure how good this hash function is tbh
        size_t h = 0;
        for (const auto &kv : channel.entries()) {
            size_t hk = this->_hash_k(kv.first);
            size_t hv = this->_hash_v(kv.second);
            // a magical hash combining algorithm from Boost
            h ^= hk ^ (hv + 0x9e3779b9 + (hk << 6) + (hk >> 2));
        }
        return h;
    }

private:

    hash<K> _hash_k;

    hash<T> _hash_v;

};

}

#endif
