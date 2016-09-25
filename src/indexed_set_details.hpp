template<typename K,
          typename H,
          typename E>
IndexedSet<K, H, E>::IndexedSet(size_t bucket_count,
                                const H &hash,
                                const E &equal)
    : _to_index(bucket_count, hash, equal)
{
}

template<typename K,
          typename H,
          typename E>
size_t IndexedSet<K, H, E>::size() const
{
    return this->_from_index.size();
}

template<typename K,
          typename H,
          typename E>
bool IndexedSet<K, H, E>::find(const K &key, size_t *index_out) const
{
    auto iter = this->_to_index.find(key);
    bool found = iter != this->_to_index.end();
    if (found && index_out) {
        *index_out = iter->second;
    }
    return found;
}

template<typename K,
          typename H,
          typename E>
const K &IndexedSet<K, H, E>::at(size_t index) const
{
    return this->_from_index.at(index);
}

template<typename K,
          typename H,
          typename E>
const K &IndexedSet<K, H, E>::operator[](size_t index) const
{
    assert(index < this->_from_index.size());
    return this->_from_index[index];
}

template<typename K,
          typename H,
          typename E>
bool IndexedSet<K, H, E>::insert(const K &key, size_t *index_out)
{
    bool not_found = !this->find(key, index_out);
    if (not_found) {
        size_t index = this->_from_index.size();
        this->_from_index.emplace_back(key);
        this->_to_index.emplace(key, index);
        if (index_out) {
            *index_out = index;
        }
    }
    return not_found;
}
