#ifndef UTILITY_HPP
#define UTILITY_HPP

size_t combine_hash(size_t h1, size_t h2);

inline
size_t combine_hash(size_t h1, size_t h2)
{
    // a magical hash combining algorithm from Boost
    return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
}

#endif
