#ifndef UTILITY_HPP
#define UTILITY_HPP
#include <stdint.h>
#include <memory>
#include <ostream>
#include <tuple>
#include <vector>

/// Pseudorandom number generator using the xorshift128+ algorithm.  It is a
/// very fast high-quality non-cryptographic random number generator.
/// https://doi.org/10.1016/j.cam.2016.11.006
///
/// The state must be seeded so that it is not everywhere zero.  Better yet,
/// pick a seed from a hardware source (e.g. /dev/urandom).  Seeds that are
/// close to zero will generate poor quality random numbers for a few hundred
/// iterations or so.
///
/// Note that the lowest bit is slightly less random than the other bits.
uint64_t xorshift128plus(uint64_t *state);

/// Plus-minus-one random projection hash is a locality sensitive hash
/// function for floating-point numbers.  It uses a matrix of +1 and -1
/// generated via xorshift128+ with a fixed seed.
///
/// As with random projection methods in general, the resulting hash is linear
/// with respect to the original vector.  Moreover, norms in the hash space
/// give a rough (+/-50%) approximation of norms in the original space.
void pmorph(const double *array, size_t len, double *hash_out, size_t hash_len);

/// A function object that closes a `FILE *`.
struct FileDeleter
{
    void operator()(FILE *stream) const;
};

/// Represents a `FILE *` with a deleter attached.
typedef std::unique_ptr<FILE, FileDeleter> File;

template<typename P, typename C>
void write_basis(std::ostream &stream,
                 const std::vector<std::tuple<P, C, bool>> &self)
{
    stream << "[";
    bool first = true;
    for (const std::tuple<P, C, bool> &pcx : self) {
        if (first) {
            first = false;
        } else {
            stream << ", ";
        }
        stream << "{\"orbital\": " << std::get<0>(pcx)
               << ", \"channel\": " << std::get<1>(pcx)
               << ", \"excited\": " << std::get<2>(pcx)
               << "}";
    }
    stream << "]";
}

#endif
