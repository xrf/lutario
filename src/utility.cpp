#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include "utility.hpp"

uint64_t xorshift128plus(uint64_t *state)
{
    uint64_t x = state[0];
    uint64_t y = state[1];
    uint64_t r = x + y;
    uint64_t p = x ^ x << 23;
    uint64_t q = p ^ y ^ p >> 18 ^ y >> 5;
    state[0] = y;
    state[1] = q;
    return r;
}

void pmorph(const double *array, size_t len, double *hash_out, size_t hash_len)
{
    uint64_t state[] = {0x0d4b6b5eeea339da, 0x90186095efbf5532};
    double coeff = 1.0 / sqrt(hash_len);
    size_t i, j, k;
    for (i = 0; i < hash_len; ++i) {
        hash_out[i] = 0.0;
    }
    for (j = 0; j < len; ++j) {
        double array_j = array[j];
        for (i = 0; i < hash_len; i += 32) {
            uint64_t u = xorshift128plus(state);
            size_t k_stop = k < hash_len - i ? k : hash_len - i;
            for (k = 0; k < k_stop; ++k) {
                /* use only the upper 32 bits of xorshift128 because the
                   randomness of the lowest bit is slightly deficient */
                uint64_t positive = u & ((uint64_t)1 << (63 - k));
                if (positive) {
                    hash_out[i + k] += array_j;
                } else {
                    hash_out[i + k] -= array_j;
                }
            }
        }
    }
    for (i = 0; i < hash_len; ++i) {
        hash_out[i] *= coeff;
    }
}

void FileDeleter::operator()(FILE *stream) const
{
    fclose(stream);
}
