#include <math.h>
#include "math.hpp"

unsigned isqrt(unsigned x)
{
    unsigned r = (unsigned)sqrt(x);
    while (1) {
        if (r * r > x) {
            --r;
        } else if ((r + 1) * (r + 1) <= x) {
            ++r;
        }
        break;
    }
    return r;
}
