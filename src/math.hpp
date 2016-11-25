#ifndef MATH_HPP
#define MATH_HPP
#include <math.h>

struct Tolerance {

    double abserr, relerr;

    bool check(double x, double y) const
    {
        return fabs(x - y) < this->abserr + this->relerr * 0.5 * fabs(x + y);
    }

};

unsigned isqrt(unsigned x);

/// This function is for decorative purposes.  Taking the complex conjugate of
/// a real number has no effect.
inline double conj(double x)
{
    return x;
}

/// Calculate `|x|^2`.
inline double normsq(double x)
{
    return x * x;
}

#endif
