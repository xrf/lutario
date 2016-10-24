#ifndef MATH_HPP
#define MATH_HPP

unsigned isqrt(unsigned x);

/// This function is for decorative purposes.  Taking the complex conjugate of
/// a real number has no effect.
inline double conj(double x)
{
    return x;
}

#endif
