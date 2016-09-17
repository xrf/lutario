#ifndef BLAS_HPP
#define BLAS_HPP
#include <cblas.h>

// we need to figure out what integer type the BLAS implementation uses;
// either we get it from a CBLAS_INT macro defined elsewhere, or we are
// going to have to guess it ourselves using C++ template magics :/
#ifndef CBLAS_INT

namespace _details {

template<typename I>
I cblas_deduce_int_type(double (*)(I, const double *, I))
{
    return I();
}

}

/// Main integer type used by the BLAS implementation.
typedef decltype(_details::cblas_deduce_int_type(&cblas_dasum)) CBLAS_INT;

// define it to itself to so that we can tell it exists;
// this definition has no other effect
#define CBLAS_INT CBLAS_INT

#endif

#endif
