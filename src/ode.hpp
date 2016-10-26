#ifndef ODE_HPP
#define ODE_HPP
#include <stddef.h>
#include <functional>
#include <memory>
#include <sg_ode.h>
#include "math.hpp"

/// Defines the state of an ordinary diferential equation.
struct Ode {

    typedef std::function<void (double, const double *, double *)> DerivFn;

    size_t size;

    double x;

    double *y;

    DerivFn deriv;

    Ode(size_t size, double x, double *y, DerivFn deriv);

    static void deriv_fn(void *ctx, double x, const double *y, double *dy_out);

    void *deriv_ctx() const;

};

class ShampineGordon {

public:

    /// Input flags.
    enum class Flags {

        /// Nothing special.
        Normal = 0,

        /// Do not exceed the target value.
        Strict = SG_ODE_FSTRICT

    };

    /// Status of the solver.
    enum class Status {

        /// Integration was successful.
        Ok = 0,

        /// Integration did not reach target because the error tolerances were
        /// too small.
        ToleranceTooSmall = SG_ODE_ETOLER,

        /// Integration did not reach target because too many (> 500) steps
        /// were taken.
        TooManySteps = SG_ODE_ESTEPS,

        /// Integration did not reach target because the equations appear to
        /// be stiff.
        TooStiff = SG_ODE_ESTIFF

    };

    ShampineGordon(Ode ode);

    const Ode &ode() const;

    Ode &ode();

    Status step(double x_target,
                const Tolerance &tolerance,
                Flags flags = Flags::Normal);

private:

    Ode _ode;

    std::unique_ptr<double[]> _work;

    int _iwork[5];

    static size_t _work_size(size_t num_equations);

};

#endif
