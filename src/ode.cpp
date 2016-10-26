#include <assert.h>
#include <stddef.h>
#include <functional>
#include <memory>
#include <sg_ode.h>
#include "math.hpp"
#include "ode.hpp"

Ode::Ode(size_t size, double x, double *y, DerivFn deriv)
    : size(size)
    , x(x)
    , y(y)
    , deriv(std::move(deriv))
{
}

void Ode::deriv_fn(void *ctx, double x, const double *y, double *dy_out)
{
    const DerivFn *f = (const DerivFn *)ctx;
    assert(f != nullptr);
    (*f)(x, y, dy_out);
}

void *Ode::deriv_ctx() const
{
    return const_cast<DerivFn *>(&this->deriv);
}

ShampineGordon::ShampineGordon(Ode ode)
    : _ode(std::move(ode))
    , _work(new double[this->_work_size(ode.size)]())
    , _iwork()
{
}

const Ode &ShampineGordon::ode() const
{
    return this->_ode;
}

Ode &ShampineGordon::ode()
{
    return this->_ode;
}

ShampineGordon::Status ShampineGordon::step(
    double x_target,
    const Tolerance &tolerance,
    ShampineGordon::Flags flags)
{
    int r = sg_ode(this->ode().deriv_ctx(),
                   &Ode::deriv_fn,
                   this->ode().size,
                   this->ode().y,
                   &this->ode().x,
                   x_target,
                   tolerance.relerr,
                   tolerance.abserr,
                   (int)flags,
                   this->_work.get(),
                   this->_iwork);
    if (r == SG_ODE_EINVAL) {
        throw std::logic_error("ShampineGordon::step(): invalid argument");
    }
    return (Status)r;
}

size_t ShampineGordon::_work_size(size_t num_equations) {
    return 100 + 21 * num_equations;
}
