import functools
import numpy as np
import scipy.integrate as spi

def zero_op(num_p):
    return np.array([np.zeros((num_p,) * (2 * r)) for r in range(3)],
                    dtype=object)

def flatten_op(op):
    return np.concatenate([
        op[0].flatten(),
        op[1].flatten(),
        op[2].flatten(),
    ])

def unflatten_op(num_p, op):
    return np.array([
        op[:1].reshape((num_p,) * 0),
        op[1:(1 + num_p * num_p)].reshape((num_p,) * 2),
        op[(1 + num_p * num_p):].reshape((num_p,) * 4),
    ], dtype=object)

def pairing_hamil(num_n, g):
    h = zero_op(num_p)

    for n in range(num_n):
        for ums in range(2):
            h[1][2 * n + ums, 2 * n + ums] = n

    for n1 in range(num_n):
        for n2 in range(num_n):
            for ums1 in range(2):
                ums2 = 1 - ums1
                x = g / 2
                h[2][2 * n1 + ums1, 2 * n1 + ums2,
                     2 * n2 + ums1, 2 * n2 + ums2] = -x
                h[2][2 * n1 + ums1, 2 * n1 + ums2,
                     2 * n2 + ums2, 2 * n2 + ums1] = x
    return h

def normal_order(num_p, pf, h):
    hn = zero_op(num_p)
    hn[0] = (
        h[0]
        + np.einsum("i i", h[1][:pf, :pf])
        + np.einsum("i j i j", h[2][:pf, :pf, :pf, :pf]) / 2
    )
    hn[1] = h[1] + np.einsum("p i q i", h[2][:, :pf, :, :pf])
    hn[2] = h[2]
    return hn

def denom2_mp(h):
    return (
        np.diag(h[1])[:, np.newaxis, np.newaxis, np.newaxis]
        + np.diag(h[1])[np.newaxis, :, np.newaxis, np.newaxis]
        - np.diag(h[1])[np.newaxis, np.newaxis, :, np.newaxis]
        - np.diag(h[1])[np.newaxis, np.newaxis, np.newaxis, :]
    )

def denom2_en(h):
    return denom_mp(h) + (
        + np.einsum("a b a b -> a b", h[2])[:, :, np.newaxis, np.newaxis]
        - np.einsum("a i a i -> a i", h[2])[:, np.newaxis, :, np.newaxis]
        - np.einsum("b i b i -> b i", h[2])[np.newaxis, :, :, np.newaxis]
        + np.einsum("i j i j -> i j", h[2])[np.newaxis, np.newaxis, :, :]
        - np.einsum("a j a j -> a j", h[2])[:, np.newaxis, np.newaxis, :]
        - np.einsum("b j b j -> b j", h[2])[np.newaxis, :, np.newaxis, :]
    )

def white_generator(denom2, num_p, pf, h):
    eta = zero_op(num_p)

    eta[1][pf:, :pf] = h[1][pf:, :pf] / (
        np.diag(h[1])[:, np.newaxis]
        - np.diag(h[1])[np.newaxis, :]
        - np.einsum("a i a i -> a i", h[2])
    )[pf:, :pf]
    eta[1] = eta[1] - np.einsum("a i -> i a", eta[1])

    eta[2][pf:, pf:, :pf, :pf] = (h[2][pf:, pf:, :pf, :pf] /
                                  denom2(h)[pf:, pf:, :pf, :pf])
    eta[2] = eta[2] - np.einsum("a b i j -> i j a b", eta[2])

    return eta

def linked_product(num_p, pf, a, b):
    d = zero_op(num_p)
    d[0] = (
        np.einsum("i a, a i", a[1][:pf, pf:], b[1][pf:, :pf])
        + 0.25 * np.einsum("i j a b, a b i j",
                           a[2][:pf, :pf, pf:, pf:],
                           b[2][pf:, pf:, :pf, :pf])
    )
    d[1] = (
        np.einsum("p r, r q", a[1], b[1])
        - 0.5 * np.einsum("i j a q, a p i j",
                          a[2][:pf, :pf, pf:, :],
                          b[2][pf:, :, :pf, :pf])
        + 0.5 * np.einsum("i p a b, a b i q",
                          a[2][:pf, :, pf:, pf:],
                          b[2][pf:, pf:, :pf, :])
        + np.einsum("i a, a p i q",
                    a[1][:pf, pf:],
                    b[2][pf:, :, :pf, :])
        + np.einsum("i p a q, a i",
                    a[2][:pf, :, pf:, :],
                    b[1][pf:, :pf])
    )
    d[2] = (
        np.einsum("i p a r, a q i s",
                  a[2][:pf, :, pf:, :],
                  b[2][pf:, :, :pf, :])
        - np.einsum("i q a r, a p i s",
                  a[2][:pf, :, pf:, :],
                  b[2][pf:, :, :pf, :])
        - np.einsum("i p a s, a q i r",
                  a[2][:pf, :, pf:, :],
                  b[2][pf:, :, :pf, :])
        + np.einsum("i q a s, a p i r",
                  a[2][:pf, :, pf:, :],
                  b[2][pf:, :, :pf, :])
        + 0.5 * np.einsum("i j r s, p q i j",
                          a[2][:pf, :pf, :, :],
                          b[2][:, :, :pf, :pf])
        + 0.5 * np.einsum("p q a b, a b r s",
                          a[2][:, :, pf:, pf:],
                          b[2][pf:, pf:, :, :])
        + np.einsum("q t, p t r s", a[1], b[2])
        - np.einsum("p t, q t r s", a[1], b[2])
        + np.einsum("p q r t, t s", a[2], b[1])
        - np.einsum("p q s t, t r", a[2], b[1])
    )
    return d

def commutator(num_p, pf, a, b):
    return linked_product(num_p, pf, a, b) - linked_product(num_p, pf, b, a)

def deriv(num_p, num_particles, generator, s, h):
    h = unflatten_op(num_p, h)
    eta = generator(num_p, num_particles, h)
    dh = commutator(num_p, num_particles, eta, h)
    return flatten_op(dh)

# p = 2 * n + ums
# ums = (tms + 1) / 2
# ms = tms / 2 (note: ms is a half-integer)

num_n = 2
num_particles = 4
g = 1.0
num_p = num_n * 2

h = pairing_hamil(num_n, g)
hn = normal_order(num_p, num_particles, h)

generator = functools.partial(white_generator, denom2_mp)
ode = spi.ode(functools.partial(deriv, num_p, num_particles, generator))
ode.set_initial_value(flatten_op(hn), 0.0)
last_e = ode.y[0]
while True:
    ode.integrate(ode.t + 1.0)
    e = ode.y[0]
    print(e)
    if abs(e - last_e) < 1e-8:
        break
    last_e = e
