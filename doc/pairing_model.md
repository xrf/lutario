# Pairing model

## Basis

States are defined by two quantum numbers:

  - principal quantum number `n = {0, 1, 2, …}`, and
  - spin quantum number `s = {−1, +1}`.

Note that here we use the convention of representing spin by an integer twice its normal value.  Therefore, even though these are spin-½ particles, we denote its two possible spin quantum numbers as −1 and +1 instead of −½ and +½.

## Hamiltonian

Hamiltonian:

    H = U + V

where the one-body operator `U` is given by:

    U = ∑[n s] (n − 1) a†[n s] a[n s]

and the two-body operator `V` is given by:

    V = −(g / 2) ∑[n1 n2] a†[n1 +] a†[n1 −] a[n2 −] a[n2 +]

## Conservation law

The spin of each energy level `n` is conserved:

    S[n] = ∑[s] s a†[n s] a[n s]
