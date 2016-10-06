/// This function is for decorative purposes.  Taking the complex conjugate of
/// a real number has no effect.
inline double conj(double x)
{
    return x;
}

template<typename C>
void calc_white_generator(const ManyBodyBasis<C> &b, ManyBodyOperator h,
                          ManyBodyOperator eta_out)
{
    /* SELECT i, a WHERE x(i) = 0 AND x(a) = 1 AND g(i, a) AND g(i, i) AND g(a, a) AND G(i, a, i, a) */
    for (ITER_CHANNELS(li, b, RANK_1)) {
        for (ITER_AUXILIARY(ua, li, 0, 1, b, RANK_1)) {
            for (ITER_AUXILIARY(ui, li, 0, 1, b, RANK_1)) {
                size_t lii = b.add(RANK_1, RANK_1, RANK_2, li, li);
                size_t uia = b.combine(RANK_1, RANK_1, li, ui, li, ua);
                double z = b.get(h, RANK_1, li, ui, ua) /
                           (b.get(h, RANK_2, lii, uia, uia) +
                            b.get(h, RANK_1, li, ui, ui) - b.get(h, 1, li, ua, ua));
                b.get(eta_out, RANK_1, li, ui, ua) = z;
                b.get(eta_out, RANK_1, li, ua, ui) = -conj(z);
            }
        }
    }
    for (ITER_CHANNELS(lij, b, RANK_2)) {
        for (ITER_AUXILIARY(uab, lij, 3, 4, b, RANK_2)) {
            for (ITER_AUXILIARY(uij, lij, 0, 1, b, RANK_2)) {
                size_t li, ui, lj, uj;
                b.split_2(lij, uij, li, ui, lj, uj);
                size_t la, ua, lb, ub;
                b.split_2(lij, uab, la, ua, lb, ub);

                size_t lia = b.add(RANK_1, RANK_1, RANK_2, li, la);
                size_t lib = b.add(RANK_1, RANK_1, RANK_2, li, lb);
                size_t lja = b.add(RANK_1, RANK_1, RANK_2, lj, la);
                size_t ljb = b.add(RANK_1, RANK_1, RANK_2, lj, lb);

                size_t uia = b.combine(RANK_1, RANK_1, li, ui, la, ua);
                size_t uib = b.combine(RANK_1, RANK_1, li, ui, lb, ub);
                size_t uja = b.combine(RANK_1, RANK_1, lj, uj, la, ua);
                size_t ujb = b.combine(RANK_1, RANK_1, lj, uj, lb, ub);
                double z =
                    b.get(h, 2, lij, uij, uab) /
                    (b.get(h, 1, li, ui, ui) + b.get(h, 1, lj, uj, uj) -
                     b.get(h, 1, la, ua, ua) - b.get(h, 1, lb, ub, ub) +
                     b.get(h, 2, lia, uia, uia) + b.get(h, 2, lib, uib, uib) +
                     b.get(h, 2, lja, uja, uja) + b.get(h, 2, ljb, ujb, ujb) -
                     b.get(h, 2, lij, uij, uij) - b.get(h, 2, lij, uab, uab));
                b.get(eta_out, 2, lij, uij, uab) = z;
                b.get(eta_out, 2, lij, uab, uij) = -conj(z);
            }
        }
    }
}

/// The White generator.
template<class T>
struct WhiteGenerator {

    // use the reduced form of the commutator that uses only hole-particle
    // matrix elements (hp or hhpp)
    hermitian_commutator<adjoint_symmetry::antihermitian,
                         adjoint_symmetry::hermitian, T, BasisM, true>
        _commutator;

    /// Internal buffer used to store the generator matrix.
    std::unique_ptr<T[]> _gen_data;

    /// Constructor.
    ///
    /// @param basis_m  The many-particle basis.
    WhiteGenerator(const BasisM &basis_m)
        : _basis_m(basis_m), _commutator(_basis_m)
    {
        _gen_data = alloc<T>(matrix_m_size(_basis_m));
        gen = make_matrix_m_view(_basis_m, unmove(_gen_data.get()));
    }

    /// Calculates the generator.
    ///
    /// @param      in      The hermitian operator being evolved.
    template<class MatrixM_in>
    void calc_generator(const MatrixM_in &in)
    {
    }

    /// Calculates the generator and evolves the hermitian operator using the
    /// generator.
    ///
    /// @param      in      The hermitian operator being evolved.
    /// @param[out] out     The result.
    template<class MatrixM_in, class MatrixM_out>
    void operator()(const MatrixM_in &in, MatrixM_out &out)
    {
        calc_generator(in);
        _commutator(gen, in, out);
    }
    ntuple_t<std::vector<matrix_view<T>>, BasisM::max_arity> gen;
};