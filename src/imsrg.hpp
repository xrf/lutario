// IDEA:

// let's write a simple test suite to compare results with existing impl
// do this ONE TERM AT A TIME (!!) starting with the C222(0)

template<typename C>
void imsrg(const ManyBodyBasis<C> &many_body_basis, const ManyBodyOperator &A,
           const ManyBodyOperator &B, Operator &D, double alpha)
{
    // D21 & D22
    D21.clear();
    D22.clear();
    if (!reduced) {
        // D1 <-> C222(-2)
        // D2 <-> C222(+2)
        for (CHANNEL(p, q)) {
            const auto &dom_hh = basis_2.subindices(lpq, 0);
            const auto &dom_pp = basis_2.subindices(lpq, 2);
            commutr(B[lpq], A[lpq], dom_hh, D1[lpq]);
            commutr(A[lpq], B[lpq], dom_pp, D2[lpq]);
        }
    } else {
        for (CHANNEL(p, q)) {
            const auto &dom_aa = basis_2.subindices(lpq);
            const auto &dom_hh = basis_2.subindices(lpq, 0);
            const auto &dom_pp = basis_2.subindices(lpq, 2);
            commutr(B[lpq], A[lpq], dom_hh, dom_aa, dom_pp, D1[lpq]);
            commutr(A[lpq], B[lpq], dom_pp, dom_hh, dom_aa, D2[lpq]);
        }
    }

    // NOTE:
    // - we are only calculating HALF of the commutator (a.k.a. the connected operator product)
    // - we need to make sure alpha is present in all results in just the right amount
    // - we need to make sure the dependencies of the intermediate contractions are correct (the temporary wXXXX operators!)
    // - how do we deal with memory allocation?

    // C020
    w() += alpha * trace(w1111, PART_0);

    // C040
    w() += alpha * trace(w1311, PART_0) / 2.0;

    // C111(-1)
    gemm(TRANS, TRANS, TRANS, PART_ALL, PART_ALL, PART_0, -alpha, x, y, w[1]);

    // C111(+1)
    zero(w1111);
    gemm(NORM, NORM, NORM, PART_ALL, PART_ALL, PART_1, 1.0, x, y, w1111);
    axpy(PART_0, alpha, w1111, w);

    // C131(-1)
    for (CHANNEL(p)) {
        size_t l_q = l_p;
        size_t l_r = l_p;
        for (STATE(p)) {
            for (STATE(q)) {
                double z = 0.0;
                for (CHANNEL(a)) {
                    for (STATE(a)) {
                        z -= w2220(a, p, a, q);
                    }
                }
                w(p, q) += alpha * z;
            }
        }
    }

    // C131(+1)
    for (CHANNEL(p)) {
        size_t l_q = l_p;
        size_t l_r = l_p;
        for (STATE(p, {0, 2})) {
            for (STATE(q, {0, 2})) {
                double z = 0.0;
                for (CHANNEL(i)) {
                    for (STATE(i, {0, 1})) {
                        z += w2222(i, p, i, q);
                    }
                }
                w(p, q) += alpha * z;
            }
        }
    }

    // C022
    for (CHANNEL(p)) {
        size_t l_q = l_p;
        size_t l_r = l_p;
        for (STATE(p)) {
            for (STATE(q)) {
                double z = 0.0;
                for (CHANNEL(i)) {
                    for (STATE(i)) {
                        z += x(i, a) * y(a, p, i, q);
                    }
                }
                w(p, q) += alpha * z;
            }
        }
    }

    // 2-particle terms

    for (CHANNEL(p, q))
        for (STATE(p, q, {0, 4})) /* lpq <- lpq */
            for (STATE(r, s, {0, 4})) /* lrs <- lpq */ {
                auto &&z = w(p, q, r, s);

                auto &&lp_up_lq_uq = lu_split_2(basis_m, lpq, upq);
                auto &&lp = get<0>(lp_up_lq_uq);
                auto &&up = get<1>(lp_up_lq_uq);
                auto &&lq = get<2>(lp_up_lq_uq);
                auto &&uq = get<3>(lp_up_lq_uq);
                auto &&lr_ur_ls_us = lu_split_2(basis_m, lpq, urs);
                auto &&lr = get<0>(lr_ur_ls_us);
                auto &&ur = get<1>(lr_ur_ls_us);
                auto &&ls = get<2>(lr_ur_ls_us);
                auto &&us = get<3>(lr_ur_ls_us);

                auto &&upq = to_signed(upq);
                auto &&urs = to_signed(urs);

                // C20
                for (STATE(t, {0, 2})) /* lt <- lp */ {
                    auto &&utq = u_fuse1(basis_m, lp, ut, lq, uq);
                    z += x(p, t) * y(t, q, r, s) -
                         y(p, t) * x(t, q, r, s);
                }
                for (STATE(t, {0, 2})) /* lt <- lq */ {
                    auto &&utp = u_fuse1(basis_m, lq, ut, lp, up);
                    z -= x(q, t) * y(t, p, r, s) -
                         y(q, t) * x(t, p, r, s);
                }
                for (STATE(t, {0, 2})) /* lt <- lr */ {
                    auto &&uts = u_fuse1(basis_m, lr, ut, ls, us);
                    z += y(t, r) * x(p, q, t, s) -
                         x(t, r) * y(p, q, t, s);
                }
                for (STATE(t, {0, 2})) /* lt <- ls */ {
                    auto &&utr = u_fuse1(basis_m, ls, ut, lr, ur);
                    z += x(t, s) * y(p, q, t, r) -
                         y(t, s) * x(p, q, t, r);
                }

                // C21
                z += GET_2(D1, p, q, r, s);

                // C22
                z += GET_2(D2, p, q, r, s);
            }

    for (CHANNEL(p, q))
        for (STATE(p, q, {0, 4})) /* lpq <- lpq */
            for (STATE(r, s, {0, 4})) /* lrs <- lpq */ {
                auto &z = w(p, q, r, s);
                const auto lp_up_lq_uq = lu_split_2(basis_m, lpq, upq);
                const auto lp = get<0>(lp_up_lq_uq);
                const auto up = get<1>(lp_up_lq_uq);
                const auto lq = get<2>(lp_up_lq_uq);
                const auto uq = get<3>(lp_up_lq_uq);
                const auto lr_ur_ls_us = lu_split_2(basis_m, lpq, urs);
                const auto lr = get<0>(lr_ur_ls_us);
                const auto ur = get<1>(lr_ur_ls_us);
                const auto ls = get<2>(lr_ur_ls_us);
                const auto us = get<3>(lr_ur_ls_us);

                int xpq;
                if (upq < basis_2.excitation_offset(lpq, 1)) {
                    xpq = 0;
                } else if (upq < basis_2.excitation_offset(lpq, 2)) {
                    xpq = 1;
                } else {
                    xpq = 2;
                }
                int xrs;
                if (urs < basis_2.excitation_offset(lpq, 1)) {
                    xrs = 0;
                } else if (urs < basis_2.excitation_offset(lpq, 2)) {
                    xrs = 1;
                } else {
                    xrs = 2;
                }
                const bool x_01_2 =
                    !reduced || ((xpq == 0 || xpq == 1) && xrs == 2);
                const bool x_12_0 =
                    !reduced || ((xpq == 1 || xpq == 2) && xrs == 0);
                const bool x_01_12 = !reduced || ((xpq == 0 || xpq == 1) &&
                                                  (xrs == 1 || xrs == 2));
                const bool x_12_01 = !reduced || ((xpq == 1 || xpq == 2) &&
                                                  (xrs == 0 || xrs == 1));
                const bool x_0_12 =
                    !reduced || (xpq == 0 && (xrs == 1 || xrs == 2));
                const bool x_2_01 =
                    !reduced || (xpq == 2 && (xrs == 0 || xrs == 1));
                const bool x_0_2 = !reduced || (xpq == 0 && xrs == 2);
                const bool x_2_0 = !reduced || (xpq == 2 && xrs == 0);

                // C23
                for (CHANNEL(a)) {
                    const auto lpa = l_add_11(basis_m, lp, la);
                    const auto lqa = l_add_11(basis_m, lq, la);
                    const auto lra = l_add_11(basis_m, lr, la);
                    const auto lsa = l_add_11(basis_m, ls, la);
                    auto li = l_sub_21(basis_m, lra, lp);
                    if (li < basis_1.num_channels())
                        for (STATE(a, {1, 2})) /* la <- la */ {
                            const auto uqa = u_fuse_11(basis_m, lq, uq, la, ua);
                            const auto ura = u_fuse_11(basis_m, lr, ur, la, ua);
                            for (STATE(i, {0, 1})) /* li <- li */ {
                                const auto uip =
                                    u_fuse_11(basis_m, li, ui, lp, up);
                                const auto uis =
                                    u_fuse_11(basis_m, li, ui, ls, us);
                                // xpq = 0 or 1
                                // xrs = 2
                                // p = i, r = a
                                if (x_01_2)
                                    z += x(i, p, r, a) *
                                         y(q, a, i, s);
                                // xpq = 1 or 2
                                // xrs = 0
                                // s = i, q = a
                                if (x_12_0)
                                    z -= y(i, p, r, a) *
                                         x(q, a, i, s);
                            }
                        }
                    li = l_sub_21(basis_m, lsa, lp);
                    if (li < basis_1.num_channels())
                        for (STATE(a, {1, 2})) /* la <- la */ {
                            const auto uqa = u_fuse_11(basis_m, lq, uq, la, ua);
                            const auto usa = u_fuse_11(basis_m, ls, us, la, ua);
                            for (STATE(i, {0, 1})) /* li <- li */ {
                                const auto uip =
                                    u_fuse_11(basis_m, li, ui, lp, up);
                                const auto uir =
                                    u_fuse_11(basis_m, li, ui, lr, ur);
                                // xpq = 0 or 1
                                // xrs = 1 or 2
                                // p = i, s = a
                                if (x_01_12)
                                    z -= x(i, p, s, a) *
                                         y(q, a, i, r);
                                // xpq = 1 or 2
                                // xrs = 0 or 1
                                // r = i, q = a
                                if (x_12_01)
                                    z += y(i, p, s, a) *
                                         x(q, a, i, r);
                            }
                        }
                    li = l_sub_21(basis_m, lsa, lq);
                    if (li < basis_1.num_channels())
                        for (STATE(a, {1, 2})) /* la <- la */ {
                            const auto upa = u_fuse_11(basis_m, lp, up, la, ua);
                            const auto usa = u_fuse_11(basis_m, ls, us, la, ua);
                            for (STATE(i, {0, 1})) /* li <- li */ {
                                const auto uiq =
                                    u_fuse_11(basis_m, li, ui, lq, uq);
                                const auto uir =
                                    u_fuse_11(basis_m, li, ui, lr, ur);
                                // xpq = 0
                                // xrs = 1 or 2
                                // q = i, s = a
                                if (x_0_12)
                                    z += x(i, q, s, a) *
                                         y(p, a, i, r);
                                // xpq = 2
                                // xrs = 0 or 1
                                // r = i, p = a
                                if (x_2_01)
                                    z -= y(i, q, s, a) *
                                         x(p, a, i, r);
                            }
                        }
                    li = l_sub_21(basis_m, lra, lq);
                    if (li < basis_1.num_channels())
                        for (STATE(a, {1, 2})) /* la <- la */ {
                            const auto upa = u_fuse_11(basis_m, lp, up, la, ua);
                            const auto ura = u_fuse_11(basis_m, lr, ur, la, ua);
                            for (STATE(i, {0, 1})) /* li <- li */ {
                                const auto uiq =
                                    u_fuse_11(basis_m, li, ui, lq, uq);
                                const auto uis =
                                    u_fuse_11(basis_m, li, ui, ls, us);
                                // xpq = 0
                                // xrs = 2
                                // q = i, r = a
                                if (x_0_2)
                                    z -= x(i, q, r, a) *
                                         y(p, a, i, s);
                                // xpq = 2
                                // xrs = 0
                                // s = i, p = a
                                if (x_2_0)
                                    z += y(i, q, r, a) *
                                         x(p, a, i, s);
                            }
                        }
                }
            }
}
