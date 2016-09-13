template<typename C>
void imsrg(const ManyBodyBasis<C> &many_body_basis,
           const ManyBodyOperator &A,
           const ManyBodyOperator &B,
           Operator &D)
{
    // D21 & D22
    D21.clear();
    D22.clear();
    if (!reduced) {
        for (auto&& lpq : basis_2.channels()) {
            const auto& dom_hh = basis_2.subindices(lpq, 0);
            const auto& dom_pp = basis_2.subindices(lpq, 2);
            commutr(B2[lpq], A2[lpq], dom_hh, D21[lpq]);
            commutr(A2[lpq], B2[lpq], dom_pp, D22[lpq]);
        }
    } else {
        for (auto&& lpq : basis_2.channels()) {
            const auto& dom_aa = basis_2.subindices(lpq);
            const auto& dom_hh = basis_2.subindices(lpq, 0);
            const auto& dom_pp = basis_2.subindices(lpq, 2);
            commutr(B2[lpq], A2[lpq], dom_hh, dom_aa, dom_pp, D21[lpq]);
            commutr(A2[lpq], B2[lpq], dom_pp, dom_hh, dom_aa, D22[lpq]);
        }
    }

    // 0-particle terms

    // C00
    auto&& c0 = C0[0](0u, 0u);
    for (auto&& li : basis_1.channels())
    for (auto&& ua : basis_1.subindices(li, 1))
    for (auto&& ui : basis_1.subindices(li, 0))
        c0 += re(A1[li](ui, ua) * brfl(B1[li](ui, ua)));

    // C01
    for (auto&& lij : basis_2.channels())
    for (auto&& uab : basis_2.subindices(lij, 2))
    for (auto&& uij : basis_2.subindices(lij, 0))
        c0 += re(A2[lij](uij, uab) * brfl(B2[lij](uij, uab)));

    c0 *= 2;

    // 1-particle terms

    for (auto&& lp : basis_1.channels())
    for (auto&& up : basis_1.subindices(lp))
    for (auto&& uq : basis_1.subindices(lp)) {
        auto&& z = C1[lp](up, uq);

        // C10
        for (auto&& ur : basis_1.subindices(lp))
            // could perhaps be calculated using rank2k update instead
            // if you have the full matrix of lp
            //
            // either that or matrix multiplication
            z += A1[lp](up, ur) * brfl(B1[lp](uq, ur))
               - B1[lp](up, ur) * arfl(A1[lp](uq, ur));

        // C11
        for (auto&& li : basis_1.channels())
        for (auto&& ua : basis_1.subindices(li, 1))
        for (auto&& ui : basis_1.subindices(li, 0)) {
            auto&& lip = l_add_11(basis_m, li, lp);
            auto&& uip = u_fuse_11(basis_m, li, ui, lp, up);
            auto&& uiq = u_fuse_11(basis_m, li, ui, lp, uq);
            auto&& upa = u_fuse_11(basis_m, lp, up, li, ua);
            auto&& uqa = u_fuse_11(basis_m, lp, uq, li, ua);
            z += arfl(A1[li](ui, ua)) *      B2[lip](uip, uqa)
               - brfl(B1[li](ui, ua)) *      A2[lip](uip, uqa)
               -      A1[li](ui, ua)  * brfl(B2[lip](uiq, upa))
               +      B1[li](ui, ua)  * arfl(A2[lip](uiq, upa));
        }

        // C12
        for (auto&& la : basis_1.channels())
        for (auto&& ua : basis_1.subindices(la, 1)) {
            auto&& lpa = l_add_11(basis_m, lp, la);
            auto&& upa = u_fuse_11(basis_m, lp, up, la, ua);
            auto&& uqa = u_fuse_11(basis_m, lp, uq, la, ua);
            z -= D21[lpa](upa, uqa);
        }

        // C13
        for (auto&& li : basis_1.channels())
        for (auto&& ui : basis_1.subindices(li, 0)) {
            auto&& lip = l_add_11(basis_m, li, lp);
            auto&& uip = u_fuse_11(basis_m, li, ui, lp, up);
            auto&& uiq = u_fuse_11(basis_m, li, ui, lp, uq);
            z += D22[lip](uip, uiq);
        }
    }

    // 2-particle terms

    for (auto&& lpq : basis_2.channels())
    for (auto&& upq : basis_2.subindices(lpq))
    for (auto&& urs : basis_2.subindices(lpq)) {
        auto&& z = C2[lpq](upq, urs);

        auto&& lp_up_lq_uq = lu_split_2(basis_m, lpq, upq);
        auto&& lp = get<0>(lp_up_lq_uq);
        auto&& up = get<1>(lp_up_lq_uq);
        auto&& lq = get<2>(lp_up_lq_uq);
        auto&& uq = get<3>(lp_up_lq_uq);
        auto&& lr_ur_ls_us = lu_split_2(basis_m, lpq, urs);
        auto&& lr = get<0>(lr_ur_ls_us);
        auto&& ur = get<1>(lr_ur_ls_us);
        auto&& ls = get<2>(lr_ur_ls_us);
        auto&& us = get<3>(lr_ur_ls_us);

        auto&& upq_ = to_signed(upq);
        auto&& urs_ = to_signed(urs);

        // C20
        for (auto&& ut : basis_1.subindices(lp)) {
            auto&& utq = u_fuse_11(basis_m, lp, ut, lq, uq);
            z += arfl(A1[lp](ut, up)) * B2[lpq](utq, urs_)
               - brfl(B1[lp](ut, up)) * A2[lpq](utq, urs_);
        }
        for (auto&& ut : basis_1.subindices(lq)) {
            auto&& utp = u_fuse_11(basis_m, lq, ut, lp, up);
            z -= arfl(A1[lq](ut, uq)) * B2[lpq](utp, urs_)
               - brfl(B1[lq](ut, uq)) * A2[lpq](utp, urs_);
        }
        for (auto&& ut : basis_1.subindices(lr)) {
            auto&& uts = u_fuse_11(basis_m, lr, ut, ls, us);
            z += B1[lr](ut, ur) * A2[lpq](upq_, uts)
               - A1[lr](ut, ur) * B2[lpq](upq_, uts);
        }
        for (auto&& ut : basis_1.subindices(ls)) {
            auto&& utr = u_fuse_11(basis_m, ls, ut, lr, ur);
            z += A1[ls](ut, us) * B2[lpq](upq_, utr)
               - B1[ls](ut, us) * A2[lpq](upq_, utr);
        }

        // C21
        z += D21[lpq](upq, urs);

        // C22
        z += D22[lpq](upq, urs);
    }

    for (auto&& lpq : basis_2.channels())
    for (auto&& upq : basis_2.subindices(lpq))
    for (auto&& urs : basis_2.subindices(lpq)) {
        auto& z = C2[lpq](upq, urs);
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
        const bool x_01_2  = !reduced || ((xpq == 0 || xpq == 1) &&
                                          xrs == 2);
        const bool x_12_0  = !reduced || ((xpq == 1 || xpq == 2) &&
                                          xrs == 0);
        const bool x_01_12 = !reduced || ((xpq == 0 || xpq == 1) &&
                                          (xrs == 1 || xrs == 2));
        const bool x_12_01 = !reduced || ((xpq == 1 || xpq == 2) &&
                                          (xrs == 0 || xrs == 1));
        const bool x_0_12  = !reduced || (xpq == 0 &&
                                          (xrs == 1 || xrs == 2));
        const bool x_2_01  = !reduced || (xpq == 2 &&
                                          (xrs == 0 || xrs == 1));
        const bool x_0_2   = !reduced || (xpq == 0 && xrs == 2);
        const bool x_2_0   = !reduced || (xpq == 2 && xrs == 0);

        // C23
        for (auto&& la : basis_1.channels()) {
            const auto lpa = l_add_11(basis_m, lp, la);
            const auto lqa = l_add_11(basis_m, lq, la);
            const auto lra = l_add_11(basis_m, lr, la);
            const auto lsa = l_add_11(basis_m, ls, la);
            auto li = l_sub_21(basis_m, lra, lp);
            if (li < basis_1.num_channels())
              for (auto&& ua : basis_1.subindices(la, 1)) {
                const auto uqa = u_fuse_11(basis_m, lq, uq, la, ua);
                const auto ura = u_fuse_11(basis_m, lr, ur, la, ua);
                for (auto&& ui : basis_1.subindices(li, 0)) {
                    const auto uip = u_fuse_11(basis_m, li, ui, lp, up);
                    const auto uis = u_fuse_11(basis_m, li, ui, ls, us);
                    // xpq = 0 or 1
                    // xrs = 2
                    // p = i, r = a
                    if (x_01_2)
                    z +=        A2[lra](uip, ura)
                         * brfl(B2[lqa](uis, uqa));
                    // xpq = 1 or 2
                    // xrs = 0
                    // s = i, q = a
                    if (x_12_0)
                    z -=        B2[lra](uip, ura)
                         * arfl(A2[lqa](uis, uqa));
                }
            }
            li = l_sub_21(basis_m, lsa, lp);
            if (li < basis_1.num_channels())
              for (auto&& ua : basis_1.subindices(la, 1)) {
                const auto uqa = u_fuse_11(basis_m, lq, uq, la, ua);
                const auto usa = u_fuse_11(basis_m, ls, us, la, ua);
                for (auto&& ui : basis_1.subindices(li, 0)) {
                    const auto uip = u_fuse_11(basis_m, li, ui, lp, up);
                    const auto uir = u_fuse_11(basis_m, li, ui, lr, ur);
                    // xpq = 0 or 1
                    // xrs = 1 or 2
                    // p = i, s = a
                    if (x_01_12)
                    z -=        A2[lsa](uip, usa)
                         * brfl(B2[lqa](uir, uqa));
                    // xpq = 1 or 2
                    // xrs = 0 or 1
                    // r = i, q = a
                    if (x_12_01)
                    z +=        B2[lsa](uip, usa)
                         * arfl(A2[lqa](uir, uqa));
                }
            }
            li = l_sub_21(basis_m, lsa, lq);
            if (li < basis_1.num_channels())
              for (auto&& ua : basis_1.subindices(la, 1)) {
                const auto upa = u_fuse_11(basis_m, lp, up, la, ua);
                const auto usa = u_fuse_11(basis_m, ls, us, la, ua);
                for (auto&& ui : basis_1.subindices(li, 0)) {
                    const auto uiq = u_fuse_11(basis_m, li, ui, lq, uq);
                    const auto uir = u_fuse_11(basis_m, li, ui, lr, ur);
                    // xpq = 0
                    // xrs = 1 or 2
                    // q = i, s = a
                    if (x_0_12)
                    z +=        A2[lsa](uiq, usa)
                         * brfl(B2[lpa](uir, upa));
                    // xpq = 2
                    // xrs = 0 or 1
                    // r = i, p = a
                    if (x_2_01)
                    z -=        B2[lsa](uiq, usa)
                         * arfl(A2[lpa](uir, upa));
                }
            }
            li = l_sub_21(basis_m, lra, lq);
            if (li < basis_1.num_channels())
              for (auto&& ua : basis_1.subindices(la, 1)) {
                const auto upa = u_fuse_11(basis_m, lp, up, la, ua);
                const auto ura = u_fuse_11(basis_m, lr, ur, la, ua);
                for (auto&& ui : basis_1.subindices(li, 0)) {
                    const auto uiq = u_fuse_11(basis_m, li, ui, lq, uq);
                    const auto uis = u_fuse_11(basis_m, li, ui, ls, us);
                    // xpq = 0
                    // xrs = 2
                    // q = i, r = a
                    if (x_0_2)
                    z -=        A2[lra](uiq, ura)
                         * brfl(B2[lpa](uis, upa));
                    // xpq = 2
                    // xrs = 0
                    // s = i, p = a
                    if (x_2_0)
                    z +=        B2[lra](uiq, ura)
                         * arfl(A2[lpa](uis, upa));
                }
            }
        }
    }
}
