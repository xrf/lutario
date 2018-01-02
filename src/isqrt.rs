/// Calculate the integer square root `⌊√n⌋`.
pub fn isqrt_i64(n: i64) -> i64 {
    assert!(n >= 0);
    isqrt_u64(n as _) as _
}

/// Calculate the integer square root `⌊√n⌋`.
pub fn isqrt_u64(n: u64) -> u64 {
    if n == 0 {
        return 0;
    }
    // use floating-point calculations to get an initial guess
    let mut r = (n as f64).sqrt() as _;
    // apply integer variant of Newton's method to refine guess
    loop { // [invariant] n > 0 && r > 0
        let r_new = (r + n / r) / 2;
        // check for either no change or a +1 increase (the latter condition
        // is sufficient but not necessary to ensure r is valid; it's needed
        // to avoid nonconverging cycles)
        if r == r_new || r == r_new - 1 {
            return r;
        }
        r = r_new;
    }
}

#[test]
fn test() {
    use self::isqrt_u64 as isqrt;
    for n in 0 .. 65535 {
        assert!((isqrt(n)).pow(2) <= n, "⌊√{n}⌋^2 ≤ {n}", n = n);
        assert!((isqrt(n) + 1).pow(2) > n,
                "(⌊√{n}⌋ + 1)^2 > {n}", n = n);
    }
    // test some large square roots
    for r in 0 .. 65535 {
        if r > 0 {
            let n = r * r - 1;
            assert_eq!(isqrt(n), r - 1, "⌊√{}⌋ == {}", n, r - 1);
        }
        let n = r * r;
        assert_eq!(isqrt(n), r, "⌊√{}⌋ == {}", n, r);
        if r > 0 {
            let n = r * r + 1;
            assert_eq!(isqrt(n), r, "⌊√{}⌋ == {}", n, r);
        }
    }
    // test some extreme cases
    for r in 4294967000 .. 4294967296 {
        if r > 0 {
            let n = r * r - 1;
            assert_eq!(isqrt(n), r - 1, "⌊√{}⌋ == {}", n, r - 1);
        }
        let n = r * r;
        assert_eq!(isqrt(n), r, "⌊√{}⌋ == {}", n, r);
        if r > 0 {
            let n = r * r + 1;
            assert_eq!(isqrt(n), r, "⌊√{}⌋ == {}", n, r);
        }
    }
    for n in 0xffffffffffffff00 .. 0xffffffffffffffff {
        assert_eq!(isqrt(n), 0xffffffff);
    }
}
