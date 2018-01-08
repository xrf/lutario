use std::{ascii, cmp, fmt, mem, panic, ptr, process};
use std::ops::{Add, Range, Sub};
use std::collections::{BTreeMap, HashMap};
use std::hash::{BuildHasher, Hash};
use conv::ValueInto;
use num::{self, Bounded, One, ToPrimitive, Zero};
use take_mut;
use super::parity;
use super::basis::HashChart; // FIXME: HashChart doesn't belong in basis

/// Helper struct for writing `Debug` implementations.
pub struct DebugWith<F>(pub F);

impl<F: Fn(&mut fmt::Formatter) -> fmt::Result> fmt::Debug for DebugWith<F> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.0(f)
    }
}

/// No-op trait: no constraints; satisfied by all types.
pub trait Erased {}

impl<T: ?Sized> Erased for T {}

pub trait Offset {
    unsafe fn offset(self, count: isize) -> Self;
}

impl<T> Offset for *const T {
    unsafe fn offset(self, count: isize) -> Self {
        self.offset(count)
    }
}

impl<T> Offset for *mut T {
    unsafe fn offset(self, count: isize) -> Self {
        self.offset(count)
    }
}

/// Maximum possible `Range`, which necessarily excludes the maximum value.
pub fn max_range<T: Bounded>() -> Range<T> {
    Bounded::min_value() .. Bounded::max_value()
}

pub fn cast_range<T: ValueInto<U>, U>(r: Range<T>) -> Range<U> {
    cast(r.start) .. cast(r.end)
}

/// A more sanely defined addition trait to avoid throwing the compiler into
/// an infinite loop.
pub trait RefAdd {
    fn ref_add(&self, rhs: &Self) -> Self;
}

impl<T> RefAdd for T where
    for<'a, 'b> &'a T: Add<&'b T, Output = Self>,
{
    fn ref_add(&self, rhs: &Self) -> Self {
        self + rhs
    }
}

pub fn euclid_div(a: i32, b: i32) -> i32 {
    let r = a / b;
    if a % b < 0 {
        r + if b < 0 { 1 } else { -1 }
    } else {
        r
    }
}

pub fn euclid_mod(a: i32, b: i32) -> i32 {
    let r = a % b;
    if r < 0 {
        r + b.abs()
    } else {
        r
    }
}

/// Temporary definition until `std::ops::RangeInclusive` is stabilized.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RangeInclusive<Idx> {
    pub start: Idx,
    pub end: Idx,
}

impl<A: Add<Output = A> + Sub<Output = A> + One + Clone> RangeInclusive<A> {
    pub fn len(&self) -> A {
        self.end.clone() - self.start.clone() + One::one()
    }
}

impl<A> Iterator for RangeInclusive<A>
    where A: Clone + PartialOrd + Zero + One + ToPrimitive
{
    type Item = A;

    #[inline]
    fn next(&mut self) -> Option<A> {
        match self.start.partial_cmp(&self.end) {
            Some(cmp::Ordering::Less) => {
                let n = self.start.clone() + One::one();
                Some(mem::replace(&mut self.start, n))
            },
            Some(cmp::Ordering::Equal) => {
                let last = mem::replace(&mut self.start, One::one());
                self.end = Zero::zero();
                Some(last)
            },
            _ => None,
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let bound = match self.start.to_i64() {
            Some(a) => self.end.to_i64()
                .and_then(|b| b.checked_sub(a))
                .and_then(|b| ToPrimitive::to_usize(&b)),
            None => match self.start.to_u64() {
                Some(a) => self.end.to_u64()
                    .and_then(|b| b.checked_sub(a))
                    .and_then(|b| ToPrimitive::to_usize(&b)),
                None => None
            }
        };
        match bound {
            Some(b) => (
                b.checked_add(One::one()).unwrap_or(b),
                b.checked_add(One::one()),
            ),
            None => (0, None)
        }
    }
}

pub fn intersect_range_inclusive<Idx: Ord>(
    range1: RangeInclusive<Idx>,
    range2: RangeInclusive<Idx>,
) -> RangeInclusive<Idx> {
    RangeInclusive {
        start: cmp::max(range1.start, range2.start),
        end: cmp::min(range1.end, range2.end),
    }
}

pub fn swap<T>((x, y): (T, T)) -> (T, T) {
    (y, x)
}

pub fn swap_if<T>(cond: bool, val: (T, T)) -> (T, T) {
    if cond {
        swap(val)
    } else {
        val
    }
}

pub fn with_tuple2_ref<A, B, F, R>(a: &A, b: &B, f: F) -> R
    where F: FnOnce(&(A, B)) -> R
{
    unsafe {
        f(&mem::ManuallyDrop::new((ptr::read(a), ptr::read(b))))
    }
}

/// Shorthand for casting numbers.  Panics if out of range.
pub fn cast<T: ValueInto<U>, U>(x: T) -> U {
    try_cast(x).expect("integer conversion failure")
}

pub fn try_cast<T: ValueInto<U>, U>(x: T) -> Option<U> {
    x.value_into().ok()
}

pub fn take_and_get<F, T, R>(mut_ref: &mut T, closure: F) -> R
    where F: FnOnce(T) -> (R, T),
{
    let mut ret = None;
    take_mut::take(mut_ref, |t| {
        let (r, t) = closure(t);
        ret = Some(r);
        t
    });
    ret.expect("unreachable")
}

pub fn chop_slice<'a, T>(
    slice: &mut &'a [T],
    index: usize,
) -> Option<&'a [T]> {
    if slice.len() >= index {
        let (chopped, rest) = slice.split_at(index);
        *slice = rest;
        Some(chopped)
    } else {
        None
    }
}

pub fn chop_slice_mut<'a, T>(
    slice: &mut &'a mut [T],
    index: usize,
) -> Option<&'a mut [T]> {
    if slice.len() >= index {
        Some(take_and_get(slice, |slice| {
            slice.split_at_mut(index)
        }))
    } else {
        None
    }
}

/// Works just like `HashMap::default()` but avoids unnecessary bounds on `K`.
pub fn default_hash_map<K, V, S: BuildHasher + Default>() -> HashMap<K, V, S> {
    // very sketchy! (assuming Default
    let m = HashMap::<(), (), S>::default();
    let r = unsafe { mem::transmute_copy(&m) };
    assert_eq!(mem::size_of_val(&m), mem::size_of_val(&r));
    mem::forget(m);
    r
}

pub fn pretty_bytes(bytes: &[u8]) -> String {
    bytes.iter()
        .cloned()
        .flat_map(ascii::escape_default)
        .map(|c| c as char)
        .collect()
}

pub trait Key<V> {
    type Map: Map<Self, Value = V>;
}

pub trait Map<K: ?Sized> {
    type Value;
    fn get(&self, key: &K) -> Option<Self::Value>;
}

pub trait MapMut<K>: Map<K> {
    fn new() -> Self;
    fn insert(&mut self, key: K, value: Self::Value);
}

impl<K, V, F: Fn(&K) -> Option<V>> Map<K> for F {
    type Value = V;
    fn get(&self, key: &K) -> Option<Self::Value> {
        self(key)
    }
}

impl<V: Clone> Map<usize> for Vec<V> {
    type Value = V;
    fn get(&self, key: &usize) -> Option<Self::Value> {
        (**self).get(*key).cloned()
    }
}

impl<V: Clone> Map<usize> for Box<[V]> {
    type Value = V;
    fn get(&self, key: &usize) -> Option<Self::Value> {
        (**self).get(*key).cloned()
    }
}

impl<K: Ord, V: Clone> Map<K> for BTreeMap<K, V> {
    type Value = V;
    fn get(&self, key: &K) -> Option<Self::Value> {
        self.get(key).cloned()
    }
}

impl<K: Eq + Hash, V: Clone, S: BuildHasher> Map<K> for HashMap<K, V, S> {
    type Value = V;
    fn get(&self, key: &K) -> Option<Self::Value> {
        self.get(key).cloned()
    }
}

pub fn gcd(mut a: i32, mut b: i32) -> i32 {
    // avoid funky results on negative numbers
    a = a.abs();
    b = b.abs();
    while b != 0 {
        let old_b = b;
        b = a % b;
        a = old_b;
    }
    return a;
}

/// An ordered sequence of numbers `[ start + k × step | 0 ≤ k < len ]`
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RangeSet {
    pub start: i32,
    pub step: i32,
    pub len: i32,
}

impl Default for RangeSet {
    fn default() -> Self {
        Self { start: 0, step: 1, len: 0 }
    }
}

impl RangeSet {
    pub fn len(&self) -> usize {
        self.len as _
    }

    pub fn get(&self, i: usize) -> Option<i32> {
        if i < self.len() {
            Some(self.unchecked_get(i))
        } else {
            None
        }
    }

    pub fn unchecked_get(&self, i: usize) -> i32 {
        self.start + i as i32 * self.step
    }

    pub fn first(&self) -> Option<i32> {
        if self.len() == 0 {
            None
        } else {
            Some(self.unchecked_get(0))
        }
    }

    pub fn last(&self) -> Option<i32> {
        if self.len() == 0 {
            None
        } else {
            Some(self.unchecked_get(self.len() - 1))
        }
    }

    pub fn contains(&self, i: i32) -> bool {
        let offset = i - self.start;
        offset >= 0
            && offset % self.step == 0
            && offset / self.step < self.len
    }

    pub fn position(&self, i: i32) -> Option<usize> {
        if self.contains(i) {
            Some(self.unchecked_position(i))
        } else {
            None
        }
    }

    pub fn unchecked_position(&self, i: i32) -> usize {
        let offset = i - self.start;
        (offset / self.step) as _
    }

    pub fn ceil(&self, i: i32) -> Option<i32> {
        self.get(0.max(-euclid_div(self.start - i, self.step)) as _)
    }

    pub fn floor(&self, i: i32) -> Option<i32> {
        let pos = (self.len - 1)
            .min(euclid_div(i - self.start, self.step));
        if pos < 0 {
            None
        } else {
            self.get(pos as _)
        }
    }

    /// Expand the range so as to include the given integer.
    pub fn insert(&mut self, i: i32) {
        if self.len == 0 {
            self.start = i;
            self.len = 1;
        } else if self.len == 1 {
            if self.start != i {
                let (_, a, b) = parity::sort2(self.start, i);
                self.start = a;
                self.step = b - a;
                self.len = 2;
            }
        } else {
            let offset = i - self.start;
            let new_step = gcd(offset, self.step);
            let (_, k1, _, k3) = parity::sort3(
                0,
                offset,
                (self.len - 1) * self.step,
            );
            self.start += k1;
            self.step = new_step;
            self.len = (k3 - k1) / new_step + 1;
        }
    }
}

impl IntoIterator for RangeSet {
    type Item = i32;
    type IntoIter = num::iter::RangeStep<i32>;
    fn into_iter(self) -> Self::IntoIter {
        num::range_step(
            self.start,
            self.start + self.len * self.step,
            self.step,
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Toler {
    pub relerr: f64,
    pub abserr: f64,
}

/// Default is `(1e-15, 1e-15)`.
impl Default for Toler {
    fn default() -> Self {
        Self {
            relerr: 1e-15,
            abserr: 1e-15,
        }
    }
}

impl Toler {
    pub fn check(self, error: f64, value: f64) -> bool {
        error.is_finite() && // avoid "inf <= inf"
        error.abs() <= value.abs() * self.relerr + self.abserr
    }

    pub fn is_eq(self, value1: f64, value2: f64) -> bool {
        self.check(value1 - value2, 0.5 * (value1.abs() + value2.abs()))
    }
}

/// Zigzag integer encoding, which maps signed integers to unsigned integers.
///
/// ```text
///  0 → 0
/// -1 → 1
///  1 → 2
/// -2 → 3
///  2 → 4
///  etc…
/// `text
///
/// The name "zigzag" originates from the Google Protocol Buffers library,
/// which uses this encoding to store variable-width integers.
///
/// The ordering is defined by the encoded value, not the original value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Zigzag<T>(pub T);

impl From<i32> for Zigzag<u32> {
    fn from(i: i32) -> Self {
        Zigzag(((i << 1) ^ (i >> 31)) as _)
    }
}

impl From<Zigzag<u32>> for i32 {
    fn from(Zigzag(i): Zigzag<u32>) -> Self {
        (i >> 1) as i32 ^ -((i & 1) as i32)
    }
}

pub fn encode_number<T: Clone>(mut n: usize, chart: &HashChart<T>) -> Vec<T> {
    let mut s = Vec::default();
    loop {
        let r = n % chart.len();
        if r == 0 {
            break;
        }
        n /= chart.len();
        s.push(chart.decode(r).unwrap().clone());
    }
    s.reverse();
    if s.is_empty() {
        s.push(chart.decode(0).unwrap().clone());
    }
    s
}

pub fn decode_number<T: Hash + Eq>(
    s: &mut Iterator<Item = T>,
    chart: &HashChart<T>,
) -> Result<usize, &'static str> {
    let mut n = 0;
    for c in s {
        n *= chart.len();
        n += *chart.encode(&c).ok_or("character not in alphabet")?;
    }
    Ok(n)
}

// Remove this when feature(from_ref) stabilizes
pub mod slice {
    use std::slice;
    pub fn from_ref<T>(s: &T) -> &[T] {
        unsafe { slice::from_raw_parts(s, 1) }
    }
    pub fn from_ref_mut<T>(s: &mut T) -> &mut [T] {
        unsafe { slice::from_raw_parts_mut(s, 1) }
    }
}

/// Prevent unwinding in foreign code (which is undefined behavior).
pub fn abort_on_unwind<F, R>(f: F) -> R where
    F: FnOnce() -> R,
{
    panic::catch_unwind(panic::AssertUnwindSafe(f))
        .unwrap_or_else(|_| process::abort())
}

#[cfg(test)]
mod tests {
    use std::collections::hash_map;
    use super::*;

    #[test]
    fn hash_map() {
        let mut m: HashMap<_, _, hash_map::RandomState> = default_hash_map();
        m.insert(42, ":3");
        assert_eq!(m.get(&42), Some(&":3"));
        assert_eq!(m.get(&0), None);
    }

    #[test]
    fn test_euclid_div_mod() {
        assert_eq!(euclid_div(10, 5), 10 / 5);
        assert_eq!(euclid_mod(10, 5), 10 % 5);
        assert_eq!(euclid_div(-10, 5), -10 / 5);
        assert_eq!(euclid_mod(-10, 5), -10 % 5);
        assert_eq!(euclid_div(10, -5), 10 / -5);
        assert_eq!(euclid_mod(10, -5), 10 % -5);
        assert_eq!(euclid_div(-10, -5), -10 / -5);
        assert_eq!(euclid_mod(-10, -5), -10 % -5);
        assert_eq!(euclid_div(10, 7), 10 / 7);
        assert_eq!(euclid_mod(10, 7), 10 % 7);
        assert_eq!(euclid_div(-10, 7), -2);
        assert_eq!(euclid_mod(-10, 7), 4);
        assert_eq!(euclid_div(10, -7), 10 / -7);
        assert_eq!(euclid_mod(10, -7), 10 % -7);
        assert_eq!(euclid_div(-10, -7), 2);
        assert_eq!(euclid_mod(-10, -7), 4);
    }

    #[test]
    fn zigzag() {
        for i in -100 .. 101 {
            assert_eq!(i32::from(Zigzag::from(i)), i);
        }
        for &i in &[-0x80000000, -0x7fffffff, -0x7ffffffe,
                    0x7ffffffe, 0x7fffffff] {
            assert_eq!(i32::from(Zigzag::from(i)), i);
        }
    }

    #[test]
    fn test_gcd() {
        assert_eq!(gcd(0, 20), 20);
        assert_eq!(gcd(20, 0), 20);
        assert_eq!(gcd(6, 3), 3);
        assert_eq!(gcd(3, 6), 3);
        assert_eq!(gcd(1, 1), 1);
        assert_eq!(gcd(1, -1), 1);
        assert_eq!(gcd(-1, 1), 1);
        assert_eq!(gcd(147, 462), 21);
    }

    #[test]
    fn test_range_set() {
        use std::iter::FromIterator;
        for &s in &[
            &[78, 27, -70, -66, 92, -90, 95, -71, -76, 72, -47, 8, 13, 37, -46, -40],
            &[-58, 61, 62, -10, 53, -28, -35, 21, 94, 28, -84, -93, 97, -53, 81, 41],
            &[12, -93, 52, -30, -2, 73, 83, 70, -15, 34, 69, -49, 74, 42, -50, 62],
            &[22, 39, -20, 72, 30, -68, 26, -1, -70, 65, 47, -82, -82, -79, -60, 44],
        ] {
            let mut past = Vec::new();
            let mut set = RangeSet::default();
            assert_eq!(Vec::from_iter(set).len(), 0);
            for &x in s {
                set.insert(x);
                past.push(x);
                let v = Vec::from_iter(set);
                for y in &past {
                    assert_eq!(v.iter().cloned().min(),
                               past.iter().cloned().min());
                    assert_eq!(v.iter().cloned().max(),
                               past.iter().cloned().max());
                    assert!(v.contains(&y));
                }
            }
        }
    }
}
