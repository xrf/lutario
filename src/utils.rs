use std::fmt;
use conv::ValueInto;
use take_mut;

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

/// Temporary definition until `std::ops::RangeInclusive` is stabilized.
#[derive(Clone, Debug)]
pub struct RangeInclusive<Idx> {
    pub start: Idx,
    pub end: Idx,
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
