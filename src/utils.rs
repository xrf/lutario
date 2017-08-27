use conv::ValueInto;

/// Shorthand for casting numbers.  Panics if out of range.
pub fn cast<T: ValueInto<U>, U>(x: T) -> U {
    x.value_into().expect("integer conversion failure")
}
