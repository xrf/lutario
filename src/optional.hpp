#ifndef OPTIONAL_HPP
#define OPTIONAL_HPP
#include <utility>

/// A minimal implementation of `std::optional` from C++17.  To keep the
/// implementation simple, we require `T` to be default-constructible.
///
/// An optional value can be thought of as a simple container that contains
/// either one value or no value at all, similar to `Option` in Rust or
/// `Maybe` in Haskell.  Optional types make it explicit that a given function
/// needs or returns a value that may or may not exist, reducing the chance of
/// programmer mistakes.
///
/// There are two ways to construct an optional value:
///
///   - `Optional()`: constructs an empty optional, which does not contain a value.
///   - `Optional(value)`: constructs an optional with the given value.
///
/// There are two ways to inspect an optional value:
///
///   - `(bool)opt`: casting an optional to `bool` returns whether the
///     optional contains a value.  Inside the condition of an `if`-statement,
///     an explicit cast is not necessary.
///   - `*opt`: dereferencing the optional retrieves the value of the
///     optional.  However, the optional must not be empty.
///
template<typename T>
class Optional {

    T _value;

    bool _has_value;

public:

    /// Construct an optional that does not contain a value.
    Optional()
        : _has_value(false)
    {
    }

    /// Construct an optional that contains the given value.
    Optional(T value)
        : _value(std::move(value))
        , _has_value(true)
    {
    }

    /// Check if a value is present.
    explicit operator bool() const
    {
        return this->_has_value;
    }

    /// Extract the value.  The value must exist.
    const T &operator*() const
    {
        assert(static_cast<bool>(*this));
        return this->_value;
    }

};

/// Try to get the value of an optional value and store it in `out`.  If `out`
/// is null, do nothing.  The return value is always that of `has_value()`.
///
/// This is a convenience function for inspecting optional values.
///
template<typename O, typename T>
bool try_get(const O &opt, T *out)
{
    bool has_value = static_cast<bool>(opt);
    if (has_value && out) {
        *out = *opt;
    }
    return has_value;
}

#endif
