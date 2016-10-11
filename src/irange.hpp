#ifndef IRANGE_HPP
#define IRANGE_HPP
#include <iterator>
#include <type_traits>
#include <utility>

/// An `RandomAccessIterator` that stores an integer value of type `T`.  The
/// dereferenced value of the iterator is the integer itself.
///
/// @tparam T
/// An integer type to be wrapped.  It must support an associated signed type,
/// obtained via `std::make_signed<T>::type`.
///
template<typename T>
struct IntegerIterator {

    /// Value type.
    typedef T value_type;

    /// Difference_type.
    typedef typename std::make_signed<T>::type difference_type;

    /// Pointer type.
    typedef const T *pointer;

    /// Reference type.
    typedef const T &reference;

    /// Iterator category.
    typedef std::random_access_iterator_tag iterator_category;

    /// Value represented by the iterator.
    T value;

    /// Constructs an iterator with the given value.
    explicit IntegerIterator(T value = T())
        : value(std::move(value))
    {
    }

    /// Returns the integer value of the iterator.
    const T &operator*() const
    {
        return this->value;
    }

    /// Member access of the object pointed to by the iterator.
    const T *operator->() const
    {
        return &*this;
    }

    /// Returns an iterator with value equal to the current value plus an
    /// integer.
    const T &operator[](const difference_type &offset) const
    {
        return *(*this + offset);
    }

    /// Compares the value of the iterators.
    bool operator==(const IntegerIterator &other) const
    {
        return **this == *other;
    }

    /// Compares the value of the iterators.
    bool operator!=(const IntegerIterator &other) const
    {
        return **this != *other;
    }

    /// Compares the value of the iterators.
    bool operator<=(const IntegerIterator &other) const
    {
        return **this <= *other;
    }

    /// Compares the value of the iterators.
    bool operator>=(const IntegerIterator &other) const
    {
        return **this >= *other;
    }

    /// Compares the value of the iterators.
    bool operator<(const IntegerIterator &other) const
    {
        return **this < *other;
    }

    /// Compares the value of the iterators.
    bool operator>(const IntegerIterator &other) const
    {
        return **this > *other;
    }

    /// Pre-increments the value of the iterator.
    IntegerIterator &operator++()
    {
        return *this += 1;
    }

    /// Post-increments the value of the iterator.
    IntegerIterator operator++(int)
    {
        const IntegerIterator old = *this;
        ++*this;
        return old;
    }

    /// Increases the value of the iterator.
    IntegerIterator &operator+=(const difference_type &offset)
    {
        this->value = static_cast<T>(static_cast<difference_type>(**this) + offset);
        return *this;
    }

    /// Pre-decrements the value of the iterator.
    IntegerIterator &operator--()
    {
        return *this -= 1;
    }

    /// Post-decrements the value of the iterator.
    IntegerIterator operator--(int)
    {
        const IntegerIterator old = *this;
        --*this;
        return old;
    }

    /// Decreases the value of the iterator.
    IntegerIterator &operator-=(const difference_type &offset)
    {
        this->value = static_cast<T>(static_cast<difference_type>(**this) - offset);
        return *this;
    }

    /// Adds an integer to the value of the iterator.
    IntegerIterator operator+(const difference_type &offset) const
    {
        return IntegerIterator(*this) += offset;
    }

    /// Subtracts an integer from the value of the iterator.
    IntegerIterator operator-(const difference_type &offset) const
    {
        return IntegerIterator(*this) -= offset;
    }

    /// Subtracts the values of the two iterators.
    difference_type operator-(const IntegerIterator &other) const
    {
        return static_cast<difference_type>(**this) - static_cast<difference_type>(*other);
    }

};

/// Stores a pair of integers denoting some range that can be iterated over,
/// compatible with range-based `for` loops.
///
/// Here's an example that prints the numbers from 0 to 10 (but not including
/// 10 itself):
///
///     for (int i : IRange<int>(0, 10)) {
///         std::cout << i << std::endl;
///     }
///
template<typename T>
struct IRange {

    /// The iteration starts at this value.
    T start;

    /// The iteration stops just before reaching this value.
    T stop;

    /// Constructor.
    IRange(T start, T stop)
        : start(std::move(start))
        , stop(std::move(stop))
    {
    }

    /// Returns the beginning iterator.
    IntegerIterator<T> begin() const
    {
        return IntegerIterator<T>(this->start);
    }

    /// Span of the range.
    size_t size() const
    {
        return this->stop - this->start;
    }

    /// Returns the end iterator.
    IntegerIterator<T> end() const
    {
        return IntegerIterator<T>(this->stop);
    }

};

typedef IRange<size_t> IndexRange;

#endif
