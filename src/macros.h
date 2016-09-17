// a header filled with macro hackery :)

/// Concatenate two tokens, but allow the tokens to expand first.
#define CAT(x, y) CAT_(x, y)
#define CAT_(x, y) x##y

/// Force macro expansion to occur.  A terrible workaround for broken
/// compilers like MSVC that don't handle `__VA_ARGS__` correctly.
/// http://stackoverflow.com/q/5134523
#define EXPAND(x) x

/// Obtain the length of a variadic macro argument list, up to a maximum of 9
/// arguments.
#define VA_SIZE(...)                                                 \
    EXPAND(VA_SIZE_(__VA_ARGS__, 9, 8, 7, 6, 5, 4, 3, 2, 1))
#define VA_SIZE_(x1, x2, x3, x4, x5, x6, x7, x8, x9, n, ...) n

/// Dispatch a variadic macro call to a macro whose name is the given prefix
/// followed by the number of arguments.  For example, if there are three
/// arguments, and `prefix` is `FOO`, it will call `FOO3`.
#define VA_CALL(prefix, ...)                                         \
    EXPAND(CAT(prefix, VA_SIZE(__VA_ARGS__))(__VA_ARGS__))

// TEST MACRO DO NOT USE YET
// #define CHANNEL(...) VA_CALL(CHANNEL_, __VA_ARGS__)
// #define CHANNEL_2(x, y) x == y
