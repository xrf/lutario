#ifndef UTILITY_HPP
#define UTILITY_HPP
#include <memory>
#include <ostream>
#include <tuple>
#include <vector>

/// A function object that closes a `FILE *`.
struct FileDeleter
{
    void operator()(FILE *stream) const;
};

/// Represents a `FILE *` with a deleter attached.
typedef std::unique_ptr<FILE, FileDeleter> File;

template<typename P, typename C>
void write_basis(std::ostream &stream,
                 const std::vector<std::tuple<P, C, bool>> &self)
{
    stream << "[";
    bool first = true;
    for (const std::tuple<P, C, bool> &pcx : self) {
        if (first) {
            first = false;
        } else {
            stream << ", ";
        }
        stream << "{\"orbital\": " << std::get<0>(pcx)
               << ", \"channel\": " << std::get<1>(pcx)
               << ", \"excited\": " << std::get<2>(pcx)
               << "}";
    }
    stream << "]";
}

#endif
