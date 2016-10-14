#ifndef UTILITY_HPP
#define UTILITY_HPP

template<typename P, typename C>
std::ostream &write_basis(std::ostream &stream,
                          const std::vector<std::tuple<P, C, bool>> &self)
{
    stream << "({";
    bool first = true;
    for (const std::tuple<P, C, bool> &pcx : self) {
        if (first) {
            first = false;
        } else {
            stream << ", ";
        }
        stream << "{" << std::get<0>(pcx)
               << ", " << std::get<1>(pcx)
               << ", " << std::get<2>(pcx)
               << "}";
    }
    stream << "})";
    return stream;
}

#endif
