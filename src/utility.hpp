#ifndef UTILITY_HPP
#define UTILITY_HPP

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
