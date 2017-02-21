#undef NDEBUG
#include <ios>
#include <fstream>
#include "irange.hpp"

int main()
{
    std::ofstream file{"out_irange_test.txt", std::ios_base::in};
    for (int i : IRange<int>(0, 10)) {
        file << i << std::endl;
    }
    file.flush();
}
