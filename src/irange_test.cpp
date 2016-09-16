#include <iostream>
#include "irange.hpp"

int main(void)
{
    for (int i : IRange<int>(0, 10)) {
        std::cout << i << std::endl;
    }
}
