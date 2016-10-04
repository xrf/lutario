#include <iostream>
#include "alloc.hpp"
#include "matrix.hpp"

int main()
{
    {
        Matrix<double> mat;
        std::unique_ptr<double[]> buf = alloc(mat.alloc_req(100, 100), &mat);
        assert(mat.data());
    }
    {
        Matrix<double> mat2;
        CompactArena<double> arena;
        arena.async_alloc(mat2.alloc_req(100, 100),
                          [&](Matrix<double> mat) { mat2 = mat; });
        assert(!mat2.data());
        arena.reify();
        assert(mat2.data());
    }
}
