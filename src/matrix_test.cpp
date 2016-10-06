#include <iostream>
#include "alloc.hpp"
#include "matrix.hpp"

int main()
{
    {
        Matrix<double> mat;
        std::unique_ptr<double[]> buf = alloc(mat.alloc_req(100, 100));
        assert(mat.data());
    }
    {
        Matrix<double> mat1, mat2;
        AllocReqBatch<double> reqs;
        reqs.push(mat1.alloc_req(100, 100));
        reqs.push(mat2.alloc_req(100, 100));
        assert(!mat1.data());
        assert(!mat2.data());
        std::unique_ptr<double[]> buf = alloc(std::move(reqs));
        assert(mat1.data());
        assert(mat2.data());
    }
}
