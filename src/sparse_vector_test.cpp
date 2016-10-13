#include "sparse_vector.hpp"

int main()
{
    SparseVector<int, int> c1({{1, -1}});
    SparseVector<int, int> c2({{1, 1}});
    SparseVector<int, int> c3({{0, 1}});
    c1 + c2;
    c1 - c2;
    c1 + c2;
    c1 - c2;
    c1 + c2;
    c1 - c2;
    c1 + c2;
    c1 - c2;
    c1 + c3;
    c1 - c3;
    SparseVector<int, int> c4 = std::move(c3);
    abort();
    c1 + c4;
    c1 - c4;
}
