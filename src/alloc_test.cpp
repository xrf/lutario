#undef NDEBUG
#include <assert.h>
#include "alloc.hpp"

static const size_t N = 42;

class MockAllocReq : public GenericAllocReq<int> {

    size_t _size;

    int *&_ptr;

public:

    MockAllocReq(size_t size, int *&ptr)
        : _size(size)
        , _ptr(ptr)
    {
    }

    size_t size() const override
    {
        return this->_size;
    }

    void fulfill(int *ptr) override
    {
        assert(ptr != nullptr);
        this->_ptr = ptr;
        for (int i = 0; i < (int)this->size(); ++i) {
            ptr[i] = i;
        }
    }

};

int main()
{

    {
        int *p;
        std::unique_ptr<int[]> u = alloc(MockAllocReq(N, p));
        assert(p == u.get());
    }

    {
        int *p0, *p1;
        AllocReqBatch<int> reqs;
        reqs.push(MockAllocReq(N, p0));
        reqs.push(MockAllocReq(N, p1));
        assert(reqs.size() == N * 2);
        std::unique_ptr<int[]> u = alloc(std::move(reqs));
        assert(p0 == u.get());
        assert(p1 == u.get() + N);
        for (int i = 0; i < (int)N; ++i) {
            assert(p0[i] == i);
        }
        for (int i = 0; i < (int)N; ++i) {
            assert(p1[i] == i);
        }
        assert(p0 == u.get());
    }

    return 0;
}
