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

    void fulfill(int *ptr) const override
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
        Stage<int> stage;
        stage.prepare(MockAllocReq(N, p0));
        stage.prepare(MockAllocReq(N, p1));
        stage.execute();
        assert(stage.size() == N * 2);
        assert(p0 == stage.data());
        assert(p1 == stage.data() + N);
        for (int i = 0; i < (int)N; ++i) {
            assert(p0[i] == i);
        }
        for (int i = 0; i < (int)N; ++i) {
            assert(p1[i] == i);
        }
        std::unique_ptr<int[]> u = stage.release();
        assert(p0 == u.get());
    }

    return 0;
}
