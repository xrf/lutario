#ifndef ALLOC_HPP
#define ALLOC_HPP
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

/// Base class that defines the interface for all allocation requests.
///
/// Allocation requests are used to request memory from an allocator.  They
/// are consumed by `alloc` and `BatchAllocator`.
///
template<typename T>
class GenericAllocReq {

public:

    virtual ~GenericAllocReq()
    {
    }

    /// The amount of memory that is requested.
    virtual size_t size() const = 0;

    /// An arbitrary callback function that is called once the request is
    /// fulfilled.
    virtual void fulfill(T *) = 0;

};

/// Fulfill an allocation request directly via `new`.
template<typename T>
std::unique_ptr<T[]> alloc(GenericAllocReq<T> &&req)
{
    std::unique_ptr<T[]> p(new T[req.size()]());
    req.fulfill(p.get());
    return p;
}

/// Allows multiple sub-objects to be allocated in a single contiguous block
/// of memory.
///
/// This is a two-stage process.  Firstly, sub-objects register their interest
/// in the block of memory via the `reserve` member function.  At this point,
/// no memory has been allocated.  Only the amount of memory that is needed is
/// tallied.  Then, all of the memory that has been request will be allocated
/// at once using `alloc`.  The memory is apportioned in the same order in
/// which the `reserve` calls were made (FIFO), triggering the callbacks
/// registered earlier via `reserve`.
///
/// Here is a schematic example that allocates two `MyString` sub-objects in a
/// single block of memory owned by a `unique_ptr`:
///
///     MyString x, y;
///     AllocReqBatch a;
///     a.emplace_back(x.alloc_req("hello"));
///     a.emplace_back(y.alloc_req("world"));
///     // until now, neither x and y have been allocated
///     std::unique_ptr<char> buf = alloc(a);
///     // now, both x and y have been allocated
///     // the lifetime of x and y is tied to that of buf
///
template<typename T>
class AllocReqBatch : public GenericAllocReq<T> {

    size_t _size;

    std::vector<std::unique_ptr<GenericAllocReq<T>>> _reqs;

public:

    AllocReqBatch(std::vector<std::unique_ptr<GenericAllocReq<T>>> reqs = {})
        : _size()
        , _reqs(std::move(reqs))
    {
        for (const std::unique_ptr<GenericAllocReq<T>> &req : this->_reqs) {
            this->_size += req->size();
        }
    }

    /// Return the number of elements that needs to be allocated.
    size_t size() const override
    {
        return this->_size;
    }

    /// Adds another sub-object that is to be allocated within the memory
    /// block.  The `R` type must be derived from `GenericAllocReq`.
    template<typename R>
    void emplace_back(R req)
    {
        this->_reqs.emplace_back(new auto(std::move(req)));
    }

    /// Use the pointer given to allocate all of the sub-objects.
    void fulfill(T *data) override
    {
        for (const std::unique_ptr<GenericAllocReq<T>> &req : this->_reqs) {
            size_t s = req->size();
            req->fulfill(data);
            data += s;
        }
        *this = AllocReqBatch();
    }

};

#endif
