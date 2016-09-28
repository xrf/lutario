#ifndef ALLOCATION_HPP
#define ALLOCATION_HPP
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

template<typename T>
class GenericAllocReq {

public:

    virtual ~GenericAllocReq()
    {
    }

    virtual size_t size() const = 0;

    virtual void fulfill(T *) const = 0;

};

template<typename T>
std::unique_ptr<T[]> alloc(const GenericAllocReq<T> &req)
{
    std::unique_ptr<T[]> p(new T[req.size()]);
    req.fulfill(p.get());
    return p;
}

template<typename T>
class Stage {

    std::unique_ptr<T[]> _data;

    size_t _size = 0;

    std::vector<std::unique_ptr<GenericAllocReq<T>>> _requests;

public:

    size_t size() const
    {
        return this->_size;
    }

    const T *data() const
    {
        return this->_data.get();
    }

    T *data()
    {
        return this->_data.get();
    }

    template<typename R>
    void prepare(R req)
    {
        if (this->_data) {
            throw std::logic_error("Stage.prepare must no be called after "
                                   "Stage.execute has already occurred");
        }
        this->_size += req.size();
        this->_requests.emplace_back(new R(std::move(req)));
    }

    void execute()
    {
        if (this->_data) {
            throw std::logic_error("cannot execute more than once");
        }
        this->_data.reset(new T[this->size()]());
        T *ptr = this->_data.get();
        for (const auto &req : this->_requests) {
            req->fulfill(ptr);
            ptr += req->size();
        }
        this->_requests.clear();
    }

    std::unique_ptr<T[]> release()
    {
        std::unique_ptr<T[]> ptr = std::move(this->_data);
        *this = Stage();
        return ptr;
    }

};

#endif
