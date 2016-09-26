#ifndef ALLOCATION_HPP
#define ALLOCATION_HPP
#include <functional>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

template<typename R>
std::unique_ptr<typename R::value_type[]>
alloc(R request, typename R::object_type *out)
{
    typedef typename R::value_type value_type;
    std::unique_ptr<value_type[]> ptr(new value_type[request.size()]);
    if (out) {
        *out = request.construct(ptr.get());
    }
    return ptr;
}

template<typename T>
class CompactArena {

    std::unique_ptr<T[]> _data;

    size_t _size = 0;

    std::vector<std::function<T * (T *)>> _callbacks;

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

    std::unique_ptr<T[]> release() && {
        std::unique_ptr<T[]> p;
        std::swap(p, this->_data);
        this->_size = 0;
        this->_callbacks.clear();
        //this->_callbacks.shrink_to_fit(); // TODO: put this back
        return p;
    }

    void reify()
    {
        if (this->_data) {
            throw std::logic_error("cannot reify more than once");
        }
        this->_data.reset(new T[this->size()]());
        T *ptr = this->_data.get();
        for (const std::function<T * (T *)> &callback : this->_callbacks) {
            ptr = callback(ptr);
        }
        this->_callbacks.clear();
        //this->_callbacks.shrink_to_fit(); // TODO: put this back
    }

    template<typename R>
    void async_alloc(R request,
                     std::function<void (typename R::object_type)> callback)
    {
        using std::placeholders::_1;
        typedef std::function<void (typename R::object_type)> callback_type;
        static_assert(std::is_same<typename R::value_type, T>::value,
                      "R::value_type and T must be the same type");
        if (this->_data) {
            throw std::logic_error("cannot allocate after arena "
                                   "has already been reified");
        }
        this->_size += request.size();
        this->_callbacks.emplace_back(std::bind(
            [](R request, callback_type callback, T *ptr)
            {
                callback(request.construct(ptr));
                return ptr + request.size();
            },
            std::move(request),
            std::move(callback),
            _1
        ));
    }

};

#endif
