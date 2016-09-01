/*\(\b\w\.o\w\)\[\(\w*\)\](*/
/*b.get(\1, \2, */

template<typename T>
struct MatView {
    T *data;
    size_t stride;
    MatView(T *data, size_t stride)
        : data(data)
        , stride(stride)
    {
    }
    T &operator()(size_t i, size_t j) const
    {
        return data[i * stride + j];
    }
};
