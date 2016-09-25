#undef NDEBUG
#include <assert.h>
#include "indexed_set.hpp"

int main()
{
    IndexedSet<int> s;
    size_t i;

    i = 999;
    assert(s.insert(320, &i) == true);
    assert(i == 0);
    assert(s.size() == 1);

    i = 999;
    assert(s.insert(320, &i) == false);
    assert(i == 0);
    assert(s.size() == 1);

    i = 999;
    assert(s.insert(250, &i) == true);
    assert(i == 1);
    assert(s.size() == 2);

    assert(s.insert(100, nullptr) == true);
    assert(s.size() == 3);

    assert(s.insert(250, nullptr) == false);
    assert(s.size() == 3);

    i = 999;
    assert(s.find(1000, &i) == false);
    assert(i == 999);

    i = 999;
    assert(s.find(100, &i) == true);
    assert(i == 2);

    assert(s[0] == 320);
    assert(s[1] == 250);
    assert(s[2] == 100);

    return 0;
}
