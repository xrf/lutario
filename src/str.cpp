#include <ctype.h>
#include <string>

std::string trim(const std::string &s)
{
    size_t i = 0;
    while (i < s.size() && isspace(s[i])) {
        ++i;
    }

    size_t j = s.size();
    while (j-- > i && isspace(s[j])) {
    }

    return s.substr(i, j - i);
}
