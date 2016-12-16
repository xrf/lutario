#include <stdio.h>
#include "utility.hpp"

void FileDeleter::operator()(FILE *stream) const
{
    if (stream != NULL) {
        fclose(stream);
    }
}
