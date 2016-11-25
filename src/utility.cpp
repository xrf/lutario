#include <stdio.h>
#include "utility.hpp"

void FileDeleter::operator()(FILE *stream) const
{
    fclose(stream);
}
