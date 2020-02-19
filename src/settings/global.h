#ifndef GLOBAL_H
#define GLOBAL_H
#include <string>

namespace global{

void license();

const std::string line_separator(70,'-');
extern const std::string line_separator;

extern double mem_size(size_t sz, const int fac=8);

}

#endif
