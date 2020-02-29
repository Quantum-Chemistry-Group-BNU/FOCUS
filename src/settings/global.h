#ifndef GLOBAL_H
#define GLOBAL_H

#include <string>

namespace global{

const std::string line_separator(70,'-');
extern const std::string line_separator;

void license();

double mem_size(size_t sz, const int fac=8);

extern int print_level; // just declaration
   
}

#endif
