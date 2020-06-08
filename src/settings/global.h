#ifndef GLOBAL_H
#define GLOBAL_H

#include <string>
#include <chrono>
#include <iomanip>

namespace global{

// global variables
const std::string line_separator(70,'-');
extern const std::string line_separator;

const std::string line_separator2(70,'=');
extern const std::string line_separator2;

extern int print_level; // just declaration

// useful functions
void license();

double mem_size(size_t sz, const int fac=8);

// timing
std::chrono::high_resolution_clock::time_point get_time();

template<typename T>
double get_duration(T t){
   return std::chrono::duration_cast<std::chrono::milliseconds>(t).count()*0.001; 
}

} // global

#endif
