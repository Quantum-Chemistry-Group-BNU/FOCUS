#ifndef GLOBAL_HEADER_H
#define GLOBAL_HEADER_H
#include <string>
#include <iostream>

const std::string line_separator = "***************************************************************";
extern const std::string line_separator;

#ifdef Complex
#define DType std::complex<double>
#else
#define DType double
#endif

//int commrank = 0; // will be changed later
//extern int commrank;

//#define pout if (commrank == 0) std::cout
#define pout std::cout

#endif
