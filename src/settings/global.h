#ifndef GLOBAL_H
#define GLOBAL_H

#include <string>
#include <iostream>
#include <Eigen/Core>

namespace global{

void license();

const std::string line_separator(30,'-');
extern const std::string line_separator;

#ifdef Complex
using DType = std::complex<double>;
using MType = Eigen::MatrixXcd; 
#else
using DType = double;
using MType = Eigen::MatrixXd;
#endif

extern double memSize(size_t sz, const int fac=8);

}

#endif
