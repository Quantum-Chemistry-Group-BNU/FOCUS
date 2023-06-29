#pragma once
#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>

#include "core/integral.h"

using namespace std;
namespace py = pybind11;

// void foo(int &i, int &j) { i++; j++;}
void PyIntegral(py::module &m);