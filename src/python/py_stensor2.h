#pragma once
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "qtensor/stensor2.h"

namespace py = pybind11;
void PyStensor2(py::module &m);