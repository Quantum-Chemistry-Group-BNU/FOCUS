#pragma once
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "qtensor/qnum_qbond.h"

namespace py = pybind11;
void PyQbond(py::module &m);