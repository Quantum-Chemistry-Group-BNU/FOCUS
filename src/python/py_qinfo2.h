#pragma once
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "qtensor/qinfo2.h"

namespace py = pybind11;
void PyQinfo2(py::module &m);
