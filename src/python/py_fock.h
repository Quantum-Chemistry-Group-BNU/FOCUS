#pragma once
#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>


#include "core/hamiltonian.h"
#include "core/onstate.h"

using namespace std;
using namespace fock;
namespace py = pybind11;

fock::onspace convert_numpy_space(py::array_t<uint8_t> bra, const int sorb);

void PyFock(py::module &m);