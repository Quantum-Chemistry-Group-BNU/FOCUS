#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <complex>

#include "../core/integral.h"

using namespace std;
namespace py = pybind11;

PYBIND11_MODULE(py_integral, m) {
  m.doc() = "This is integral";
  // m.def("load", &integral::load<double>, "Load the integral information from
  // the *.info"); m.def("load", &integral::load<complex<double>>, "Load the
  // integral information from the *.info"); Using a small wrapper lambda
  // function, return tuple with all output argument
  m.def(
      "load",
      [](integral::two_body<double> int2e, integral::one_body<double> int1e,
         double ecore, const std::string s) {
        integral::load(int2e, int1e, ecore, s);
        return std::make_tuple(int2e, int1e, ecore);
      },
      "Load the integral information from the *.info");
  m.def(
      "load",
      [](integral::two_body<complex<double>> int2e,
         integral::one_body<complex<double>> int1e, double ecore,
         const std::string s) {
        integral::load(int2e, int1e, ecore, s);
        return std::make_tuple(int2e, int1e, ecore);
      },
      "Load the integral information from the *.info");
  // m.def("foo", [](int i, int j) {foo(i,j); return std::make_tuple(i, j); },
  // "Load the integral information from the *.info");
  py::class_<integral::one_body<double>>(m, "one_body")
      .def(py::init())
      .def("size", &integral::one_body<double>::size)
      .def("get", &integral::one_body<double>::get)
      .def("set", &integral::one_body<double>::set)
      .def("set_real", &integral::one_body<double>::set_real)
      .def("set_zero", &integral::one_body<double>::set_zero)
      .def("print", &integral::one_body<double>::print)
      .def_property_readonly(
          "data", []() { return &integral::one_body<double>::data; }, "<i|Q|j>")
      .def_property_readonly(
          "sorb", []() { return &integral::one_body<double>::sorb; },
          "The number of spin orbital");
  // .def_property_readonly("data", &integral::one_body<double>::_data,
  // "<i|Q|j>"T .def_property_readonly("sorb",
  // &integral::one_body<double>::_sorb, "The number of spin orbital");

  py::class_<integral::one_body<complex<double>>>(m, "one_body_c")
      .def(py::init())
      .def("size", &integral::one_body<complex<double>>::size)
      .def("get", &integral::one_body<complex<double>>::get)
      .def("set", &integral::one_body<complex<double>>::set)
      .def("set_real", &integral::one_body<complex<double>>::set_real)
      .def("set_zero", &integral::one_body<complex<double>>::set_zero)
      .def("print", &integral::one_body<complex<double>>::print)
      .def_property_readonly(
          "data", []() { return &integral::one_body<complex<double>>::data; },
          "<i|Q|j>")
      .def_property_readonly(
          "sorb", []() { return &integral::one_body<complex<double>>::sorb; },
          "The number of spin orbital");

  py::class_<integral::two_body<double>>(m, "two_body")
      .def(py::init())
      .def("size", &integral::two_body<double>::size)
      .def("get", &integral::two_body<double>::get)
      .def("set_real", &integral::two_body<double>::set_real)
      .def("set_zero", &integral::two_body<double>::set_zero)
      .def("get_Q", &integral::two_body<double>::getQ)
      .def("print", &integral::two_body<double>::print)
      .def_property_readonly(
          "sorb", []() { return &integral::two_body<double>::sorb; },
          "The number of spin orbital")
      .def_property_readonly(
          "data", []() { return &integral::two_body<double>::data; },
          "<ij||kl>")
      .def_property_readonly(
          "Q", []() { return &integral::two_body<double>::Q; },
          "Q_ij=<ij||ij>");

  py::class_<integral::two_body<complex<double>>>(m, "two_body_c")
      .def(py::init())
      .def("size", &integral::two_body<complex<double>>::size)
      .def("get", &integral::two_body<complex<double>>::get)
      .def("size", &integral::two_body<complex<double>>::size)
      .def("set_real", &integral::two_body<complex<double>>::set_real)
      .def("set_zero", &integral::two_body<complex<double>>::set_zero)
      .def("get_Q", &integral::two_body<complex<double>>::getQ)
      .def("print", &integral::two_body<complex<double>>::print)
      .def_property_readonly(
          "sorb", []() { return &integral::two_body<complex<double>>::sorb; },
          "The number of spin orbital")
      .def_property_readonly(
          "data", []() { return &integral::two_body<complex<double>>::data; },
          "<ij||kl>")
      .def_property_readonly(
          "Q", []() { return &integral::two_body<complex<double>>::Q; },
          "Q_ij=<ij||ij>");
}