#include "py_qinfo2.h"

#include <pybind11/pybind11.h>

#include <string>

#include "py_utils.h"
#include "qtensor/qinfo2.h"

void PyQinfo2(py::module &m) {
  py::class_<ctns::qinfo2<double>>(m, "Qinfo2")
      .def(py::init<>())
      .def("print", &ctns::qinfo2<double>::print)
      //.def("sym1", &ctns::qinfo2<double>::sym) // error, why struct?
      .def_readonly("sym", &ctns::qinfo2<double>::sym, "Qsym class")
      .def_readonly("qrow", &ctns::qinfo2<double>::qrow,
                    "Qbond class, list[(N, Sz, dim)]")  // vector({N,Sz, dim})
      .def_readonly("qcol", &ctns::qinfo2<double>::qcol,
                    "Qbond class, list[(N, Sz, dim)]")
      .def(
          "nnzaddr",
          [](ctns::qinfo2<double> &self) {
            return as_pyarray(std::move(self._nnzaddr));
          },
          "nnzaddr")
      .def_readonly("size", &ctns::qinfo2<double>::_size)
      .def("__str__", [](ctns::qinfo2<double> &self) {
        self.print("Qinfo2");
        return " ";
      });
}