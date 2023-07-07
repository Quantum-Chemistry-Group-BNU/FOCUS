#include "py_stensor2.h"

#include <vector>

#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "qtensor/stensor2.h"

void PyStensor2(py::module &m) {
  py::class_<ctns::stensor2<double>>(m, "Stensor2")
      .def(py::init())
      .def("rows", &ctns::stensor2<double>::rows)
      .def("cols", &ctns::stensor2<double>::cols)
      // .def("data", &ctns::stensor2<double>::data)
      .def("data",
           [](ctns::stensor2<double> &self) {
             auto data_ptr = self.data();
             auto count = self.size();
             return py::array(count, data_ptr);
           })
      .def("size", &ctns::stensor2<double>::size)
      .def("info", [](ctns::stensor2<double> &self) { return self.info; })
      //.def("info", &ctns::stensor2<double>::get_info)
      .def("shape", &ctns::stensor2<double>::get_shape)

      ;
}
