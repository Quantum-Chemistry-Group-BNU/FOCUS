#include "py_qbond.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <tuple>
#include <vector>

#include "qtensor/qnum_qbond.h"

void PyQbond(py::module &m) {
  py::class_<ctns::qbond>(m, "Qbond")
      .def(py::init())
      .def(py::init<const std::vector<std::pair<ctns::qsym, int>> &>())
      .def("size", &ctns::qbond::size)
      .def("get_sym", &ctns::qbond::get_sym)
      .def("get_dim", &ctns::qbond::get_dim)
      .def("get_parity", &ctns::qbond::get_parity)
      .def("get_dimAll", &ctns::qbond::get_dimAll)
      .def("print", &ctns::qbond::print)
      .def("__str__", [](ctns::qbond &self) { self.print("Qbond"); return " "; })
      .def("__len__", &ctns::qbond::size)
      .def("__getitem__",
           [](ctns::qbond &self, const int i) {
             auto qsym_i = self.get_sym(i);
             short isym = qsym_i.isym();
             short ne = qsym_i.ne();
             short tm = qsym_i.tm();
             int dim = self.get_dim(i);
             auto qsym_tuple = std::make_tuple(isym, ne, tm);
             return std::make_tuple(qsym_tuple, dim);
             //  return std::vector<int>{isym, ne, tm, dim};
           })
      .def("data",
           [](ctns::qbond &self) {
             const int n = self.size();
             using qsym_tuple = std::tuple<short, short, short>;
             std::vector<qsym_tuple> result_qsym;
             std::vector<int> result_dim;
             for (int i = 0; i < n; i++) {
               auto qsym_i = self.get_sym(i);
               short isym = qsym_i.isym();
               short ne = qsym_i.ne();
               short tm = qsym_i.tm();
               result_qsym.push_back(std::make_tuple(isym, ne, tm));
               result_dim.push_back(self.get_dim(i));
             }
             return std::make_tuple(result_qsym, result_dim);
           })

      ;
}