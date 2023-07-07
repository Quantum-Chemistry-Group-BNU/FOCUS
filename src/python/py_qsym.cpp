#include "py_qsym.h"

#include <tuple>

#include "qtensor/qnum_qsym.h"

void PyQsym(py::module &m) {
  py::class_<ctns::qsym>(m, "Qsym")
      .def(py::init())
      .def(py::init<const short>())
      .def(py::init<const short, const short>())
      .def(py::init<const short, const short, const short>())
      .def("is_zero", &ctns::qsym::is_zero)
      .def("is_nonzero", &ctns::qsym::is_nonzero)
      .def("isym", &ctns::qsym::isym)
      .def("ne", &ctns::qsym::ne, "na + nb")
      .def("tm", &ctns::qsym::tm, "na - nb")
      .def(
          "data",
          [](ctns::qsym &self) {
            auto isym = self.isym();
            auto ne = self.ne();
            auto tm = self.tm();
            return std::make_tuple(isym, ne, tm);
          },
          "(isym, ne, tm)");
}