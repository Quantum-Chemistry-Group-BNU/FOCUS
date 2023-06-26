#include <pybind11/pybind11.h>

#include "../post/post_header.h"
#include "../qtensor/qtensor.h"

namespace py = pybind11;
using namespace ctns;

PYBIND11_MODULE(py_mps, m) {
  m.doc() = "MPS class";
  m.def("CIcoeff", &mps_CIcoeff<qkind::rNSz>);
  m.def("mps_random", &mps_random<qkind::rNSz>);

  py::class_<mps<qkind::rNSz>>(m, "MPS")
      .def(py::init())
      .def("load", &mps<qkind::rNSz>::load, "load MPS file")
      .def("print", &mps<qkind::rNSz>::print)
      .def("get_pindex", &mps<qkind::rNSz>::get_pindex)

      .def_readonly("nphysical", &mps<qkind::rNSz>::nphysical);
}