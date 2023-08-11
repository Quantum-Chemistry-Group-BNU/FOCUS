#pragma once
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sys/types.h>

#include <algorithm>
#include <cstdint>
#include <tuple>
#include <utility>
#include <vector>

#include "core/onspace.h"
#include "core/onstate.h"
#include "post/mps.h"
#include "post/post_header.h"
#include "py_utils.h"
#include "pybind11/cast.h"
#include "qtensor/qnum_qkind.h"
#include "qtensor/qtensor.h"

namespace py = pybind11;
using namespace ctns;
using onspace = std::vector<fock::onstate>;

template <typename Qm, typename Tm>
py::array_t<Tm> nbatch_mps_CIcoeff(const mps<Qm, Tm> &imps, const int iroot,
                                   const py::array_t<uint8_t> bra,
                                   const int sorb) {
  const int nbatch = bra.shape(0);  // [nbatch, sorb]
  const unsigned long *bra_ptr =
      reinterpret_cast<unsigned long *>(bra.request().ptr);
  onspace space = fock::convert_space(bra_ptr, sorb, nbatch);
  auto result = std::vector<Tm>(nbatch, 0);
  for (int i = 0; i < nbatch; i++) {
    result[i] = ctns::mps_CIcoeff<Qm, Tm>(imps, iroot, space[i]);
  }
  return as_pyarray(std::move(result));
}

template <typename Qm, typename Tm>
auto nbatch_mps_random(const mps<Qm, Tm> &imps, const int iroot,
                       const int nbatch, const int sorb,
                       const bool debug = false) {
  int _len = (sorb - 1) / 64 + 1;
  auto result_state = std::vector<unsigned long>(nbatch * _len, 0);
  auto result_coeff0 = std::vector<Tm>(nbatch, 0);
  for (int i = 0; i < nbatch; i++) {
    // c++17 new feature
    auto [state, coeff0] = ctns::mps_random<Qm, Tm>(imps, iroot, debug);
    std::copy_n(state.get_data(), _len,
                &result_state[i * _len]);  // sorb maybe great 64
    result_coeff0[i] = coeff0;
  }
  // vector to numpy array
  py::array_t<Tm> array_coeff0 = as_pyarray(std::move(result_coeff0));
  // unsigned long to uint8 -> memory must is contiguous
  py::array_t<uint8_t> array_state =
      as_pyarray(std::move(result_state)).reshape({nbatch, _len}).view("uint8");
  return std::make_tuple(array_state, array_coeff0);
}

inline auto read_file(const std::string filename) {
  std::ifstream fin(filename);
  if (!fin) {
    std::cout << "failed to open " << filename << "\n";
    exit(1);
  }
  std::vector<int> topology;
  std::string line;
  while (!fin.eof()) {
    line.clear();
    std::getline(fin, line);
    if (line.empty() || line[0] == '#') {
      continue;
    } else {
      int i = std::stoi(line);
      topology.push_back(2 * i);
      topology.push_back(2 * i + 1);
    }
  }
  fin.close();
  return topology;
}

void PyMps(py::module &m) {
  // TODO: how to simultaneously load mps and topology;
  m.doc() = "MPS class";
  // m.def("CIcoeff", &mps_CIcoeff<qkind::qNSz, double>);
  m.def("CIcoeff", &nbatch_mps_CIcoeff<qkind::qNSz, double>, py::arg("imps"),
        py::arg("iroot"), py::arg("bra"), py::arg("sorb"));
  m.def("mps_random", &nbatch_mps_random<qkind::qNSz, double>, py::arg("imps"),
        py::arg("iroot"), py::arg("nbatch"), py::arg("sorb"),
        py::arg("debug") = false);

  py::class_<mps<qkind::qNSz, double>>(m, "MPS")
      .def(py::init())
      .def("load", &mps<qkind::qNSz, double>::load, "load MPS file")
      .def("print", &mps<qkind::qNSz, double>::print)
      .def("get_pindex", &mps<qkind::qNSz, double>::get_pindex)
      .def("convert", &mps<qkind::qNSz, double>::convert,
           py::arg("debug") = false)
      .def_static(
          "load_topology",
          [](const std::string ftopo) {
            std::cout << "load topology file: " << ftopo << " " << std::endl;
            return read_file(ftopo);
          },
          "load topology file")

      .def_readwrite("nphysical", &mps<qkind::qNSz, double>::nphysical,
                     "space orbital")
      .def_readwrite("image2", &mps<qkind::qNSz, double>::image2, "topology");
}