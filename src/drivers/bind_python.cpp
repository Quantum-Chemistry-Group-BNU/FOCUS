#include <iomanip>
#include <iostream>
#include <string>

#include "ci/ci_header.h"
#include "ctns/ctns_header.h"
#include "io/input.h"
#include "io/io.h"
#include "post/post_header.h"
#include "python/py_header.h"
#include "vmc/vmc_header.h"

PYBIND11_MODULE(qubic, m) {
  m.doc() = "This code module";
  auto m_fock = m.def_submodule("fock", "fock module");
  PyFock(m_fock);
  auto m_integral = m.def_submodule("integral", "integral module");
  PyIntegral(m_integral);
  auto m_mps = m.def_submodule("post", "ctns post mps module");
  PyMps(m_mps);
}