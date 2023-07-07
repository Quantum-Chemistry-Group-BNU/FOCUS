#ifdef PYTHON_BINDING

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
  auto m_qtensor = m.def_submodule("qtensor", "qtensor module");
  PyStensor2(m_qtensor);
  PyQinfo2(m_qtensor);
  PyQbond(m_qtensor);
  PyQsym(m_qtensor);
}

#endif
