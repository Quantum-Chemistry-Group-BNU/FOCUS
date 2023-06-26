#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../io/input.h"

namespace py = pybind11;
using namespace input;

PYBIND11_MODULE(schedule, m) {
  py::class_<schedule>(m, "schd")
      .def(py::init())
      .def("read", &input::schedule::read, "Read from input from file",
           py::arg("fname") = "input.data")  // default params
      .def("print", &input::schedule::print, "print params")

      // public member
      .def_readwrite("scratch", &schedule::scratch)
      .def_readwrite("nelec", &schedule::nelec)
      .def_readwrite("dtype", &schedule::dtype)
      .def_readwrite("twoms", &schedule::twoms)
      .def_readwrite("integral_file", &schedule::integral_file)
      .def_readwrite("sci", &schedule::sci)
      .def_readwrite("ctns", &schedule::ctns)
      .def_readwrite("post", &schedule::post)
      .def_readwrite("vmc", &schedule::vmc);

  py::class_<params_sci>(m, "sci")
      .def(py::init())
      .def("read", &params_sci::read)  // ifstream
      .def("print", &params_sci::print)

      // public member
      .def_readwrite("run", &params_sci::run)  // default public member
      .def_readwrite("nroot", &params_sci::nroots)
      .def_readwrite("det_seeds", &params_sci::det_seeds, "initial dets")
      .def_readwrite("nseeds", &params_sci::nseeds)
      .def_readwrite("flip", &params_sci::flip)
      .def_readwrite("eps0", &params_sci::eps0)
      .def_readwrite("eps1", &params_sci::eps1)
      .def_readwrite("eps2", &params_sci::eps2)
      .def_readwrite("miniter", &params_sci::miniter)
      .def_readwrite("maxiter", &params_sci::maxiter)
      .def_readwrite("deltaE", &params_sci::deltaE)
      .def_readwrite("cisolver", &params_sci::cisolver)
      .def_readwrite("maxcycle", &params_sci::maxcycle)
      .def_readwrite("crit_v", &params_sci::crit_v)
      .def_readwrite("ifpt2", &params_sci::ifpt2)
      .def_readwrite("iroot", &params_sci::iroot)
      .def_readwrite("load", &params_sci::load)
      .def_readwrite("ci_file", &params_sci::ci_file)
      .def_readwrite("cthrd", &params_sci::cthrd);

  py::class_<params_sweep>(m, "sweep")
      .def(py::init())
      .def("print", &params_sweep::print)

      // public member
      .def_readwrite("isweep", &params_sweep::isweep)
      .def_readwrite("dots", &params_sweep::dots)
      .def_readwrite("dcut", &params_sweep::dcut)
      .def_readwrite("eps", &params_sweep::eps)
      .def_readwrite("noise", &params_sweep::noise);

  py::class_<params_ctns>(m, "ctns")
      .def(py::init())
      .def("read", &params_ctns::read)  // ifstream
      .def("print", &params_sweep::print)

      // public member
      .def_readwrite("run", &params_ctns::run)
      .def_readwrite("run", &params_ctns::run)
      .def_readwrite("qkind", &params_ctns::qkind)
      .def_readwrite("topology_file", &params_ctns::topology_file)
      .def_readwrite("verbose", &params_ctns::verbose)
      .def_readwrite("task_init", &params_ctns::task_init)
      .def_readwrite("task_sdiag", &params_ctns::task_sdiag)
      .def_readwrite("task_ham", &params_ctns::task_ham)
      .def_readwrite("task_opt", &params_ctns::task_opt)
      .def_readwrite("task_vmc", &params_ctns::task_vmc)
      .def_readwrite("restart_sweep", &params_ctns::restart_sweep)
      .def_readwrite("restart_bond", &params_ctns::restart_bond)
      .def_readwrite("timestamp", &params_ctns::timestamp)
      .def_readwrite("maxdets", &params_ctns::maxdets)
      .def_readwrite("thresh_proj", &params_ctns::thresh_proj)
      .def_readwrite("thresh_ortho", &params_ctns::thresh_ortho)
      .def_readwrite("rdm_svd", &params_ctns::rdm_svd)
      .def_readwrite("nroots", &params_ctns::nroots)
      .def_readwrite("guess", &params_ctns::guess)
      .def_readwrite("dbranch", &params_ctns::dbranch)
      .def_readwrite("maxsweep", &params_ctns::maxsweep)
      .def_readwrite("maxbond", &params_ctns::maxbond)
      .def_readwrite("ctrls", &params_ctns::ctrls)
      .def_readwrite("alg_hvec", &params_ctns::alg_hvec)
      .def_readwrite("alg_hinter", &params_ctns::alg_hinter)
      .def_readwrite("alg_hcoper", &params_ctns::alg_hcoper)
      .def_readwrite("alg_renorm", &params_ctns::alg_renorm)
      .def_readwrite("alg_rinter", &params_ctns::alg_rinter)
      .def_readwrite("alg_rcoper", &params_ctns::alg_rcoper)
      .def_readwrite("alg_decim", &params_ctns::alg_decim)
      .def_readwrite("ifdist1", &params_ctns::ifdist1)
      .def_readwrite("ifdistc", &params_ctns::ifdistc)
      .def_readwrite("save_formulae", &params_ctns::save_formulae)
      .def_readwrite("sort_formulae", &params_ctns::sort_formulae)
      .def_readwrite("save_mmtask", &params_ctns::save_mmtask)
      .def_readwrite("batchhvec", &params_ctns::batchhvec)
      .def_readwrite("batchrenorm", &params_ctns::batchrenorm)
      .def_readwrite("batchmem", &params_ctns::batchmem)
      .def_readwrite("cisolver", &params_ctns::cisolver)
      .def_readwrite("maxcycle", &params_ctns::maxcycle)
      .def_readwrite("nbuff", &params_ctns::nbuff)
      .def_readwrite("damping", &params_ctns::damping)
      .def_readwrite("precond", &params_ctns::precond)
      .def_readwrite("rcanon_load", &params_ctns::rcanon_load)
      .def_readwrite("rcanon_file", &params_ctns::rcanon_file)
      .def_readwrite("iomode", &params_ctns::iomode)
      .def_readwrite("async_fetch", &params_ctns::async_fetch)
      .def_readwrite("async_save", &params_ctns::async_save)
      .def_readwrite("async_remove", &params_ctns::async_remove)
      .def_readwrite("async_tocpu", &params_ctns::async_tocpu)
      .def_readwrite("ifnccl", &params_ctns::ifnccl)
      .def_readwrite("iroot", &params_ctns::iroot)
      .def_readwrite("nsample", &params_ctns::nsample)
      .def_readwrite("ndetprt", &params_ctns::ndetprt);

  py::class_<params_post>(m, "post")
      .def(py::init())
      .def("read", &params_post::read)  // ifstream
      .def("print", &params_post::print)

      // public member
      .def_readwrite("run", &params_post::run)
      .def_readwrite("qkind", &params_post::qkind)
      .def_readwrite("topology_file", &params_post::topology_file)
      .def_readwrite("verbose", &params_post::verbose)
      .def_readwrite("task_ovlp", &params_post::task_ovlp)
      .def_readwrite("task_cicoeff", &params_post::task_cicoeff)
      .def_readwrite("task_expect", &params_post::task_expect)
      .def_readwrite("bra", &params_post::bra)
      .def_readwrite("ket", &params_post::ket)
      .def_readwrite("opname", &params_post::opname)
      .def_readwrite("integral_file", &params_post::integral_file)
      .def_readwrite("iroot", &params_post::iroot)
      .def_readwrite("nsample", &params_post::nsample)
      .def_readwrite("ndetprt", &params_post::ndetprt)
      .def_readwrite("eps2", &params_post::eps2);

  py::class_<params_vmc>(m, "vmc")
      .def(py::init())
      .def("read", &params_vmc::read) // ifstream
      .def("print", &params_vmc::print)

      // public member
      .def_readwrite("run", &params_vmc::run)
      .def_readwrite("ansatz", &params_vmc::ansatz)
      .def_readwrite("nhiden", &params_vmc::nhiden)
      .def_readwrite("iscale", &params_vmc::iscale)
      .def_readwrite("exactopt", &params_vmc::exactopt)
      .def_readwrite("nsample", &params_vmc::nsample)
      .def_readwrite("maxiter", &params_vmc::maxiter)
      .def_readwrite("optimizer", &params_vmc::optimizer)
      .def_readwrite("lr", &params_vmc::lr)
      .def_readwrite("history", &params_vmc::history)
      .def_readwrite("wf_load", &params_vmc::wf_load)
      .def_readwrite("wf_file", &params_vmc::wf_file);
}
