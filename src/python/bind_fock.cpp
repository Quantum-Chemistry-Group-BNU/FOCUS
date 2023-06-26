#include "../core/hamiltonian.h"
#include "../core/onstate.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>
#include <pybind11/chrono.h>
#include <pybind11/functional.h>


using namespace std; 
using namespace fock;
namespace py = pybind11;


PYBIND11_MODULE(py_fock, m){
    m.doc() = "This is python interface for cpp code";
    m.def("get_Hij",&fock::get_Hij<double>, "H_ij");
    m.def("get_fci_space", py::overload_cast<const int>(&fock::get_fci_space),"FCI-CI space");
    m.def("get_fci_space", py::overload_cast<const int, const int>(&fock::get_fci_space),"FCI-CI space, ks, na");
    m.def("get_fci_space", py::overload_cast<const int, const int, const int>(&fock::get_fci_space),"Full-CI space construct ks, na, nb");
    m.def("check_space", &fock::check_space, "print Full-CI space");
    // m.def("get_Hmat", &fock::get_Hmat<double>, "generate representation of H in this space");

    py::class_<fock::bit_proxy>(m, "bit_proxy")
        .def(py::init<unsigned long&, unsigned long>());

    py::class_<fock::onstate>(m, "onstate")
        .def(py::init<int>())
        .def(py::init<const string&>())
        // .def(py::init<fock::onstate, fock::onstate>)
       // .def_property_readonly("len", &fock::onstate::len)
        .def("__setitem__",[](fock::onstate &self, const int idx, const int val){return self[idx] = val;})
        .def("__getitem__",[](fock::onstate &self, const int idx){return self[idx];})
        .def_property_readonly("size", &fock::onstate::size)
        .def_property_readonly("nelec_a", &fock::onstate::nelec_a) 
        .def_property_readonly("nelec_b", &fock::onstate::nelec_b)
        .def_property_readonly("twoms", &fock::onstate::twoms)
        .def_property_readonly("zvec", &fock::onstate::get_zvec)
        .def("join", &fock::onstate::join)
        .def("to_string", &fock::onstate::to_string)
        .def("to_string2", &fock::onstate::to_string2)
        .def("diff_type", &fock::onstate::diff_type)
        .def("ann", &fock::onstate::ann)
        .def("cre", &fock::onstate::cre)
        .def_static("diff_orb_d",[](const fock::onstate& bra, const fock::onstate& ket){
            std::vector<int> cre, ann;
            bra.diff_orb(ket, cre, ann);
            int sgn = bra.parity(cre[0]) * bra.parity(cre[1]) 
                      * ket.parity(ann[0]) * ket.parity(ann[1]);
            return std::make_tuple(cre, ann, sgn);})
        .def_static("diff_orb_s",[](const fock::onstate& bra, const fock::onstate& ket){
            std::vector<int> cre, ann;
            bra.diff_orb(ket, cre, ann);
            int sgn = bra.parity(cre[0]) * ket.parity(ann[0]);
            return std::make_tuple(cre, ann, sgn);});
        // .def("parity", &fock::onstate::parity);
    
}   
