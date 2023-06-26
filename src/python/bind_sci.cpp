#include <complex>
#include <pybind11/pybind11.h>

#include <../ci/ci_header.h>
#include <../drivers/sci.cpp>


PYBIND11_MODULE(SCI, m){
    m.doc() = "Select CI";
    m.def("SCI_real", &SCI<double>, "Selected SCI");
    m.def("SCI_complex", &SCI<std::complex<double>>, "Selected SCI ");
}