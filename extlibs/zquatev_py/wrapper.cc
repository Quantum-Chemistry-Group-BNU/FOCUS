#include <iostream>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <iomanip>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "zquatev.h"
#include "f77.h"

namespace py = pybind11;
using namespace std;

py::array_t<complex<double>> gen_array(const int n) { 

  const int n2 = n*2;
  const int nld = n*2;

  unique_ptr<complex<double>[]> A(new complex<double>[n*n]);
  unique_ptr<complex<double>[]> B(new complex<double>[n*n]);
  unique_ptr<complex<double>[]> C(new complex<double>[nld*n2]);

  srand(32);
  for (size_t i = 0; i != n; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      const array<double,4> t = {{rand()%10000 * 0.0001, rand()%10000 * 0.0001,
                                  rand()%10000 * 0.0001, rand()%10000 * 0.0001}};
      A[j+n*i] = j == i ? t[0] : complex<double>(t[0], t[1]);
      A[i+n*j] = conj(A[j+n*i]);
      B[j+n*i] = j == i ? 0.0 : complex<double>(t[2], t[3]);
      B[i+n*j] = -B[j+n*i];
    }
  }

  for (size_t i = 0; i != n; ++i) {
    for (size_t j = 0; j != n; ++j) {
      C[j+nld*i] = A[j+n*i];
      C[n+j+nld*(n+i)] = conj(A[j+n*i]);
      C[j+nld*(n+i)] = B[j+n*i];
      C[n+j+nld*(i)] = -conj(B[j+n*i]);
    }
  }

  // return results
  py::array_t<complex<double>> result = py::array_t<complex<double>>(nld*n2);
  auto buf = result.request();
  complex<double> *ptr = (complex<double> *) buf.ptr;
  copy_n(C.get(), nld*n2, ptr);
  result.resize({nld,n2});
  return result;
}

int test(py::array_t<complex<double>> eigvec,
         py::array_t<double> eigval) {

  auto buf1 = eigvec.request(); 
  complex<double> *ptr1 = (complex<double> *) buf1.ptr;
  auto buf2 = eigval.request(); 
  double *ptr2 = (double *) buf2.ptr;

  const int nld = buf1.shape[0];
  const int n = nld/2;
  const int n2 = buf1.shape[1];

  unique_ptr<complex<double>[]> D(new complex<double>[nld*n2]);
  unique_ptr<complex<double>[]> E(new complex<double>[nld*n2]);

  unique_ptr<double[]> eig(new double[n2]);
  unique_ptr<double[]> eig2(new double[n2]);

  // D,E hold data, E will be used for test
  copy_n(ptr1, nld*n2, D.get());
  copy_n(ptr1, nld*n2, E.get());

  cout << endl;
  cout << " **** using zheev **** " << endl;
  {
    int lwork = -1;
    unique_ptr<double[]> rwork(new double[6*n-2]);
    complex<double> lworkopt;
    int info;
    zheev_("V", "U", &n2, ptr1, &nld, eig.get(), &lworkopt, &lwork, rwork.get(), &info);
    lwork = static_cast<int>(lworkopt.real());
    unique_ptr<complex<double>[]> work(new complex<double>[lwork]);
    zheev_("V", "U", &n2, ptr1, &nld, eig.get(), work.get(), &lwork, rwork.get(), &info);
    if (info) throw runtime_error("zheev failed");
  }

  cout << " **** using zquatev **** " << endl;
  const int info2 = ts::zquatev(n2, D.get(), nld, eig2.get());
  if (info2) throw runtime_error("zquatev failed");
  for (size_t i = 0; i != nld*n2; ++i)
  	cout << D[i] << endl;

  cout << " **** Check solutions **** " << endl; // D^+ED
  zgemm_("N", "N", n2, n2, n2, 1.0, E.get(), n2, D.get(), n2, 0.0, ptr1, n2);
  zgemm_("C", "N", n2, n2, n2, 1.0, D.get(), n2, ptr1, n2, 0.0, E.get(), n2);
  for (size_t i = 0; i != n2; ++i)
    E[i+i*n2] -= eig2[i%n];
  const double error = real(zdotc_(n2*n2, E.get(), 1, E.get(), 1));

  double maxdev = 0.0;
  for (size_t i = 0; i != n; ++i)
    maxdev = max(maxdev, abs(eig[2*i]-eig2[i]));
  cout << setprecision(5) << scientific << endl;
  cout << " * Max deviation of the eigenvalues: " << maxdev << endl;
  cout << " * Errors in the eigenvectors      : " << error << endl;
  cout << endl;

  for (size_t i = 0; i != nld*n2; ++i)
  	cout << E[i] << endl;
  for (size_t i = 0; i != nld/2; ++i)  
    eig2[nld/2+i] = eig2[i];
  for (size_t i = 0; i != n2; ++i)
  	cout << eig2[i] << endl;

  copy_n(E.get(), nld*n2, ptr1);
  copy_n(eig2.get(), n2, ptr2);
  eigvec.resize({nld,n2});

  return info2;
}

int eigh(py::array_t<complex<double>> eigvec,
         py::array_t<double> eigval) {

  auto buf1 = eigvec.request(); 
  complex<double> *ptr1 = (complex<double> *) buf1.ptr;
  auto buf2 = eigval.request(); 
  double *ptr2 = (double *) buf2.ptr;

  const int nld = buf1.shape[0];
  const int n2 = buf1.shape[1];

  unique_ptr<complex<double>[]> D(new complex<double>[nld*n2]);
  copy_n(ptr1, nld*n2, D.get());
  //const int info2 = ts::zquatev(n2, ptr1, nld, ptr2);
  const int info2 = ts::zquatev(n2, D.get(), nld, ptr2);
  if (info2) throw runtime_error("zquatev failed");
/*
  for (size_t i = 0; i != n2; ++i){
     for (size_t j = 0; j != n2; ++j){
  	cout << D[i*n2+j] << " "; 
     }
     cout << endl;
  }
*/
  copy_n(D.get(), nld*n2, ptr1); // copy back
  eigvec.resize({nld,n2});
  for (int i = 0; i != nld/2; ++i) // eigvalues 
    ptr2[nld/2+i] = ptr2[i];
  return info2;
}

PYBIND11_MODULE(libzquatev, m) {
	m.doc() = "Interface to zquatev package";
	m.def("gen_array", &gen_array, "Gen complex matrix");
	m.def("test", &test, "Test diagonalization");
	m.def("eigh", &eigh, "Diagonalize quaternionic matrix");
}

