#ifndef TEST_DVDSON_H
#define TEST_DVDSON_H

#include <complex>
#include <functional> // for std::function
#include <iostream>
#include "../settings/global.h"
#include "../core/tools.h"
#include "../core/matrix.h"
#include "../core/linalg.h"
#include "../core/dvdson.h"

using namespace std;
using namespace linalg;

template <typename Tm>
vector<double> get_diag(const matrix<Tm>& mat){
   int n = mat.rows();
   vector<double> diag(n);
   for(int i=0; i<n; i++){
      diag[i] = real(mat(i,i));
   }
   return diag;
}

template <typename Tm>
void get_Mx(Tm* y, const Tm* x, const matrix<Tm>& M){
   int n = M.rows();
   for(int i=0; i<n; i++){
      y[i] = 0.0;
      for(int j=0; j<n; j++){
         y[i] += M(i,j)*x[j]; 
      }
   }
}

template <typename Tm>
int iter_solver(const matrix<Tm>& mat, vector<double>& es, matrix<Tm>& vs){	
   // Davidson solver 
   dvdsonSolver<Tm> solver;
   solver.ndim = mat.rows();
   solver.neig = es.size();
   // diag
   auto Diag = get_diag(mat);
   solver.Diag = Diag.data(); 
   using std::placeholders::_1;
   using std::placeholders::_2;
   solver.HVec = bind(&get_Mx<Tm>, _1, _2, cref(mat));
   // solve
   //solver.solve_diag(es.data(), vs.data());
   //solver.solve_iter(es.data(), vs.data(), vs.data());
   solver.solve_iter(es.data(), vs.data());
   return 0;
}

#endif
