#include <functional> // for std::function
#include <complex>
#include <iostream>
#include "tools.h"
#include "matrix.h"
#include "linalg.h"
#include "dvdson.h"
#include "tests_core.h"

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

int tests::test_dvdson(){
   cout << endl;
   cout << tools::line_separator << endl;	
   cout << "tests::test_dvdson" << endl;
   cout << tools::line_separator << endl;
  
   const int n = 100;
   const int m = 3;
   const double thresh = 1.e-10;
   
   // --- real matrix --- 
   cout << "\nreal version" << endl; 
   matrix<double> rd1 = random_matrix(n,n);
   matrix<double> rd2 = random_matrix(n,n);
   rd1 = 0.5*(rd1 + rd1.H()) + identity_matrix<double>(n);
   //rd1.print("rd1");

   vector<double> e(n);
   matrix<double> v;
   eig_solver(rd1, e, v);
   cout << "e: " << setprecision(12) << e[0] << " " << e[1] << endl;
   //v.print("v");
   
   vector<double> es(m);
   matrix<double> vs(n,m);
   iter_solver(rd1, es, vs);
   for(int i=0; i<m; i++){
      auto diff = es[i]-e[i];
      cout << "e(iter): " << setprecision(12) << es[i] << " " << e[i] << " diff=" << diff << endl;
      assert(diff < thresh);
   }

   // --- complex matrix ---
   cout << "\ncomplex version" << endl; 
   const complex<double> i0(1.0,0.0), i1(0.0,1.0);
   matrix<complex<double>> cmat = i0*rd1 + i1*rd2;
   cmat = cmat + cmat.H();
   //cmat.print("cmat");

   vector<double> ec(n);
   matrix<complex<double>> vc;
   eig_solver(cmat, ec, vc);
   cout << "ec: " << ec[0] << " " << ec[1] << endl;
   
   vector<double> esc(m);
   matrix<complex<double>> vsc(n,m);
   iter_solver(cmat, esc, vsc);
   for(int i=0; i<m; i++){
      auto diff = esc[i]-ec[i];
      cout << "e(iter): " << esc[i] << " " << ec[i] << " diff=" << diff << endl;
      assert(diff < thresh);
   }

   return 0;
}
