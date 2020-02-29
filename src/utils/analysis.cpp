#include <iomanip>
#include <cmath>
#include "analysis.h"
#include "tools.h"
#include "matrix.h"

using namespace std;
using namespace fock;

void fock::coefficients(const onspace& space, 
			const vector<double>& civec, 
			const double thresh){
   cout << "\nfock::coefficients thresh=" << thresh << endl;
   cout << "onstate / nelec / single / idx / ci / pi" << endl;
   cout << setprecision(10);
   double ne = 0.0, na = 0.0, nb = 0.0;
   double pi, psum = 0.0;
   vector<int> idx;
   idx = tools::sort_index_abs(civec);
   for(const auto& i : idx){ 
      pi = pow(civec[i],2);
      psum += pi;
      // Measurement in Z-basis 
      ne += pi*space[i].nelec();
      na += pi*space[i].nelec_a();
      nb += pi*space[i].nelec_b();
      if(abs(civec[i]) > thresh){ 
         cout << space[i] << " "
              << space[i].to_string2() << " ("
              << space[i].nelec() << ","
              << space[i].nelec_a() << ","
              << space[i].nelec_b() << ") "
              << space[i].norb_single() << " | "
              << i << " "
              << civec[i] << " " 
              << pi << endl;
      }
   }
   cout << "psum=" << psum << endl;
   cout << "(Ne,Na,Nb)=" << ne << "," 
	   		 << na << "," 
			 << nb << endl; 
}

double fock::vonNeumann_entropy(const vector<double>& sigs, const double cutoff){
   double psum = 0.0, ssum = 0.0;
   for(const auto& sig : sigs){
      if(sig < cutoff) continue;
      psum += sig;
      ssum -= sig*log2(sig);
   }
   cout << "\nfock::vonNeumann_entropy" << endl;
   cout << "psum=" << psum << " ssum=" << ssum << endl; 
   return ssum;
}

using namespace linalg;

void fock::get_rdm1(const onspace& space,
		    const vector<double>& civec1,
		    const vector<double>& civec2,
		    matrix& rdm1){
}

void get_rdm2(const onspace& space,
	      const vector<double>& civec1,
	      const vector<double>& civec2,
	      matrix& rdm2){
}
