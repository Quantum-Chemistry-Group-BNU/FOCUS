#include <iomanip>
#include <cmath>
#include "analysis.h"
#include "tools.h"
#include "matrix.h"
#include "integral.h"
#include "linalg.h"

using namespace std;
using namespace fock;
using namespace linalg;

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
   cout << "fock::vonNeumann_entropy" << endl;
   cout << defaultfloat << setprecision(10);	  
   cout << "Psum=" << psum << " SvN=" << ssum << endl; 
   return ssum;
}

// <Psi1|p^+q|Psi2>
void fock::get_rdm1(const onspace& space,
 		    const vector<double>& civec1,
		    const vector<double>& civec2,
		    matrix& rdm1){
   cout << "\nfock:get_rdm1" << endl;
   for(size_t i=0; i<space.size(); i++){
      // c1[i]<Di|p^+q|Di>c2[i]
      vector<int> olst;
      space[i].get_olst(olst);
      for(int p : olst){
         rdm1(p,p) += civec1[i]*civec2[i];
      }
      // c1[i]<Di|p^+q|Dj>c2[j] + c1[j]<Dj|p^+q|Di>c2[i] (j<i)
      for(size_t j=0; j<i; j++){
         if(space[i].diff_num(space[j]) != 2) continue;
         vector<int> cre,ann;
         space[i].diff_orb(space[j],cre,ann);
	 auto p0 = cre[0];
	 auto q0 = ann[0];
         auto sgn = space[i].parity(p0)*space[j].parity(q0);
         rdm1(p0,q0) += sgn*civec1[i]*civec2[j];
         rdm1(q0,p0) += sgn*civec1[j]*civec2[i];
      }
   }
}

// <Psi|p0^+p1^+q1q0|Psi> (p0>p1, q0>q1)
void fock::get_rdm2(const onspace& space,
	            const vector<double>& civec1,
	            const vector<double>& civec2,
		    matrix& rdm2){
   cout << "\nfock:get_rdm2" << endl;
   for(size_t i=0; i<space.size(); i++){
      // c1[i]<Di|p0^+p1^+p1p0|Di>c2[i]
      vector<int> olst;
      space[i].get_olst(olst);
      for(int idx=0; idx<olst.size(); idx++){
         auto p0 = olst[idx]; 
	 for(int jdx=0; jdx<idx; jdx++){
            auto p1 = olst[jdx];
	    auto p01 = tools::canonical_pair0(p0,p1);
	    rdm2(p01,p01) += civec1[i]*civec2[i]; 
	 }
      }
      // c1[i]<Di|p0^+p1^+q1q0|Dj>c2[j] + c1[j]<Dj|p0^+p1^+q1q0|Di>c2[i] (j<i)
      for(size_t j=0; j<i; j++){
         auto ndiff = space[i].diff_num(space[j]); 
	 // <Di|p0^+k^+kq0|Dj>
	 if(ndiff == 2){
            vector<int> cre,ann;
            space[i].diff_orb(space[j],cre,ann);
	    auto p0 = cre[0];
	    auto q0 = ann[0];
            auto sgn0 = space[i].parity(p0)*space[j].parity(q0);
	    for(int idx=0; idx<olst.size(); idx++){
               auto p1 = olst[idx];
	       if(p1 == p0) continue; 
               auto sgn = sgn0;
	       auto p01 = tools::canonical_pair0(p0,p1);
	       if(p0 < p1) sgn *= -1; // sign coming from ordering of operators
	       // p1 must be not identical to q0, otherwise it cannot be in olst 
	       auto q01 = tools::canonical_pair0(q0,p1);
	       if(q0 < p1) sgn *= -1; 
               rdm2(p01,q01) += sgn*civec1[i]*civec2[j];
               rdm2(q01,p01) += sgn*civec1[j]*civec2[i];
	    }
	 // <Di|p0^+p1^+q1q0|Dj>
	 }else if(ndiff == 4){
            vector<int> cre,ann;
            space[i].diff_orb(space[j],cre,ann);
	    auto p0 = cre[0], p1 = cre[1];
	    auto q0 = ann[0], q1 = ann[1];
	    auto p01 = tools::canonical_pair0(p0,p1);
	    auto q01 = tools::canonical_pair0(q0,q1);
            auto sgn = space[i].parity(p0)*space[i].parity(p1)
		     * space[j].parity(q0)*space[j].parity(q1);
            rdm2(p01,q01) += sgn*civec1[i]*civec2[j];
            rdm2(q01,p01) += sgn*civec1[j]*civec2[i];
	 }
      } // j
   } // i
}

// from rdm2 for particle number conserving wf
matrix fock::get_rdm1_from_rdm2(const matrix& rdm2){
   cout << "\nfock:get_rdm1_from_rdm2" << endl;
   int k2 = rdm2.rows();
   auto pr = tools::inverse_pair0(k2-1);
   assert(pr.first == pr.second+1);
   int k = pr.first+1;   
   matrix rdm1(k,k);
   // <p^+r^+rq>
   for(int p=0; p<k; p++){
      for(int q=0; q<k; q++){
         for(int r=0; r<k; r++){
            if(r == p || r == q) continue;		 
	    auto pr = tools::canonical_pair0(p,r);
	    auto qr = tools::canonical_pair0(q,r);
	    double sgn_pr = p>r? 1 : -1;
	    double sgn_qr = q>r? 1 : -1;
	    rdm1(p,q) += sgn_pr*sgn_qr*rdm2(pr,qr);
	 }
      }
   }
   double diff = symmetric_diff(rdm1);
   assert(diff < 1.e-8);
   // normalization
   double dn2 = rdm1.trace(); // tr(rdm2) is normalized to n(n-1)
   int n2 = round(dn2);
   if(abs(dn2 - n2)>1.e-8){
      cout << scientific << setprecision(12);
      cout << "error in get_rdm1_from_rdm2: non-integer electron number" << endl;
      cout << "n2=n(n-1)" << n2 << " while dn2=" << dn2 << " diff=" << dn2-n2 << endl;
      exit(1);
   }
   if(n2 == 0){
      cout << "error in get_rdm1_from_rdm2: not work for ne = 1" << endl;
      exit(1);
   }
   // find nelec
   auto pp = tools::inverse_pair0(n2/2-1);
   assert(pp.first == pp.second+1);
   int ne = pp.first+1;
   rdm1 *= 1.0/(ne-1.0);
   return rdm1;
}

// E1 = h[p,q]*<p^+q>
double fock::get_e1(const matrix& rdm1,
	      	    const integral::one_body& int1e){
   double e1 = 0.0;
   int k = int1e.sorb;
   assert(k == rdm1.rows());
   for(int j=0; j<k; j++){
      for(int i=0; i<k; i++){
	 e1 += rdm1(i,j)*int1e.get(i,j);
      }
   }
   return e1;
}

// E2 = <p0p1||q0q1>*<p0^+p1^+q1q0> (p0>p1,q0>q1)
double fock::get_e2(const matrix& rdm2,
	     	    const integral::two_body& int2e){
   double e2 = 0.0;
   int k = int2e.sorb;
   assert(k*(k-1)/2 == rdm2.rows());
   for(int p0=0; p0<k; p0++){
      for(int p1=0; p1<p0; p1++){
         for(int q0=0; q0<k; q0++){
            for(int q1=0; q1<q0; q1++){
	       auto p01 = tools::canonical_pair0(p0,p1);
	       auto q01 = tools::canonical_pair0(q0,q1);
	       e2 += rdm2(p01,q01)*(int2e.get(p0,q0,p1,q1)-int2e.get(p0,q1,p1,q0));
	    }
	 }
      }
   }
   return e2;
}

// Etot = h[p,q]*<p^+q> + <p0p1||q0q1>*<p0^+p1^+q1q0>
double fock::get_etot(const matrix& rdm1,
		      const matrix& rdm2,
	     	      const integral::two_body& int2e,
	     	      const integral::one_body& int1e,
		      const double ecore){
   double e1 = get_e1(rdm1, int1e);
   double e2 = get_e2(rdm2, int2e);
   return ecore+e1+e2;
}

double fock::get_etot(const matrix& rdm2,
	     	      const integral::two_body& int2e,
	     	      const integral::one_body& int1e,
		      const double ecore){
   auto rdm1 = get_rdm1_from_rdm2(rdm2);
   return get_etot(rdm1, rdm2, int2e, int1e, ecore);
}
