#include <iostream>
#include <vector>
#include "hamiltonian.h"

using namespace std;

// <Di|H|Di> =  hpp + \sum_{p>q}<pq||pq> 
double fock::get_Hii(const onstate& state1,
		     const integral::two_body& int2e,
		     const integral::one_body& int1e){
   double Hii = 0.0;
   vector<int> olst;
   state1.get_occ(olst);
   for(int i=0; i<olst.size(); i++){
      int p = olst[i];
      Hii += int1e(p,p);
      for(int j=0; j<i; j++){
         int q = olst[j];
	 Hii += int2e(p,p,q,q)-int2e(p,q,q,p); 
      }
   } 
   return Hii;
}

// <Di|H|Dj>
double fock::get_Hij(const onstate& state1, const onstate& state2,
	             const integral::two_body& int2e,
	             const integral::one_body& int1e){
   double Hij = 0.0;
   vector<int> cre,ann;
   int ndiff = state1.diff_num(state2);
   if(ndiff == 0){
      Hij = fock::get_Hii(state1,int2e,int1e);
   }else if(ndiff == 2){
      fock::diff_orb(state1,state2,cre,ann);
      int p0 = cre[0];
      int q0 = ann[0];
      Hij += int1e(p0,q0); // hpq
      vector<int> olst;
      state2.get_occ(olst);
      for(const auto& k : olst){
	 Hij += int2e(p0,q0,k,k) - int2e(p0,k,k,q0); // <pk||qk>
      } 
      Hij *= state1.parity(p0)*state2.parity(q0);
   }else if(ndiff == 4){
      fock::diff_orb(state1,state2,cre,ann);
      int p0 = cre[0], p1 = cre[1];
      int q0 = ann[0], q1 = ann[1]; 
      // <p0p1||q0q1> = [p0q0|p1q1]-[p0q1|p1q0]
      Hij = state1.parity(p0)*state1.parity(p1)
	  * state2.parity(q0)*state2.parity(q1)
	  * (int2e(p0,q0,p1,q1)-int2e(p0,q1,p1,q0)); 
   }
   return Hij;
}
