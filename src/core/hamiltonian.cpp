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
      Hii += int1e.get(p,p);
      for(int j=0; j<i; j++){
         int q = olst[j];
	 Hii += int2e.get(p,p,q,q)-int2e.get(p,q,q,p); 
      }
   }
   return Hii;
}

// single - fast version (20200312) 
double fock::get_HijS(const onstate& state1, 
		      const onstate& state2,
	              const integral::two_body& int2e,
	              const integral::one_body& int1e){
   int p[1], q[1];
   state1.diff_orb(state2,p,q);
   double Hij = int1e.get(p[0],q[0]); // hpq
   // loop over occupied state in state2
#ifdef GNU
   for(int i=0; i<state2.len(); i++){
      unsigned long repr = state2.repr(i);
      while(repr != 0){
         int j = 63-__builtin_clzl(repr);
	 int k = i*64+j;
	 Hij += int2e.get(p[0],q[0],k,k) 
	      - int2e.get(p[0],k,k,q[0]); // <pk||qk>
	 repr &= ~(1ULL<<j);
      }
   }
#else
   for(int k=0; k<state2.size(); k++){
      if(state2[k]){
         Hij += int2e.get(p[0],q[0],k,k)
              - int2e.get(p[0],k,k,q[0]); 
      }
   }
#endif
   Hij *= state1.parity(p[0])*state2.parity(q[0]);
   return Hij;
}

// double: <p0p1||q0q1> = [p0q0|p1q1]-[p0q1|p1q0]
double fock::get_HijD(const onstate& state1, const onstate& state2,
	              const integral::two_body& int2e,
	              const integral::one_body& int1e){
   int p[2], q[2];
   state1.diff_orb(state2,p,q);
   return state1.parity(p[0])*state1.parity(p[1])
         *state2.parity(q[0])*state2.parity(q[1])
         *(int2e.get(p[0],q[0],p[1],q[1])-int2e.get(p[0],q[1],p[1],q[0]));
}

// <Di|H|Dj>
double fock::get_Hij(const onstate& state1, const onstate& state2,
	             const integral::two_body& int2e,
	             const integral::one_body& int1e){
   double Hij = 0.0;
   auto pr = state1.diff_type(state2);
   if(pr == make_pair(0,0)){
      Hij = fock::get_Hii(state1,int2e,int1e);
   }else if(pr == make_pair(1,1)){
      Hij = fock::get_HijS(state1,state2,int2e,int1e);
   }else if(pr == make_pair(2,2)){
      Hij = fock::get_HijD(state1,state2,int2e,int1e);
   }
   return Hij;
}
