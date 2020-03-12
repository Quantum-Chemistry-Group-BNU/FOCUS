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

// single 
double fock::get_HijS(const onstate& state1, const onstate& state2,
	              const integral::two_body& int2e,
	              const integral::one_body& int1e,
		      const int iop){
   vector<int> cre, ann;
   state1.diff_orb(state2,cre,ann);
   int p0 = cre[0];
   int q0 = ann[0];
   vector<int> olst;
   state2.get_occ(olst);
   double Hij = 0.0;
   // for even string
   if(iop == 0){
      int pp0 = 2*p0, qq0 = 2*q0;
      Hij = int1e.get(pp0,qq0); // hpq
      for(int k : olst){
	 int kk = 2*k;
	 Hij += int2e.get(pp0,qq0,kk,kk) - int2e.get(pp0,kk,kk,qq0); // <pk||qk>
      }
   // for odd string
   }else if(iop == 1){
      int pp0 = 2*p0+1, qq0 = 2*q0+1;
      Hij = int1e.get(pp0,qq0); // hpq
      for(int k : olst){
	 int kk = 2*k+1;
	 Hij += int2e.get(pp0,qq0,kk,kk) - int2e.get(pp0,kk,kk,qq0); // <pk||qk>
      }
   // for full det
   }else if(iop == 2){
      Hij = int1e.get(p0,q0); // hpq
      for(int k : olst){
	 Hij += int2e.get(p0,q0,k,k) - int2e.get(p0,k,k,q0); // <pk||qk>
      }
   }
   Hij *= state1.parity(p0)*state2.parity(q0);
   return Hij;
}

// double: <p0p1||q0q1> = [p0q0|p1q1]-[p0q1|p1q0]
double fock::get_HijD(const onstate& state1, const onstate& state2,
	              const integral::two_body& int2e,
	              const integral::one_body& int1e,
		      const int iop){
   int p[2], q[2];
   state1.diff_orb(state2,p,q);
   double Hij;
   if(iop == 0){ 
      int p0=2*p[0], p1=2*p[1], q0=2*q[0], q1=2*q[1];
      Hij = int2e.get(p0,q0,p1,q1)
	  - int2e.get(p0,q1,p1,q0);
   }else if(iop == 1){
      int p0=2*p[0]+1, p1=2*p[1]+1, q0=2*q[0]+1, q1=2*q[1]+1;
      Hij = int2e.get(p0,q0,p1,q1)
	  - int2e.get(p0,q1,p1,q0);
   }else if(iop == 2){
      Hij = int2e.get(p[0],q[0],p[1],q[1])
	  - int2e.get(p[0],q[1],p[1],q[0]);
   }
   Hij *= state1.parity(p[0])*state1.parity(p[1])*
	  state2.parity(q[0])*state2.parity(q[1]);
   return Hij;
}

// <Di|H|Dj>
double fock::get_Hij(const onstate& state1, const onstate& state2,
	             const integral::two_body& int2e,
	             const integral::one_body& int1e){
   double Hij = 0.0;
   int ndiff = state1.diff_num(state2);
   if(ndiff == 0){
      Hij = fock::get_Hii(state1,int2e,int1e);
   }else if(ndiff == 2){
      Hij = fock::get_HijS(state1,state2,int2e,int1e,2);
   }else if(ndiff == 4){
      Hij = fock::get_HijD(state1,state2,int2e,int1e,2);
   }
   return Hij;
}

// single - fast version (20200312) 
double fock::get_HijS_fast(const onstate& state1, const onstate& state2,
	                   const integral::two_body& int2e,
	                   const integral::one_body& int1e){
   unsigned long idiff,icre,iann;
   int p0, q0;
   for(int i=0; i<state2.len(); i++){
      idiff = state1.repr(i) ^ state2.repr(i);
      icre = idiff & state1.repr(i);
      iann = idiff & state2.repr(i);
      if(icre != 0){
         p0 = __builtin_ffsl(icre)-1+i*64;
      }
      if(iann != 0){
	 q0 = __builtin_ffsl(iann)-1+i*64;
      }
   }
   double Hij = int1e.get(p0,q0); // hpq
   for(int i=0; i<state2.len(); i++){
      unsigned long repr = state2.repr(i);
      while(repr != 0){
         int pos = __builtin_ffsl(repr);
	 int k = pos-1+i*64;
	 Hij += int2e.get(p0,q0,k,k) - int2e.get(p0,k,k,q0); // <pk||qk>
	 repr &= ~(1ULL<<(pos-1));
      }
   }   
   Hij *= state1.parity(p0)*state2.parity(q0);
   return Hij;
}

double fock::get_HijD_fast(const onstate& state1, const onstate& state2,
	                   const integral::two_body& int2e,
	                   const integral::one_body& int1e){
   int p[2], q[2];
   state1.diff_orb(state2,p,q);
/*
   unsigned long idiff,icre,iann;
   int p[2], q[2];
   int ip = 0, iq = 0;
   for(int i=0; i<state2.len(); i++){
      idiff = state1.repr(i) ^ state2.repr(i);
      icre = idiff & state1.repr(i);
      iann = idiff & state2.repr(i);
      while(icre != 0){
         // Returns one plus the index of the least significant 1-bit of x, 
	 // or if x is zero, returns zero.
         int pos = __builtin_ffsl(icre)-1;
	 p[ip] = pos+i*64;
	 ip++;
	 icre &= ~(1ULL<<pos);
      }
      while(iann != 0){
	 int pos = __builtin_ffsl(iann)-1;
	 q[iq] = pos+i*64;
	 iq++;
	 iann &= ~(1ULL<<pos);
      }
   }
*/

/*
   int p[2], q[2];
   int ip = 0, iq = 0;
   unsigned long idiff,icre,iann;
   for(int i=state1.len()-1; i>=0; i--){
      idiff = state1.repr(i) ^ state2.repr(i);
      icre = idiff & state1.repr(i);
      iann = idiff & state2.repr(i);
      for(int j=63; j>=0; j--){
         if(icre & 1ULL<<j){
	    p[ip] = i*64+j;
	    ip++;
	 }
      }
      for(int j=63; j>=0; j--){
	 if(iann & 1ULL<<j){
	    q[iq] = i*64+j;
	    iq++;
	 }
      }
   }
*/

   return state1.parity(p[0])*state1.parity(p[1])
         *state2.parity(q[0])*state2.parity(q[1])
         *(int2e.get(p[0],q[0],p[1],q[1])-int2e.get(p[0],q[1],p[1],q[0]));
}
