#include "../core/linalg.h"
#include "../core/hamiltonian.h"
#include "../core/tools.h"
#include <iostream>
#include "vmc.h"

using namespace std;
using namespace linalg;
using namespace fock;
using namespace vmc;

// operator
vecmat vmc::operator *(const double fac, const vecmat& tmp){
   vecmat tmp1 = tmp;
   tmp1 *= fac;
   return tmp1;
}

vecmat vmc::operator *(const vecmat& tmp, const double fac){
   return fac*tmp;
}

vecmat vmc::operator +(const vecmat& tmp1, const vecmat& tmp2){
   vecmat tmp = tmp1;
   tmp += tmp2;
   return tmp;
}

vecmat vmc::operator -(const vecmat& tmp1, const vecmat& tmp2){
   vecmat tmp = tmp1;
   tmp -= tmp2;
   return tmp;
}

// Ansatz |Psi>=tr(A[p1]*...*A[pn])|p1...pn> ~ O(KD^2)
double vmc::get_Ws(const onstate& state,
	           const vecmat& mps){
   // ensure olst is ordered 
   vector<int> olst;
   state.get_olst(olst);
   int nelec = olst.size();
   auto tmp = mps.site[olst[0]];
   for(int idx=1; idx<nelec; idx++){
      int i = olst[idx];
      tmp = dgemm("N","N",tmp,mps.site[i]);
   }
   return tmp.trace();
}

// compute local energy E[n] = <n|H|Psi>/<n|Psi> 
// and gradient info delta_a[n] = <n|Psi_a>/<n|Psi>
void vmc::local_egrad(const onstate& state,
		      const vector<int>& olst,
		      const vector<int>& vlst,
		      const vecmat& mps,
	    	      const integral::two_body& int2e,
	    	      const integral::one_body& int1e,
		      const double ecore,
		      double& wt,
		      double& elocal,
		      vecmat& delta){
   int D = mps.D;
   int n = olst.size();
   int k = state.size();
   // weight
   wt = get_Ws(state,mps);
   double wtinv = 1.0/wt; 
   // local energy E[n] = <n|H|Psi>/<n|Psi>
   // 			= <n|H|n> + <n|H|ex><ex|Psi>/<ex|Psi>
   elocal = fock::get_Hii(state,int2e,int1e)+ecore;
   // single
   auto state1 = state;
   for(int i : olst){
      for(int a : vlst){
	 state1[i] = 0;
	 state1[a] = 1;
	 auto pr = fock::get_HijS(state,state1,int2e,int1e);
         elocal += pr.first*get_Ws(state1,mps)*wtinv;
	 state1[i] = 1;
	 state1[a] = 0;
      }
   }
   // double
   auto state2 = state;
   for(int idx=0; idx<n; idx++){
      int i = olst[idx];
      for(int jdx=0; jdx<idx; jdx++){
	 int j = olst[jdx];
	 for(int adx=0; adx<k-n; adx++){
	    int a = vlst[adx];
	    for(int bdx=0; bdx<adx; bdx++){
	       int b = vlst[bdx];
	       state2[i] = 0;
	       state2[j] = 0;
	       state2[a] = 1;
	       state2[b] = 1;
	       auto pr = fock::get_HijD(state,state2,int2e,int1e);
               elocal += pr.first*get_Ws(state2,mps)*wtinv;
	       state2[i] = 1;
	       state2[j] = 1;
	       state2[a] = 0;
	       state2[b] = 0;
	    } // b
	 } // a
      } // j
   } // i
   // gradient info <n|Psi_a>/<n|Psi>
   // left & right environment 
   vector<matrix> Lenv(n);
   Lenv[0] = identity_matrix(D);
   for(int idx=1; idx<n; idx++){
      int i = olst[idx-1];
      Lenv[idx] = dgemm("N","N",Lenv[idx-1],mps.site[i]);
   } 
   vector<matrix> Renv(n);
   Renv[n-1] = identity_matrix(D);
   for(int idx=n-2; idx>-1; idx--){
      int i = olst[idx+1];
      Renv[idx] = dgemm("N","N",mps.site[i],Renv[idx+1]);
   }
   // <n|Psi_a> = (Ra*La)^T = L^T*R^T (/<n|Psi>)
   for(int idx=0; idx<n; idx++){
      int i = olst[idx];
      delta.site[i] = dgemm("T","T",Lenv[idx],Renv[idx])*wtinv;
   }
   for(int adx=0; adx<n; adx++){
      int a = vlst[adx];
      delta.site[a] = 0.0;
   }
}

// VMC estimate of energy and gradients
void vmc::estimate_egrad(const int mcmc,
		         const onstate& seed,
		         const vecmat& mps,
	    	         const integral::two_body& int2e,
	    	         const integral::one_body& int1e,
			 const double ecore,
			 double& emps,
		         vecmat& grad){
   // settings
   std::uniform_real_distribution<double> dist(0,1);
   int D = mps.D;
   // initial 
   onstate state = seed;
   int k = state.size();
   int n = state.nelec();
   cout << "\nvmc::estimate_egrad" << endl;
   cout << "k=" << k << " n=" << n << " D=" << D
	<< " seed=" << seed
	<< endl;
   // compute local quantities
   vector<int> olst(n),vlst(k-n);
   state.get_olst(olst.data());
   state.get_vlst(vlst.data());
   double wt, elocal;
   vecmat delta(k,D,"zero");
   local_egrad(state,olst,vlst,mps,int2e,int1e,ecore,
	       wt,elocal,delta);
   // MCMC
   double acpt = 0;
   double esum = 0.0;
   vecmat dsum(k,D,"zero");
   vecmat edsum(k,D,"zero");
   for(int it=1; it<=mcmc; it++){

      // MC move: 
      onstate state1 = state;
      double sd = dist(linalg::generator);
      if(sd < 0.5){
         // randomly pick a single excitation i->a
         int idx = n*dist(linalg::generator);
         int adx = (k-n)*dist(linalg::generator);
         state1[olst[idx]] = 0;
         state1[vlst[adx]] = 1;
      }else{
         // randomly pick a single excitation i->a
         int ijdx = n*(n-1)/2*dist(linalg::generator);
	 auto pij = tools::inverse_pair0(ijdx);
	 int idx = pij.first, jdx = pij.second;
         int abdx = (k-n)*(k-n-1)/2*dist(linalg::generator);
	 auto pab = tools::inverse_pair0(abdx);
	 int adx = pab.first, bdx = pab.second; 
         state1[olst[idx]] = 0;
         state1[olst[jdx]] = 0;
         state1[vlst[adx]] = 1;
         state1[vlst[bdx]] = 1;
      }
      double wt1 = get_Ws(state1,mps);
      // Metropolis step 
      double prob = min(1.0,pow(wt1/wt,2));
      double rand = dist(linalg::generator);
      // if accepted, compute new local quantities
      if(prob > rand){
         acpt += 1;
         state = state1;
         state.get_olst(olst);
         state.get_vlst(vlst);
         local_egrad(state,olst,vlst,mps,int2e,int1e,ecore,
           	     wt,elocal,delta);
      }
      // accumulate
      esum += elocal;
      dsum += delta;
      edsum += elocal*delta;
      if(it%10000 == 0){
         cout << "it=" << it 
	      << " ratio=" << setprecision(3) << acpt/it
              << " e=" << setprecision(10) << esum/it
              << endl;
      }
   } // it
   double fac = 1.0/mcmc;
   emps = fac*esum;
   grad = 2.0*fac*(edsum-emps*dsum);
}

// update MPS
void vmc::update_mps(const double step,
	   	     const vecmat& grad,
		     vecmat& mps){
   // settings
   std::uniform_real_distribution<double> dist(0,1);
   int D = mps.D;
   int k = mps.size();
   for(int i=0; i<k; i++){
      for(int m=0; m<D; m++){
	 for(int n=0; n<D; n++){
	    mps.site[i](n,m) -= copysign(step*dist(linalg::generator),grad.site[i](n,m));
	    //mps.site[i](n,m) -= step*grad.site[i](n,m);
	 }
      }
   }
}

// optimize
void vmc::solver(const input::schedule& schd,
	         const integral::two_body& int2e,
	         const integral::one_body& int1e,
	         const double ecore){
   cout << "\nvmc::solver" << endl; 
   auto t0 = global::get_time();

   // settings
   int k = int1e.sorb;
   int n = schd.nelec;
   int maxcycle = 100;
   int mcmc = 100000;
   int D = 10;
  
   // initial state 
   vecmat mps(k,D,"random");
   mps *= 0.1;
   onstate seed(k);
   for(const auto& det : schd.det_seeds){
      for(int i : det){
         seed[i] = 1;
	 mps.site[i](0,0) = 1.0;
      }
      break;
   }
   cout << "seed=" << seed << endl;
   cout << "energy=" << defaultfloat << setprecision(12)
	<< fock::get_Hii(seed,int2e,int1e)+ecore << endl;
  
   double emps; 
   vecmat grad(k,D,"zero");
   for(int iter=1; iter<=maxcycle; iter++){
      // compute Energy and Gradient
      estimate_egrad(mcmc,seed,mps,int2e,int1e,ecore,emps,grad);
      // update sites
      double Q = 0.95;
      double step = 0.05*(pow(Q,iter));
      update_mps(step,grad,mps);
      // print
      cout << "iter=" << iter << " e=" << emps 
	   << " |g|=" << grad.norm() << " step=" << step << endl;
      // debug
      cout << "mps = ";
      for(int i=0; i<k; i++){
	 cout << defaultfloat << setprecision(3) << mps.site[i](0,0) << " ";
      }
      cout << endl;
   } // iter

   auto t1 = global::get_time();
   cout << "timing for vmc::solver : " << setprecision(2) 
	<< global::get_duration(t1-t0) << " s" << endl;
}   
