#ifndef SWEEP_TWODOT_SIGMA_H
#define SWEEP_TWODOT_SIGMA_H

#ifdef _OPENMP
#include <omp.h>
#endif

#include "oper_dict.h" 
#include "oper_timer.h"

namespace ctns{
   
const bool debug_twodot_sigma = false; 
extern const bool debug_twodot_sigma;

//
// construct H*wf: if ifkr=True, construct skeleton sigma vector 
//
//  Local terms:
//   1. H^LC1
//   2. H^C2R
//  One-index operators
//   3. sum_p1 p1^+[LC1]*Sp1^[C2R] + h.c. 
//   4. sum_q2 q2^+[C2R]*Sq2^[LC1] + h.c. = -Sq2^[LC1]*q2^+[C2R] + h.c. 
//  Two-index operators
//   5. Apq^1*Ppq^2 + h.c. / Prs^1+Ars^2+ + h.c.
//   6. Bps^1*Qps^2 / Qqr^1*Bqr^2
//
template <typename Tm>
stensor3<Tm> twodot_Hx_CS(const int index,
		  	  const int iformula,
	          	  const oper_dict<Tm>& lqops,
	          	  const oper_dict<Tm>& rqops,
	          	  const oper_dict<Tm>& c1qops,
	          	  const oper_dict<Tm>& c2qops,
	          	  const integral::two_body<Tm>& int2e,
	          	  const stensor4<Tm>& wf,
	          	  const int& size,
	          	  const int& rank,
			  const bool& ifdist1){
   if(debug_twodot_sigma) std::cout << "twodot_Hx_CS index=" << index << std::endl;
   const bool dagger = true;
   auto wf2 = wf.merge_lc1(); // wf2[lc1,r,c2]
   // p1^L1C1+*Sp1^C2R
   auto qt3n = oper_compxwf_opS("cr",wf2,c2qops,rqops,int2e,index,size,rank,ifdist1);
   qt3n.row_signed(); // due to action of opS passing lc1
   qt3n = qt3n.merge_cr().split_lc(wf.info.qrow, wf.info.qmid); // wf2[lc1,r,c2] => wf1[l,c2r,c1]
   auto Hwf1 = oper_normxwf_opC("lc",qt3n,lqops,c1qops,index,iformula); 
   // -Sp1^C2R+*p1^L1C1
   auto qt3h = oper_compxwf_opS("cr",wf2,c2qops,rqops,int2e,index,size,rank,ifdist1,dagger); 
   qt3h.row_signed();
   qt3h = qt3h.merge_cr().split_lc(wf.info.qrow, wf.info.qmid);
   Hwf1 -= oper_normxwf_opC("lc",qt3h,lqops,c1qops,index,iformula,dagger);
   return Hwf1;
}
 
template <typename Tm>
stensor3<Tm> twodot_Hx_SC(const int index,
		  	  const int iformula,
	          	  const oper_dict<Tm>& lqops,
	          	  const oper_dict<Tm>& rqops,
	          	  const oper_dict<Tm>& c1qops,
	          	  const oper_dict<Tm>& c2qops,
	          	  const integral::two_body<Tm>& int2e,
	          	  const stensor4<Tm>& wf,
	          	  const int& size,
	          	  const int& rank,
			  const bool& ifdist1){
   if(debug_twodot_sigma) std::cout << "twodot_Hx_SC index=" << index << std::endl;
   const bool dagger = true;
   auto wf2 = wf.merge_lc1(); // wf2[lc1,c2,r]
   // q2^C2R+*Sq2^LC1 = -Sq2^LC1*q2^C2R+
   auto qt3n = oper_normxwf_opC("cr",wf2,c2qops,rqops,index,iformula);
   qt3n.row_signed();
   qt3n = qt3n.merge_cr().split_lc(wf.info.qrow, wf.info.qmid);
   auto Hwf1 = oper_compxwf_opS("lc",qt3n,lqops,c1qops,int2e,index,size,rank,ifdist1);
   Hwf1 *= -1.0;
   // Sq2^LC1+*q2^C2R
   auto qt3h = oper_normxwf_opC("cr",wf2,c2qops,rqops,index,iformula,dagger);
   qt3h.row_signed();
   qt3h = qt3h.merge_cr().split_lc(wf.info.qrow, wf.info.qmid);
   Hwf1 += oper_compxwf_opS("lc",qt3h,lqops,c1qops,int2e,index,size,rank,ifdist1,dagger);
   return Hwf1;
}

template <typename Tm>
stensor3<Tm> twodot_Hx_AP(const int index,
		  	  const int iformula,
	          	  const oper_dict<Tm>& lqops,
	          	  const oper_dict<Tm>& rqops,
	          	  const oper_dict<Tm>& c1qops,
	          	  const oper_dict<Tm>& c2qops,
	          	  const integral::two_body<Tm>& int2e,
	          	  const stensor4<Tm>& wf,
	          	  const int& size,
	          	  const int& rank){
   if(debug_twodot_sigma) std::cout << "twodot_Hx_AP index=" << index << std::endl;
   const bool dagger = true;
   auto wf2 = wf.merge_lc1(); // wf2[lc1,c2,r]
   // Apq*Ppq
   auto qt3n = oper_compxwf_opP("cr",wf2,c2qops,rqops,int2e,index);
   qt3n = qt3n.merge_cr().split_lc(wf.info.qrow, wf.info.qmid); 
   auto Hwf1 = oper_normxwf_opA("lc",qt3n,lqops,c1qops,index,iformula);
   // (Apq*Ppq)^H
   auto qt3h = oper_compxwf_opP("cr",wf2,c2qops,rqops,int2e,index,dagger);
   qt3h = qt3h.merge_cr().split_lc(wf.info.qrow, wf.info.qmid); 
   Hwf1 += oper_normxwf_opA("lc",qt3h,lqops,c1qops,index,iformula,dagger);
   const Tm wt = lqops.ifkr? wfacAP(index) : 1.0;
   Hwf1 *= wt;
   return Hwf1;
}

template <typename Tm>
stensor3<Tm> twodot_Hx_BQ(const int index,
		  	  const int iformula,
	          	  const oper_dict<Tm>& lqops,
	          	  const oper_dict<Tm>& rqops,
	          	  const oper_dict<Tm>& c1qops,
	          	  const oper_dict<Tm>& c2qops,
	          	  const integral::two_body<Tm>& int2e,
	          	  const stensor4<Tm>& wf,
	          	  const int& size,
	          	  const int& rank){
   if(debug_twodot_sigma) std::cout << "twodot_Hx_BQ index=" << index << std::endl;
   const bool dagger = true;
   auto wf2 = wf.merge_lc1(); // wf2[lc1,c2,r]
   // Bpq*Qpq
   auto qt3n = oper_compxwf_opQ("cr",wf2,c2qops,rqops,int2e,index);
   qt3n = qt3n.merge_cr().split_lc(wf.info.qrow, wf.info.qmid); 
   auto Hwf1 = oper_normxwf_opB("lc",qt3n,lqops,c1qops,index,iformula);
   // (Bpq*Qpq)^H
   auto qt3h = oper_compxwf_opQ("cr",wf2,c2qops,rqops,int2e,index,dagger);
   qt3h = qt3h.merge_cr().split_lc(wf.info.qrow, wf.info.qmid); 
   Hwf1 += oper_normxwf_opB("lc",qt3h,lqops,c1qops,index,iformula,dagger);
   const Tm wt = lqops.ifkr? wfacBQ(index) : wfac(index);
   Hwf1 *= wt;
   return Hwf1;
}

template <typename Tm>
stensor3<Tm> twodot_Hx_PA(const int index,
			  const int iformula,
	          	  const oper_dict<Tm>& lqops,
	          	  const oper_dict<Tm>& rqops,
	          	  const oper_dict<Tm>& c1qops,
	          	  const oper_dict<Tm>& c2qops,
	          	  const integral::two_body<Tm>& int2e,
	          	  const stensor4<Tm>& wf,
	          	  const int& size,
	          	  const int& rank){
   if(debug_twodot_sigma) std::cout << "twodot_Hx_PA index=" << index << std::endl;
   const bool dagger = true;
   auto wf2 = wf.merge_lc1(); // wf2[lc1,c2,r]
   // Prs*Ars
   auto qt3n = oper_normxwf_opA("cr",wf2,c2qops,rqops,index,iformula);
   qt3n = qt3n.merge_cr().split_lc(wf.info.qrow, wf.info.qmid); 
   auto Hwf1 = oper_compxwf_opP("lc",qt3n,lqops,c1qops,int2e,index);
   // (Prs*Ars)^H
   auto qt3h = oper_normxwf_opA("cr",wf2,c2qops,rqops,index,iformula,dagger);
   qt3h = qt3h.merge_cr().split_lc(wf.info.qrow, wf.info.qmid); 
   Hwf1 += oper_compxwf_opP("lc",qt3h,lqops,c1qops,int2e,index,dagger);
   const Tm wt = lqops.ifkr? wfacAP(index) : 1.0;
   Hwf1 *= wt;
   return Hwf1;
}

template <typename Tm>
stensor3<Tm> twodot_Hx_QB(const int index,
		  	  const int iformula,
	          	  const oper_dict<Tm>& lqops,
	          	  const oper_dict<Tm>& rqops,
	          	  const oper_dict<Tm>& c1qops,
	          	  const oper_dict<Tm>& c2qops,
	          	  const integral::two_body<Tm>& int2e,
	          	  const stensor4<Tm>& wf,
	          	  const int& size,
	          	  const int& rank){
   if(debug_twodot_sigma) std::cout << "twodot_Hx_QB index=" << index << std::endl;
   const bool dagger = true;
   auto wf2 = wf.merge_lc1(); // wf2[lc1,c2,r]
   // Prs*Ars
   auto qt3n = oper_normxwf_opB("cr",wf2,c2qops,rqops,index,iformula);
   qt3n = qt3n.merge_cr().split_lc(wf.info.qrow, wf.info.qmid); 
   auto Hwf1 = oper_compxwf_opQ("lc",qt3n,lqops,c1qops,int2e,index);
   // (Prs*Ars)^H
   auto qt3h = oper_normxwf_opB("cr",wf2,c2qops,rqops,index,iformula,dagger);
   qt3h = qt3h.merge_cr().split_lc(wf.info.qrow, wf.info.qmid); 
   Hwf1 += oper_compxwf_opQ("lc",qt3h,lqops,c1qops,int2e,index,dagger);
   const Tm wt = lqops.ifkr? wfacBQ(index) : wfac(index);
   Hwf1 *= wt;
   return Hwf1;
}

// --- driver ---

template <typename Tm>
stensor3<Tm> twodot_Hx_local(const oper_dict<Tm>& lqops,
	          	     const oper_dict<Tm>& rqops,
			     const oper_dict<Tm>& c1qops,
	          	     const oper_dict<Tm>& c2qops,
			     const double& ecore,
			     const stensor4<Tm>& wf,
	          	     const int& size,
	          	     const int& rank,
			     const bool& ifdist1){
   if(debug_twodot_sigma) std::cout << "twodot_Hx_local" << std::endl;
   const Tm scale = lqops.ifkr? 0.5 : 1.0;
   // 1. H^LC1
   auto wf1 = wf.merge_c2r(); // wf1[l,c1,c2r]
   auto Hwf1 = oper_compxwf_opH("lc",wf1,lqops,c1qops,size,rank,ifdist1);
   // 2. H^C2R
   auto wf2 = wf.merge_lc1(); // wf2[lc1,r,c2]
   auto Hwf2 = oper_compxwf_opH("cr",wf2,c2qops,rqops,size,rank,ifdist1);
   Hwf1 += Hwf2.split_lc1(wf.info.qrow, wf.info.qmid).merge_c2r();
   // scale for kr case
   Hwf1 *= scale; 
   // 3. add const term
   if(rank == 0){
      const Tm fac = scale*ecore;
      linalg::xaxpy(wf1.size(), fac, wf1.data(), Hwf1.data());
   }
   return Hwf1; 
}

// Collect all Hx_funs 
template <typename Tm>
Hx_functors<Tm> twodot_Hx_functors(const oper_dictmap<Tm>& qops_dict,
	                           const integral::two_body<Tm>& int2e,
				   const double& ecore,
	                           const stensor4<Tm>& wf,
	                           const int& size,
	                           const int& rank,
				   const bool& ifdist1,
				   const bool debug=false){
   const auto& lqops = qops_dict.at("l");
   const auto& rqops = qops_dict.at("r");
   const auto& c1qops = qops_dict.at("c1");
   const auto& c2qops = qops_dict.at("c2");
   Hx_functors<Tm> Hx_funs;
   // Local terms:
   Hx_functor<Tm> Hx("Hloc", 0, 0);
   Hx.opxwf = bind(&twodot_Hx_local<Tm>, 
		   std::cref(lqops), std::cref(rqops), std::cref(c1qops), std::cref(c2qops), 
		   std::cref(ecore), std::cref(wf), std::cref(size), std::cref(rank),
		   std::cref(ifdist1));
   Hx_funs.push_back(Hx);
   // One-index terms:
   // 3. sum_p1 p1^+[LC1]*Sp1^[C2R] + h.c.
   auto infoC1 = oper_combine_opC(lqops.cindex, c1qops.cindex);
   for(const auto& pr : infoC1){
      int index = pr.first;
      int iformula = pr.second;
      Hx_functor<Tm> Hx("CS", index, iformula);
      Hx.opxwf = bind(&twodot_Hx_CS<Tm>, index, iformula, 
           	      std::cref(lqops), std::cref(rqops), std::cref(c1qops), std::cref(c2qops), 
           	      std::cref(int2e), std::cref(wf), std::cref(size), std::cref(rank),
		      std::cref(ifdist1));
      Hx_funs.push_back(Hx); 
   }
   // 4. sum_q2 q2^+[C2R]*Sq2^[LC1] + h.c. = -Sq2^[LC1]*q2^+[C2R] + h.c.
   auto infoC2 = oper_combine_opC(c2qops.cindex, rqops.cindex);
   for(const auto& pr : infoC2){
      int index = pr.first;
      int iformula = pr.second;
      Hx_functor<Tm> Hx("SC", index, iformula);
      Hx.opxwf = bind(&twodot_Hx_SC<Tm>, index, iformula, 
           	      std::cref(lqops), std::cref(rqops), std::cref(c1qops), std::cref(c2qops),  
           	      std::cref(int2e), std::cref(wf), std::cref(size), std::cref(rank),
		      std::cref(ifdist1));
      Hx_funs.push_back(Hx); 
   }
   // Two-index terms:
   int slc1 = lqops.cindex.size() + c1qops.cindex.size();
   int sc2r = c2qops.cindex.size() + rqops.cindex.size();
   const bool ifNC = (slc1 <= sc2r);
   const bool ifkr = lqops.ifkr;
   auto ainfo = ifNC? oper_combine_opA(lqops.cindex, c1qops.cindex, ifkr) :
      		      oper_combine_opA(c2qops.cindex, rqops.cindex, ifkr);
   auto binfo = ifNC? oper_combine_opB(lqops.cindex, c1qops.cindex, ifkr) :
      		      oper_combine_opB(c2qops.cindex, rqops.cindex, ifkr);
   auto afun = ifNC? &twodot_Hx_AP<Tm> : &twodot_Hx_PA<Tm>; 
   auto bfun = ifNC? &twodot_Hx_BQ<Tm> : &twodot_Hx_QB<Tm>;
   auto alabel = ifNC? "AP" : "PA";
   auto blabel = ifNC? "BQ" : "QB"; 
   // 5. Apq^LC1*Ppq^C2R + h.c. or Ars^C2R*Prs^LC1 + h.c.
   for(const auto& pr : ainfo){
      int index = pr.first;
      int iformula = pr.second;
      int iproc = distribute2(ifkr,size,index);
      if(iproc == rank){
         Hx_functor<Tm> Hx(alabel, index, iformula);
         Hx.opxwf = bind(afun, index, iformula,
              	         std::cref(lqops), std::cref(rqops), std::cref(c1qops), std::cref(c2qops), 
              	         std::cref(int2e), std::cref(wf), std::cref(size), std::cref(rank));
         Hx_funs.push_back(Hx); 
      } // iproc
   }
   // 6. Bps^LC1*Qps^C2R + h.c. or Qqr^LC1*Bqr^C2R
   for(const auto& pr : binfo){
      int index = pr.first;
      int iformula = pr.second;
      int iproc = distribute2(ifkr,size,index);
      if(iproc == rank){
         Hx_functor<Tm> Hx(blabel, index, iformula);
         Hx.opxwf = bind(bfun, index, iformula, 
              	         std::cref(lqops), std::cref(rqops), std::cref(c1qops), std::cref(c2qops), 
              	         std::cref(int2e), std::cref(wf), std::cref(size), std::cref(rank));
         Hx_funs.push_back(Hx);
      } // iproc
   }
   // debug
   if(rank == 0 and debug){
      std::cout << "twodot_Hx_functors: size=" << Hx_funs.size() 
                << " CS:" << infoC1.size()
                << " SC:" << infoC2.size()
                << " " << alabel << ":" << ainfo.size()
                << " " << blabel << ":" << binfo.size()
                << std::endl; 
   }
   return Hx_funs;
}

template <typename Tm> 
void twodot_Hx(Tm* y,
	       const Tm* x,
	       Hx_functors<Tm>& Hx_funs,
	       stensor4<Tm>& wf,
	       const int size,
	       const int rank){
   auto t0 = tools::get_time();
#ifdef _OPENMP
   int maxthreads = omp_get_max_threads();
#else
   int maxthreads = 1;
#endif
   if(rank == 0 && debug_twodot_sigma){ 
      std::cout << "ctns::twodot_Hx size=" << size 
                << " maxthreads=" << maxthreads
                << std::endl;
   }
   //=======================
   // Parallel evaluation
   //=======================
   wf.from_array(x);
   // initialization
   auto wf1 = wf.merge_c2r(); // wf1[l,c1,c2r]
   std::vector<stensor3<Tm>> Hwfs(maxthreads);
   for(int i=0; i<maxthreads; i++){
      Hwfs[i].init(wf1.info);
   }
   auto t1 = tools::get_time();
   // compute
#ifdef _OPENMP
   #pragma omp parallel for schedule(dynamic)
#endif
   for(int i=0; i<Hx_funs.size(); i++){
#ifdef _OPENMP
      int omprank = omp_get_thread_num();
#else
      int omprank = 0;
#endif
      Hwfs[omprank] += Hx_funs[i]();
   }
   auto t2 = tools::get_time();
   // reduction & save
   for(int i=1; i<maxthreads; i++){
      Hwfs[0] += Hwfs[i];
   }
   auto Hwf = Hwfs[0].split_c2r(wf.info.qver, wf.info.qcol);
   Hwf.to_array(y);
   auto t3 = tools::get_time();
   oper_timer.tHxInit += tools::get_duration(t1-t0);
   oper_timer.tHxCalc += tools::get_duration(t2-t1);
   oper_timer.tHxFinl += tools::get_duration(t3-t2);
}

} // ctns

#endif
