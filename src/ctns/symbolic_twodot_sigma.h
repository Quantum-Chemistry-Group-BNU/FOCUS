#ifndef SYMBOLIC_TWODOT_SIGMA_H
#define SYMBOLIC_TWODOT_SIGMA_H

#ifdef _OPENMP
#include <omp.h>
#endif

#include "symbolic_oper.h"
#include "symbolic_normxwf.h"
#include "symbolic_compxwf.h"

namespace ctns{

// generate all formulea for constructing H*x as a list of terms
// organizing principle: recursive partition
template <typename Tm>
symbolic_task<Tm> symbolic_twodot_Hx_functors(const oper_dict<Tm>& lqops,
	                                      const oper_dict<Tm>& rqops,
			                      const oper_dict<Tm>& c1qops,
	                                      const oper_dict<Tm>& c2qops,
	                                      const integral::two_body<Tm>& int2e,
	                                      const int& size,
	                                      const int& rank){
   const int level = 0;
   const bool debug = true;
   std::cout << "symbolic_twodot_Hx_functors" << std::endl;

   const bool ifkr = lqops.ifkr;
   const int isym = lqops.isym;
   symbolic_task<Tm> formulae;

   // Local terms:
   const double scale = ifkr? 0.5 : 1.0;
   // H[lc1]
   auto Hlc1 = symbolic_compxwf_opH<Tm>("l", "c1", lqops.cindex, c1qops.cindex, 
		                        ifkr, size, rank, scale);
   if(rank == 0) Hlc1.display("Hlc1", level);
   formulae.join(Hlc1);
   // H[c2r]
   auto Hc2r = symbolic_compxwf_opH<Tm>("c2", "r", c2qops.cindex, rqops.cindex, 
		                        ifkr, size, rank, scale);
   if(rank == 0) Hc2r.display("Hc2r", level);
   formulae.join(Hc2r);

   // One-index terms:
   // 3. sum_p1 p1^+[LC1]*Sp1^[C2R] + h.c.
   auto infoC1 = oper_combine_opC(lqops.cindex, c1qops.cindex);
   for(const auto& pr : infoC1){
      int index = pr.first;
      int iformula = pr.second;
      // p1^L1C1+*Sp1^C2R & -p1^L1C1*Sp1^C2R+
      auto Clc1 = symbolic_normxwf_opC<Tm>("l", "c1", index, iformula);
      auto Sc2r = symbolic_compxwf_opS<Tm>("c2", "r", c2qops.cindex, rqops.cindex,
		                           index, ifkr, size, rank);
      auto Clc1_Sc2r = Clc1.outer_product(Sc2r);
      if(rank == 0) Clc1_Sc2r.display("Clc1_Sc2r : "+std::to_string(index), level);
      formulae.join(Clc1_Sc2r);
   }
   // 4. sum_q2 q2^+[C2R]*Sq2^[LC1] + h.c. = -Sq2^[LC1]*q2^+[C2R] + h.c.
   auto infoC2 = oper_combine_opC(c2qops.cindex, rqops.cindex);
   for(const auto& pr : infoC2){
      int index = pr.first;
      int iformula = pr.second;
      // q2^C2R+*Sq2^LC1 = -Sq2^LC1*q2^C2R+ & Sq2^LC1+*q2^C2R
      auto Slc1 = symbolic_compxwf_opS<Tm>("l", "c1", lqops.cindex, c1qops.cindex,
		                           index, ifkr, size, rank);
      auto Cc2r = symbolic_normxwf_opC<Tm>("c2", "r", index, iformula);
      auto Slc1_Cc2r = Slc1.outer_product(Cc2r);
      if(rank == 0) Slc1_Cc2r.display("Slc1_Cc2r : "+std::to_string(index), level);
      formulae.join(Slc1_Cc2r);
   }

   // Two-index terms:
   int slc1 = lqops.cindex.size() + c1qops.cindex.size();
   int sc2r = c2qops.cindex.size() + rqops.cindex.size();
   const bool ifNC = (slc1 <= sc2r);
   auto ainfo = ifNC? oper_combine_opA(lqops.cindex, c1qops.cindex, ifkr) :
      		      oper_combine_opA(c2qops.cindex, rqops.cindex, ifkr);
   auto binfo = ifNC? oper_combine_opB(lqops.cindex, c1qops.cindex, ifkr) :
      		      oper_combine_opB(c2qops.cindex, rqops.cindex, ifkr);
   // 5. Apq^LC1*Ppq^C2R + h.c. or Ars^C2R*Prs^LC1 + h.c.
   for(const auto pr : ainfo){
      int index = pr.first;
      int iformula = pr.second;
      int iproc = distribute2(index,size);
      if(iproc == rank){
	 // Apq*Ppq + Apq^+*Ppq^+
         auto Alc1 = symbolic_normxwf_opA<Tm>("l", "c1", index, iformula);
         auto Pc2r = symbolic_compxwf_opP<Tm>("c2", "r", c2qops.cindex, rqops.cindex,
	 				      int2e, index, isym, ifkr);
	 const double wt = ifkr? wfacAP(index) : 1.0;
	 Pc2r.scale(wt);
	 auto Alc1_Pc2r = Alc1.outer_product(Pc2r);
         if(rank == 0) Alc1_Pc2r.display("Alc1_Pc2r : "+std::to_string(index), level);
	 formulae.join(Alc1_Pc2r);
      } // iproc
   }
   // 6. Bps^LC1*Qps^C2R + h.c. or Qqr^LC1*Bqr^C2R
   for(const auto pr : binfo){
      int index = pr.first;
      int iformula = pr.second;
      int iproc = distribute2(index,size);
      if(iproc == rank){
	 auto Blc1 = symbolic_normxwf_opB<Tm>("l", "c1", index, iformula);
	 auto Qc2r = symbolic_compxwf_opQ<Tm>("c2", "r", c2qops.cindex, rqops.cindex,
			                      int2e, index, isym, ifkr);
	 // Bpq*Qpq + Bpq^+*Qpq^+
         const double wt = ifkr? wfacBQ(index) : wfac(index);
	 Qc2r.scale(wt);
	 auto Blc1_Qc2r = Blc1.outer_product(Qc2r);
	 if(rank == 0) Blc1_Qc2r.display("Blc1_Qc2r : "+std::to_string(index), level);
	 formulae.join(Blc1_Qc2r);
      } // iproc
   }

   if(rank == 0) formulae.display("total", debug);
   return formulae;
}

const bool debug_twodot_sigma1 = false; 
extern const bool debug_twodot_sigma1;

template <typename Tm> 
void twodot_Hx1(Tm* y,
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
   if(rank == 0 && debug_twodot_sigma1){ 
      std::cout << "ctns::twodot_Hx1 size=" << size 
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
