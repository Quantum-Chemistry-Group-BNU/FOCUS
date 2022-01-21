#ifndef SYMBOLIC_TWODOT_FORMULAE_H
#define SYMBOLIC_TWODOT_FORMULAE_H

#include "symbolic_oper.h"
#include "symbolic_normxwf.h"
#include "symbolic_compxwf.h"

namespace ctns{

// generate all formulea for constructing H*x as a list of terms
// organizing principle: recursive partition
template <typename Tm>
symbolic_task<Tm> symbolic_twodot_formulae(const oper_dict<Tm>& lqops,
	                                   const oper_dict<Tm>& rqops,
			                   const oper_dict<Tm>& c1qops,
	                                   const oper_dict<Tm>& c2qops,
	                                   const integral::two_body<Tm>& int2e,
	                                   const int& size,
	                                   const int& rank){
   const bool debug = true;
   const int print_level = 0;
   const int isym = lqops.isym;
   const bool ifkr = lqops.ifkr;
   int slc1 = lqops.cindex.size() + c1qops.cindex.size();
   int sc2r = c2qops.cindex.size() + rqops.cindex.size();
   const bool ifNC = (slc1 <= sc2r);
   auto ainfo = ifNC? oper_combine_opA(lqops.cindex, c1qops.cindex, ifkr) :
      		      oper_combine_opA(c2qops.cindex, rqops.cindex, ifkr);
   auto binfo = ifNC? oper_combine_opB(lqops.cindex, c1qops.cindex, ifkr) :
      		      oper_combine_opB(c2qops.cindex, rqops.cindex, ifkr);
   if(debug) std::cout << "symbolic_twodot_formulae"
	               << " isym=" << isym
		       << " ifkr=" << ifkr
		       << " ifNC=" << ifNC
	               << std::endl;

   symbolic_task<Tm> formulae;
   int idx = 0;
   // Local terms:
   // H[lc1]
   auto Hlc1 = symbolic_compxwf_opH<Tm>("l", "c1", lqops.cindex, c1qops.cindex, 
		                        ifkr, size, rank);
   formulae.join(Hlc1);
   if(rank == 0){ 
      std::cout << " idx=" << idx++ << " "; 
      Hlc1.display("Hlc1", print_level);
   }
   // H[c2r]
   auto Hc2r = symbolic_compxwf_opH<Tm>("c2", "r", c2qops.cindex, rqops.cindex, 
		                        ifkr, size, rank);
   formulae.join(Hc2r);
   if(rank == 0){ 
      std::cout << " idx=" << idx++ << " ";
      Hc2r.display("Hc2r", print_level);
   }
/*
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
      formulae.join(Clc1_Sc2r);
      if(rank == 0){ 
	 std::cout << " idx=" << idx++ << " ";
	 Clc1_Sc2r.display("Clc1_Sc2r["+std::to_string(index)+"]", print_level);
      }
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
      Cc2r.scale(-1.0);
      auto Slc1_Cc2r = Slc1.outer_product(Cc2r);
      formulae.join(Slc1_Cc2r);
      if(rank == 0){ 
	 std::cout << " idx=" << idx++ << " ";
	 Slc1_Cc2r.display("Slc1_Cc2r["+std::to_string(index)+"]", print_level);
      }
   }

   // Two-index terms:
   // 5. Apq^LC1*Ppq^C2R + h.c. or Ars^C2R*Prs^LC1 + h.c.
   for(const auto pr : ainfo){
      int index = pr.first;
      int iformula = pr.second;
      int iproc = distribute2(index,size);
      if(iproc == rank){
	 // Apq*Ppq + Apq^+*Ppq^+
         auto Alc1 = symbolic_normxwf_opA<Tm>("l", "c1", index, iformula, ifkr);
         auto Pc2r = symbolic_compxwf_opP<Tm>("c2", "r", c2qops.cindex, rqops.cindex,
	 				      int2e, index, isym, ifkr);
	 const double wt = ifkr? wfacAP(index) : 1.0;
	 Pc2r.scale(wt);
	 auto Alc1_Pc2r = Alc1.outer_product(Pc2r);
	 formulae.join(Alc1_Pc2r);
         if(rank == 0){ 
	    std::cout << " idx=" << idx++ << " ";
	    Alc1_Pc2r.display("Alc1_Pc2r["+std::to_string(index)+"]", print_level);
	 }
      } // iproc
   }
   // 6. Bps^LC1*Qps^C2R + h.c. or Qqr^LC1*Bqr^C2R
   for(const auto pr : binfo){
      int index = pr.first;
      int iformula = pr.second;
      int iproc = distribute2(index,size);
      if(iproc == rank){
	 auto Blc1 = symbolic_normxwf_opB<Tm>("l", "c1", index, iformula, ifkr);
	 auto Qc2r = symbolic_compxwf_opQ<Tm>("c2", "r", c2qops.cindex, rqops.cindex,
			                      int2e, index, isym, ifkr);
	 // Bpq*Qpq + Bpq^+*Qpq^+
         const double wt = ifkr? wfacBQ(index) : wfac(index);
	 Qc2r.scale(wt);
	 auto Blc1_Qc2r = Blc1.outer_product(Qc2r);
	 if(rank == 0){ 
	    std::cout << " idx=" << idx++ << " ";
            Blc1_Qc2r.display("Blc1_Qc2r["+std::to_string(index)+"]", print_level);
	 }
	 formulae.join(Blc1_Qc2r);
      } // iproc
   }
*/
   if(rank == 0) formulae.display("total", debug);
   return formulae;
}

} // ctns

#endif
