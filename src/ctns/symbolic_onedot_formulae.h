#ifndef SYMBOLIC_ONEDOT_FORMULAE_H
#define SYMBOLIC_ONEDOT_FORMULAE_H

#include "symbolic_oper.h"
#include "symbolic_normxwf.h"
#include "symbolic_compxwf.h"

namespace ctns{

// generate all formulea for constructing H*x as a list of terms
// organizing principle: recursive partition
template <typename Tm>
symbolic_task<Tm> symbolic_onedot_formulae(const oper_dict<Tm>& lqops,
	                                   const oper_dict<Tm>& rqops,
	                                   const oper_dict<Tm>& cqops,
	                                   const integral::two_body<Tm>& int2e,
	                                   const int& size,
	                                   const int& rank){
   const int level = 0;
   const bool debug = true;
   std::cout << "symbolic_onedot_formulae" << std::endl;

   const bool ifkr = lqops.ifkr;
   const int isym = lqops.isym;
   symbolic_task<Tm> formulae;

   // Local terms:
   const double scale = ifkr? 0.5 : 1.0;
   const bool ifNC = lqops.cindex.size() <= rqops.cindex.size();
   auto aindex = ifNC? oper_index_opA(lqops.cindex, ifkr) : oper_index_opA(rqops.cindex, ifkr);
   auto bindex = ifNC? oper_index_opB(lqops.cindex, ifkr) : oper_index_opB(rqops.cindex, ifkr);
   int idx = 0;
   if(ifNC){
      // partition = l|cr
      // 1. H^l + 2. H^cr
      auto Hl = symbolic_term<Tm>(symbolic_oper("l","H",0), scale);
      formulae.add(Hl);
      auto Hcr = symbolic_compxwf_opH<Tm>("c", "r", cqops.cindex, rqops.cindex, 
		                          ifkr, size, rank, scale);
      formulae.join(Hcr);
      if(rank == 0){
	 std::cout << " idx=" << idx++ << " ";
	 formulae.display("Hl+Hcr", level);
      }
      // One-index terms:
      // 3. p1^l+*Sp1^cr + h.c.
      for(const auto& index : lqops.cindex){
         auto Cl = symbolic_task<Tm>(symbolic_term<Tm>(symbolic_oper("l","C",index)));
         auto Scr = symbolic_compxwf_opS<Tm>("c", "r", cqops.cindex, rqops.cindex,
           	                             index, ifkr, size, rank);
         auto Cl_Scr = Cl.outer_product(Scr);
         formulae.join(Cl_Scr);
         if(rank == 0){ 
	    std::cout << " idx=" << idx++ << " ";
            Cl_Scr.display("Cl_Scr["+std::to_string(index)+"]", level);
	 }
      }
      // 4. q2^cr+*Sq2^l + h.c. = -Sq2^l*q2^cr + h.c.
      auto infoC = oper_combine_opC(cqops.cindex, rqops.cindex);
      for(const auto& pr : infoC){
         int index = pr.first;
         int iformula = pr.second;
         auto Sl = symbolic_task<Tm>(symbolic_term<Tm>(symbolic_oper("l","S",index)));
         auto Ccr = symbolic_normxwf_opC<Tm>("c", "r", index, iformula);
	 Ccr.scale(-1.0);
         auto Sl_Ccr = Sl.outer_product(Ccr);
         formulae.join(Sl_Ccr);
         if(rank == 0){ 
	    std::cout << " idx=" << idx++ << " ";
            Sl_Ccr.display("Sl_Ccr["+std::to_string(index)+"]", level);
	 }
      }
      // 5. Apq^l*Ppq^cr + h.c.
      for(const auto& index : aindex){
         int iproc = distribute2(index,size);
         if(iproc == rank){
            auto Al = symbolic_task<Tm>(symbolic_term<Tm>(symbolic_oper("l","A",index)));
            auto Pcr = symbolic_compxwf_opP<Tm>("c", "r", cqops.cindex, rqops.cindex,
	 				        int2e, index, isym, ifkr);
	    const double wt = ifkr? wfacAP(index) : 1.0;
	    Pcr.scale(wt);
            auto Al_Pcr = Al.outer_product(Pcr);
            formulae.join(Al_Pcr);
            if(rank == 0){ 
	       std::cout << " idx=" << idx++ << " ";
	       Al_Pcr.display("Al_Pcr["+std::to_string(index)+"]", level);
	    }
         } // iproc
      }
      // 6. Bps^l*Qps^cr (using Hermicity)
      for(const auto& index : bindex){
         int iproc = distribute2(index,size);
         if(iproc == rank){
            auto Bl = symbolic_task<Tm>(symbolic_term<Tm>(symbolic_oper("l","B",index)));
	    auto Qcr = symbolic_compxwf_opQ<Tm>("c", "r", cqops.cindex, rqops.cindex,
	           	                        int2e, index, isym, ifkr);
            const double wt = ifkr? wfacBQ(index) : wfac(index);
	    Qcr.scale(wt);
	    auto Bl_Qcr = Bl.outer_product(Qcr);
	    formulae.join(Bl_Qcr);
	    if(rank == 0){ 
	       std::cout << " idx=" << idx++ << " ";
	       Bl_Qcr.display("Bl_Qcr["+std::to_string(index)+"]", level);
	    }
	 } // iproc
      }
   }else{
      // partition = lc|r
      // 1. H^lc + 2. H^r
      auto Hlc = symbolic_compxwf_opH<Tm>("l", "c", lqops.cindex, cqops.cindex, 
           	                          ifkr, size, rank, scale);
      formulae.join(Hlc);
      auto Hr = symbolic_term<Tm>(symbolic_oper("r","H",0), scale);
      if(rank == 0){ 
	 std::cout << " idx=" << idx++ << " ";
	 formulae.display("Hlc+Hr", level);
      }
      // One-index terms:
      // 3. q2^r+*Sq2^lc + h.c. = -Sq2^lc*q2^r + h.c.
      for(const auto& index : rqops.cindex){
         auto Slc = symbolic_compxwf_opS<Tm>("l", "c", lqops.cindex, cqops.cindex,
			 		     index, ifkr, size, rank);
	 Slc.scale(-1.0);
	 auto Cr = symbolic_task<Tm>(symbolic_term<Tm>(symbolic_oper("r","C",index)));
	 auto Slc_Cr = Slc.outer_product(Cr);
	 formulae.join(Slc_Cr);
	 if(rank == 0){ 
	    std::cout << " idx=" << idx++ << " ";
            Slc_Cr.display("Slc_Cr["+std::to_string(index)+"]", level);
	 }
      }
      // 4. p1^lc+*Sp1^r + h.c.
      auto infoC = oper_combine_opC(lqops.cindex, cqops.cindex);
      for(const auto& pr : infoC){
         int index = pr.first;
         int iformula = pr.second;
         auto Clc = symbolic_normxwf_opC<Tm>("l", "c", index, iformula);
         auto Sr = symbolic_task<Tm>(symbolic_term<Tm>(symbolic_oper("r","S",index)));
         auto Clc_Sr = Clc.outer_product(Sr);
         formulae.join(Clc_Sr);
         if(rank == 0){ 
	    std::cout << " idx=" << idx++ << " ";
	    Clc_Sr.display("Clc_Sr["+std::to_string(index)+"]", level);
	 }
      }
      // 5. Ars^r*Prs^lc + h.c.
      for(const auto& index : aindex){
         int iproc = distribute2(index,size);
         if(iproc == rank){
            auto Plc = symbolic_compxwf_opP<Tm>("l", "c", lqops.cindex, cqops.cindex,
	 				        int2e, index, isym, ifkr);
            auto Ar = symbolic_task<Tm>(symbolic_term<Tm>(symbolic_oper("r","A",index)));
	    const double wt = ifkr? wfacAP(index) : 1.0;
	    Plc.scale(wt);
            auto Plc_Ar = Plc.outer_product(Ar);
            formulae.join(Plc_Ar);
            if(rank == 0){ 
	       std::cout << " idx=" << idx++ << " ";
	       Plc_Ar.display("Plc_Ar["+std::to_string(index)+"]", level);
	    }
         } // iproc
      }
      // 6. Qqr^lc*Bqr^r (using Hermicity)
      for(const auto& index : bindex){
         int iproc = distribute2(index,size);
         if(iproc == rank){
	    auto Qlc = symbolic_compxwf_opQ<Tm>("l", "c", lqops.cindex, cqops.cindex,
	           	                        int2e, index, isym, ifkr);
            auto Br = symbolic_task<Tm>(symbolic_term<Tm>(symbolic_oper("r","B",index)));
            const double wt = ifkr? wfacBQ(index) : wfac(index);
	    Qlc.scale(wt);
	    auto Qlc_Br = Qlc.outer_product(Br);
	    formulae.join(Qlc_Br);
	    if(rank == 0){ 
	       std::cout << " idx=" << idx++ << " ";
	       Qlc_Br.display("Qlc_Br["+std::to_string(index)+"]", level);
	    }
	 } // iproc
      }
   }

   if(rank == 0) formulae.display("total", debug);
   return formulae;
}

} // ctns

#endif
