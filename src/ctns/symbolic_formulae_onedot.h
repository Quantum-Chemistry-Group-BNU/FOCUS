#ifndef SYMBOLIC_FORMULAE_ONEDOT_H
#define SYMBOLIC_FORMULAE_ONEDOT_H

#include "../core/tools.h"
#include "oper_dict.h"
#include "symbolic_task.h"
#include "symbolic_normxwf.h"
#include "symbolic_compxwf.h"

namespace ctns{

template <typename Tm>
symbolic_task<Tm> preprocess_formulae_onedot(const std::vector<int>& cindex_l,
					     const std::vector<int>& cindex_r,
					     const std::vector<int>& cindex_c,
					     const int isym,
					     const bool ifkr,
	                                     const integral::two_body<Tm>& int2e,
	                                     const int& size,
	                                     const int& rank,
					     const bool ifdist1,
					     const bool ifsave,
   				             std::map<std::string,int>& counter){
   const int print_level = 1;
   const bool ifNC = cindex_l.size() <= cindex_r.size();
   const auto& cindex = ifNC? cindex_l : cindex_r;
   auto aindex = oper_index_opA(cindex, ifkr);
   auto bindex = oper_index_opB(cindex, ifkr);
   
   symbolic_task<Tm> formulae;
   int idx = 0;
   counter["CS"] = 0;
   counter["SC"] = 0;
   
   if(ifNC){
      // partition = l|cr
      counter["AP"] = 0;
      counter["BQ"] = 0;
      // 1. H^l 
      const double scale = ifkr? 0.25 : 0.5;
      auto Hl = symbolic_prod<Tm>(symbolic_oper("l",'H',0), scale);
      formulae.append(Hl);
      // 2. H^cr
      auto Hcr = symbolic_compxwf_opH<Tm>("c", "r", cindex_c, cindex_r, 
		                          ifkr, size, rank, ifdist1);
      formulae.join(Hcr);
      if(ifsave){
	 std::cout << "idx=" << idx++;
	 formulae.display("Hl+Hcr", print_level);
      }
      // One-index terms:
      // 3. p1^l+*Sp1^cr + h.c.
      for(const auto& index : cindex_l){
         auto Cl = symbolic_task<Tm>(symbolic_prod<Tm>(symbolic_oper("l",'C',index)));
         auto Scr = symbolic_compxwf_opS<Tm>("c", "r", cindex_c, cindex_r,
           	                             int2e, index, isym, ifkr, size, rank, ifdist1);
         auto Cl_Scr = Cl.outer_product(Scr);
         formulae.join(Cl_Scr);
         if(ifsave){ 
	    std::cout << "idx=" << idx++;
            Cl_Scr.display("Cl_Scr["+std::to_string(index)+"]", print_level);
	    counter["CS"] += 1;
	 }
      }
      // 4. q2^cr+*Sq2^l + h.c. = -Sq2^l*q2^cr + h.c.
      auto infoC = oper_combine_opC(cindex_c, cindex_r);
      for(const auto& pr : infoC){
         int index = pr.first;
	 int iproc = distribute1(index,size);
	 if(!ifdist1 or iproc==rank){ 
            int iformula = pr.second;
            auto Sl = symbolic_task<Tm>(symbolic_prod<Tm>(symbolic_oper("l",'S',index)));
            auto Ccr = symbolic_normxwf_opC<Tm>("c", "r", index, iformula);
	    Ccr.scale(-1.0);
            auto Sl_Ccr = Sl.outer_product(Ccr);
            formulae.join(Sl_Ccr);
            if(ifsave){ 
	       std::cout << "idx=" << idx++;
               Sl_Ccr.display("Sl_Ccr["+std::to_string(index)+"]", print_level);
	       counter["SC"] += 1;
	    }
	 }
      }
      // 5. Apq^l*Ppq^cr + h.c.
      for(const auto& index : aindex){
         int iproc = distribute2(index,size);
         if(iproc == rank){
            auto Al = symbolic_task<Tm>(symbolic_prod<Tm>(symbolic_oper("l",'A',index)));
            auto Pcr = symbolic_compxwf_opP<Tm>("c", "r", cindex_c, cindex_r,
	 				        int2e, index, isym, ifkr);
	    const double wt = ifkr? wfacAP(index) : 1.0;
	    Pcr.scale(wt);
            auto Al_Pcr = Al.outer_product(Pcr);
            formulae.join(Al_Pcr);
            if(ifsave){ 
	       std::cout << "idx=" << idx++;
	       Al_Pcr.display("Al_Pcr["+std::to_string(index)+"]", print_level);
	       counter["AP"] += 1;
	    }
         } // iproc
      }
      // 6. Bps^l*Qps^cr (using Hermicity)
      for(const auto& index : bindex){
         int iproc = distribute2(index,size);
         if(iproc == rank){
            auto Bl = symbolic_task<Tm>(symbolic_prod<Tm>(symbolic_oper("l",'B',index)));
	    auto Qcr = symbolic_compxwf_opQ<Tm>("c", "r", cindex_c, cindex_r,
	           	                        int2e, index, isym, ifkr);
            const double wt = ifkr? wfacBQ(index) : wfac(index);
	    Qcr.scale(wt);
	    auto Bl_Qcr = Bl.outer_product(Qcr);
	    formulae.join(Bl_Qcr);
	    if(ifsave){ 
	       std::cout << "idx=" << idx++;
	       Bl_Qcr.display("Bl_Qcr["+std::to_string(index)+"]", print_level);
	       counter["BQ"] += 1;
	    }
	 } // iproc
      }
   }else{
      // partition = lc|r
      counter["PA"] = 0;
      counter["QB"] = 0;
      // 1. H^lc 
      auto Hlc = symbolic_compxwf_opH<Tm>("l", "c", cindex_l, cindex_c, 
           	                          ifkr, size, rank, ifdist1);
      formulae.join(Hlc);
      // 2. H^r
      const double scale = ifkr? 0.25 : 0.5;
      auto Hr = symbolic_prod<Tm>(symbolic_oper("r",'H',0), scale);
      formulae.append(Hr);
      if(ifsave){ 
	 std::cout << "idx=" << idx++;
	 formulae.display("Hlc+Hr", print_level);
      }
      // One-index terms:
      // 3. q2^r+*Sq2^lc + h.c. = -Sq2^lc*q2^r + h.c.
      for(const auto& index : cindex_r){
         auto Slc = symbolic_compxwf_opS<Tm>("l", "c", cindex_l, cindex_c,
			 		     int2e, index, isym, ifkr, size, rank, ifdist1);
	 Slc.scale(-1.0);
	 auto Cr = symbolic_task<Tm>(symbolic_prod<Tm>(symbolic_oper("r",'C',index)));
	 auto Slc_Cr = Slc.outer_product(Cr);
	 formulae.join(Slc_Cr);
	 if(ifsave){ 
	    std::cout << "idx=" << idx++;
            Slc_Cr.display("Slc_Cr["+std::to_string(index)+"]", print_level);
	    counter["SC"] += 1;
	 }
      }
      // 4. p1^lc+*Sp1^r + h.c.
      auto infoC = oper_combine_opC(cindex_l, cindex_c);
      for(const auto& pr : infoC){
         int index = pr.first;
	 int iproc = distribute1(index,size);
	 if(!ifdist1 or iproc==rank){ 
            int iformula = pr.second;
            auto Clc = symbolic_normxwf_opC<Tm>("l", "c", index, iformula);
            auto Sr = symbolic_task<Tm>(symbolic_prod<Tm>(symbolic_oper("r",'S',index)));
            auto Clc_Sr = Clc.outer_product(Sr);
            formulae.join(Clc_Sr);
            if(ifsave){ 
	       std::cout << "idx=" << idx++;
	       Clc_Sr.display("Clc_Sr["+std::to_string(index)+"]", print_level);
	       counter["CS"] += 1;
	    }
	 }
      }
      // 5. Ars^r*Prs^lc + h.c.
      for(const auto& index : aindex){
         int iproc = distribute2(index,size);
         if(iproc == rank){
            auto Plc = symbolic_compxwf_opP<Tm>("l", "c", cindex_l, cindex_c,
	 				        int2e, index, isym, ifkr);
            auto Ar = symbolic_task<Tm>(symbolic_prod<Tm>(symbolic_oper("r",'A',index)));
	    const double wt = ifkr? wfacAP(index) : 1.0;
	    Plc.scale(wt);
            auto Plc_Ar = Plc.outer_product(Ar);
            formulae.join(Plc_Ar);
            if(ifsave){ 
	       std::cout << "idx=" << idx++;
	       Plc_Ar.display("Plc_Ar["+std::to_string(index)+"]", print_level);
	       counter["PA"] += 1;
	    }
         } // iproc
      }
      // 6. Qqr^lc*Bqr^r (using Hermicity)
      for(const auto& index : bindex){
         int iproc = distribute2(index,size);
         if(iproc == rank){
	    auto Qlc = symbolic_compxwf_opQ<Tm>("l", "c", cindex_l, cindex_c,
	           	                        int2e, index, isym, ifkr);
            auto Br = symbolic_task<Tm>(symbolic_prod<Tm>(symbolic_oper("r",'B',index)));
            const double wt = ifkr? wfacBQ(index) : wfac(index);
	    Qlc.scale(wt);
	    auto Qlc_Br = Qlc.outer_product(Br);
	    formulae.join(Qlc_Br);
	    if(ifsave){ 
	       std::cout << "idx=" << idx++;
	       Qlc_Br.display("Qlc_Br["+std::to_string(index)+"]", print_level);
	       counter["QB"] += 1;
	    }
	 } // iproc
      }
   } // ifNC
   return formulae;
}

// primitive form (without factorization)
template <typename Tm>
symbolic_task<Tm> symbolic_formulae_onedot(const oper_dictmap<Tm>& qops_dict,
	                                   const integral::two_body<Tm>& int2e,
	                                   const int& size,
	                                   const int& rank,
					   const std::string fname,
					   const bool sort_formulae,
					   const bool ifdist1){
   auto t0 = tools::get_time();
   const int print_level = 1;
   const auto& lqops = qops_dict.at("l");
   const auto& rqops = qops_dict.at("r");
   const auto& cqops = qops_dict.at("c");
   const auto& cindex_l = lqops.cindex;
   const auto& cindex_r = rqops.cindex;
   const auto& cindex_c = cqops.cindex;
   const bool ifNC = cindex_l.size() <= cindex_r.size();
   const int isym = lqops.isym;
   const bool ifkr = lqops.ifkr;
   std::streambuf *psbuf, *backup;
   std::ofstream file;
   bool ifsave = !fname.empty();
   if(ifsave){
      if(rank == 0){
         std::cout << "ctns::symbolic_formulae_onedot"
                   << " mpisize=" << size
                   << " fname=" << fname 
           	   << std::endl;
      }
      // http://www.cplusplus.com/reference/ios/ios/rdbuf/
      file.open(fname);
      backup = std::cout.rdbuf(); // back up cout's streambuf
      psbuf = file.rdbuf(); // get file's streambuf
      std::cout.rdbuf(psbuf); // assign streambuf to cout
      std::cout << "ctns::symbolic_formulae_onedot"
	        << " isym=" << isym
	  	<< " ifkr=" << ifkr
		<< " mpisize=" << size
		<< " mpirank=" << rank 
	        << std::endl;
   }
   // generation of Hx
   std::map<std::string,int> counter;
   auto formulae = preprocess_formulae_onedot(cindex_l,cindex_r,cindex_c,isym,ifkr,
		   		   	      int2e,size,rank,ifdist1,ifsave,counter);
   // reorder if necessary
   if(sort_formulae){
      std::map<std::string,int> dims = {{"l",lqops.qket.get_dimAll()},
                                        {"r",rqops.qket.get_dimAll()},
                                        {"c",cqops.qket.get_dimAll()}};
      formulae.sort(dims);
   }
   if(ifsave){
      std::cout << "\nSUMMARY size=" << formulae.size();
      if(ifNC){
         std::cout << " CSnc:" << counter["CS"] << " SCnc:" << counter["SC"]
             	   << " APnc:" << counter["AP"] << " BQnc:" << counter["BQ"]
           	   << std::endl;
      }else{
         std::cout << " SCcn:" << counter["SC"] << " CScn:" << counter["CS"]
           	   << " PAcn:" << counter["PA"] << " QBcn:" << counter["QB"]
           	   << std::endl;
      }
      formulae.display("total");
      std::cout.rdbuf(backup); // restore cout's original streambuf
      file.close();
   }
   if(rank == 0){
      auto t1 = tools::get_time();
      int size = formulae.size();
      tools::timing("symbolic_formulae_onedot with size="+std::to_string(size), t0, t1);
   }
   return formulae;
}

// bipartite form (with factorization)
template <typename Tm>
bipart_task<Tm> symbolic_formulae_onedot2(const oper_dictmap<Tm>& qops_dict,
	                                  const integral::two_body<Tm>& int2e,
	                                  const int& size,
	                                  const int& rank,
					  const std::string fname,
					  const bool sort_formulae,
					  const bool ifdist1){
   auto t0 = tools::get_time();
   const int print_level = 1;
   const auto& lqops = qops_dict.at("l");
   const auto& rqops = qops_dict.at("r");
   const auto& cqops = qops_dict.at("c");
   const auto& cindex_l = lqops.cindex;
   const auto& cindex_r = rqops.cindex;
   const auto& cindex_c = cqops.cindex;
   const int isym = lqops.isym;
   const bool ifkr = lqops.ifkr;
   const bool ifNC = cindex_l.size() <= cindex_r.size();
   std::streambuf *psbuf, *backup;
   std::ofstream file;
   bool ifsave = !fname.empty();
   if(ifsave){
      if(rank == 0){
         std::cout << "ctns::symbolic_formulae_onedot2"
                   << " mpisize=" << size
                   << " fname=" << fname 
           	   << std::endl;
      }
      // http://www.cplusplus.com/reference/ios/ios/rdbuf/
      file.open(fname);
      backup = std::cout.rdbuf(); // back up cout's streambuf
      psbuf = file.rdbuf(); // get file's streambuf
      std::cout.rdbuf(psbuf); // assign streambuf to cout
      std::cout << "ctns::symbolic_formulae_onedot2"
	        << " isym=" << isym
	  	<< " ifkr=" << ifkr
		<< " ifNC=" << ifNC
		<< " mpisize=" << size
		<< " mpirank=" << rank 
	        << std::endl;
   }
   const auto& cindex = ifNC? cindex_l : cindex_r;
   auto aindex = oper_index_opA(cindex, ifkr);
   auto bindex = oper_index_opB(cindex, ifkr);
   
   bipart_task<Tm> formulae;
   int idx = 0;
   std::map<std::string,int> counter;
   counter["CS"] = 0;
   counter["SC"] = 0;

   if(ifNC){
      // partition = l|cr
      counter["AP"] = 0;
      counter["BQ"] = 0;
      // 1. H^l 
      const double scale = ifkr? 0.25 : 0.5;
      auto Hl = symbolic_task<Tm>(symbolic_prod<Tm>(symbolic_oper("l",'H',0), scale));
      auto Hl_Ir = bipart_oper('l',Hl,"Hl_Ir");
      assert(Hl_Ir.parity == 0);
      formulae.push_back(Hl_Ir);
      if(ifsave){
         std::cout << "idx=" << idx++;
	 Hl_Ir.display(print_level);
      }
      // 2. H^cr
      auto Hcr = symbolic_compxwf_opH<Tm>("c", "r", cindex_c, cindex_r, 
		                          ifkr, size, rank, ifdist1);
      auto Il_Hcr = bipart_oper('r',Hcr,"Il_Hcr");
      assert(Il_Hcr.parity == 0);
      formulae.push_back(Il_Hcr);
      if(ifsave){
	 std::cout << "idx=" << idx++;
	 Il_Hcr.display(print_level);
      }
      // One-index terms:
      // 3. p1^l+*Sp1^cr + h.c.
      for(const auto& index : cindex_l){
         auto Cl = symbolic_task<Tm>(symbolic_prod<Tm>(symbolic_oper("l",'C',index)));
         auto Scr = symbolic_compxwf_opS<Tm>("c", "r", cindex_c, cindex_r,
           	                             int2e, index, isym, ifkr, size, rank, ifdist1);
         auto Cl_Scr = bipart_oper(Cl,Scr,"Cl_Scr["+std::to_string(index)+"]");
	 assert(Cl_Scr.parity == 0);
	 formulae.push_back(Cl_Scr);
         if(ifsave){ 
	    std::cout << "idx=" << idx++;
            Cl_Scr.display(print_level);
	    counter["CS"] += 1;
	 }
      }
      // 4. q2^cr+*Sq2^l + h.c. = -Sq2^l*q2^cr + h.c.
      auto infoC = oper_combine_opC(cindex_c, cindex_r);
      for(const auto& pr : infoC){
         int index = pr.first;
         int iformula = pr.second;
         auto Sl = symbolic_task<Tm>(symbolic_prod<Tm>(symbolic_oper("l",'S',index)));
         auto Ccr = symbolic_normxwf_opC<Tm>("c", "r", index, iformula);
	 Ccr.scale(-1.0);
	 auto Sl_Ccr = bipart_oper(Sl,Ccr,"Sl_Ccr["+std::to_string(index)+"]");
         assert(Sl_Ccr.parity == 0);
         formulae.push_back(Sl_Ccr);
         if(ifsave){ 
	    std::cout << "idx=" << idx++;
            Sl_Ccr.display(print_level);
	    counter["SC"] += 1;
	 }
      }
      // 5. Apq^l*Ppq^cr + h.c.
      for(const auto& index : aindex){
         int iproc = distribute2(index,size);
         if(iproc == rank){
            auto Al = symbolic_task<Tm>(symbolic_prod<Tm>(symbolic_oper("l",'A',index)));
            auto Pcr = symbolic_compxwf_opP<Tm>("c", "r", cindex_c, cindex_r,
	 				        int2e, index, isym, ifkr);
	    const double wt = ifkr? wfacAP(index) : 1.0;
	    Pcr.scale(wt);
            auto Al_Pcr = bipart_oper(Al,Pcr,"Al_Pcr["+std::to_string(index)+"]");
            assert(Al_Pcr.parity == 0);
            formulae.push_back(Al_Pcr);
            if(ifsave){ 
	       std::cout << "idx=" << idx++;
	       Al_Pcr.display(print_level);
	       counter["AP"] += 1;
	    }
         } // iproc
      }
      // 6. Bps^l*Qps^cr (using Hermicity)
      for(const auto& index : bindex){
         int iproc = distribute2(index,size);
         if(iproc == rank){
            auto Bl = symbolic_task<Tm>(symbolic_prod<Tm>(symbolic_oper("l",'B',index)));
	    auto Qcr = symbolic_compxwf_opQ<Tm>("c", "r", cindex_c, cindex_r,
	           	                        int2e, index, isym, ifkr);
            const double wt = ifkr? wfacBQ(index) : wfac(index);
	    Qcr.scale(wt);
	    auto Bl_Qcr = bipart_oper(Bl,Qcr,"Bl_Qcr["+std::to_string(index)+"]");
            assert(Bl_Qcr.parity == 0);
	    formulae.push_back(Bl_Qcr);
	    if(ifsave){ 
	       std::cout << "idx=" << idx++;
	       Bl_Qcr.display(print_level);
	       counter["BQ"] += 1;
	    }
	 } // iproc
      }
   }else{
      // partition = lc|r
      counter["PA"] = 0;
      counter["QB"] = 0;
      // 1. H^lc 
      auto Hlc = symbolic_compxwf_opH<Tm>("l", "c", cindex_l, cindex_c, 
           	                          ifkr, size, rank, ifdist1);
      auto Hlc_Ir = bipart_oper('l',Hlc,"Hlc_Ir");
      assert(Hlc_Ir.parity == 0);
      formulae.push_back(Hlc_Ir);
      if(ifsave){
         std::cout << "idx=" << idx++;
	 Hlc_Ir.display(print_level);
      }
      // 2. H^r
      const double scale = ifkr? 0.25 : 0.5;
      auto Hr = symbolic_task<Tm>(symbolic_prod<Tm>(symbolic_oper("r",'H',0), scale));
      auto Ilc_Hr = bipart_oper('r',Hr,"Ilc_Hr");
      assert(Ilc_Hr.parity == 0);
      formulae.push_back(Ilc_Hr);
      if(ifsave){
	 std::cout << "idx=" << idx++;
	 Ilc_Hr.display(print_level);
      }
      // One-index terms:
      // 3. q2^r+*Sq2^lc + h.c. = -Sq2^lc*q2^r + h.c.
      for(const auto& index : cindex_r){
         auto Slc = symbolic_compxwf_opS<Tm>("l", "c", cindex_l, cindex_c,
			 		     int2e, index, isym, ifkr, size, rank, ifdist1);
	 Slc.scale(-1.0);
	 auto Cr = symbolic_task<Tm>(symbolic_prod<Tm>(symbolic_oper("r",'C',index)));
	 auto Slc_Cr = bipart_oper(Slc,Cr,"Slc_Cr["+std::to_string(index)+"]");
         assert(Slc_Cr.parity == 0);
	 formulae.push_back(Slc_Cr);
	 if(ifsave){ 
	    std::cout << "idx=" << idx++;
            Slc_Cr.display(print_level);
	    counter["SC"] += 1;
	 }
      }
      // 4. p1^lc+*Sp1^r + h.c.
      auto infoC = oper_combine_opC(cindex_l, cindex_c);
      for(const auto& pr : infoC){
         int index = pr.first;
         int iformula = pr.second;
         auto Clc = symbolic_normxwf_opC<Tm>("l", "c", index, iformula);
         auto Sr = symbolic_task<Tm>(symbolic_prod<Tm>(symbolic_oper("r",'S',index)));
         auto Clc_Sr = bipart_oper(Clc,Sr,"Clc_Sr["+std::to_string(index)+"]");
         assert(Clc_Sr.parity == 0);
         formulae.push_back(Clc_Sr);
         if(ifsave){ 
	    std::cout << "idx=" << idx++;
	    Clc_Sr.display(print_level);
	    counter["CS"] += 1;
	 }
      }
      // 5. Ars^r*Prs^lc + h.c.
      for(const auto& index : aindex){
         int iproc = distribute2(index,size);
         if(iproc == rank){
            auto Plc = symbolic_compxwf_opP<Tm>("l", "c", cindex_l, cindex_c,
	 				        int2e, index, isym, ifkr);
            auto Ar = symbolic_task<Tm>(symbolic_prod<Tm>(symbolic_oper("r",'A',index)));
	    const double wt = ifkr? wfacAP(index) : 1.0;
	    Plc.scale(wt);
            auto Plc_Ar = bipart_oper(Plc,Ar,"Plc_Ar["+std::to_string(index)+"]");
            assert(Plc_Ar.parity == 0);
            formulae.push_back(Plc_Ar);
            if(ifsave){ 
	       std::cout << "idx=" << idx++;
	       Plc_Ar.display(print_level);
	       counter["PA"] += 1;
	    }
         } // iproc
      }
      // 6. Qqr^lc*Bqr^r (using Hermicity)
      for(const auto& index : bindex){
         int iproc = distribute2(index,size);
         if(iproc == rank){
	    auto Qlc = symbolic_compxwf_opQ<Tm>("l", "c", cindex_l, cindex_c,
	           	                        int2e, index, isym, ifkr);
            auto Br = symbolic_task<Tm>(symbolic_prod<Tm>(symbolic_oper("r",'B',index)));
            const double wt = ifkr? wfacBQ(index) : wfac(index);
	    Qlc.scale(wt);
	    auto Qlc_Br = bipart_oper(Qlc,Br,"Qlc_Br["+std::to_string(index)+"]");
            assert(Qlc_Br.parity == 0);
	    formulae.push_back(Qlc_Br);
	    if(ifsave){ 
	       std::cout << "idx=" << idx++;
	       Qlc_Br.display(print_level);
	       counter["QB"] += 1;
	    }
	 } // iproc
      }
   } // ifNC

   if(sort_formulae){
      std::map<std::string,int> dims = {{"l",lqops.qket.get_dimAll()},
                                        {"r",rqops.qket.get_dimAll()},
                                        {"c",cqops.qket.get_dimAll()}};
      sort(formulae, dims);
   }
   if(ifsave){
      std::cout << "\nSUMMARY size=" << idx;
      if(ifNC){
         std::cout << " CSnc:" << counter["CS"] << " SCnc:" << counter["SC"]
             	   << " APnc:" << counter["AP"] << " BQnc:" << counter["BQ"]
           	   << std::endl;
      }else{
         std::cout << " SCcn:" << counter["SC"] << " CScn:" << counter["CS"]
           	   << " PAcn:" << counter["PA"] << " QBcn:" << counter["QB"]
           	   << std::endl;
      }
      display(formulae, "total");
      std::cout.rdbuf(backup); // restore cout's original streambuf
      file.close();
   }
   if(rank == 0){
      auto t1 = tools::get_time();
      int size = formulae.size();
      tools::timing("symbolic_formulae_onedot2 with size="+std::to_string(size), t0, t1);
   }
   return formulae;
}

} // ctns

#endif
