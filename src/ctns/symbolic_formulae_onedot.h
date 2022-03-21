#ifndef SYMBOLIC_ONEDOT_FORMULAE_H
#define SYMBOLIC_ONEDOT_FORMULAE_H

#include "symbolic_task.h"
#include "symbolic_normxwf.h"
#include "symbolic_compxwf.h"
#include "../core/tools.h"

namespace ctns{

// generate all formulea for constructing H*x as a list of terms
// organizing principle: recursive partition
template <typename Tm>
symbolic_task<Tm> symbolic_formulae_onedot(const oper_dict<Tm>& lqops,
	                                   const oper_dict<Tm>& rqops,
	                                   const oper_dict<Tm>& cqops,
	                                   const integral::two_body<Tm>& int2e,
	                                   const int& size,
	                                   const int& rank,
					   const std::string fname){
   auto t0 = tools::get_time();
   const int print_level = 0;
   const int isym = lqops.isym;
   const bool ifkr = lqops.ifkr;
   const bool ifNC = lqops.cindex.size() <= rqops.cindex.size();
   const auto& cindex = ifNC? lqops.cindex : rqops.cindex;
   auto aindex = oper_index_opA(cindex, ifkr);
   auto bindex = oper_index_opB(cindex, ifkr);
   std::streambuf *psbuf, *backup;
   std::ofstream file;
   bool ifsave = !fname.empty() and rank == 0;
   if(ifsave){
      std::cout << "ctns::symbolic_formulae_onedot"
	        << " mpisize=" << size
	        << " fname=" << fname 
		<< std::endl;
      // http://www.cplusplus.com/reference/ios/ios/rdbuf/
      file.open(fname);
      backup = std::cout.rdbuf(); // back up cout's streambuf
      psbuf = file.rdbuf(); // get file's streambuf
      std::cout.rdbuf(psbuf); // assign streambuf to cout
      std::cout << "ctns::symbolic_formulae_onedot"
	        << " isym=" << isym
	  	<< " ifkr=" << ifkr
		<< " ifNC=" << ifNC
		<< " mpisize=" << size
	        << std::endl;
   }
   
   symbolic_task<Tm> formulae;
   
   int idx = 0;
   std::map<std::string,int> counter;
   
   if(ifNC){
      // partition = l|cr
      counter = {{"CS",0},{"SC",0},{"AP",0},{"BQ",0}};
      // 1. H^l 
      const double scale = ifkr? 0.25 : 0.5;
      auto Hl = symbolic_prod<Tm>(symbolic_oper("l",'H',0), scale);
      formulae.append(Hl);
      // 2. H^cr
      auto Hcr = symbolic_compxwf_opH<Tm>("c", "r", cqops.cindex, rqops.cindex, 
		                          ifkr, size, rank);
      formulae.join(Hcr);
      if(ifsave){
	 std::cout << "idx=" << idx++;
	 formulae.display("Hl+Hcr", print_level);
      }
      // One-index terms:
      // 3. p1^l+*Sp1^cr + h.c.
      for(const auto& index : lqops.cindex){
         auto Cl = symbolic_task<Tm>(symbolic_prod<Tm>(symbolic_oper("l",'C',index)));
         auto Scr = symbolic_compxwf_opS<Tm>("c", "r", cqops.cindex, rqops.cindex,
           	                             index, ifkr, size, rank);
         auto Cl_Scr = Cl.outer_product(Scr);
         formulae.join(Cl_Scr);
         if(ifsave){ 
	    std::cout << "idx=" << idx++;
            Cl_Scr.display("Cl_Scr["+std::to_string(index)+"]", print_level);
	    counter["CS"] += 1;
	 }
      }
      // 4. q2^cr+*Sq2^l + h.c. = -Sq2^l*q2^cr + h.c.
      auto infoC = oper_combine_opC(cqops.cindex, rqops.cindex);
      for(const auto& pr : infoC){
         int index = pr.first;
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
      // 5. Apq^l*Ppq^cr + h.c.
      for(const auto& index : aindex){
         int iproc = distribute2(index,size);
         if(iproc == rank){
            auto Al = symbolic_task<Tm>(symbolic_prod<Tm>(symbolic_oper("l",'A',index)));
            auto Pcr = symbolic_compxwf_opP<Tm>("c", "r", cqops.cindex, rqops.cindex,
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
	    auto Qcr = symbolic_compxwf_opQ<Tm>("c", "r", cqops.cindex, rqops.cindex,
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
      counter = {{"CS",0},{"SC",0},{"PA",0},{"QB",0}};
      // 1. H^lc 
      auto Hlc = symbolic_compxwf_opH<Tm>("l", "c", lqops.cindex, cqops.cindex, 
           	                          ifkr, size, rank);
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
      for(const auto& index : rqops.cindex){
         auto Slc = symbolic_compxwf_opS<Tm>("l", "c", lqops.cindex, cqops.cindex,
			 		     index, ifkr, size, rank);
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
      auto infoC = oper_combine_opC(lqops.cindex, cqops.cindex);
      for(const auto& pr : infoC){
         int index = pr.first;
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
      // 5. Ars^r*Prs^lc + h.c.
      for(const auto& index : aindex){
         int iproc = distribute2(index,size);
         if(iproc == rank){
            auto Plc = symbolic_compxwf_opP<Tm>("l", "c", lqops.cindex, cqops.cindex,
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
	    auto Qlc = symbolic_compxwf_opQ<Tm>("l", "c", lqops.cindex, cqops.cindex,
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

   std::map<std::string,int> dims = {{"l",lqops.qket.get_dimAll()},
                                     {"r",rqops.qket.get_dimAll()},
                                     {"c",cqops.qket.get_dimAll()}};
   formulae.sort(dims);
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

// generate all formulea for constructing H*x as a list of terms
// organizing principle: recursive partition
template <typename Tm>
bipart_task<Tm> symbolic_formulae_onedot2(const oper_dict<Tm>& lqops,
	                                  const oper_dict<Tm>& rqops,
	                                  const oper_dict<Tm>& cqops,
	                                  const integral::two_body<Tm>& int2e,
	                                  const int& size,
	                                  const int& rank,
					  const std::string fname){
   auto t0 = tools::get_time();
   const int print_level = 1;
   const int isym = lqops.isym;
   const bool ifkr = lqops.ifkr;
   const bool ifNC = lqops.cindex.size() <= rqops.cindex.size();
   const auto& cindex = ifNC? lqops.cindex : rqops.cindex;
   auto aindex = oper_index_opA(cindex, ifkr);
   auto bindex = oper_index_opB(cindex, ifkr);
   std::streambuf *psbuf, *backup;
   std::ofstream file;
   bool ifsave = !fname.empty() and rank == 0;
   if(ifsave){
      std::cout << "ctns::symbolic_formulae_onedot2"
	        << " mpisize=" << size
	        << " fname=" << fname 
		<< std::endl;
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
	        << std::endl;
   }
   
   bipart_task<Tm> formulae;

   int idx = 0;
   std::map<std::string,int> counter;
   
   if(ifNC){
      // partition = l|cr
      counter = {{"CS",0},{"SC",0},{"AP",0},{"BQ",0}};
      // 1. H^l 
      const double scale = ifkr? 0.25 : 0.5;
      auto Hl = symbolic_task<Tm>(symbolic_prod<Tm>(symbolic_oper("l",'H',0), scale));
      auto Hl_Ir = bipart_oper('l',Hl,"Hl_Ir");
      formulae.push_back(Hl_Ir);
      if(ifsave){
         std::cout << "idx=" << idx++;
	 Hl_Ir.display(print_level);
      }
      // 2. H^cr
      auto Hcr = symbolic_compxwf_opH<Tm>("c", "r", cqops.cindex, rqops.cindex, 
		                          ifkr, size, rank);
      auto Il_Hcr = bipart_oper('r',Hcr,"Il_Hcr");
      formulae.push_back(Il_Hcr);
      if(ifsave){
	 std::cout << "idx=" << idx++;
	 Il_Hcr.display(print_level);
      }
      // One-index terms:
      // 3. p1^l+*Sp1^cr + h.c.
      for(const auto& index : lqops.cindex){
         auto Cl = symbolic_task<Tm>(symbolic_prod<Tm>(symbolic_oper("l",'C',index)));
         auto Scr = symbolic_compxwf_opS<Tm>("c", "r", cqops.cindex, rqops.cindex,
           	                             index, ifkr, size, rank);
         auto Cl_Scr = bipart_oper(Cl,Scr,"Cl_Scr["+std::to_string(index)+"]");
         formulae.push_back(Cl_Scr);
         if(ifsave){ 
	    std::cout << "idx=" << idx++;
            Cl_Scr.display(print_level);
	    counter["CS"] += 1;
	 }
      }
      // 4. q2^cr+*Sq2^l + h.c. = -Sq2^l*q2^cr + h.c.
      auto infoC = oper_combine_opC(cqops.cindex, rqops.cindex);
      for(const auto& pr : infoC){
         int index = pr.first;
         int iformula = pr.second;
         auto Sl = symbolic_task<Tm>(symbolic_prod<Tm>(symbolic_oper("l",'S',index)));
         auto Ccr = symbolic_normxwf_opC<Tm>("c", "r", index, iformula);
	 Ccr.scale(-1.0);
	 auto Sl_Ccr = bipart_oper(Sl,Ccr,"Sl_Ccr["+std::to_string(index)+"]");
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
            auto Pcr = symbolic_compxwf_opP<Tm>("c", "r", cqops.cindex, rqops.cindex,
	 				        int2e, index, isym, ifkr);
	    const double wt = ifkr? wfacAP(index) : 1.0;
	    Pcr.scale(wt);
            auto Al_Pcr = bipart_oper(Al,Pcr,"Al_Pcr["+std::to_string(index)+"]");
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
	    auto Qcr = symbolic_compxwf_opQ<Tm>("c", "r", cqops.cindex, rqops.cindex,
	           	                        int2e, index, isym, ifkr);
            const double wt = ifkr? wfacBQ(index) : wfac(index);
	    Qcr.scale(wt);
	    auto Bl_Qcr = bipart_oper(Bl,Qcr,"Bl_Qcr["+std::to_string(index)+"]");
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
      counter = {{"CS",0},{"SC",0},{"PA",0},{"QB",0}};
      // 1. H^lc 
      auto Hlc = symbolic_compxwf_opH<Tm>("l", "c", lqops.cindex, cqops.cindex, 
           	                          ifkr, size, rank);
      auto Hlc_Ir = bipart_oper('l',Hlc,"Hlc_Ir");
      formulae.push_back(Hlc_Ir);
      if(ifsave){
         std::cout << "idx=" << idx++;
	 Hlc_Ir.display(print_level);
      }
      // 2. H^r
      const double scale = ifkr? 0.25 : 0.5;
      auto Hr = symbolic_task<Tm>(symbolic_prod<Tm>(symbolic_oper("r",'H',0), scale));
      auto Il_Hr = bipart_oper('r',Hr,"Il_Hr");
      formulae.push_back(Il_Hr);
      if(ifsave){
	 std::cout << "idx=" << idx++;
	 Il_Hr.display(print_level);
      }
      // One-index terms:
      // 3. q2^r+*Sq2^lc + h.c. = -Sq2^lc*q2^r + h.c.
      for(const auto& index : rqops.cindex){
         auto Slc = symbolic_compxwf_opS<Tm>("l", "c", lqops.cindex, cqops.cindex,
			 		     index, ifkr, size, rank);
	 Slc.scale(-1.0);
	 auto Cr = symbolic_task<Tm>(symbolic_prod<Tm>(symbolic_oper("r",'C',index)));
	 auto Slc_Cr = bipart_oper(Slc,Cr,"Slc_Cr["+std::to_string(index)+"]");
	 formulae.push_back(Slc_Cr);
	 if(ifsave){ 
	    std::cout << "idx=" << idx++;
            Slc_Cr.display(print_level);
	    counter["SC"] += 1;
	 }
      }
      // 4. p1^lc+*Sp1^r + h.c.
      auto infoC = oper_combine_opC(lqops.cindex, cqops.cindex);
      for(const auto& pr : infoC){
         int index = pr.first;
         int iformula = pr.second;
         auto Clc = symbolic_normxwf_opC<Tm>("l", "c", index, iformula);
         auto Sr = symbolic_task<Tm>(symbolic_prod<Tm>(symbolic_oper("r",'S',index)));
         auto Clc_Sr = bipart_oper(Clc,Sr,"Clc_Sr["+std::to_string(index)+"]");
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
            auto Plc = symbolic_compxwf_opP<Tm>("l", "c", lqops.cindex, cqops.cindex,
	 				        int2e, index, isym, ifkr);
            auto Ar = symbolic_task<Tm>(symbolic_prod<Tm>(symbolic_oper("r",'A',index)));
	    const double wt = ifkr? wfacAP(index) : 1.0;
	    Plc.scale(wt);
            auto Plc_Ar = bipart_oper(Plc,Ar,"Plc_Ar["+std::to_string(index)+"]");
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
	    auto Qlc = symbolic_compxwf_opQ<Tm>("l", "c", lqops.cindex, cqops.cindex,
	           	                        int2e, index, isym, ifkr);
            auto Br = symbolic_task<Tm>(symbolic_prod<Tm>(symbolic_oper("r",'B',index)));
            const double wt = ifkr? wfacBQ(index) : wfac(index);
	    Qlc.scale(wt);
	    auto Qlc_Br = bipart_oper(Qlc,Br,"Qlc_Br["+std::to_string(index)+"]");
	    formulae.push_back(Qlc_Br);
	    if(ifsave){ 
	       std::cout << "idx=" << idx++;
	       Qlc_Br.display(print_level);
	       counter["QB"] += 1;
	    }
	 } // iproc
      }
   } // ifNC

   std::map<std::string,int> dims = {{"l",lqops.qket.get_dimAll()},
                                     {"r",rqops.qket.get_dimAll()},
                                     {"c",cqops.qket.get_dimAll()}};
   sort(formulae, dims);
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
   exit(1);
   return formulae;
}

} // ctns

#endif
