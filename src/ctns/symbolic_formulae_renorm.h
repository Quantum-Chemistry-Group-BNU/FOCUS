#ifndef SYMBOLIC_RENORM_FORMULAE_H
#define SYMBOLIC_RENORM_FORMULAE_H

#include "symbolic_task.h"
#include "symbolic_normxwf.h"
#include "symbolic_compxwf.h"
#include "../core/tools.h"

namespace ctns{

template <typename Tm>
renorm_tasks<Tm> symbolic_formulae_renorm(const std::string superblock,
	     		                  const integral::two_body<Tm>& int2e,
			                  const oper_dict<Tm>& qops1,
			                  const oper_dict<Tm>& qops2,
			                  const oper_dict<Tm>& qops,
					  const std::string fname){
   auto t0 = tools::get_time();
   const int print_level = 1;
   const int isym = qops.isym;
   const bool ifkr = qops.ifkr;
   const std::string block1 = superblock.substr(0,1);
   const std::string block2 = superblock.substr(1,2);
   std::streambuf *psbuf, *backup;
   std::ofstream file;
   bool ifsave = !fname.empty() and qops.mpirank == 0;
   if(ifsave){
      std::cout << "ctns::symbolic_formulae_renorm"
	        << " mpisize=" << qops.mpisize
		<< " fname=" << fname
		<< std::endl;
      // http://www.cplusplus.com/reference/ios/ios/rdbuf/
      file.open(fname);
      backup = std::cout.rdbuf(); // back up cout's streambuf
      psbuf = file.rdbuf(); // get file's streambuf
      std::cout.rdbuf(psbuf); // assign streambuf to cout
      std::cout << "ctns::symbolic_formulae_renorm"
	        << " isym=" << isym
		<< " ifkr=" << ifkr
		<< " block1=" << block1
		<< " block2=" << block2
		<< " mpisize=" << qops.mpisize
		<< std::endl;
   }

   renorm_tasks<Tm> formulae;

   int idx = 0;
   // opC
   if(qops.oplist.find('C') != std::string::npos){
      auto info = oper_combine_opC(qops1.cindex, qops2.cindex);
      for(const auto& pr : info){
         int index = pr.first, iformula = pr.second;
	 auto opC = symbolic_normxwf_opC<Tm>(block1, block2, index, iformula);
	 formulae.append(std::make_tuple('C', index, opC));
         if(ifsave){
	    std::cout << " idx=" << idx++;
	    opC.display("opC["+std::to_string(index)+"]", print_level);
	 }
      }
   }
   // opA
   if(qops.oplist.find('A') != std::string::npos){
      auto ainfo = oper_combine_opA(qops1.cindex, qops2.cindex, qops.ifkr);
      for(const auto& pr : ainfo){
         int index = pr.first, iformula = pr.second;
         int iproc = distribute2(index, qops.mpisize);
         if(iproc == qops.mpirank){
	    auto opA = symbolic_normxwf_opA<Tm>(block1, block2, index, iformula, ifkr);
	    formulae.append(std::make_tuple('A', index, opA));
            if(ifsave){
	       std::cout << " idx=" << idx++;
	       opA.display("opA["+std::to_string(index)+"]", print_level);
	    }
         }
      }
   }
   // opB
   if(qops.oplist.find('B') != std::string::npos){
      auto binfo = oper_combine_opB(qops1.cindex, qops2.cindex, qops.ifkr);
      for(const auto& pr : binfo){
         int index = pr.first, iformula = pr.second;
         int iproc = distribute2(index, qops.mpisize);
         if(iproc == qops.mpirank){
	    auto opB = symbolic_normxwf_opB<Tm>(block1, block2, index, iformula, ifkr);
	    formulae.append(std::make_tuple('B', index, opB));
            if(ifsave){
	       std::cout << " idx=" << idx++;
	       opB.display("opB["+std::to_string(index)+"]", print_level);
	    }
         }
      }
   }
   // opP
   if(qops.oplist.find('P') != std::string::npos){
      for(const auto& pr : qops('P')){
         int index = pr.first;
	 auto opP = symbolic_compxwf_opP<Tm>(block1, block2, qops1.cindex, qops2.cindex,
			 	             int2e, index, isym, ifkr);
	 formulae.append(std::make_tuple('P', index, opP));
	 if(ifsave){
	    std::cout << " idx=" << idx++;
            opP.display("opP["+std::to_string(index)+"]", print_level);
	 }
      }
   }
   // opQ
   if(qops.oplist.find('Q') != std::string::npos){
      for(const auto& pr : qops('Q')){
         int index = pr.first;
	 auto opQ = symbolic_compxwf_opQ<Tm>(block1, block2, qops1.cindex, qops2.cindex,
			 	             int2e, index, isym, ifkr);
	 formulae.append(std::make_tuple('Q', index, opQ));
	 if(ifsave){
	    std::cout << " idx=" << idx++;
            opQ.display("opQ["+std::to_string(index)+"]", print_level);
	 }
      }
   }
   // opS
   if(qops.oplist.find('S') != std::string::npos){
      for(const auto& pr : qops('S')){
         int index = pr.first;
	 auto opS = symbolic_compxwf_opS<Tm>(block1, block2, qops1.cindex, qops2.cindex,
			 	             index, ifkr, qops.mpisize, qops.mpirank);
	 formulae.append(std::make_tuple('S', index, opS));
	 if(ifsave){
	    std::cout << " idx=" << idx++;
	    opS.display("opS["+std::to_string(index)+"]", print_level);
	 }
      }
   }
   // opH
   if(qops.oplist.find('H') != std::string::npos){
      auto opH = symbolic_compxwf_opH<Tm>(block1, block2, qops1.cindex, qops2.cindex,
	 	      		          ifkr, qops.mpisize, qops.mpirank);
      formulae.append(std::make_tuple('H', 0, opH));
      if(ifsave){
	 std::cout << " idx=" << idx++;
         opH.display("opH", print_level);
      }
   }

   std::map<std::string,int> dims = {{block1,qops1.qket.get_dimAll()},
	   			     {block2,qops2.qket.get_dimAll()}};
   formulae.sort(dims);
   if(ifsave){
      std::cout << "renormalization summary:" << std::endl;
      qops.print("qops",2);
      std::cout.rdbuf(backup); // restore cout's original streambuf
      file.close();
   }
   if(qops.mpirank == 0){
      auto t1 = tools::get_time();
      int size = formulae.size();
      tools::timing("symbolic_formulae_renorm with size="+std::to_string(size), t0, t1);
   }
   return formulae;
}

} // ctns

#endif
