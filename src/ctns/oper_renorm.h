#ifndef OPER_RENORM_H
#define OPER_RENORM_H

#ifdef _OPENMP
#include <omp.h>
#endif

#include "oper_timer.h"
#include "oper_functors.h"
#include "oper_normxwf.h"
#include "oper_compxwf.h"
#include "oper_rbasis.h"

namespace ctns{

// renormalize ops
template <typename Km, typename Tm>
void oper_renorm_opAll(const std::string superblock,
		       const comb<Km>& icomb,
		       const comb_coord& p,
		       const integral::two_body<Tm>& int2e,
		       const integral::one_body<Tm>& int1e,
		       const oper_dict<Tm>& qops1,
		       const oper_dict<Tm>& qops2,
		       oper_dict<Tm>& qops){
   const bool debug = false;
   int size = 1, rank = 0;
#ifndef SERIAL
   size = icomb.world.size();
   rank = icomb.world.rank();
#endif   
   const int isym = Km::isym;
   const bool ifkr = qkind::is_kramers<Km>();
   if(rank == 0){ 
      std::cout << "ctns::oper_renorm_opAll coord=" << p 
	        << " superblock=" << superblock 
	     	<< " isym=" << isym << " ifkr=" << ifkr
		<< " mpisize=" << size << std::endl;
   }
   auto t0 = tools::get_time();
   // setup basic information
   qops.isym = isym;
   qops.ifkr = ifkr;
   qops.cindex = oper_combine_cindex(qops1.cindex, qops2.cindex);
   // rest of spatial orbital indices
   const auto& node = icomb.topo.get_node(p);
   const auto& rindex = icomb.topo.rindex;
   const auto& site = (superblock=="cr")? icomb.rsites[rindex.at(p)] : 
	   				  icomb.lsites[rindex.at(p)];
   if(superblock == "cr"){
      qops.krest = node.lsupport;
      qops.qbra = site.info.qrow;
      qops.qket = site.info.qrow; 
   }else if(superblock == "lc"){
      auto pr = node.right;
      qops.krest = icomb.topo.get_node(pr).rsupport;
      qops.qbra = site.info.qcol;
      qops.qket = site.info.qcol;
   }else if(superblock == "lr"){
      auto pc = node.center;
      qops.krest = icomb.topo.get_node(pc).rsupport;
      qops.qbra = site.info.qmid;
      qops.qket = site.info.qmid;
   }
   qops.oplist = "CABPQSH";
   qops.mpisize = size;
   qops.mpirank = rank;
   qops.ifdist2 = true;
   // initialize memory 
   qops.allocate_memory();
/*
   std::cout << "try save start" << std::endl;
   std::string fname = oper_fname("./scratch", p, "r");
   oper_save(fname, qops);
   std::cout << "try save end" << std::endl;
*/

   // compute renormalized operators 
   oper_timer.clear();
   auto Hx_funs = oper_renorm_functors(superblock, site, qops1, qops2,
		   		       qops, int2e, int1e);
   if(debug){
      std::cout << "rank=" << rank << " size[Hx_funs]=" << Hx_funs.size() << std::endl;
      for(int i=0; i<Hx_funs.size(); i++){
         std::cout << "rank=" << rank << " i=" << i << Hx_funs[i] << std::endl;
      }
   }

#ifdef _OPENMP
   #pragma omp parallel for schedule(dynamic)
#endif
   for(int i=0; i<Hx_funs.size(); i++){
      char key = Hx_funs[i].label[0];
      int index = Hx_funs[i].index; 
      if(debug) std::cout << "cal: rank=" << rank << " i=" << i << " key=" << key << " index=" << index << std::endl;
      auto opxwf = Hx_funs[i]();
      auto op = contract_qt3_qt3(superblock, site, opxwf);
      linalg::xcopy(op.size(), op.data(), qops(key)[index].data());
   }

   // check operators against explicit construction
   const bool ifcheck = false;
   if(ifcheck){
      for(const auto& key : qops.oplist){
	 if(key == 'C' || key == 'A' || key == 'B'){
	    oper_check_rbasis(icomb, icomb, p, qops, key);
         }else{
	    oper_check_rbasis(icomb, icomb, p, qops, key, int2e, int1e);
	 }
      }
   }

/*
   // // consistency check for Hamiltonian
   // const auto& H = qops('H').at(0);
   // auto diffH = (H-H.H()).normF();
   // if(diffH > 1.e-10){
   //    H.print("H",2);
   //    std::string msg = "error: H-H.H() is too large! diffH=";
   //    tools::exit(msg+std::to_string(diffH));
   // }
*/
   auto t1 = tools::get_time();
   if(rank == 0){ 
      qops.print("qops");
      oper_timer.analysis();
      tools::timing("ctns::oper_renorm_opAll", t0, t1);
   }
}

template <typename Tm>
Hx_functors<Tm> oper_renorm_functors(const std::string superblock, 
				     const stensor3<Tm>& site, 
				     const oper_dict<Tm>& qops1, 
				     const oper_dict<Tm>& qops2, 
				     const oper_dict<Tm>& qops, 
	             		     const integral::two_body<Tm>& int2e,
	             		     const integral::one_body<Tm>& int1e){
   Hx_functors<Tm> Hx_funs;
   // opC
   if(qops.oplist.find('C') != std::string::npos){
      auto info = oper_combine_opC(qops1.cindex, qops2.cindex);
      for(const auto& pr : info){
         int index = pr.first, iformula = pr.second;
         Hx_functor<Tm> Hx("C", index, iformula);
         Hx.opxwf = bind(&oper_normxwf_opC<Tm>, 
           	         std::cref(superblock), std::cref(site), 
           	         std::ref(qops1), std::ref(qops2),
           	         index, iformula, false);
         Hx_funs.push_back(Hx);
      }
   }
   // opA
   if(qops.oplist.find('A') != std::string::npos){
      auto ainfo = oper_combine_opA(qops1.cindex, qops2.cindex, qops.ifkr);
      for(const auto& pr : ainfo){
         int index = pr.first, iformula = pr.second;
         int iproc = distribute2(index, qops.mpisize);
         if(iproc == qops.mpirank){
            Hx_functor<Tm> Hx("A", index, iformula);
            Hx.opxwf = bind(&oper_normxwf_opA<Tm>, 
           	            std::cref(superblock), std::cref(site), 
           	            std::ref(qops1), std::ref(qops2),
           		    index, iformula, false);
            Hx_funs.push_back(Hx);
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
            Hx_functor<Tm> Hx("B", index, iformula);
            Hx.opxwf = bind(&oper_normxwf_opB<Tm>, 
           	            std::cref(superblock), std::cref(site), 
           	            std::ref(qops1), std::ref(qops2),
           		    index, iformula, false);
            Hx_funs.push_back(Hx);
         }
      }
   }
   // opP
   if(qops.oplist.find('P') != std::string::npos){
      for(const auto& pr : qops('P')){
         int index = pr.first;
         Hx_functor<Tm> Hx("P", index);
         Hx.opxwf = bind(&oper_compxwf_opP<Tm>,
           	         std::cref(superblock), std::cref(site),
           	         std::ref(qops1), std::ref(qops2),
           	         std::cref(int2e), std::cref(int1e),
           	         index, false);
         Hx_funs.push_back(Hx);
      }
   }
   // opQ
   if(qops.oplist.find('Q') != std::string::npos){
      for(const auto& pr : qops('Q')){
         int index = pr.first;
         Hx_functor<Tm> Hx("Q", index);
         Hx.opxwf = bind(&oper_compxwf_opQ<Tm>,
           	         std::cref(superblock), std::cref(site),
           	         std::ref(qops1), std::ref(qops2),
           	         std::cref(int2e), std::cref(int1e),
           	         index, false);
         Hx_funs.push_back(Hx);
      }
   }
   // opS
   if(qops.oplist.find('S') != std::string::npos){
      for(const auto& pr : qops('S')){
         int index = pr.first;
         Hx_functor<Tm> Hx("S", index);
         Hx.opxwf = bind(&oper_compxwf_opS<Tm>,
           	         std::cref(superblock), std::cref(site),
           	         std::ref(qops1), std::ref(qops2),
           	         std::cref(int2e), std::cref(int1e),
           	         index, qops.mpisize, qops.mpirank, false);
         Hx_funs.push_back(Hx);
      }
   }
   // opH
   if(qops.oplist.find('H') != std::string::npos){
      Hx_functor<Tm> Hx("H");
      Hx.opxwf = bind(&oper_compxwf_opH<Tm>, 
           	      std::cref(superblock), std::cref(site),
           	      std::ref(qops1), std::ref(qops2),
           	      std::cref(int2e), std::cref(int1e),
           	      qops.mpisize, qops.mpirank);
      Hx_funs.push_back(Hx);
   }
   return Hx_funs;
}

} // ctns

#endif
