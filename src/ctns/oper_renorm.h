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
#include "symbolic_kernel_renorm.h"
#include "symbolic_kernel_renorm2.h"

namespace ctns{

const bool debug_oper_renorm = true;
extern const bool debug_oper_renorm;

// renormalize operators
template <typename Km, typename Tm>
void oper_renorm_opAll(const std::string superblock,
		       const comb<Km>& icomb,
		       const comb_coord& p,
		       const integral::two_body<Tm>& int2e,
		       const integral::one_body<Tm>& int1e,
		       const oper_dict<Tm>& qops1,
		       const oper_dict<Tm>& qops2,
		       oper_dict<Tm>& qops,
		       const std::string fname,
		       const int alg_renorm,
		       const bool sort_formulae){
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
	     	<< " isym=" << isym 
		<< " ifkr=" << ifkr
	        << " alg_renorm=" << alg_renorm	
		<< " mpisize=" << size
		<< std::endl;
   }
   auto ti = tools::get_time();
   
   // 0. setup basic information for qops
   qops.isym = isym;
   qops.ifkr = ifkr;
   qops.cindex = oper_combine_cindex(qops1.cindex, qops2.cindex);
   // rest of spatial orbital indices
   const auto& node = icomb.topo.get_node(p);
   const auto& rindex = icomb.topo.rindex;
   const auto& site = (superblock=="cr")? icomb.rsites[rindex.at(p)] : 
	   				  icomb.lsites[rindex.at(p)];
   if(superblock == "cr"){
      qops.krest = node.lorbs;
      qops.qbra = site.info.qrow;
      qops.qket = site.info.qrow; 
   }else if(superblock == "lc"){
      qops.krest = node.rorbs;
      qops.qbra = site.info.qcol;
      qops.qket = site.info.qcol;
   }else if(superblock == "lr"){
      qops.krest = node.corbs;
      qops.qbra = site.info.qmid;
      qops.qket = site.info.qmid;
   }
   qops.oplist = "CABPQSH";
   qops.mpisize = size;
   qops.mpirank = rank;
   qops.ifdist2 = true;
   // initialize memory 
   qops.allocate_memory();
      
   // 1. start renormalization
   oper_timer.clear();
   if(alg_renorm == 0){
      auto rfuns = oper_renorm_functors(superblock, site, int2e, qops1, qops2, qops);
      oper_renorm_kernel(superblock, rfuns, site, qops, debug);
   }else if(alg_renorm == 1){
      auto rtasks = symbolic_formulae_renorm(superblock, int2e, qops1, qops2, qops, 
		                             size, rank, fname, sort_formulae);
      symbolic_kernel_renorm(superblock, rtasks, site, qops1, qops2, qops, debug);
   }else if(alg_renorm == 2){
      auto rtasks = symbolic_formulae_renorm(superblock, int2e, qops1, qops2, qops, 
		                             size, rank, fname, sort_formulae);
      symbolic_kernel_renorm2(superblock, rtasks, site, qops1, qops2, qops, debug);
   }

   // 2. reduce 
   auto ta = tools::get_time();
   if(ifdistribute1 and size > 1){
      std::vector<Tm> top(qops._opsize);
      // Sp[iproc] += \sum_i Sp[i]
      auto opS_index = qops.oper_index_op('S');
      for(auto& p : opS_index){
         int iproc = distribute1(p,size);
         auto& opS = qops('S')[p];
         int opsize = opS.size();
         memset(top.data(), 0, opsize*sizeof(Tm));
         boost::mpi::reduce(icomb.world, opS.data(), opsize, 
           	            top.data(), std::plus<Tm>(), iproc);
         if(iproc == rank) linalg::xcopy(opsize, top.data(), opS.data());
      }
      // H[0] += \sum_i H[i]
      auto& opH = qops('H')[0];
      int opsize = opH.size();
      memset(top.data(), 0, opsize*sizeof(Tm));
      boost::mpi::reduce(icomb.world, opH.data(), opsize,
		         top.data(), std::plus<Tm>(), 0);
      if(rank == 0) linalg::xcopy(opsize, top.data(), opH.data());
   }

   // 3. consistency check for Hamiltonian
   const auto& opH = qops('H').at(0);
   auto diffH = (opH-opH.H()).normF();
   if(diffH > 1.e-10){
      opH.print("H",2);
      std::string msg = "error: H-H.H() is too large! diffH=";
      tools::exit(msg+std::to_string(diffH));
   }

   // check against explicit construction
   if(debug_oper_renorm){
      for(const auto& key : qops.oplist){
	 if(key == 'C' || key == 'A' || key == 'B'){
	    //oper_check_rbasis(icomb, icomb, p, qops, key, size, rank);
         }else{
	    if(key == 'S'){
	    oper_check_rbasis(icomb, icomb, p, qops, key, int2e, int1e, size, rank);
	    }
	 }
      }
   }

   icomb.world.barrier();
   std::cout << "####### rank=" << rank << " #########" << std::endl;
   icomb.world.barrier();

   auto tf = tools::get_time();
   if(rank == 0){ 
      qops.print("qops");
      if(alg_renorm == 0) oper_timer.analysis();
      double t_tot = tools::get_duration(tf-ti); 
      double t_cal = tools::get_duration(ta-ti);
      double t_comm = tools::get_duration(tf-ta);
      std::cout << "T(tot/cal/comm)=" 
	        << t_tot << "," << t_cal << "," << t_comm
	        << std::endl;	
      tools::timing("ctns::oper_renorm_opAll", ti, tf);
   }
}

template <typename Tm>
void oper_renorm_kernel(const std::string superblock,
		        const Hx_functors<Tm>& rfuns,
		        const stensor3<Tm>& site,
		        oper_dict<Tm>& qops,
			const bool debug){
   if(debug) std::cout << "rank=" << qops.mpirank 
	               << " size[rfuns]=" << rfuns.size() 
		       << std::endl;
#ifdef _OPENMP
   #pragma omp parallel for schedule(dynamic)
#endif
   for(int i=0; i<rfuns.size(); i++){
      char key = rfuns[i].label[0];
      int index = rfuns[i].index; 
      if(debug){
         std::cout << "cal: rank=" << qops.mpirank 
                   << " i=" << i 
                   << " key=" << key 
		   << " index=" << index 
		   << std::endl;
      }
      auto opxwf = rfuns[i]();
      auto op = contract_qt3_qt3(superblock, site, opxwf);
      linalg::xcopy(op.size(), op.data(), qops(key)[index].data());
   } // i
}

template <typename Tm>
Hx_functors<Tm> oper_renorm_functors(const std::string superblock,
				     const stensor3<Tm>& site,
	             		     const integral::two_body<Tm>& int2e,
				     const oper_dict<Tm>& qops1,
				     const oper_dict<Tm>& qops2,
				     const oper_dict<Tm>& qops){
   Hx_functors<Tm> rfuns;
   // opC
   if(qops.oplist.find('C') != std::string::npos){
      auto info = oper_combine_opC(qops1.cindex, qops2.cindex);
      for(const auto& pr : info){
         int index = pr.first, iformula = pr.second;
         Hx_functor<Tm> Hx("C", index, iformula);
         Hx.opxwf = bind(&oper_normxwf_opC<Tm>, 
           	         std::cref(superblock), std::cref(site), 
           	         std::cref(qops1), std::cref(qops2),
           	         index, iformula, false);
         rfuns.push_back(Hx);
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
           	            std::cref(qops1), std::cref(qops2),
           		    index, iformula, false);
            rfuns.push_back(Hx);
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
           	            std::cref(qops1), std::cref(qops2),
           		    index, iformula, false);
            rfuns.push_back(Hx);
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
           	         std::cref(qops1), std::cref(qops2), std::cref(int2e), 
			 index, false);
         rfuns.push_back(Hx);
      }
   }
   // opQ
   if(qops.oplist.find('Q') != std::string::npos){
      for(const auto& pr : qops('Q')){
         int index = pr.first;
         Hx_functor<Tm> Hx("Q", index);
         Hx.opxwf = bind(&oper_compxwf_opQ<Tm>,
           	         std::cref(superblock), std::cref(site),
           	         std::cref(qops1), std::cref(qops2), std::cref(int2e), 
			 index, false);
         rfuns.push_back(Hx);
      }
   }
   // opS
   if(qops.oplist.find('S') != std::string::npos){
      for(const auto& pr : qops('S')){
         int index = pr.first;
         Hx_functor<Tm> Hx("S", index);
         Hx.opxwf = bind(&oper_compxwf_opS<Tm>,
           	         std::cref(superblock), std::cref(site),
           	         std::cref(qops1), std::cref(qops2), std::cref(int2e),
			 index, qops.mpisize, qops.mpirank, false);
         rfuns.push_back(Hx);
      }
   }
   // opH
   if(qops.oplist.find('H') != std::string::npos){
      Hx_functor<Tm> Hx("H");
      Hx.opxwf = bind(&oper_compxwf_opH<Tm>, 
           	      std::cref(superblock), std::cref(site),
           	      std::cref(qops1), std::cref(qops2),
           	      qops.mpisize, qops.mpirank);
      rfuns.push_back(Hx);
   }
   return rfuns;
}

} // ctns

#endif
