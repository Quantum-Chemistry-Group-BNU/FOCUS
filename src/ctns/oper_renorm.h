#ifndef OPER_RENORM_H
#define OPER_RENORM_H

#include "oper_rbasis.h"
#include "oper_combine.h"
#include "oper_normxwf.h"
#include "oper_compxwf.h"
#include "oper_timer.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ctns{

// kernel for computing renormalized ap^+
template <typename Km, typename Tm>
void oper_renorm_opC(const std::string& superblock,
		     const comb<Km>& icomb,
		     const comb_coord& p,
		     const qtensor3<Tm>& site,
		     oper_dict<Tm>& qops1,
		     oper_dict<Tm>& qops2,
		     oper_dict<Tm>& qops){
   if(debug_oper_dict) std::cout << "\nctns::oper_renorm_opC" << std::endl;
   auto t0 = tools::get_time();
   // preprocess
   auto info = oper_combine_opC(qops1.cindex, qops2.cindex);
   auto ta = tools::get_time();

   // compute
#ifdef _OPENMP
   int maxthreads = omp_get_max_threads();
#else
   int maxthreads = 1;
#endif
   std::vector<std::vector<int>> indices(maxthreads);
   std::vector<std::vector<qtensor2<Tm>>> tops(maxthreads);
#ifdef _OPENMP
   #pragma omp parallel for schedule(dynamic)
#endif 
   for(const auto pr : info){
      int iformula = pr.first;
      int index = pr.second;
      auto opwf = oper_normxwf_opC(superblock,site,qops1,qops2,iformula,index); 
      auto tmp = oper_kernel_renorm(superblock,site,opwf); 
#ifdef _OPENMP
      int omprank = omp_get_thread_num();
#else 
      int omprank = 0;
#endif
      indices[omprank].push_back(index);
      tops[omprank].push_back(tmp);
/* 
      std::cout << "id=" << omp_get_thread_num() 
                << " iformula/index=" << iformula << "," << index 
                << std::endl; 
*/
   }
   auto tb = tools::get_time();

   for(int i=0; i<maxthreads; i++){
      for(int j=0; j<indices[i].size(); j++){
         int index = indices[i][j];
         qops('C')[index] = tops[i][j];
      }
   }
   auto tc = tools::get_time();
   //exit(1);

   auto t1 = tools::get_time();
   if(debug_oper_dict) tools::timing("ctns::oper_renorm_opC", t0, t1);

   std::cout << "opC: n=" << info.size() 
             << " tot=" << tools::get_duration(t1-t0) << " S"
             << " info=" << tools::get_duration(ta-t0) << " S"
             << " calc=" << tools::get_duration(tb-ta) << " S"
             << " save=" << tools::get_duration(tc-tb) << " S"
             << std::endl; 
}

// kernel for computing renormalized Apq=ap^+aq^+
template <typename Km, typename Tm>
void oper_renorm_opA(const std::string& superblock,
		     const comb<Km>& icomb,
		     const comb_coord& p,
		     const qtensor3<Tm>& site,
		     oper_dict<Tm>& qops1,
		     oper_dict<Tm>& qops2,
		     oper_dict<Tm>& qops,
		     const bool& ifkr){
   int size = 1, rank = 0;
#ifndef SERIAL
   size = icomb.world.size();
   rank = icomb.world.rank();
#endif   
   if(debug_oper_dict) std::cout << "\nctns::oper_renorm_opA" << std::endl;
   auto t0 = tools::get_time();
   // preprocess
   auto info = oper_combine_opA(qops1.cindex, qops2.cindex, ifkr);
   auto ta = tools::get_time();

   // compute
#ifdef _OPENMP
   int maxthreads = omp_get_max_threads();
#else
   int maxthreads = 1;
#endif
   std::vector<std::vector<int>> indices(maxthreads);
   std::vector<std::vector<qtensor2<Tm>>> tops(maxthreads);
#ifdef _OPENMP
   #pragma omp parallel for schedule(dynamic)
#endif 
   for(const auto pr : info){
      int iformula = pr.first;
      int index = pr.second;
      int iproc = distribute2(index, size);
      if(iproc == rank){
         auto opwf = oper_normxwf_opA(superblock,site,qops1,qops2,ifkr,iformula,index);
         auto tmp = oper_kernel_renorm(superblock,site,opwf);
#ifdef _OPENMP
         int omprank = omp_get_thread_num();
#else
	 int omprank = 0;
#endif
         indices[omprank].push_back(index);
         tops[omprank].push_back(tmp);
      }
   }
   auto tb = tools::get_time();

   for(int i=0; i<maxthreads; i++){
      for(int j=0; j<indices[i].size(); j++){
         int index = indices[i][j];
         qops('A')[index] = tops[i][j];
      }
   }
   auto tc = tools::get_time();
   //exit(1);

   if(debug_oper_para){
      std::cout << " opA: coord=" << p << " no.=" << info.size()
	        << " size,rank=" << size << "," << rank 
		<< " no.=" << qops('A').size() << std::endl;
   }
   auto t1 = tools::get_time();
   if(debug_oper_dict) tools::timing("ctns::oper_renorm_opA", t0, t1);

   std::cout << "opA: n=" << info.size() 
             << " tot=" << tools::get_duration(t1-t0) << " S"
             << " info=" << tools::get_duration(ta-t0) << " S"
             << " calc=" << tools::get_duration(tb-ta) << " S"
             << " save=" << tools::get_duration(tc-tb) << " S"
             << std::endl; 
}

// kernel for computing renormalized ap^+aq
template <typename Km, typename Tm>
void oper_renorm_opB(const std::string& superblock,
		     const comb<Km>& icomb,
		     const comb_coord& p,
		     const qtensor3<Tm>& site,
		     oper_dict<Tm>& qops1,
		     oper_dict<Tm>& qops2,
		     oper_dict<Tm>& qops,
		     const bool& ifkr){
   int size = 1, rank = 0;
#ifndef SERIAL
   size = icomb.world.size();
   rank = icomb.world.rank();
#endif   
   if(debug_oper_dict) std::cout << "\nctns::oper_renorm_opB" << std::endl;
   auto t0 = tools::get_time();
   // preprocess
   auto info = oper_combine_opB(qops1.cindex, qops2.cindex, ifkr);
   auto ta = tools::get_time();
   
   // compute
#ifdef _OPENMP
   int maxthreads = omp_get_max_threads();
#else
   int maxthreads = 1;
#endif
   std::vector<std::vector<int>> indices(maxthreads);
   std::vector<std::vector<qtensor2<Tm>>> tops(maxthreads);
#ifdef _OPENMP
   #pragma omp parallel for schedule(dynamic)
#endif 
   for(const auto pr : info){
      int iformula = pr.first;
      int index = pr.second;
      int iproc = distribute2(index, size);
      if(iproc == rank){
         auto opwf = oper_normxwf_opB(superblock,site,qops1,qops2,ifkr,iformula,index);
         auto tmp = oper_kernel_renorm(superblock,site,opwf);
#ifdef _OPENMP
         int omprank = omp_get_thread_num();
#else
 	 int omprank = 0;
#endif
         indices[omprank].push_back(index);
         tops[omprank].push_back(tmp);
      }
   }
   auto tb = tools::get_time();

   for(int i=0; i<maxthreads; i++){
      for(int j=0; j<indices[i].size(); j++){
         int index = indices[i][j];
         qops('B')[index] = tops[i][j];
      }
   }
   auto tc = tools::get_time();
   //exit(1);

   if(debug_oper_para){
      std::cout << " opB: coord=" << p << " no.=" << info.size()
	        << " size,rank=" << size << "," << rank 
		<< " no.=" << qops('B').size() << std::endl;
   }
   auto t1 = tools::get_time();
   if(debug_oper_dict) tools::timing("ctns::oper_renorm_opB", t0, t1);
   
   std::cout << "opB: n=" << info.size() 
             << " tot=" << tools::get_duration(t1-t0) << " S"
             << " info=" << tools::get_duration(ta-t0) << " S"
             << " calc=" << tools::get_duration(tb-ta) << " S"
             << " save=" << tools::get_duration(tc-tb) << " S"
             << std::endl; 
}

// Ppq = <pq||sr> aras [r>s] (p<q)
template <typename Km, typename Tm>
void oper_renorm_opP(const std::string& superblock,	
		     const comb<Km>& icomb,
		     const comb_coord& p,
		     const qtensor3<Tm>& site,
		     oper_dict<Tm>& qops1,
		     oper_dict<Tm>& qops2,
		     oper_dict<Tm>& qops,
		     const int& isym,
		     const bool& ifkr,
		     const std::vector<int>& krest,
	             const integral::two_body<Tm>& int2e,
	             const integral::one_body<Tm>& int1e){
   int size = 1, rank = 0;
#ifndef SERIAL
   size = icomb.world.size();
   rank = icomb.world.rank();
#endif   
   if(debug_oper_dict) std::cout << "\nctns::oper_renorm_opP" << std::endl;
   auto t0 = tools::get_time();
   // preprocess
   auto info = oper_combine_opP(krest, ifkr);
   auto ta = tools::get_time();
  
   // compute
#ifdef _OPENMP
   int maxthreads = omp_get_max_threads();
#else 
   int maxthreads = 1;
#endif
   std::vector<std::vector<int>> indices(maxthreads);
   std::vector<std::vector<qtensor2<Tm>>> tops(maxthreads);
#ifdef _OPENMP
   #pragma omp parallel for schedule(dynamic)
#endif 
   for(const auto index : info){
      int iproc = distribute2(index, size);
      if(iproc == rank){
         auto opwf = oper_compxwf_opP(superblock,site,qops1,qops2,isym,ifkr,int2e,int1e,index);
         auto tmp = oper_kernel_renorm(superblock,site,opwf);
#ifdef _OPENMP
         int omprank = omp_get_thread_num();
#else
	 int omprank = 0;
#endif
         indices[omprank].push_back(index);
         tops[omprank].push_back(tmp);
      }
   }
   auto tb = tools::get_time();

   for(int i=0; i<maxthreads; i++){
      for(int j=0; j<indices[i].size(); j++){
         int index = indices[i][j];
         qops('P')[index] = tops[i][j];
      }
   }
   auto tc = tools::get_time();
   //exit(1);
 
   auto t1 = tools::get_time();
   if(debug_oper_para){
      std::cout << " opP: coord=" << p << " no.=" << info.size()
	        << " size,rank=" << size << "," << rank 
		<< " no.=" << qops('P').size() << std::endl;
   }
   if(debug_oper_dict) tools::timing("ctns::oper_renorm_opP", t0, t1);
   
   std::cout << "opP: n=" << info.size() 
             << " tot=" << tools::get_duration(t1-t0) << " S"
             << " info=" << tools::get_duration(ta-t0) << " S"
             << " calc=" << tools::get_duration(tb-ta) << " S"
             << " save=" << tools::get_duration(tc-tb) << " S"
             << std::endl; 
}

// Qps = <pq||sr> aq^+ar
template <typename Km, typename Tm>
void oper_renorm_opQ(const std::string& superblock,
		     const comb<Km>& icomb,
		     const comb_coord& p,
		     const qtensor3<Tm>& site,
		     oper_dict<Tm>& qops1,
		     oper_dict<Tm>& qops2,
		     oper_dict<Tm>& qops,
		     const int& isym,
		     const bool& ifkr,
		     const std::vector<int>& krest,
	             const integral::two_body<Tm>& int2e,
	             const integral::one_body<Tm>& int1e){
   int size = 1, rank = 0;
#ifndef SERIAL
   size = icomb.world.size();
   rank = icomb.world.rank();
#endif   
   if(debug_oper_dict) std::cout << "\nctns::oper_renorm_opQ" << std::endl;
   auto t0 = tools::get_time();
   // preprocess
   auto info = oper_combine_opQ(krest, ifkr);
   auto ta = tools::get_time();
   
   // compute
#ifdef _OPENMP
   int maxthreads = omp_get_max_threads();
#else
   int maxthreads = 1;
#endif
   std::vector<std::vector<int>> indices(maxthreads);
   std::vector<std::vector<qtensor2<Tm>>> tops(maxthreads);
#ifdef _OPENMP
   #pragma omp parallel for schedule(dynamic)
#endif 
   for(const int index : info){
      int iproc = distribute2(index, size);
      if(iproc == rank){
         auto opwf = oper_compxwf_opQ(superblock,site,qops1,qops2,isym,ifkr,int2e,int1e,index);
         auto tmp = oper_kernel_renorm(superblock,site,opwf);
#ifdef _OPENMP
         int omprank = omp_get_thread_num();
#else
	 int omprank = 0;
#endif
         indices[omprank].push_back(index);
         tops[omprank].push_back(tmp);
      }
   }
   auto tb = tools::get_time();

   for(int i=0; i<maxthreads; i++){
      for(int j=0; j<indices[i].size(); j++){
         int index = indices[i][j];
         qops('Q')[index] = tops[i][j];
      }
   }
   auto tc = tools::get_time();
   //exit(1);
 
   if(debug_oper_para){
      std::cout << " opQ: coord=" << p << " no.=" << info.size()
	        << " size,rank=" << size << "," << rank 
		<< " no.=" << qops('Q').size() << std::endl;
   }
   auto t1 = tools::get_time();
   if(debug_oper_dict) tools::timing("ctns::oper_renorm_opQ", t0, t1);
   
   std::cout << "opQ: n=" << info.size() 
             << " tot=" << tools::get_duration(t1-t0) << " S"
             << " info=" << tools::get_duration(ta-t0) << " S"
             << " calc=" << tools::get_duration(tb-ta) << " S"
             << " save=" << tools::get_duration(tc-tb) << " S"
             << std::endl; 
}

// Sp = 1/2 hpq aq + <pq||sr> aq^+aras [r>s]
template <typename Km, typename Tm>
void oper_renorm_opS(const std::string& superblock,
		     const comb<Km>& icomb,
		     const comb_coord& p,
		     const qtensor3<Tm>& site,
		     oper_dict<Tm>& qops1,
		     oper_dict<Tm>& qops2,
		     oper_dict<Tm>& qops,
		     const int& isym,
		     const bool& ifkr,
		     const std::vector<int>& krest,
	             const integral::two_body<Tm>& int2e,
	             const integral::one_body<Tm>& int1e){
   int size = 1, rank = 0;
#ifndef SERIAL
   size = icomb.world.size();
   rank = icomb.world.rank();
#endif   
   if(debug_oper_dict) std::cout << "\nctns::oper_renorm_opS" << std::endl;
   auto t0 = tools::get_time();
   // preprocess
   auto info = oper_combine_opS(krest, ifkr);
   auto ta = tools::get_time();
   
   // compute
#ifdef _OPENMP
   int maxthreads = omp_get_max_threads();
#else
   int maxthreads = 1; 
#endif
   std::vector<std::vector<int>> indices(maxthreads);
   std::vector<std::vector<qtensor2<Tm>>> tops(maxthreads);
#ifdef _OPENMP
   #pragma omp parallel for schedule(dynamic)
#endif 
   for(const int index : info){
      auto opwf = oper_compxwf_opS(superblock,site,qops1,qops2,isym,ifkr,int2e,int1e,index,size,rank);
      auto tmp = oper_kernel_renorm(superblock,site,opwf);
#ifdef _OPENMP
      int omprank = omp_get_thread_num();
#else
      int omprank = 0;
#endif
      indices[omprank].push_back(index);
      tops[omprank].push_back(tmp);
   }
   auto tb = tools::get_time();

   for(int i=0; i<maxthreads; i++){
      for(int j=0; j<indices[i].size(); j++){
         int index = indices[i][j];
         qops('S')[index] = tops[i][j];
      }
   }
   auto tc = tools::get_time();
   //exit(1);

   auto t1 = tools::get_time();
   if(debug_oper_dict) tools::timing("ctns::oper_renorm_opS", t0, t1);
   
   std::cout << "opS: n=" << info.size() 
             << " tot=" << tools::get_duration(t1-t0) << " S"
             << " info=" << tools::get_duration(ta-t0) << " S"
             << " calc=" << tools::get_duration(tb-ta) << " S"
             << " save=" << tools::get_duration(tc-tb) << " S"
             << std::endl; 
}

template <typename Km, typename Tm>
void oper_renorm_opH(const std::string& superblock,
		     const comb<Km>& icomb,
		     const comb_coord& p,
		     const qtensor3<Tm>& site,
		     oper_dict<Tm>& qops1,
		     oper_dict<Tm>& qops2,
		     oper_dict<Tm>& qops,
		     const int& isym,
		     const bool& ifkr,
	             const integral::two_body<Tm>& int2e,
	             const integral::one_body<Tm>& int1e){
   int size = 1, rank = 0;
#ifndef SERIAL
   size = icomb.world.size();
   rank = icomb.world.rank();
#endif   
   if(debug_oper_dict) std::cout << "\nctns::oper_renorm_opH" << std::endl;
   auto t0 = tools::get_time();
   // compute 
   auto opwf = oper_compxwf_opH(superblock,site,qops1,qops2,isym,ifkr,int2e,int1e,size,rank);
   qops('H')[0] = oper_kernel_renorm(superblock,site,opwf);
   auto t1 = tools::get_time();
   if(debug_oper_dict) tools::timing("ctns::oper_renorm_opH", t0, t1);
}

// renormalize ops
template <typename Km, typename Tm>
void oper_renorm_opAll(const std::string& superblock,
		       const comb<Km>& icomb,
		       const comb_coord& p,
		       const integral::two_body<Tm>& int2e,
		       const integral::one_body<Tm>& int1e,
		       oper_dict<Tm>& qops1,
		       oper_dict<Tm>& qops2,
		       oper_dict<Tm>& qops){
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
		<< " size=" << size
	        << std::endl;
   }
   auto t0 = tools::get_time();
  
   // support for index of complementary ops 
   auto& node = icomb.topo.get_node(p);
   std::vector<int> krest; 
   if(superblock == "cr"){
      krest = node.lsupport;
   }else if(superblock == "lc"){
      auto pr = node.right;
      krest = icomb.topo.get_node(pr).rsupport;
   }else if(superblock == "lr"){
      auto pc = node.center;
      krest = icomb.topo.get_node(pc).rsupport;
   }

   oper_timer.clear();
   
   // combine cindex first 
   qops.cindex = oper_combine_cindex(qops1.cindex, qops2.cindex);
   const auto& site = (superblock=="cr")? icomb.rsites.at(p) : icomb.lsites.at(p);
   const bool ifcheck = false; // check operators against explicit construction
   // C
   oper_renorm_opC(superblock, icomb, p, site, qops1, qops2, qops);
   if(ifcheck) oper_check_rbasis(icomb, icomb, p, qops, 'C');
   // A
   oper_renorm_opA(superblock, icomb, p, site, qops1, qops2, qops, ifkr);
   if(ifcheck) oper_check_rbasis(icomb, icomb, p, qops, 'A');
   // B
   oper_renorm_opB(superblock, icomb, p, site, qops1, qops2, qops, ifkr);
   if(ifcheck) oper_check_rbasis(icomb, icomb, p, qops, 'B');
   // P
   oper_renorm_opP(superblock, icomb, p, site, qops1, qops2, qops, isym, ifkr, krest, int2e, int1e);
   if(ifcheck) oper_check_rbasis(icomb, icomb, p, qops, 'P', int2e, int1e);
   // Q
   oper_renorm_opQ(superblock, icomb, p, site, qops1, qops2, qops, isym, ifkr, krest, int2e, int1e);
   if(ifcheck) oper_check_rbasis(icomb, icomb, p, qops, 'Q', int2e, int1e);
   // S
   oper_renorm_opS(superblock, icomb, p, site, qops1, qops2, qops, isym, ifkr, krest, int2e, int1e);
   if(ifcheck) oper_check_rbasis(icomb, icomb, p, qops, 'S', int2e, int1e);
   // H
   oper_renorm_opH(superblock, icomb, p, site, qops1, qops2, qops, isym, ifkr, int2e, int1e);
   if(ifcheck) oper_check_rbasis(icomb, icomb, p, qops, 'H', int2e, int1e);

   // consistency check for Hamiltonian
   const auto& H = qops('H').at(0);
   auto diffH = (H-H.H()).normF();
   if(diffH > 1.e-10){
      H.print("H",2);
      std::string msg = "error: H-H.H() is too large! diffH=";
      tools::exit(msg+std::to_string(diffH));
   }

   auto t1 = tools::get_time();
   if(rank == 0){ 
      qops.print("qops");
      oper_timer.analysis();
      tools::timing("ctns::oper_renorm_opAll", t0, t1);
   }
}

} // ctns

#endif
