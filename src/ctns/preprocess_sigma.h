#ifndef PREPROCESS_SIGMA_H
#define PREPROCESS_SIGMA_H

#include "preprocess_inter.h"
#include "preprocess_hmu.h"

namespace ctns{

template <typename Tm, typename QTm>
size_t preprocess_formulae_sigma(const oper_dictmap<Tm>& qops_dict,
				 const std::map<std::string,int>& oploc,
		 	         const symbolic_task<Tm>& H_formulae,
			         const QTm& wf,
				 intermediates<Tm>& inter,
   			         Hxlist<Tm>& Hxlst,
				 const int hxorder,
				 const bool debug){
   auto t0 = tools::get_time();

   // 1. form intermediate operators 
   inter.init(qops_dict, H_formulae, debug);
   auto ta = tools::get_time();

   // 2. preprocess formulae to Hmu
   int hsize = H_formulae.size();
   std::vector<Hmu_ptr<Tm>> Hmu_vec(hsize);
   for(int it=0; it<hsize; it++){
      Hmu_vec[it].init(it, H_formulae, qops_dict, inter, oploc);
   } // it
   auto tb = tools::get_time();

   // 3. from Hmu to Hxlst in expanded block forms
   size_t blksize = 0;
   double cost = 0.0;
   for(int it=0; it<hsize; it++){
      Hmu_vec[it].gen_Hxlist(wf.info, Hxlst, blksize, cost, false);
      Hmu_vec[it].gen_Hxlist(wf.info, Hxlst, blksize, cost, true);
   }
   auto tc = tools::get_time();

   // 4. reorder & gen MMlst
   if(hxorder == 1){ // sort by cost
      std::stable_sort(Hxlst.begin(), Hxlst.end(),
           	       [](const Hxblock<Tm>& t1, const Hxblock<Tm>& t2){ 
		       	  return t1.cost > t2.cost; });
   }else if(hxorder == 2){ // sort by cost
      std::stable_sort(Hxlst.begin(), Hxlst.end(),
           	       [](const Hxblock<Tm>& t1, const Hxblock<Tm>& t2){ 
		       	  return t1.cost < t2.cost; });
   }else if(hxorder == 3){ // sort by offout
      std::stable_sort(Hxlst.begin(), Hxlst.end(),
           	       [](const Hxblock<Tm>& t1, const Hxblock<Tm>& t2){ 
		          return t1.offout < t2.offout; });
   }else if(hxorder == 4){ // sort by offin
      std::stable_sort(Hxlst.begin(), Hxlst.end(),
           	       [](const Hxblock<Tm>& t1, const Hxblock<Tm>& t2){ 
		          return t1.offin < t2.offin; });
   } // hxorder
   for(int i=0; i<Hxlst.size(); i++){
      Hxlst[i].get_MMlist();
   }
   auto td = tools::get_time();

   if(debug){
      auto t1 = tools::get_time();
      size_t hxsize = Hxlst.size(); 
      std::cout << "size(Hxlst)=" << hxsize
                << " size(formulae)=" << hsize
	        << " hxsize/hsize=" << hxsize/double(hsize)
		<< " nnzblk=" << wf.info._nnzaddr.size()
		<< " cost=" << cost 
	        << std::endl;
      std::cout << "T(inter/Hmu/Hxlist/sort/tot)="
	        << tools::get_duration(ta-t0) << ","
	        << tools::get_duration(tb-ta) << ","
	        << tools::get_duration(tc-tb) << ","
	        << tools::get_duration(td-tc) << ","
		<< tools::get_duration(t1-t0) 
		<< std::endl;
      tools::timing("preprocess_formulae_sigma", t0, t1);
   }
   return blksize;
}

// for Davidson diagonalization
template <typename Tm> 
void preprocess_Hx(Tm* y,
	           const Tm* x,
		   const Tm& scale,
		   const int& size,
	           const int& rank,
		   const size_t& ndim,
	           const size_t& blksize,
	           Hxlist<Tm>& Hxlst,
		   Tm** opaddr,
		   Tm* workspace){
   const bool debug = false;
#ifdef _OPENMP
   int maxthreads = omp_get_max_threads();
#else
   int maxthreads = 1;
#endif
   if(rank == 0 && debug){
      std::cout << "ctns::preprocess_Hx"
	        << " mpisize=" << size 
                << " maxthreads=" << maxthreads
                << std::endl;
   }

   // initialization
   memset(y, 0, ndim*sizeof(Tm));

   // compute
#ifdef _OPENMP
   #pragma omp parallel for schedule(dynamic)
#endif
   for(int i=0; i<Hxlst.size(); i++){
#ifdef _OPENMP
      int omprank = omp_get_thread_num();
#else
      int omprank = 0;
#endif
      auto& Hxblk = Hxlst[i];
      Tm* wptr = &workspace[omprank*blksize*2];
      Hxblk.kernel(x, opaddr, wptr);
      Tm* rptr = &workspace[omprank*blksize*2+Hxblk.offres];
#ifdef _OPENMP
      #pragma omp critical
#endif
      {
         linalg::xaxpy(Hxblk.size, Hxblk.coeff, rptr, y+Hxblk.offout);
      }
   } // i

   // add const term
   if(rank == 0) linalg::xaxpy(ndim, scale, x, y);
}

} // ctns

#endif
