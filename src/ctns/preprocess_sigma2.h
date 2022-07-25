#ifndef PREPROCESS_SIGMA2_H
#define PREPROCESS_SIGMA2_H

#include "preprocess_inter.h"
#include "preprocess_hmu.h"

namespace ctns{

template <typename Tm, typename QTm>
void preprocess_formulae_sigma2(const oper_dictmap<Tm>& qops_dict,
				  const std::map<std::string,int>& oploc,
		 	          const symbolic_task<Tm>& H_formulae,
			          const QTm& wf,
				  intermediates<Tm>& inter,
   			          Hxlist2<Tm>& Hxlst2,
				  size_t& blksize,
				  double& cost,
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

   // 3. from Hmu to expanded block forms
   blksize = 0;
   cost = 0.0;
   int nnzblk = wf.info._nnzaddr.size();
   Hxlst2.resize(nnzblk);
   for(int it=0; it<hsize; it++){
      Hmu_vec[it].gen_Hxlist2(wf.info, Hxlst2, blksize, cost, false);
      Hmu_vec[it].gen_Hxlist2(wf.info, Hxlst2, blksize, cost, true);
   }
   auto tc = tools::get_time();

   // 4. reorder & gen MMlst
   for(int i=0; i<nnzblk; i++){
      auto& Hxlst = Hxlst2[i];
      if(hxorder == 1){ // sort by cost
         std::stable_sort(Hxlst.begin(), Hxlst.end(),
           	          [](const Hxblock<Tm>& t1, const Hxblock<Tm>& t2){ 
			     return t1.cost > t2.cost; });
      }else if(hxorder == 2){ // sort by cost
         std::stable_sort(Hxlst.begin(), Hxlst.end(),
           	          [](const Hxblock<Tm>& t1, const Hxblock<Tm>& t2){ 
			     return t1.cost < t2.cost; });
      }else if(hxorder == 4){ // sort by offin
         std::stable_sort(Hxlst.begin(), Hxlst.end(),
           	          [](const Hxblock<Tm>& t1, const Hxblock<Tm>& t2){ 
			     return t1.offin < t2.offin; });
      } // hxorder
      for(int j=0; j<Hxlst.size(); j++){
	 Hxlst[j].get_MMlist();
      }
   } // i
   auto td = tools::get_time();

   if(debug){
      auto t1 = tools::get_time();
      std::cout << "T(inter/Hmu/Hxlist/sort/tot)="
	        << tools::get_duration(ta-t0) << ","
	        << tools::get_duration(tb-ta) << ","
	        << tools::get_duration(tc-tb) << ","
	        << tools::get_duration(td-tc) << ","
		<< tools::get_duration(t1-t0) 
		<< std::endl;
      tools::timing("preprocess_formulae_sigma2", t0, t1);
   }
}

// for Davidson diagonalization
template <typename Tm> 
void preprocess_Hx2(Tm* y,
	           const Tm* x,
		   const Tm& scale,
		   const int& size,
	           const int& rank,
		   const size_t& ndim,
	           const size_t& blksize,
	           Hxlist2<Tm>& Hxlst2,
		   Tm** opaddr){
   const bool debug = false;
#ifdef _OPENMP
   int maxthreads = omp_get_max_threads();
#else
   int maxthreads = 1;
#endif
   if(rank == 0 && debug){
      std::cout << "ctns::preprocess_Hx2"
	        << " mpisize=" << size 
                << " maxthreads=" << maxthreads
                << std::endl;
   }

   // initialization
   memset(y, 0, ndim*sizeof(Tm));

   // compute Y[I] = \sum_J H[I,J] X[J]
#ifdef _OPENMP
   #pragma omp parallel
   {
#endif

   Tm* work = new Tm[blksize*3];
   for(int i=0; i<Hxlst2.size(); i++){
      memset(work, 0, blksize*sizeof(Tm));
#ifdef _OPENMP
      #pragma omp for schedule(dynamic) nowait
#endif
      for(int j=0; j<Hxlst2[i].size(); j++){
         auto& Hxblk = Hxlst2[i][j];
         Tm* wptr = &work[blksize];
         Hxblk.kernel(x, opaddr, wptr);
	 Tm* rptr = &work[blksize+Hxblk.offres];
	 // save to local memory
         linalg::xaxpy(Hxblk.size, Hxblk.coeff, rptr, work);
      } // j
      if(Hxlst2[i].size()>0){
         const auto& Hxblk = Hxlst2[i][0];
#ifdef _OPENMP
         #pragma omp critical
#endif
         linalg::xaxpy(Hxblk.size, 1.0, work, y+Hxblk.offout);
      }
   } // i
   delete[] work;

#ifdef _OPENMP
   }
#endif

   // add const term
   if(rank == 0) linalg::xaxpy(ndim, scale, x, y);
}

} // ctns

#endif
