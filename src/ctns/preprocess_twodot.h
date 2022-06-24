#ifndef PREPROCESS_TWODOT_H
#define PREPROCESS_TWODOT_H

#include "preprocess_inter.h"
#include "preprocess_hmu.h"

namespace ctns{

template <typename Tm>
size_t preprocess_formulae_twodot(const oper_dictmap<Tm>& qops_dict,
		 	          const symbolic_task<Tm>& H_formulae,
			          const stensor4<Tm>& wf,
				  intermediates<Tm>& inter,
   			          Hxlist<Tm>& Hxlst,
				  const bool debug){
   auto t0 = tools::get_time();
   const auto& lqops = qops_dict.at("l");
   const auto& rqops = qops_dict.at("r");
   const auto& c1qops = qops_dict.at("c1");
   const auto& c2qops = qops_dict.at("c2");

   // 1. form intermediate operators 
   inter.init(qops_dict,H_formulae,debug);
   auto ta = tools::get_time();

   // 2. preprocess formulae to Hmu
   const std::map<std::string,int> posmap = {{"l",0},{"r",1},{"c1",2},{"c2",3}};
   int hsize = H_formulae.size();
   std::vector<Hmu_ptr<Tm>> Hmu_vec(hsize);
   for(int it=0; it<hsize; it++){
      const auto& HTerm = H_formulae.tasks[it];
      auto& Hmu = Hmu_vec[it];
      for(int idx=HTerm.size()-1; idx>=0; idx--){
         const auto& sop = HTerm.terms[idx];
	 const auto& sop0 = sop.sums[0].second;
         const auto& parity = sop0.parity;
         const auto& dagger = sop0.dagger;
         const auto& block = sop0.block;
         const auto& label = sop0.label;
	 const auto& index0 = sop0.index;
	 const auto& qops = qops_dict.at(block); 
	 const auto& op0 = qops(label).at(index0);
	 int pos = posmap.at(block); 
	 Hmu.parity[pos] = parity;
	 Hmu.dagger[pos] = dagger;
	 Hmu.info[pos] = const_cast<qinfo2<Tm>*>(&op0.info);
	 if(sop.size() == 1){
	    Hmu.location[pos] = pos;
	    Hmu.coeff *= sop.sums[0].first;
	    Hmu.offop[pos] = qops._offset.at(std::make_pair(label,index0));
	 }else{
	    Hmu.location[pos] = 4; // {l,r,c1,c2,i} 
            Hmu.offop[pos] = inter._offset.at(std::make_pair(it,idx));
         }
      } // idx
      Hmu.coeffH = Hmu.coeff*HTerm.Hsign(); 
   } // it
   auto tb = tools::get_time();

   // 3. from Hmu to expanded block forms
   size_t blksize = 0; 
   for(int it=0; it<hsize; it++){
      size_t tmpsize0 = Hmu_vec[it].gen_Hxlist(wf.info, Hxlst, false);
      blksize = std::max(blksize, tmpsize0);
      size_t tmpsize1 = Hmu_vec[it].gen_Hxlist(wf.info, Hxlst, true);
      blksize = std::max(blksize, tmpsize1);
   }
   auto tc = tools::get_time();

   if(debug){
      auto t1 = tools::get_time();
      std::cout << "size(formulae)=" << hsize
	        << " size(Hxlst)=" << Hxlst.size() 
                << " T(inter/Hmu/Hxlist/tot)="
	        << tools::get_duration(ta-t0) << ","
	        << tools::get_duration(tb-ta) << ","
	        << tools::get_duration(tc-tb) << ","
		<< tools::get_duration(t1-t0) 
		<< std::endl;
      tools::timing("preprocess_formulae_twodot", t0, t1);
   }
   return blksize;
}

// for Davidson diagonalization
template <typename Tm> 
void preprocess_twodot_Hx(Tm* y,
	                  const Tm* x,
	                  const Hxlist<Tm>& Hxlst,
		          const Tm& scale,
		          const int& size,
	                  const int& rank,
			  const size_t& ndim,
	                  const size_t& blksize,
	   	          Tm** qops_addr,
		          Tm* workspace){
   const bool debug = false;
#ifdef _OPENMP
   int maxthreads = omp_get_max_threads();
#else
   int maxthreads = 1;
#endif
   if(rank == 0 && debug){
      std::cout << "ctns::preprocess_twodot_Hx"
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
      const auto& Hxblk = Hxlst[i];
      Tm* wptr = &workspace[omprank*blksize*2];
      Tm* rptr = Hxblk.kernel(x, blksize, qops_addr, wptr);
#ifdef _OPENMP
      #pragma omp critical
#endif
      {
         linalg::xaxpy(Hxblk.size, Hxblk.coeff, rptr, y+Hxblk.offout);
      }
   } // i

   // add const term
   if(rank == 0){
      linalg::xaxpy(ndim, scale, x, y);
   }
}

} // ctns

#endif
