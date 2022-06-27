#ifndef PREPROCESS_SIGMA_H
#define PREPROCESS_SIGMA_H

#include "preprocess_inter.h"
#include "preprocess_hmu.h"

namespace ctns{

template <typename Tm, typename QTm>
size_t preprocess_formulae_sigma(const oper_dictmap<Tm>& qops_dict,
		 	         const symbolic_task<Tm>& H_formulae,
			         const QTm& wf,
				 intermediates<Tm>& inter,
   			         Hxlist<Tm>& Hxlst,
				 const int hxorder,
				 const bool debug){
   auto t0 = tools::get_time();

   // 1. form intermediate operators 
   inter.init(qops_dict,H_formulae,debug);
   auto ta = tools::get_time();

   // 2. preprocess formulae to Hmu
   std::map<std::string,int> posmap; 
   Tm* opaddr[5]; // this can be different from qops_dict/inter 
   		     // in case of GPU implementation in future
   if(qops_dict.size() == 4){
      posmap["l"] = 0;
      posmap["r"] = 1;
      posmap["c1"] = 2;
      posmap["c2"] = 3;
      opaddr[0] = qops_dict.at("l")._data;
      opaddr[1] = qops_dict.at("r")._data;
      opaddr[2] = qops_dict.at("c1")._data;
      opaddr[3] = qops_dict.at("c2")._data;
      opaddr[4] = inter._data;
   }else if(qops_dict.size() == 3){
      posmap["l"] = 0;
      posmap["r"] = 1;
      posmap["c"] = 2;
      opaddr[0] = qops_dict.at("l")._data;
      opaddr[1] = qops_dict.at("r")._data;
      opaddr[2] = qops_dict.at("c")._data;
      opaddr[3] = inter._data;
   }
   int hsize = H_formulae.size();
   std::vector<Hmu_ptr<Tm>> Hmu_vec(hsize);
   for(int it=0; it<hsize; it++){
      Hmu_vec[it].init(it, H_formulae, qops_dict, inter, 
		       posmap, opaddr);
   } // it
   auto tb = tools::get_time();

   // 3. from Hmu to expanded block forms
   size_t blksize = 0;
   double cost = 0.0;
   for(int it=0; it<hsize; it++){
      Hmu_vec[it].gen_Hxlist(wf.info, Hxlst, blksize, cost, false);
      Hmu_vec[it].gen_Hxlist(wf.info, Hxlst, blksize, cost, true);
   }
   auto tc = tools::get_time();

   // 4. reorder hxlist
   if(hxorder == 1){
      // sort by cost
      std::stable_sort(Hxlst.begin(), Hxlst.end(),
           	       [](const Hxblock<Tm>& t1, const Hxblock<Tm>& t2){ return t1.cost > t2.cost; });
   }else if(hxorder == 2){
      // sort by cost
      std::stable_sort(Hxlst.begin(), Hxlst.end(),
           	       [](const Hxblock<Tm>& t1, const Hxblock<Tm>& t2){ return t1.cost < t2.cost; });
   }else if(hxorder == 3){
      // sort by offout
      std::stable_sort(Hxlst.begin(), Hxlst.end(),
           	       [](const Hxblock<Tm>& t1, const Hxblock<Tm>& t2){ return t1.offout < t2.offout; });
   }else if(hxorder == 4){
      // sort by offin
      std::stable_sort(Hxlst.begin(), Hxlst.end(),
           	       [](const Hxblock<Tm>& t1, const Hxblock<Tm>& t2){ return t1.offin < t2.offin; });
   }
   auto td = tools::get_time();

   if(debug){
      auto t1 = tools::get_time();
      size_t hxsize = Hxlst.size(); 
      std::cout << "size(Hxlst)=" << hxsize
                << " size(formulae)=" << hsize
	        << " average=" << hxsize/double(hsize)
		<< " nnzblks=" << wf.info._nnzaddr.size()
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
      Hxblk.kernel(x, wptr);
#ifdef _OPENMP
      #pragma omp critical
#endif
      {
         linalg::xaxpy(Hxblk.size, Hxblk.coeff, wptr+Hxblk.offres, y+Hxblk.offout);
      }
   } // i

   // add const term
   if(rank == 0){
      linalg::xaxpy(ndim, scale, x, y);
   }
}

template <typename Tm, typename QTm>
size_t preprocess_formulae_sigma2(const oper_dictmap<Tm>& qops_dict,
		 	         const symbolic_task<Tm>& H_formulae,
			         const QTm& wf,
				 intermediates<Tm>& inter,
   			         Hxlist2<Tm>& Hxlst2,
				 const int hxorder,
				 const bool debug){
   auto t0 = tools::get_time();

   // 1. form intermediate operators 
   inter.init(qops_dict,H_formulae,debug);
   auto ta = tools::get_time();

   // 2. preprocess formulae to Hmu
   std::map<std::string,int> posmap; 
   Tm* opaddr[5]; // this can be different from qops_dict/inter 
   		     // in case of GPU implementation in future
   if(qops_dict.size() == 4){
      posmap["l"] = 0;
      posmap["r"] = 1;
      posmap["c1"] = 2;
      posmap["c2"] = 3;
      opaddr[0] = qops_dict.at("l")._data;
      opaddr[1] = qops_dict.at("r")._data;
      opaddr[2] = qops_dict.at("c1")._data;
      opaddr[3] = qops_dict.at("c2")._data;
      opaddr[4] = inter._data;
   }else if(qops_dict.size() == 3){
      posmap["l"] = 0;
      posmap["r"] = 1;
      posmap["c"] = 2;
      opaddr[0] = qops_dict.at("l")._data;
      opaddr[1] = qops_dict.at("r")._data;
      opaddr[2] = qops_dict.at("c")._data;
      opaddr[3] = inter._data;
   }
   int hsize = H_formulae.size();
   std::vector<Hmu_ptr<Tm>> Hmu_vec(hsize);
   for(int it=0; it<hsize; it++){
      Hmu_vec[it].init(it, H_formulae, qops_dict, inter, 
		       posmap, opaddr);
   } // it
   auto tb = tools::get_time();

   // 3. from Hmu to expanded block forms
   int nnzblk = wf.info._nnzaddr.size();
   Hxlst2.resize(nnzblk);
   size_t blksize = 0;
   double cost = 0.0;
   for(int it=0; it<hsize; it++){
      Hmu_vec[it].gen_Hxlist2(wf.info, Hxlst2, blksize, cost, false);
      Hmu_vec[it].gen_Hxlist2(wf.info, Hxlst2, blksize, cost, true);
   }
   auto tc = tools::get_time();

   if(hxorder == 1){
      // sort by cost
      for(int i=0; i<nnzblk; i++){
	 auto& Hxlst = Hxlst2[i];
         std::stable_sort(Hxlst.begin(), Hxlst.end(),
           	          [](const Hxblock<Tm>& t1, const Hxblock<Tm>& t2){ return t1.cost > t2.cost; });
      }
   }else if(hxorder == 2){
      // sort by cost
      for(int i=0; i<nnzblk; i++){
	 auto& Hxlst = Hxlst2[i];     
         std::stable_sort(Hxlst.begin(), Hxlst.end(),
           	          [](const Hxblock<Tm>& t1, const Hxblock<Tm>& t2){ return t1.cost < t2.cost; });
      }
   }else if(hxorder == 4){
      // sort by offin
      for(int i=0; i<nnzblk; i++){
	 auto& Hxlst = Hxlst2[i];     
         std::stable_sort(Hxlst.begin(), Hxlst.end(),
           	          [](const Hxblock<Tm>& t1, const Hxblock<Tm>& t2){ return t1.offin < t2.offin; });
      }
   }
 
   auto td = tools::get_time();

   if(debug){
      auto t1 = tools::get_time();
      size_t hxsize = 0;
      for(int i=0; i<nnzblk; i++){
	 hxsize += Hxlst2[i].size();
      }
      std::cout << "size(Hxlst2)=" << nnzblk
	        << " size(Hxlst)=" << hxsize
                << " size(formulae)=" << hsize
	        << " average=" << hxsize/double(nnzblk)
		<< " cost=" << cost 
	        << std::endl;
      std::cout << "T(inter/Hmu/Hxlist/sort/tot)="
	        << tools::get_duration(ta-t0) << ","
	        << tools::get_duration(tb-ta) << ","
	        << tools::get_duration(tc-tb) << ","
	        << tools::get_duration(td-tc) << ","
		<< tools::get_duration(t1-t0) 
		<< std::endl;
      tools::timing("preprocess_formulae_sigma2", t0, t1);
   }
   return blksize;
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
		   Tm* workspace){
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

   // compute
   size_t off = maxthreads*blksize;
   for(int i=0; i<Hxlst2.size(); i++){
      memset(workspace, 0, off*sizeof(Tm));
#ifdef _OPENMP
   #pragma omp parallel for schedule(dynamic)
#endif
      for(int j=0; j<Hxlst2[i].size(); j++){
#ifdef _OPENMP
         int omprank = omp_get_thread_num();
#else
         int omprank = 0;
#endif
         auto& Hxblk = Hxlst2[i][j];
         Tm* wptr = &workspace[off+omprank*blksize*2];
         Hxblk.kernel(x, wptr);
	 // save to local memory
         linalg::xaxpy(Hxblk.size, Hxblk.coeff, wptr+Hxblk.offres, &workspace[omprank*blksize]);
      } // j
      // reduction
      const auto& Hxblk = Hxlst2[i][0];
      for(int k=0; k<maxthreads; k++){
         linalg::xaxpy(Hxblk.size, 1.0, &workspace[k*blksize], y+Hxblk.offout);
      } // k
   } // i

   // add const term
   if(rank == 0){
      linalg::xaxpy(ndim, scale, x, y);
   }
}


} // ctns

#endif
