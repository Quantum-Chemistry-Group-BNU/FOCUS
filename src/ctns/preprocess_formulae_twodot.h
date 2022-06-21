#ifndef PREPROCESS_FORMULAE_TWODOT_H
#define PREPROCESS_FORMULAE_TWODOT_H

#include "preprocess_intermediates.h"
#include "preprocess_contractions.h"

namespace ctns{

template <typename Tm>
void preprocess_formulae_twodot(const oper_dictmap<Tm>& qops_dict,
		 	        const symbolic_task<Tm>& H_formulae,
			        const stensor4<Tm>& wf, 
				const bool debug){
   auto t0 = tools::get_time();
   const auto& lqops = qops_dict.at("l");
   const auto& rqops = qops_dict.at("r");
   const auto& c1qops = qops_dict.at("c1");
   const auto& c2qops = qops_dict.at("c2");

   // 1. form intermediate operators 
   intermediates<Tm> inter;
   inter.init(qops_dict,H_formulae,true);

   // 2. preprocess formulae to Hmu
   const std::map<std::string,int> posmap = {{"l",0},{"r",1},{"c1",2},{"c2",3}};
   int hsize = H_formulae.size();
   std::vector<Hmu_ptr<Tm>> Hmu(hsize);
   for(int it=0; it<hsize; it++){
      const auto& HTerm = H_formulae.tasks[it];
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
	 Hmu[it].info[pos] = const_cast<qinfo2<Tm>*>(&op0.info);
	 Hmu[it].parity[pos] = parity;
	 Hmu[it].dagger[pos] = dagger;
	 if(sop.size() == 1){
	    Hmu[it].coeff *= sop.sums[0].first;
	    Hmu[it].data[pos] = op0.data();
	 }else{
            Hmu[it].data[pos] = inter(it,idx);
         }
      }
   } // it
  
   // 3. from Hmu to expanded block forms 
   std::cout << "hsize=" << hsize << std::endl;
   for(int it=0; it<hsize; it++){
      std::cout << "it=" << it
	        << " coeff=" << Hmu[it].coeff
		<< " lop=" << Hmu[it].info[0]
		<< " rop=" << Hmu[it].info[1]
		<< " c1op=" << Hmu[it].info[2]
		<< " c2op=" << Hmu[it].info[3]
		<< std::endl;
      Hmu[it].gen_Hxblocks(wf);
   }
   std::cout << "hsize=" << hsize << std::endl;
   exit(1);

   if(debug){
      auto t1 = tools::get_time();
      tools::timing("preprocess_formulae_twodot", t0, t1);
   }
}

/*
template <typename Tm, typename QTm, typename QInfo> 
void symbolic_Hx2(Tm* y,
	          const Tm* x,
	          const symbolic_task<Tm>& H_formulae,
	   	  const oper_dictmap<Tm>& qops_dict,
		  const double& ecore,
	          QTm& wf,
		  const int& size,
	          const int& rank,
		  const std::map<qsym,QInfo>& info_dict, 
	          const size_t& opsize,
		  const size_t& wfsize,
		  const size_t& tmpsize,
		  Tm* workspace){
   const bool debug = false;
   auto t0 = tools::get_time();
#ifdef _OPENMP
   int maxthreads = omp_get_max_threads();
#else
   int maxthreads = 1;
#endif
   if(rank == 0 && debug){
      std::cout << "ctns::symbolic_Hx2"
	        << " mpisize=" << size 
                << " maxthreads=" << maxthreads
                << std::endl;
   }
   //=======================
   // Parallel evaluation
   //=======================
   wf.from_array(x);

   // initialization
   std::vector<QTm> Hwfs(maxthreads);
   for(int i=0; i<maxthreads; i++){
      Hwfs[i].init(wf.info, false);
      Hwfs[i].setup_data(&workspace[i*tmpsize]);
      Hwfs[i].clear();
   }
   auto t1 = tools::get_time();
   // compute
#ifdef _OPENMP
   #pragma omp parallel for schedule(dynamic)
#endif
   for(int it=0; it<H_formulae.size(); it++){
#ifdef _OPENMP
      int omprank = omp_get_thread_num();
#else
      int omprank = 0;
#endif
      const auto& HTerm = H_formulae.tasks[it];
      symbolic_HxTerm2(qops_dict,it,HTerm,wf,Hwfs[omprank],
		       info_dict,opsize,wfsize,
		       &workspace[omprank*tmpsize+wfsize],
		       false);
      symbolic_HxTerm2(qops_dict,it,HTerm,wf,Hwfs[omprank],
		       info_dict,opsize,wfsize,
		       &workspace[omprank*tmpsize+wfsize],
		       true);
   } // it
   auto t2 = tools::get_time();
   // reduction & save
   for(int i=1; i<maxthreads; i++){
      Hwfs[0] += Hwfs[i];
   }
   Hwfs[0].to_array(y);

   // add const term
   if(rank == 0){
      const Tm scale = qops_dict.at("l").ifkr? 0.5 : 1.0;
      linalg::xaxpy(wf.size(), scale*ecore, x, y);
   }

   auto t3 = tools::get_time();
   oper_timer.tHxInit += tools::get_duration(t1-t0);
   oper_timer.tHxCalc += tools::get_duration(t2-t1);
   oper_timer.tHxFinl += tools::get_duration(t3-t2);
}
*/

} // ctns

#endif
