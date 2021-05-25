#ifndef SWEEP_PRDM_H
#define SWEEP_PRDM_H

namespace ctns{

const double thresh_noise = 1.e-10;
extern const double thresh_noise;

// ZL@20210420:	
// Note that for perturbative noise, the signs are not properly treated as in 
// sweep_onedot_ham.h. We assume that they are not essential for recovering
// the losing quantum number! This proved to work for simple tests at least.

template <typename Tm>
void get_prdm(const std::string& superblock,
	      const bool& ifkr,
	      const qtensor3<Tm>& wf, 
	      oper_dict<Tm>& qops1, 
	      oper_dict<Tm>& qops2, 
	      const double noise, 
	      qtensor2<Tm>& rdm,
	      const int size,
	      const int rank){
   if(noise < thresh_noise) return;
   if(rank == 0){
      std::cout << "ctns::get_prdm_"+superblock+" noise=" << std::setprecision(2) << noise;
   }
   const bool dagger = true;
   auto t0 = tools::get_time();
   // pRDM from a(+)^12|psi>: this part can also be parallelized,
   // 			      because all opC are stored locally.
   auto infoC = oper_combine_opC(qops1.cindex, qops2.cindex);
   for(const auto pr : infoC){
      int iformula = pr.first;
      int index = pr.second;
      int iproc = index%size;
      if(iproc == rank){
         auto qt3n = oper_normxwf_opC(superblock,wf,qops1,qops1,iformula,index);
         auto qt3h = oper_normxwf_opC(superblock,wf,qops2,qops2,iformula,index,dagger);
	 rdm += noise*qt3n.get_rdm(superblock);
	 rdm += noise*qt3h.get_rdm(superblock);
      }
   }
   // pRDM from A(+)^12|psi>:
   auto infoA = oper_combine_opA(qops1.cindex, qops2.cindex, ifkr);
   for(const auto pr : infoA){
      int iformula = pr.first;
      int index = pr.second;
      int iproc = distribute2(index, size);
      if(iproc == rank){
         auto qt3n = oper_normxwf_opA(superblock,wf,qops1,qops2,ifkr,iformula,index);
         auto qt3h = oper_normxwf_opA(superblock,wf,qops1,qops2,ifkr,iformula,index,dagger);
	 rdm += noise*qt3n.get_rdm(superblock);
	 rdm += noise*qt3h.get_rdm(superblock);
      }
   }
   // pRDM from B(+)^12|psi>:
   auto infoB = oper_combine_opB(qops1.cindex, qops2.cindex, ifkr);
   for(const auto pr : infoB){
      int iformula = pr.first;
      int index = pr.second;
      int iproc = distribute2(index, size);
      if(iproc == rank){
         auto qt3n = oper_normxwf_opB(superblock,wf,qops1,qops2,ifkr,iformula,index);
         auto qt3h = oper_normxwf_opB(superblock,wf,qops1,qops2,ifkr,iformula,index,dagger);
	 rdm += noise*qt3n.get_rdm(superblock);
	 rdm += noise*qt3h.get_rdm(superblock);
      }
   }
   auto t1 = tools::get_time();
   if(rank == 0){
      std::cout << " timing : " << tools::get_duration(t1-t0) << " s" << std::endl;
   } 
}

} // ctns

#endif
