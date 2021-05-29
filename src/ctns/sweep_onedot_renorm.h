#ifndef SWEEP_ONEDOT_RENORM_H
#define SWEEP_ONEDOT_RENORM_H

#include "oper_io.h"
#include "sweep_prdm.h"
#include "sweep_decimation.h"

namespace ctns{

template <typename Km>
void onedot_renorm_lc(sweep_data& sweeps,
		      const int isweep,
		      const int ibond, 
		      comb<Km>& icomb,
		      const linalg::matrix<typename Km::dtype>& vsol,
		      qtensor3<typename Km::dtype>& wf,
		      oper_dict<typename Km::dtype>& lqops,
		      oper_dict<typename Km::dtype>& cqops,
	              const integral::two_body<typename Km::dtype>& int2e, 
	              const integral::one_body<typename Km::dtype>& int1e,
	              const std::string scratch){
   int size = 1, rank = 0;
#ifndef SERIAL
   size = icomb.world.size();
   rank = icomb.world.rank();
#endif
   using Tm = typename Km::dtype; 
   const bool ifkr = kind::is_kramers<Km>();
   const auto& dbond = sweeps.seq[ibond];
   const auto& dcut  = sweeps.ctrls[isweep].dcut;
   const auto& noise = sweeps.ctrls[isweep].noise;
   if(rank == 0){
      std::cout << " |lc> (forward,cturn,dcut,inoise,noise)="
                << dbond.forward << "," << dbond.cturn << ","
                << dcut << "," << sweeps.inoise << ","
                << std::scientific << std::setprecision(1) << noise << std::endl;
   }
   auto& timing = sweeps.opt_timing[isweep][ibond];
   auto& result = sweeps.opt_result[isweep][ibond];
   // Renormalize superblock = lc
   auto qprod = qmerge(wf.qrow, wf.qmid);
   auto qlc = qprod.first;
   auto dpt = qprod.second;
   qtensor2<Tm> rdm(qsym(), qlc, qlc);
   // 1. build pRDM 
   for(int i=0; i<vsol.cols(); i++){
      wf.from_array(vsol.col(i));
      if(sweeps.inoise > 0) get_prdm("lc", ifkr, wf, lqops, cqops, noise, rdm, size, rank);
      if(rank == 0){
         if(sweeps.inoise > 1) wf.add_noise(noise);
         rdm += wf.get_rdm("lc"); // only rank-0 build RDM
      }
   }
#ifndef SERIAL
   if(size > 1){
      qtensor2<Tm> rdm2; 
      boost::mpi::reduce(icomb.world, rdm, rdm2, std::plus<qtensor2<Tm>>(), 0);
      rdm = rdm2;      
   }
#endif 
   // 2. decimation
   qtensor2<Tm> rot;
   if(rank == 0){
      rot = decimation_row(rdm, dcut, result.dwt, result.deff,
           	   	   ifkr, wf.qrow, wf.qmid, dpt);
   }
#ifndef SERIAL
   if(size > 1) boost::mpi::broadcast(icomb.world, rot, 0); 
#endif
   // 3. update site tensor
   const auto& p = sweeps.seq[ibond].p;
   icomb.lsites[p] = rot.split_lc(wf.qrow, wf.qmid, dpt);
   //-------------------------------------------------------------------	 
   assert((rot-icomb.lsites[p].merge_lc()).normF() < 1.e-10);
   auto ovlp = contract_qt3_qt3_lc(icomb.lsites[p],icomb.lsites[p]);
   assert(ovlp.check_identityMatrix(1.e-10,false)<1.e-10);
   //-------------------------------------------------------------------
   // 4. initial guess for next site within the bond
   if(rank == 0 && sweeps.guess){
      const auto& p1 = sweeps.seq[ibond].p1;	   
      icomb.psi.clear();
      for(int i=0; i<vsol.cols(); i++){
         wf.from_array(vsol.col(i));
         auto cwf = rot.H().dot(wf.merge_lc()); // <-W[alpha,r]->
         auto psi = contract_qt3_qt2_l(icomb.rsites[p1],cwf);
         icomb.psi.push_back(psi);
      }
   }
   timing.td = tools::get_time();
   // 5. renorm operators	 
   oper_dict<Tm> qops;
   oper_renorm_opAll("lc", icomb, p, int2e, int1e, lqops, cqops, qops);
   timing.te = tools::get_time();
   auto fname = oper_fname(scratch, p, "l");
   oper_save(fname, qops);
}

template <typename Km>
void onedot_renorm_lr(sweep_data& sweeps,
		      const int isweep,
		      const int ibond, 
		      comb<Km>& icomb,
		      const linalg::matrix<typename Km::dtype>& vsol,
		      qtensor3<typename Km::dtype>& wf,
		      oper_dict<typename Km::dtype>& lqops,
		      oper_dict<typename Km::dtype>& rqops,
	              const integral::two_body<typename Km::dtype>& int2e, 
	              const integral::one_body<typename Km::dtype>& int1e,
	              const std::string scratch){
   int size = 1, rank = 0;
#ifndef SERIAL
   size = icomb.world.size();
   rank = icomb.world.rank();
#endif
   using Tm = typename Km::dtype; 
   const bool ifkr = kind::is_kramers<Km>();
   const auto& dbond = sweeps.seq[ibond];
   const auto& dcut  = sweeps.ctrls[isweep].dcut;
   const auto& noise = sweeps.ctrls[isweep].noise;
   if(rank == 0){ 
      std::cout << " |lr>(comb) (forward,cturn,dcut,inoise,noise)=" 
                << dbond.forward << "," << dbond.cturn << "," 
                << dcut << "," << sweeps.inoise << ","
                << std::scientific << std::setprecision(1) << noise << std::endl;
   }
   auto& timing = sweeps.opt_timing[isweep][ibond];
   auto& result = sweeps.opt_result[isweep][ibond];
   // Renormalize superblock = lr
   auto qprod = qmerge(wf.qrow, wf.qcol);
   auto qlr = qprod.first;
   auto dpt = qprod.second;
   qtensor2<Tm> rdm(qsym(), qlr, qlr);
   // 1. build pRDM 
   for(int i=0; i<vsol.cols(); i++){
      wf.from_array(vsol.col(i));
      if(sweeps.inoise > 0) get_prdm("lr", ifkr, wf, lqops, rqops, noise, rdm, size, rank);
      if(rank == 0){
	 if(sweeps.inoise > 1) wf.add_noise(noise);
         rdm += wf.get_rdm("lr"); // only rank-0 build RDM
      }
   }
#ifndef SERIAL
   if(size > 1){
      qtensor2<Tm> rdm2; 
      boost::mpi::reduce(icomb.world, rdm, rdm2, std::plus<qtensor2<Tm>>(), 0);
      rdm = rdm2;      
   }
#endif 
   // 2. decimation
   qtensor2<Tm> rot;
   if(rank == 0){
      rot = decimation_row(rdm, dcut, result.dwt, result.deff,
		           ifkr, wf.qrow, wf.qcol, dpt);
   }
#ifndef SERIAL
   if(size > 1) boost::mpi::broadcast(icomb.world, rot, 0); 
#endif
   // 3. update site tensor
   const auto& p = sweeps.seq[ibond].p;
   icomb.lsites[p]= rot.split_lr(wf.qrow, wf.qcol, dpt);
   //-------------------------------------------------------------------	 
   assert((rot-icomb.lsites[p].merge_lr()).normF() < 1.e-10);
   auto ovlp = contract_qt3_qt3_lr(icomb.lsites[p],icomb.lsites[p]);
   assert(ovlp.check_identityMatrix(1.e-10,false)<1.e-10);
   //-------------------------------------------------------------------	
   // 4. initial guess for next site within the bond
   if(rank == 0 && sweeps.guess){	
      const auto& p1 = sweeps.seq[ibond].p1;	   
      icomb.psi.clear();
      for(int i=0; i<vsol.cols(); i++){
         wf.from_array(vsol.col(i));
         auto cwf = rot.H().dot(wf.permCR_signed().merge_lr()); // <-W[alpha,r]->
         auto psi = contract_qt3_qt2_l(icomb.rsites[p1],cwf);
         icomb.psi.push_back(psi);
      }
   }
   timing.td = tools::get_time();
   // 5. renorm operators	 
   oper_dict<Tm> qops;
   oper_renorm_opAll("lr", icomb, p, int2e, int1e, lqops, rqops, qops);
   timing.te = tools::get_time();
   auto fname = oper_fname(scratch, p, "l");
   oper_save(fname, qops);
}

template <typename Km>
void onedot_renorm_cr(sweep_data& sweeps,
		      const int isweep,
		      const int ibond, 
		      comb<Km>& icomb,
		      const linalg::matrix<typename Km::dtype>& vsol,
		      qtensor3<typename Km::dtype>& wf,
		      oper_dict<typename Km::dtype>& cqops,
		      oper_dict<typename Km::dtype>& rqops,
	              const integral::two_body<typename Km::dtype>& int2e, 
	              const integral::one_body<typename Km::dtype>& int1e,
	              const std::string scratch){
   int size = 1, rank = 0;
#ifndef SERIAL
   size = icomb.world.size();
   rank = icomb.world.rank();
#endif
   using Tm = typename Km::dtype; 
   const bool ifkr = kind::is_kramers<Km>();
   const auto& dbond = sweeps.seq[ibond];
   const auto& dcut  = sweeps.ctrls[isweep].dcut;
   const auto& noise = sweeps.ctrls[isweep].noise;
   if(rank == 0){ 
      std::cout << " |cr> (forward,cturn,dcut,inoise,noise)="
   	        << dbond.forward << "," << dbond.cturn << "," 
   	        << dcut << "," << sweeps.inoise << ","
   	        << std::scientific << std::setprecision(1) << noise << std::endl;
   }
   auto& timing = sweeps.opt_timing[isweep][ibond];
   auto& result = sweeps.opt_result[isweep][ibond];
   // Renormalize superblock = cr
   auto qprod = qmerge(wf.qmid, wf.qcol);
   auto qcr = qprod.first;
   auto dpt = qprod.second;
   qtensor2<Tm> rdm(qsym(), qcr, qcr);
   // 1. build pRDM 
   for(int i=0; i<vsol.cols(); i++){
      wf.from_array(vsol.col(i));
      if(sweeps.inoise > 0) get_prdm("cr", ifkr, wf, cqops, rqops, noise, rdm, size, rank);
      if(rank == 0){
         if(sweeps.inoise > 1) wf.add_noise(noise);
         rdm += wf.get_rdm("cr");
      }
   }
#ifndef SERIAL
   if(size > 1){
      qtensor2<Tm> rdm2; 
      boost::mpi::reduce(icomb.world, rdm, rdm2, std::plus<qtensor2<Tm>>(), 0);
      rdm = rdm2;      
   }
#endif
   // 2. decimation
   qtensor2<Tm> rot;
   if(rank == 0){
      rot = decimation_row(rdm, dcut, result.dwt, result.deff,
		           ifkr, wf.qmid, wf.qcol, dpt).T(); // permute two lines for RCF
   }
#ifndef SERIAL
   if(size > 1) boost::mpi::broadcast(icomb.world, rot, 0); 
#endif
   // 3. update site tensor
   const auto& p = sweeps.seq[ibond].p;
   icomb.rsites[p] = rot.split_cr(wf.qmid, wf.qcol, dpt);
   //-------------------------------------------------------------------	
   assert((rot-icomb.rsites[p].merge_cr()).normF() < 1.e-10);	 
   auto ovlp = contract_qt3_qt3_cr(icomb.rsites[p],icomb.rsites[p]);
   assert(ovlp.check_identityMatrix(1.e-10,false)<1.e-10);
   //-------------------------------------------------------------------	
   // 4. initial guess for next site within the bond
   if(rank == 0 && sweeps.guess){
      const auto& p0 = sweeps.seq[ibond].p0;	  
      const auto& cturn = sweeps.seq[ibond].cturn; 
      icomb.psi.clear();
      for(int i=0; i<vsol.cols(); i++){
         wf.from_array(vsol.col(i));
         auto cwf = wf.merge_cr().dot(rot.H()); // <-W[l,alpha]->
	 qtensor3<typename Km::dtype> psi;
         if(!cturn){
            psi = contract_qt3_qt2_r(icomb.lsites[p0],cwf.T());
         }else{
            // special treatment of the propagation downside to backbone
            psi = contract_qt3_qt2_c(icomb.lsites[p0],cwf.T());
            psi = psi.permCR_signed(); // |(lr)c> back to |lcr> order on backbone
         }
         icomb.psi.push_back(psi);
      }
   }
   timing.td = tools::get_time();
   // 5. renorm operators	 
   oper_dict<Tm> qops;
   oper_renorm_opAll("cr", icomb, p, int2e, int1e, cqops, rqops, qops);
   timing.te = tools::get_time();
   auto fname = oper_fname(scratch, p, "r");
   oper_save(fname, qops);
}

template <typename Km>
void onedot_renorm(sweep_data& sweeps,
		   const int isweep,
		   const int ibond, 
		   comb<Km>& icomb,
		   const linalg::matrix<typename Km::dtype>& vsol,
		   qtensor3<typename Km::dtype>& wf,
		   oper_dict<typename Km::dtype>& cqops,
		   oper_dict<typename Km::dtype>& lqops,
		   oper_dict<typename Km::dtype>& rqops,
	           const integral::two_body<typename Km::dtype>& int2e, 
	           const integral::one_body<typename Km::dtype>& int1e,
	           const std::string scratch){
   int size = 1, rank = 0;
#ifndef SERIAL
   size = icomb.world.size();
   rank = icomb.world.rank();
#endif
   if(rank == 0) std::cout << "ctns::onedot_renorm";
   // build reduced density matrix & perform renormalization
   const auto& dbond = sweeps.seq[ibond];
   if(dbond.forward){
      // update lsites & ql
      if(!dbond.cturn){
	 onedot_renorm_lc(sweeps, isweep, ibond, icomb, vsol, wf, 
			  lqops, cqops, int2e, int1e, scratch);
      }else{
	 // special for comb
         onedot_renorm_lr(sweeps, isweep, ibond, icomb, vsol, wf, 
			  lqops, rqops, int2e, int1e, scratch);
      } // cturn
   }else{
      // update rsites & qr
      onedot_renorm_cr(sweeps, isweep, ibond, icomb, vsol, wf, 
		       cqops, rqops, int2e, int1e, scratch); 
   }
}

} // ctns

#endif
