#ifndef SWEEP_ONEDOT_RENORM_H
#define SWEEP_ONEDOT_RENORM_H

#include "oper_io.h"
#include "sweep_prdm.h"
#include "sweep_decimation.h"
#include "sweep_decimation2.h"

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
   const int& dbranch = sweeps.dbranch;
   const int dcut = (dbranch>0 && dbond.p1.second>0)? dbranch : sweeps.ctrls[isweep].dcut;
   const auto& noise = sweeps.ctrls[isweep].noise;
   const auto& rdm_vs_svd = sweeps.rdm_vs_svd;
   if(rank == 0){
      std::cout << " |lc> (dbranch,dcut,inoise,noise,rdm_vs_svd)="
                << dbranch << "," << dcut << "," << sweeps.inoise << ","
                << std::scientific << std::setprecision(1) << noise << ","
		<< rdm_vs_svd
		<< std::endl;
   }
   auto& timing = sweeps.opt_timing[isweep][ibond];
   auto& result = sweeps.opt_result[isweep][ibond];

   // Renormalize superblock = lc
   qtensor2<Tm> rot;
   if(rank == 0){
      std::cout << "0. decimation" << std::endl;
      std::vector<qtensor2<Tm>> wfs2;
      for(int i=0; i<vsol.cols(); i++){
         wf.from_array(vsol.col(i));
	 if(noise > 1.e-10) wf.add_noise(noise);
         auto wf2 = wf.merge_lc();
	 wfs2.push_back(wf2);
      }
      decimation_row(ifkr, wf.qrow, wf.qmid, dcut, rdm_vs_svd, wfs2, 
		     rot, result.dwt, result.deff);

      // 4. initial guess for next site within the bond
      if(sweeps.guess){
         std::cout << "4. initial guess" << std::endl;
         const auto& p1 = sweeps.seq[ibond].p1;	   
         icomb.psi.clear();
         for(int i=0; i<vsol.cols(); i++){
            wf.from_array(vsol.col(i));
            auto cwf = rot.H().dot(wf.merge_lc()); // <-W[alpha,r]->
            auto psi = contract_qt3_qt2_l(icomb.rsites[p1],cwf);
            icomb.psi.push_back(psi);
         }
      }

   }
#ifndef SERIAL
   if(size > 1) boost::mpi::broadcast(icomb.world, rot, 0); 
#endif

   // 3. update site tensor
   if(rank == 0) std::cout << "3. update site tensor" << std::endl;
   const auto& p = sweeps.seq[ibond].p;
   icomb.lsites[p] = rot.split_lc(wf.qrow, wf.qmid);
   //-------------------------------------------------------------------	 
   assert((rot-icomb.lsites[p].merge_lc()).normF() < 1.e-10);
   auto ovlp = contract_qt3_qt3_lc(icomb.lsites[p],icomb.lsites[p]);
   assert(ovlp.check_identityMatrix(1.e-10,false)<1.e-10);
   //-------------------------------------------------------------------
   timing.td = tools::get_time();
   // 5. renorm operators	 
   if(rank == 0) std::cout << "5. renormalize operators" << std::endl;
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
   const int& dbranch = sweeps.dbranch;
   const int dcut = (dbranch>0 && dbond.p1.second>0)? dbranch : sweeps.ctrls[isweep].dcut;
   const auto& noise = sweeps.ctrls[isweep].noise;
   const auto& rdm_vs_svd = sweeps.rdm_vs_svd;
   if(rank == 0){ 
      std::cout << " |lr>(comb) (dbranch,dcut,inoise,noise,rdm_vs_svd)=" 
                << dbranch << "," << dcut << "," << sweeps.inoise << ","
                << std::scientific << std::setprecision(1) << noise << ","
		<< rdm_vs_svd
		<< std::endl;
   }
   auto& timing = sweeps.opt_timing[isweep][ibond];
   auto& result = sweeps.opt_result[isweep][ibond];
   
/*
   qtensor2<Tm> rdm(qsym(), qlr, qlr);
   if(rank == 0){ 
      std::cout << "0. start renormalizing" << std::endl;
      qlr.print("qlr");
      rdm.print_size("rdm");
      get_sys_status();
   }
   // 1. build pRDM 
   if(rank == 0) std::cout << "1. build pRDM" << std::endl;
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
      if(rank == 0) rdm = rdm2;      
   }
#endif 
   // 2. decimation
   if(rank == 0) std::cout << "2. decimation" << std::endl;
   qtensor2<Tm> rot;
   if(rank == 0){
      rot = decimation_row(rdm, dcut, result.dwt, result.deff,
		           ifkr, wf.qrow, wf.qcol, dpt);
   }
#ifndef SERIAL
   if(size > 1) boost::mpi::broadcast(icomb.world, rot, 0); 
#endif
*/
   
   // Renormalize superblock = lr
   qtensor2<Tm> rot;
   if(rank == 0){
      std::cout << "0. decimation" << std::endl;
      std::vector<qtensor2<Tm>> wfs2;
      for(int i=0; i<vsol.cols(); i++){
         wf.from_array(vsol.col(i));
	 if(noise > 1.e-10) wf.add_noise(noise);
         // Need to first bring two dimensions adjacent to each other before merge!
   	 auto wf2 = wf.permCR_signed().merge_lr();
	 wfs2.push_back(wf2);
      }
      decimation_row(ifkr, wf.qrow, wf.qcol, dcut, rdm_vs_svd, wfs2, 
		     rot, result.dwt, result.deff);

      // 4. initial guess for next site within the bond
      if(sweeps.guess){
         std::cout << "4. initial guess" << std::endl;
         const auto& p1 = sweeps.seq[ibond].p1;	   
         icomb.psi.clear();
         for(int i=0; i<vsol.cols(); i++){
            wf.from_array(vsol.col(i));
            auto cwf = rot.H().dot(wf.permCR_signed().merge_lr()); // <-W[alpha,r]->
            auto psi = contract_qt3_qt2_l(icomb.rsites[p1],cwf);
            icomb.psi.push_back(psi);
         }
      }

   }
#ifndef SERIAL
   if(size > 1) boost::mpi::broadcast(icomb.world, rot, 0); 
#endif


   // 3. update site tensor
   if(rank == 0) std::cout << "3. update site tensor" << std::endl;
   const auto& p = sweeps.seq[ibond].p;
   icomb.lsites[p]= rot.split_lr(wf.qrow, wf.qcol);
   //-------------------------------------------------------------------	 
   assert((rot-icomb.lsites[p].merge_lr()).normF() < 1.e-10);
   auto ovlp = contract_qt3_qt3_lr(icomb.lsites[p],icomb.lsites[p]);
   assert(ovlp.check_identityMatrix(1.e-10,false)<1.e-10);
   //-------------------------------------------------------------------	
   timing.td = tools::get_time();
   // 5. renorm operators	 
   if(rank == 0) std::cout << "5. renormalize operators" << std::endl;
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
   const int& dbranch = sweeps.dbranch;
   const int dcut = (dbranch>0 && dbond.p1.second>0)? dbranch : sweeps.ctrls[isweep].dcut;
   const auto& noise = sweeps.ctrls[isweep].noise;
   const auto& rdm_vs_svd = sweeps.rdm_vs_svd;
   if(rank == 0){ 
      std::cout << " |cr> (dbranch,dcut,inoise,noise,rdm_vs_svd)="
   	        << dbranch << "," << dcut << "," << sweeps.inoise << ","
   	        << std::scientific << std::setprecision(1) << noise << ","
		<< rdm_vs_svd
		<< std::endl;
   }
   auto& timing = sweeps.opt_timing[isweep][ibond];
   auto& result = sweeps.opt_result[isweep][ibond];
   
   // Renormalize superblock = cr
   qtensor2<Tm> rot;
   if(rank == 0){
      std::cout << "0. decimation" << std::endl;
      std::vector<qtensor2<Tm>> wfs2;
      for(int i=0; i<vsol.cols(); i++){
         wf.from_array(vsol.col(i));
	 if(noise > 1.e-10) wf.add_noise(noise);
         auto wf2 = wf.merge_cr().T();
	 wfs2.push_back(wf2);
      }
      decimation_row(ifkr, wf.qmid, wf.qcol, dcut, rdm_vs_svd, wfs2, 
		     rot, result.dwt, result.deff);
      rot = rot.T(); // rot[alpha,r] = (V^+)

      // 4. initial guess for next site within the bond
      if(sweeps.guess){
         std::cout << "4. initial guess" << std::endl;
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
   }
#ifndef SERIAL
   if(size > 1) boost::mpi::broadcast(icomb.world, rot, 0); 
#endif

   // 3. update site tensor
   if(rank == 0) std::cout << "3. update site tensor" << std::endl;
   const auto& p = sweeps.seq[ibond].p;
   icomb.rsites[p] = rot.split_cr(wf.qmid, wf.qcol);
   //-------------------------------------------------------------------	
   assert((rot-icomb.rsites[p].merge_cr()).normF() < 1.e-10);	 
   auto ovlp = contract_qt3_qt3_cr(icomb.rsites[p],icomb.rsites[p]);
   assert(ovlp.check_identityMatrix(1.e-10,false)<1.e-10);
   //-------------------------------------------------------------------
   timing.td = tools::get_time();
   // 5. renorm operators	 
   if(rank == 0) std::cout << "5. renormalize operators" << std::endl;
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
