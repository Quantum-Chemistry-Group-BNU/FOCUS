#ifndef SWEEP_TWODOT_RENORM_H
#define SWEEP_TWODOT_RENORM_H

#include "oper_io.h"
#include "sweep_prdm.h"
#include "sweep_decimation.h"

namespace ctns{

template <typename Km>
void twodot_renorm_lc1(sweep_data& sweeps,
		       const int isweep,
		       const int ibond, 
		       comb<Km>& icomb,
		       const linalg::matrix<typename Km::dtype>& vsol,
		       qtensor4<typename Km::dtype>& wf,
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
   if(rank == 0){
      std::cout << " |lc1> (dbranch,dcut,inoise,noise)="
                << dbranch << "," << dcut << "," << sweeps.inoise << ","
                << std::scientific << std::setprecision(1) << noise << std::endl;
   }
   auto& timing = sweeps.opt_timing[isweep][ibond];
   auto& result = sweeps.opt_result[isweep][ibond];
   // Renormalize superblock = lc1
   auto qprod = qmerge(wf.qrow, wf.qmid);
   auto qlc = qprod.first;
   auto dpt = qprod.second;

   get_sys_status("in-rank="+std::to_string(rank));

   if(rank == 0) qlc.print("qlc");
   qtensor2<Tm> rdm(qsym(), qlc, qlc);
  
   if(rank == 0) rdm.print("rdm"); 
   get_sys_status("out-rank="+std::to_string(rank));
   icomb.world.barrier();
   
   if(rank == 0){ 
      std::cout << "0. start renormalizing" << std::endl;
      qlc.print("qlc1");
      rdm.print_size("rdm");
      get_sys_status();
   }

   icomb.world.barrier();

   // 1. build pRDM 
   if(rank == 0) std::cout << "1. build pRDM" << std::endl;
   for(int i=0; i<vsol.cols(); i++){
      wf.from_array(vsol.col(i));
      //----------------------------------------------
      // Two-dot case: wf3[l,c1,c2r] = wf4[l,c1,c2,r]
      //----------------------------------------------
      auto wf3 = wf.merge_c2r();
      //----------------------------------------------
      if(sweeps.inoise > 0) get_prdm("lc", ifkr, wf3, lqops, cqops, noise, rdm, size, rank);
      if(rank == 0){
         if(sweeps.inoise > 1) wf3.add_noise(noise);
         rdm += wf3.get_rdm("lc");
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
		   	   ifkr, wf.qrow, wf.qmid, dpt);
   }
#ifndef SERIAL
   if(size > 1) boost::mpi::broadcast(icomb.world, rot, 0); 
#endif
   // 3. update site tensor
   if(rank == 0) std::cout << "3. update site tensor" << std::endl;
   const auto& p = sweeps.seq[ibond].p;
   icomb.lsites[p] = rot.split_lc(wf.qrow, wf.qmid, dpt);
   //-------------------------------------------------------------------	 
   assert((rot-icomb.lsites[p].merge_lc()).normF() < 1.e-10);
   auto ovlp = contract_qt3_qt3_lc(icomb.lsites[p],icomb.lsites[p]);
   assert(ovlp.check_identityMatrix(1.e-10,false)<1.e-10);
   //-------------------------------------------------------------------
   // 4. initial guess for next site within the bond
   if(rank == 0) std::cout << "4. initial guess" << std::endl;
   if(rank == 0 && sweeps.guess){
      icomb.psi.clear();
      for(int i=0; i<vsol.cols(); i++){
         wf.from_array(vsol.col(i));
         //------------------------------------------
         // Two-dot case: simply use cwf[alpha,c2,r]
	 //------------------------------------------
	 // wf4[l,c1,c2,r]->wf3[l,c1,c2r]
	 auto wf3 = wf.merge_c2r(); 
	 // rot.H()[alpha,lc1]*wf3[lc1,c2r]->cwf[alpha,c2r]
	 auto cwf = rot.H().dot(wf3.merge_lc()); 
	 // cwf[alpha,c2r]->psi[alpha,c2,r]
	 auto psi = cwf.split_cr(wf.qver, wf.qcol, wf.dpt_c2r().second);
	 //------------------------------------------
	 icomb.psi.push_back(psi);
      }
   }
   timing.td = tools::get_time();
   // 5. renormalize operators	 
   if(rank == 0) std::cout << "5. renormalize operators" << std::endl;
   oper_dict<Tm> qops;
   oper_renorm_opAll("lc", icomb, p, int2e, int1e, lqops, cqops, qops);
   timing.te = tools::get_time();
   auto fname = oper_fname(scratch, p, "l");
   oper_save(fname, qops);
}

template <typename Km>
void twodot_renorm_lr(sweep_data& sweeps,
		      const int isweep,
		      const int ibond, 
		      comb<Km>& icomb,
		      const linalg::matrix<typename Km::dtype>& vsol,
		      qtensor4<typename Km::dtype>& wf,
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
   if(rank == 0){ 
      std::cout << " |lr>(comb) (dbranch,dcut,inoise,noise)=" 
                << dbranch << "," << dcut << "," << sweeps.inoise << ","
                << std::scientific << std::setprecision(1) << noise << std::endl;
   }
   auto& timing = sweeps.opt_timing[isweep][ibond];
   auto& result = sweeps.opt_result[isweep][ibond];
   // Renormalize superblock = lr
   auto qprod = qmerge(wf.qrow, wf.qcol);
   auto qlr = qprod.first;
   auto dpt = qprod.second;
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
      //----------------------------------------------
      // Two-dot case: wf3[l,c1c2,r] = wf4[l,c1,c2,r]
      //----------------------------------------------
      auto wf3 = wf.merge_c1c2();
      //----------------------------------------------
      if(sweeps.inoise > 0) get_prdm("lr", ifkr, wf3, lqops, rqops, noise, rdm, size, rank);
      if(rank == 0){
         if(sweeps.inoise > 1) wf3.add_noise(noise);
         rdm += wf3.get_rdm("lr");
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
      rot  = decimation_row(rdm, dcut, result.dwt, result.deff,
		            ifkr, wf.qrow, wf.qcol, dpt);
   }
#ifndef SERIAL
   if(size > 1) boost::mpi::broadcast(icomb.world, rot, 0); 
#endif
   // 3. update site tensor
   if(rank == 0) std::cout << "3. update site tensor" << std::endl;
   const auto& p = sweeps.seq[ibond].p;
   icomb.lsites[p]= rot.split_lr(wf.qrow, wf.qcol, dpt);
   //-------------------------------------------------------------------	 
   assert((rot-icomb.lsites[p].merge_lr()).normF() < 1.e-10);
   auto ovlp = contract_qt3_qt3_lr(icomb.lsites[p],icomb.lsites[p]);
   assert(ovlp.check_identityMatrix(1.e-10,false)<1.e-10);
   //-------------------------------------------------------------------	
   // 4. initial guess for next site within the bond
   if(rank == 0) std::cout << "4. initial guess" << std::endl;
   if(rank == 0 && sweeps.guess){	
      icomb.psi.clear();
      for(int i=0; i<vsol.cols(); i++){
         wf.from_array(vsol.col(i));
         //-------------------------------------------
         // Two-dot case: simply use cwf[alpha,c1,c2]
	 //-------------------------------------------
         // wf4[l,c1,c2,r]->wf3[l,c1c2,r]
	 auto wf3 = wf.merge_c1c2();
	 // rot.H()[alpha,lr]*wf3[lr,c1c2]->cwf[alpha,c1c2]
	 auto cwf = rot.H().dot(wf3.permCR_signed().merge_lr());
	 // cwf[alpha,c1c2]->cwf[alpha,c1,c2] 
	 auto psi = cwf.split_cr(wf.qmid, wf.qver, wf.dpt_c1c2().second);
         //-------------------------------------------
         icomb.psi.push_back(psi);
      }
   }
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
void twodot_renorm_c2r(sweep_data& sweeps,
		       const int isweep,
		       const int ibond, 
		       comb<Km>& icomb,
		       const linalg::matrix<typename Km::dtype>& vsol,
		       qtensor4<typename Km::dtype>& wf,
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
   if(rank == 0){ 
      std::cout << " |c2r> (dbranch,dcut,inoise,noise)="
                << dbranch << "," << dcut << "," << sweeps.inoise << ","
                << std::scientific << std::setprecision(1) << noise << std::endl;
   }
   auto& timing = sweeps.opt_timing[isweep][ibond];
   auto& result = sweeps.opt_result[isweep][ibond];
   // Renormalize superblock = c2r
   auto qprod = qmerge(wf.qver, wf.qcol);
   auto qcr = qprod.first;
   auto dpt = qprod.second;
   qtensor2<Tm> rdm(qsym(), qcr, qcr);
   if(rank == 0){ 
      std::cout << "0. start renormalizing" << std::endl;
      qcr.print("qcr");
      rdm.print_size("rdm");
      get_sys_status();
   }
   // 1. build pRDM 
   if(rank == 0) std::cout << "1. build pRDM" << std::endl;
   for(int i=0; i<vsol.cols(); i++){
      wf.from_array(vsol.col(i));
      //----------------------------------------------
      // Two-dot case: wf3[lc1,c2,r] = wf4[l,c1,c2,r]
      //----------------------------------------------
      auto wf3 = wf.merge_lc1();
      //----------------------------------------------
      if(sweeps.inoise > 0) get_prdm("cr", ifkr, wf3, cqops, rqops, noise, rdm, size, rank);
      if(rank == 0){
         if(sweeps.inoise > 1) wf3.add_noise(noise);
         rdm += wf3.get_rdm("cr");
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
   if(rank == 0) std::cout << "2. decimation" << std::endl;
   qtensor2<Tm> rot;
   if(rank == 0){
      rot = decimation_row(rdm, dcut, result.dwt, result.deff,
		           ifkr, wf.qver, wf.qcol, dpt).T(); // permute two lines for RCF
   }
#ifndef SERIAL
   if(size > 1) boost::mpi::broadcast(icomb.world, rot, 0); 
#endif
   // 3. update site tensor
   if(rank == 0) std::cout << "3. update site tensor" << std::endl;
   const auto& p = sweeps.seq[ibond].p;
   icomb.rsites[p] = rot.split_cr(wf.qver, wf.qcol, dpt);
   //-------------------------------------------------------------------	
   assert((rot-icomb.rsites[p].merge_cr()).normF() < 1.e-10);	 
   auto ovlp = contract_qt3_qt3_cr(icomb.rsites[p],icomb.rsites[p]);
   assert(ovlp.check_identityMatrix(1.e-10,false)<1.e-10);
   //-------------------------------------------------------------------	
   // 4. initial guess for next site within the bond
   if(rank == 0) std::cout << "4. initial guess" << std::endl;
   if(rank == 0 && sweeps.guess){
      icomb.psi.clear();
      for(int i=0; i<vsol.cols(); i++){
         wf.from_array(vsol.col(i));
	 //------------------------------------------
	 // Two-dot case: simply use cwf[l,c1,alpha]
	 //------------------------------------------
         // wf4[l,c1,c2,r]->wf3[lc1,c2,r]
	 auto wf3 = wf.merge_lc1();
	 // wf3[lc1,c2r]*rot.H()[c2r,alpha]->cwf[lc1,alpha]
	 auto cwf = wf3.merge_cr().dot(rot.H());
	 // cwf[lc1,alpha]->cwf[l,c1,alpha]
	 auto psi = cwf.split_lc(wf.qrow, wf.qmid, wf.dpt_lc1().second);
	 //------------------------------------------
         icomb.psi.push_back(psi);
      }
   }
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
void twodot_renorm_c1c2(sweep_data& sweeps,
		        const int isweep,
		        const int ibond, 
		        comb<Km>& icomb,
		        const linalg::matrix<typename Km::dtype>& vsol,
		        qtensor4<typename Km::dtype>& wf,
		        oper_dict<typename Km::dtype>& c1qops,
		        oper_dict<typename Km::dtype>& c2qops,
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
   if(rank == 0){ 
      std::cout << " |c1c2>(comb) (dbranch,dcut,inoise,noise)="
                << dbranch << "," << dcut << "," << sweeps.inoise << ","
                << std::scientific << std::setprecision(1) << noise << std::endl;
   }
   auto& timing = sweeps.opt_timing[isweep][ibond];
   auto& result = sweeps.opt_result[isweep][ibond];
   // Renormalize superblock = c1c2
   auto qprod = qmerge(wf.qmid, wf.qver);
   auto qc1c2 = qprod.first;
   auto dpt = qprod.second;
   qtensor2<Tm> rdm(qsym(), qc1c2, qc1c2);
   if(rank == 0){ 
      std::cout << "0. start renormalizing" << std::endl;
      qc1c2.print("qc1c2");
      rdm.print_size("rdm");
      get_sys_status();
   }
   // 1. build pRDM 
   if(rank == 0) std::cout << "1. build pRDM" << std::endl;
   for(int i=0; i<vsol.cols(); i++){
      wf.from_array(vsol.col(i));
      if(sweeps.inoise > 1) wf.add_noise(noise);
      //----------------------------------------------
      // Two-dot case: wf3[lr,c1,c2] = wf4[l,c1,c2,r]
      //----------------------------------------------
      auto wf3 = wf.permCR_signed().merge_lr();
      //----------------------------------------------
      if(sweeps.inoise > 0) get_prdm("cr", ifkr, wf3, c1qops, c2qops, noise, rdm, size, rank);
      if(rank == 0){
         if(sweeps.inoise > 1) wf3.add_noise(noise);
	 rdm += wf3.get_rdm("cr");
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
   if(rank == 0) std::cout << "2. decimation" << std::endl;
   qtensor2<Tm> rot;
   if(rank == 0){
      rot = decimation_row(rdm, dcut, result.dwt, result.deff,
		           ifkr, wf.qmid, wf.qver, dpt).T(); // permute two lines for RCF
   }
#ifndef SERIAL
   if(size > 1) boost::mpi::broadcast(icomb.world, rot, 0); 
#endif
   // 3. update site tensor
   if(rank == 0) std::cout << "3. update site tensor" << std::endl;
   const auto& p = sweeps.seq[ibond].p;
   icomb.rsites[p] = rot.split_cr(wf.qmid, wf.qver, dpt);
   //-------------------------------------------------------------------	
   assert((rot-icomb.rsites[p].merge_cr()).normF() < 1.e-10);	 
   auto ovlp = contract_qt3_qt3_cr(icomb.rsites[p],icomb.rsites[p]);
   assert(ovlp.check_identityMatrix(1.e-10,false)<1.e-10);
   //-------------------------------------------------------------------	
   // 4. initial guess for next site within the bond
   if(rank == 0) std::cout << "4. initial guess" << std::endl;
   if(rank == 0 && sweeps.guess){
      icomb.psi.clear();
      for(int i=0; i<vsol.cols(); i++){
         wf.from_array(vsol.col(i));
	 //----------------------------------------------
         // Two-dot case: wf3[lr,c1,c2] = wf4[l,c1,c2,r]
         //----------------------------------------------
         // wf4[l,c1,c2,r]->wf3[lr,c1,c2]
	 auto wf3 = wf.permCR_signed().merge_lr();
	 // wf3[lr,c1c2]*rot.H()[c1c2,alpha]->cwf[lr,alpha]
         auto cwf = wf3.merge_cr().dot(rot.H());
	 // cwf[lr,alpha]->psi[l,alpha,r] 
 	 auto psi = cwf.split_lr(wf.qrow, wf.qcol, wf.dpt_lr().second);
	 // revert ordering of the underlying basis
	 psi = psi.permCR_signed(); 
         //----------------------------------------------
         icomb.psi.push_back(psi); // psi on backbone
      }
   }
   timing.td = tools::get_time();
   // 5. renorm operators	 
   if(rank == 0) std::cout << "5. renormalize operators" << std::endl;
   oper_dict<Tm> qops;
   oper_renorm_opAll("cr", icomb, p, int2e, int1e, c1qops, c2qops, qops);
   timing.te = tools::get_time();
   auto fname = oper_fname(scratch, p, "r");
   oper_save(fname, qops);
}

template <typename Km>
void twodot_renorm(sweep_data& sweeps,
		   const int isweep,
		   const int ibond, 
		   comb<Km>& icomb,
		   const linalg::matrix<typename Km::dtype>& vsol,
		   qtensor4<typename Km::dtype>& wf,
	           oper_dict<typename Km::dtype>& c1qops,
	           oper_dict<typename Km::dtype>& c2qops,
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
   if(rank == 0) std::cout << "ctns::twodot_renorm";
   // build reduced density matrix & perform renormalization
   const auto& dbond = sweeps.seq[ibond];
   if(dbond.forward){
      // update lsites & ql
      if(!dbond.cturn){
	 twodot_renorm_lc1(sweeps, isweep, ibond, icomb, vsol, wf, 
			   lqops, c1qops, int2e, int1e, scratch);
      }else{
	 // special for comb
         twodot_renorm_lr(sweeps, isweep, ibond, icomb, vsol, wf, 
			  lqops, rqops, int2e, int1e, scratch);
      } // cturn
   }else{
      // update rsites & qr
      if(!dbond.cturn){
         twodot_renorm_c2r(sweeps, isweep, ibond, icomb, vsol, wf, 
		           c2qops, rqops, int2e, int1e, scratch);
      }else{
	 // special for comb
	 //        c2      
	 //        |
	 //   c1---p
	 //        |
	 //    l---*---r
         twodot_renorm_c1c2(sweeps, isweep, ibond, icomb, vsol, wf, 
		            c1qops, c2qops, int2e, int1e, scratch);
      } // cturn
   } // forward
}

} // ctns

#endif
