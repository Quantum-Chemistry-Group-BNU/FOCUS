#ifndef SWEEP_TWODOT_RENORM_H
#define SWEEP_TWODOT_RENORM_H

#include "oper_io.h"
#include "sweep_prdm.h"
#include "sweep_decimation.h"

namespace ctns{

inline size_t get_rdm_size(const qbond& qrow, const qbond& qcol){
   size_t size = 0;
   for(int br=0; br<qrow.size(); br++){
      for(int bc=0; bc<qcol.size(); bc++){
         if(qrow.get_sym(br) != qcol.get_sym(bc)) continue;
         int rdim = qrow.get_dim(br);
         int cdim = qcol.get_dim(bc);
         size += rdim*cdim;
      } // bc
   } // br
   return size;
}

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
   const auto& rdm_vs_svd = sweeps.rdm_vs_svd;
   if(rank == 0){
      std::cout << " |lc1> (dbranch,dcut,inoise,noise,rdm_vs_svd)="
                << dbranch << "," << dcut << "," << sweeps.inoise << ","
                << std::scientific << std::setprecision(1) << noise << ","
		<< rdm_vs_svd
		<< std::endl;
   }
   auto& timing = sweeps.opt_timing[isweep][ibond];
   auto& result = sweeps.opt_result[isweep][ibond];

   // Renormalize superblock = lc1
   qtensor2<Tm> rot;
   if(rank == 0){
      std::cout << "0. decimation" << std::endl;
      std::vector<qtensor2<Tm>> wfs2;
      for(int i=0; i<vsol.cols(); i++){
         wf.from_array(vsol.col(i));
	 if(noise > 1.e-10) wf.add_noise(noise);
         //----------------------------------------------
         // Two-dot case: wf3[l,c1,c2r] = wf4[l,c1,c2,r]
	 // 	          wf2[lc1,c2r] = wf3[l,c1,c2r]
         //----------------------------------------------
         auto wf2 = wf.merge_c2r().merge_lc();
	 wfs2.push_back(wf2);
      }
      decimation_row(ifkr, wf.qrow, wf.qmid, dcut, rdm_vs_svd, wfs2, 
		     rot, result.dwt, result.deff);

      // 4. initial guess for next site within the bond
      if(sweeps.guess){
         std::cout << "4. initial guess" << std::endl;
         icomb.psi.clear();
         for(int i=0; i<vsol.cols(); i++){
            wf.from_array(vsol.col(i));
            //------------------------------------------
            // Two-dot case: simply use cwf[alpha,c2,r]
            //------------------------------------------
            // wf4[l,c1,c2,r]->wf3[l,c1,c2r]->wf2[lc1,c2r]
            auto wf2 = wf.merge_c2r().merge_lc(); 
            // rot.H()[alpha,lc1]*wf2[lc1,c2r]->cwf[alpha,c2r]
            auto cwf = rot.H().dot(wf2); 
            // cwf[alpha,c2r]->psi[alpha,c2,r]
            auto psi = cwf.split_cr(wf.qver, wf.qcol);
            //------------------------------------------
            icomb.psi.push_back(psi);
         }
      }

   }
#ifndef SERIAL
   if(size > 1) boost::mpi::broadcast(icomb.world, rot, 0); 
#endif


/*
   if(rank == 0){
      qlc.print("qlc");
      size_t size = get_rdm_size(qlc,qlc); 
      std::cout << "rdm_size=" << size
		<< " sizeMB=" << tools::sizeMB<Tm>(size)
                << std::endl; 
   }
   qtensor2<Tm> rdm(qsym(), qlc, qlc);
   if(rank == 0) rdm.print_size("rdm"); 
   
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

   // LZD
   if(rank == 0) rot.print_size("rot");

   if(size > 1) boost::mpi::broadcast(icomb.world, rot, 0); 
#endif
*/

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
   
   // Renormalize superblock = lr
   qtensor2<Tm> rot;
   if(rank == 0){
      std::cout << "0. decimation" << std::endl;
      std::vector<qtensor2<Tm>> wfs2;
      for(int i=0; i<vsol.cols(); i++){
         wf.from_array(vsol.col(i));
	 if(noise > 1.e-10) wf.add_noise(noise);
         //----------------------------------------------
         // Two-dot case: wf3[l,c1c2,r] = wf4[l,c1,c2,r]
	 // 		  wf2[lr,c1c2] = wf3[l,c1c2,r]
         //----------------------------------------------
	 auto wf2 = wf.merge_c1c2().permCR_signed().merge_lr();
	 wfs2.push_back(wf2);
      }
      decimation_row(ifkr, wf.qrow, wf.qcol, dcut, rdm_vs_svd, wfs2, 
		     rot, result.dwt, result.deff);

      // 4. initial guess for next site within the bond
      if(sweeps.guess){
         std::cout << "4. initial guess" << std::endl;
         icomb.psi.clear();
         for(int i=0; i<vsol.cols(); i++){
            wf.from_array(vsol.col(i));
            //-------------------------------------------
            // Two-dot case: simply use cwf[alpha,c1,c2]
            //-------------------------------------------
            // wf4[l,c1,c2,r]->wf3[l,c1c2,r]->wf2[lr,c1c2]
            auto wf2 = wf.merge_c1c2().permCR_signed().merge_lr();
            // rot.H()[alpha,lr]*wf3[lr,c1c2]->cwf[alpha,c1c2]
            auto cwf = rot.H().dot(wf2);
            // cwf[alpha,c1c2]->cwf[alpha,c1,c2] 
            auto psi = cwf.split_cr(wf.qmid, wf.qver);
            //-------------------------------------------
            icomb.psi.push_back(psi);
         }
      }

   }
#ifndef SERIAL
   if(size > 1) boost::mpi::broadcast(icomb.world, rot, 0); 
#endif



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
*/

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
   const auto& rdm_vs_svd = sweeps.rdm_vs_svd;
   if(rank == 0){ 
      std::cout << " |c2r> (dbranch,dcut,inoise,noise,rdm_vs_svd)="
                << dbranch << "," << dcut << "," << sweeps.inoise << ","
                << std::scientific << std::setprecision(1) << noise << ","
		<< rdm_vs_svd
		<< std::endl;
   }
   auto& timing = sweeps.opt_timing[isweep][ibond];
   auto& result = sweeps.opt_result[isweep][ibond];
   
   // Renormalize superblock = c2r
   qtensor2<Tm> rot;
   if(rank == 0){
      std::cout << "0. decimation" << std::endl;
      std::vector<qtensor2<Tm>> wfs2;
      for(int i=0; i<vsol.cols(); i++){
         wf.from_array(vsol.col(i));
	 if(noise > 1.e-10) wf.add_noise(noise);
         //----------------------------------------------
         // Two-dot case: wf3[lc1,c2,r] = wf4[l,c1,c2,r]
	 // 		  wf2[lc1,c2r] = wf3[lc1,c2,r]
         //----------------------------------------------
         auto wf2 = wf.merge_lc1().merge_cr().T();
	 wfs2.push_back(wf2);
      }
      decimation_row(ifkr, wf.qver, wf.qcol, dcut, rdm_vs_svd, wfs2, 
		     rot, result.dwt, result.deff);
      rot = rot.T(); // rot[alpha,r] = (V^+)

      // 4. initial guess for next site within the bond
      if(sweeps.guess){
         std::cout << "4. initial guess" << std::endl;
         icomb.psi.clear();
         for(int i=0; i<vsol.cols(); i++){
            wf.from_array(vsol.col(i));
            //------------------------------------------
            // Two-dot case: simply use cwf[l,c1,alpha]
            //------------------------------------------
            // wf4[l,c1,c2,r]->wf3[lc1,c2,r]->wf2[lc1,c2r]
            auto wf2 = wf.merge_lc1().merge_cr();
            // wf2[lc1,c2r]*rot.H()[c2r,alpha]->cwf[lc1,alpha]
            auto cwf = wf2.dot(rot.H());
            // cwf[lc1,alpha]->cwf[l,c1,alpha]
            auto psi = cwf.split_lc(wf.qrow, wf.qmid);
            //------------------------------------------
            icomb.psi.push_back(psi);
         }
      }

   }
#ifndef SERIAL
   if(size > 1) boost::mpi::broadcast(icomb.world, rot, 0); 
#endif

/*
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
*/

   // 3. update site tensor
   if(rank == 0) std::cout << "3. update site tensor" << std::endl;
   const auto& p = sweeps.seq[ibond].p;
   icomb.rsites[p] = rot.split_cr(wf.qver, wf.qcol);
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
   const auto& rdm_vs_svd = sweeps.rdm_vs_svd;
   if(rank == 0){ 
      std::cout << " |c1c2>(comb) (dbranch,dcut,inoise,noise,rdm_vs_svd)="
                << dbranch << "," << dcut << "," << sweeps.inoise << ","
                << std::scientific << std::setprecision(1) << noise << ","
		<< rdm_vs_svd
		<< std::endl;
   }
   auto& timing = sweeps.opt_timing[isweep][ibond];
   auto& result = sweeps.opt_result[isweep][ibond];
   
/*
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
*/

   // Renormalize superblock = c1c2
   qtensor2<Tm> rot;
   if(rank == 0){
      std::cout << "0. decimation" << std::endl;
      std::vector<qtensor2<Tm>> wfs2;
      for(int i=0; i<vsol.cols(); i++){
	 wf.from_array(vsol.col(i));
	 if(noise > 1.e-10) wf.add_noise(noise);
         //----------------------------------------------
         // Two-dot case: wf3[lr,c1,c2] = wf4[l,c1,c2,r]
	 //               wf2[lr,c1c2] = wf3[lr,c1,c2]
         //----------------------------------------------
	 auto wf2 = wf.permCR_signed().merge_lr().merge_cr().T();
	 wfs2.push_back(wf2);
      } // i
      decimation_row(ifkr, wf.qmid, wf.qver, dcut, rdm_vs_svd, wfs2,
		     rot, result.dwt, result.deff);
      rot = rot.T(); // permute two lines for RCF

      // 4. initial guess for next site within the bond
      if(sweeps.guess){
         std::cout << "4. initial guess" << std::endl;
         icomb.psi.clear();
         for(int i=0; i<vsol.cols(); i++){
            wf.from_array(vsol.col(i));
            //----------------------------------------------
            // Two-dot case: wf3[lr,c1,c2] = wf4[l,c1,c2,r]
            //----------------------------------------------
            // wf4[l,c1,c2,r]->wf3[lr,c1,c2]->wf2[lr,c1c2]
            auto wf2 = wf.permCR_signed().merge_lr().merge_cr();
            // wf2[lr,c1c2]*rot.H()[c1c2,alpha]->cwf[lr,alpha]
            auto cwf = wf2.dot(rot.H());
            // cwf[lr,alpha]->psi[l,alpha,r] 
            auto psi = cwf.split_lr(wf.qrow, wf.qcol);
            // revert ordering of the underlying basis
            psi = psi.permCR_signed(); 
            //----------------------------------------------
            icomb.psi.push_back(psi); // psi on backbone
         }
      }
  
   }
#ifndef SERIAL
   if(size > 1) boost::mpi::broadcast(icomb.world, rot, 0); 
#endif


   // 3. update site tensor
   if(rank == 0) std::cout << "3. update site tensor" << std::endl;
   const auto& p = sweeps.seq[ibond].p;
   icomb.rsites[p] = rot.split_cr(wf.qmid, wf.qver);
   //-------------------------------------------------------------------	
   assert((rot-icomb.rsites[p].merge_cr()).normF() < 1.e-10);	 
   auto ovlp = contract_qt3_qt3_cr(icomb.rsites[p],icomb.rsites[p]);
   assert(ovlp.check_identityMatrix(1.e-10,false)<1.e-10);
   //-------------------------------------------------------------------	
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
