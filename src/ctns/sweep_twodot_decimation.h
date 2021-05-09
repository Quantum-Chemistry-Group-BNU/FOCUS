#ifndef SWEEP_TWODOT_DECIMATION_H
#define SWEEP_TWODOT_DECIMATION_H

namespace ctns{

/*
template <typename Km>
void twodot_decimation_lc1(sweep_data& sweeps,
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
   const auto& dbond = sweeps.seq[ibond];
   const auto& dcut  = sweeps.ctrls[isweep].dcut;
   const auto& noise = sweeps.ctrls[isweep].noise;
   std::cout << " |lc1> (forward,cturn,dcut,inoise,noise)="
	     << dbond.forward << "," << dbond.cturn << ","
	     << dcut << "," << sweeps.inoise << ","
	     << std::scientific << std::setprecision(1) << noise << std::endl;
   auto& timing = sweeps.opt_timing[isweep][ibond];
   auto& result = sweeps.opt_result[isweep][ibond];
   // Renormalize superblock = lc1
   auto qprod = qmerge(wf.qrow, wf.qmid);
   auto qlc = qprod.first;
   auto dpt = qprod.second;
   qtensor2<typename Km::dtype> rdm(qsym(), qlc, qlc);
   // 1. build pRDM 
   for(int i=0; i<vsol.cols(); i++){
      wf.from_array(vsol.col(i));
      if(sweeps.inoise > 1) wf.add_noise(noise);
      //----------------------------------------------
      // Two-dot case: wf3[l,c1,c2r] = wf4[l,c1,c2,r]
      //----------------------------------------------
      auto wf3 = wf.merge_c2r();
      rdm += wf3.merge_lc().get_rdm_row();
      if(sweeps.inoise > 0) get_prdm_lc(wf3, lqops, cqops, noise, rdm);
      //----------------------------------------------
   }
   // 2. decimation
   const bool ifkr = kind::is_kramers<Km>();
   auto qt2 = decimation_row(rdm, dcut, result.dwt, result.deff,
		   	     ifkr, wf.qrow, wf.qmid, dpt);
   // 3. update site tensor
   const auto& p = sweeps.seq[ibond].p;
   icomb.lsites[p] = qt2.split_lc(wf.qrow, wf.qmid, dpt);
   //-------------------------------------------------------------------	 
   assert((qt2-icomb.lsites[p].merge_lc()).normF() < 1.e-10);
   auto ovlp = contract_qt3_qt3_lc(icomb.lsites[p],icomb.lsites[p]);
   assert(ovlp.check_identityMatrix(1.e-10,false)<1.e-10);
   //-------------------------------------------------------------------
   // 4. initial guess for next site within the bond
   if(sweeps.guess){
      icomb.psi.clear();
      for(int i=0; i<vsol.cols(); i++){
         wf.from_array(vsol.col(i));
         //------------------------------------------
         // Two-dot case: simply use cwf[alpha,c2,r]
	 //------------------------------------------
	 auto wf3 = wf.merge_c2r(); 
	 auto cwf = qt2.H().dot(wf3.merge_lc()); // <-W[alpha,r]->
	 auto psi = cwf.split_cr(wf.qver, wf.qcol, wf.dpt_c2r().second);
	 //------------------------------------------
	 icomb.psi.push_back(psi);
      }
   }
   timing.td = tools::get_time();
   // 5. renorm operators	 
   oper_dict<typename Km::dtype> qops;
   oper_renorm_opAll("lc", icomb, p, int2e, int1e, lqops, cqops, qops);
   auto fname = oper_fname(scratch, p, "lop");
   oper_save(fname, qops);
}

template <typename Km>
void twodot_decimation_lr(sweep_data& sweeps,
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
   const auto& dbond = sweeps.seq[ibond];
   const auto& dcut  = sweeps.ctrls[isweep].dcut;
   const auto& noise = sweeps.ctrls[isweep].noise; 
   std::cout << " |lr>(comb) (forward,cturn,dcut,inoise,noise)=" 
	     << dbond.forward << "," << dbond.cturn << "," 
	     << dcut << "," << sweeps.inoise << ","
	     << std::scientific << std::setprecision(1) << noise << std::endl;
   auto& timing = sweeps.opt_timing[isweep][ibond];
   auto& result = sweeps.opt_result[isweep][ibond];
   // Renormalize superblock = lr
   auto qprod = qmerge(wf.qrow, wf.qcol);
   auto qlr = qprod.first;
   auto dpt = qprod.second;
   qtensor2<typename Km::dtype> rdm(qsym(), qlr, qlr);
   // 1. build pRDM 
   for(int i=0; i<vsol.cols(); i++){
      wf.from_array(vsol.col(i));
      if(sweeps.inoise > 1) wf.add_noise(noise);
      //----------------------------------------------
      // Two-dot case: wf3[l,c1c2,r] = wf4[l,c1,c2,r]
      //----------------------------------------------
      auto wf3 = wf.merge_c1c2();
      // Note: need to first bring two dimensions adjacent to each other before merge!
      rdm += wf3.permCR_signed().merge_lr().get_rdm_row();
      if(sweeps.inoise > 0) get_prdm_lr(wf3, lqops, rqops, noise, rdm);
   }
   // 2. decimation
   const bool ifkr = kind::is_kramers<Km>();
   auto qt2 = decimation_row(rdm, dcut, result.dwt, result.deff,
		             ifkr, wf.qrow, wf.qcol, dpt);
   // 3. update site tensor
   const auto& p = sweeps.seq[ibond].p;
   icomb.lsites[p]= qt2.split_lr(wf.qrow, wf.qcol, dpt);
   //-------------------------------------------------------------------	 
   assert((qt2-icomb.lsites[p].merge_lr()).normF() < 1.e-10);
   auto ovlp = contract_qt3_qt3_lr(icomb.lsites[p],icomb.lsites[p]);
   assert(ovlp.check_identityMatrix(1.e-10,false)<1.e-10);
   //-------------------------------------------------------------------	
   // 4. initial guess for next site within the bond
   if(sweeps.guess){	
      icomb.psi.clear();
      for(int i=0; i<vsol.cols(); i++){
         wf.from_array(vsol.col(i));
         //-------------------------------------------
         // Two-dot case: simply use cwf[alpha,c1,c2]
	 //-------------------------------------------
         auto wf3 = wf.merge_c1c2();
	 auto cwf = qt2.H().dot(wf3.permCR_signed().merge_lr()); // <-W[alpha,r]->
	 auto psi = cwf.split_cr(wf.qmid, wf.qver, wf.dpt_c1c2().second);
         //-------------------------------------------
         icomb.psi.push_back(psi);
      }
   }
   timing.td = tools::get_time();
   // 5. renorm operators	 
   oper_dict<typename Km::dtype> qops;
   oper_renorm_opAll("lr", icomb, p, int2e, int1e, lqops, rqops, qops);
   auto fname = oper_fname(scratch, p, "lop");
   oper_save(fname, qops);
}

template <typename Km>
void twodot_decimation_c2r(sweep_data& sweeps,
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
   const auto& dbond = sweeps.seq[ibond];
   const auto& dcut  = sweeps.ctrls[isweep].dcut;
   const auto& noise = sweeps.ctrls[isweep].noise; 
   std::cout << " |c2r> (forward,cturn,dcut,inoise,noise)="
	     << dbond.forward << "," << dbond.cturn << "," 
	     << dcut << "," << sweeps.inoise << ","
	     << std::scientific << std::setprecision(1) << noise << std::endl;
   auto& timing = sweeps.opt_timing[isweep][ibond];
   auto& result = sweeps.opt_result[isweep][ibond];
   // Renormalize superblock = c2r
   auto qprod = qmerge(wf.qver, wf.qcol);
   auto qcr = qprod.first;
   auto dpt = qprod.second;
   qtensor2<typename Km::dtype> rdm(qsym(), qcr, qcr);
   // 1. build pRDM 
   for(int i=0; i<vsol.cols(); i++){
      wf.from_array(vsol.col(i));
      if(sweeps.inoise > 1) wf.add_noise(noise);
      //----------------------------------------------
      // Two-dot case: wf3[lc1,c2,r] = wf4[l,c1,c2,r]
      //----------------------------------------------
      auto wf3 = wf.merge_lc1();
      rdm += wf3.merge_cr().get_rdm_col();
      if(sweeps.inoise > 0) get_prdm_cr(wf3, cqops, rqops, noise, rdm);
      //----------------------------------------------
   }
   // 2. decimation
   const bool ifkr = kind::is_kramers<Km>();
   auto qt2 = decimation_row(rdm, dcut, result.dwt, result.deff,
		             ifkr, wf.qmid, wf.qcol, dpt).T(); // permute two lines for RCF
   // 3. update site tensor
   const auto& p = sweeps.seq[ibond].p;
   icomb.rsites[p] = qt2.split_cr(wf.qmid, wf.qcol, dpt);
   //-------------------------------------------------------------------	
   assert((qt2-icomb.rsites[p].merge_cr()).normF() < 1.e-10);	 
   auto ovlp = contract_qt3_qt3_cr(icomb.rsites[p],icomb.rsites[p]);
   assert(ovlp.check_identityMatrix(1.e-10,false)<1.e-10);
   //-------------------------------------------------------------------	
   // 4. initial guess for next site within the bond
   if(sweeps.guess){
      icomb.psi.clear();
      for(int i=0; i<vsol.cols(); i++){
         wf.from_array(vsol.col(i));
	 //------------------------------------------
	 // Two-dot case: simply use cwf[l,c1,alpha]
	 //------------------------------------------
         auto wf3 = wf.merge_lc1();
	 auto cwf = wf3.merge_cr().dot(qt2.H()); // <-W[l,alpha]->
	 auto psi = cwf.split_lc(wf.qrow, wf.qmid, wf.dpt_lc1().second);
	 //------------------------------------------
         icomb.psi.push_back(psi);
      }
   }
   timing.td = tools::get_time();
   // 5. renorm operators	 
   oper_dict<typename Km::dtype> qops;
   oper_renorm_opAll("cr", icomb, p, int2e, int1e, cqops, rqops, qops);
   auto fname = oper_fname(scratch, p, "rop");
   oper_save(fname, qops);
}

template <typename Km>
void twodot_decimation_c1c2(sweep_data& sweeps,
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
   const auto& dbond = sweeps.seq[ibond];
   const auto& dcut  = sweeps.ctrls[isweep].dcut;
   const auto& noise = sweeps.ctrls[isweep].noise; 
   std::cout << " |c1c2>(comb) (forward,cturn,dcut,inoise,noise)="
	     << dbond.forward << "," << dbond.cturn << "," 
	     << dcut << "," << sweeps.inoise << ","
	     << std::scientific << std::setprecision(1) << noise << std::endl;
   auto& timing = sweeps.opt_timing[isweep][ibond];
   auto& result = sweeps.opt_result[isweep][ibond];
   // Renormalize superblock = c1c2
   auto qprod = qmerge(wf.qmid, wf.qver);
   auto qcv = qprod.first;
   auto dpt = qprod.second;
   qtensor2<typename Km::dtype> rdm(qsym(), qcv, qcv);
   // 1. build pRDM 
   for(int i=0; i<vsol.cols(); i++){
      wf.from_array(vsol.col(i));
      if(sweeps.inoise > 1) wf.add_noise(noise);
      //----------------------------------------------
      // Two-dot case: wf3[lr,c1,c2] = wf4[l,c1,c2,r]
      //----------------------------------------------
      auto wf3 = wf.permCR_signed().merge_lr();
      rdm += wf3.merge_cr().get_rdm_col();
      if(sweeps.inoise > 0) get_prdm_cr(wf3, c1qops, c2qops, noise, rdm);
      //----------------------------------------------
   }
   // 2. decimation
   const bool ifkr = kind::is_kramers<Km>();
   auto qt2 = decimation_row(rdm, dcut, result.dwt, result.deff,
		             ifkr, wf.qmid, wf.qver, dpt).T(); // permute two lines for RCF
   // 3. update site tensor
   const auto& p = sweeps.seq[ibond].p;
   icomb.rsites[p] = qt2.split_cr(wf.qmid, wf.qver, dpt);
   //-------------------------------------------------------------------	
   assert((qt2-icomb.rsites[p].merge_cr()).normF() < 1.e-10);	 
   auto ovlp = contract_qt3_qt3_cr(icomb.rsites[p],icomb.rsites[p]);
   assert(ovlp.check_identityMatrix(1.e-10,false)<1.e-10);
   //-------------------------------------------------------------------	
   // 4. initial guess for next site within the bond
   if(sweeps.guess){
      icomb.psi.clear();
      for(int i=0; i<vsol.cols(); i++){
         wf.from_array(vsol.col(i));
	 //----------------------------------------------
         // Two-dot case: wf3[lr,c1,c2] = wf4[l,c1,c2,r]
         //----------------------------------------------
         auto wf3 = wf.permCR_signed().merge_lr();
         auto cwf = wf3.merge_cr().dot(qt2.H()); // cwf[lr,alpha] 
 	 auto psi = cwf.split_lr(wf.qrow, wf.qcol, wf.dpt_lr().second);
	 psi = psi.permCR_signed(); // permute underlying basis
         icomb.psi.push_back(psi); // psi on backbone
      }
   }
   timing.td = tools::get_time();
   // 5. renorm operators	 
   oper_dict<typename Km::dtype> qops;
   oper_renorm_opAll("cr", icomb, p, int2e, int1e, c1qops, c2qops, qops);
   auto fname = oper_fname(scratch, p, "rop");
   oper_save(fname, qops);
}

template <typename Km>
void twodot_decimation(sweep_data& sweeps,
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
   const auto& dbond = sweeps.seq[ibond];
   std::cout << "ctns::twodot_decimation";
   // build reduced density matrix & perform renormalization
   if(dbond.forward){
      // update lsites & ql
      if(!dbond.cturn){
	 twodot_decimation_lc1(sweeps, isweep, ibond, icomb, vsol, wf, 
			       lqops, c1qops, int2e, int1e, scratch);
      }else{
	 // special for comb
         twodot_decimation_lr(sweeps, isweep, ibond, icomb, vsol, wf, 
			      lqops, rqops, int2e, int1e, scratch);
      } // cturn
   }else{
      // update rsites & qr
      if(!dbond.cturn){
         twodot_decimation_c2r(sweeps, isweep, ibond, icomb, vsol, wf, 
		               c2qops, rqops, int2e, int1e, scratch);
      }else{
	 // special for comb
	 //        c2      
	 //        |
	 //   c1---p
	 //        |
	 //    l---*---r
         twodot_decimation_c1c2(sweeps, isweep, ibond, icomb, vsol, wf, 
		                c1qops, c2qops, int2e, int1e, scratch);
      } // cturn
   } // forward
}
*/

} // ctns

#endif
