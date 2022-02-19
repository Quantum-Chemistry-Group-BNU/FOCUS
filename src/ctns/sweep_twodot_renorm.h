#ifndef SWEEP_TWODOT_RENORM_H
#define SWEEP_TWODOT_RENORM_H

#include "oper_io.h"
#include "sweep_decimation.h"
#include "sweep_onedot_renorm.h"

namespace ctns{

template <typename Tm>
void twodot_decimation(sweep_data& sweeps,
		       const int isweep,
		       const int ibond,
		       const bool ifkr,
		       const std::string superblock,
		       const linalg::matrix<Tm>& vsol, 
		       stensor4<Tm>& wf,
	               stensor2<Tm>& rot,
	               const std::string scratch){
   const auto& dbond = sweeps.seq[ibond];
   const int& dbranch = sweeps.dbranch;
   const int dcut = (dbranch>0 && dbond.p1.second>0)? dbranch : sweeps.ctrls[isweep].dcut;
   const auto& noise = sweeps.ctrls[isweep].noise;
   const auto& rdm_vs_svd = sweeps.rdm_vs_svd;
   std::cout <<" (dbranch,dcut,noise,rdm_vs_svd)=" << dbranch << "," << dcut << ","
             << std::scientific << std::setprecision(1) << noise << "," << rdm_vs_svd
	     << std::endl;
   auto& result = sweeps.opt_result[isweep][ibond];
   int nroots = vsol.cols();
   std::vector<stensor2<Tm>> wfs2(nroots);
   std::string fname = scratch+"/decimation_"+std::to_string(isweep)
	             + "_"+std::to_string(ibond)+".txt"; 
   if(superblock == "lc1"){ 

      for(int i=0; i<nroots; i++){
         wf.from_array(vsol.col(i));
	 auto wf2 = wf.merge_lc1_c2r();
	 if(noise > thresh_noise) wf2.add_noise(noise);
	 wfs2[i] = std::move(wf2);
      }
      decimation_row(ifkr, wf.info.qrow, wf.info.qmid, 
		     dcut, rdm_vs_svd, wfs2, 
		     rot, result.dwt, result.deff, fname);

   }else if(superblock == "lr"){ 

      for(int i=0; i<nroots; i++){
         wf.from_array(vsol.col(i));
	 wf.permCR_signed();
	 auto wf2 = wf.merge_lr_c1c2();
	 if(noise > thresh_noise) wf2.add_noise(noise);
	 wfs2[i] = std::move(wf2);
      }
      decimation_row(ifkr, wf.info.qrow, wf.info.qcol, 
		     dcut, rdm_vs_svd, wfs2, 
		     rot, result.dwt, result.deff, fname);

   }else if(superblock == "c2r"){ 

      for(int i=0; i<nroots; i++){
         wf.from_array(vsol.col(i));
	 auto wf2 = wf.merge_lc1_c2r().T();
	 if(noise > thresh_noise) wf2.add_noise(noise);
	 wfs2[i] = std::move(wf2);
      }
      decimation_row(ifkr, wf.info.qver, wf.info.qcol, 
		     dcut, rdm_vs_svd, wfs2, 
		     rot, result.dwt, result.deff, fname);
      rot = rot.T(); // rot[alpha,r] = (V^+)

   }else if(superblock == "c1c2"){

      for(int i=0; i<nroots; i++){
	 wf.from_array(vsol.col(i));
	 wf.permCR_signed();
	 auto wf2 = wf.merge_lr_c1c2().T();
	 if(noise > thresh_noise) wf2.add_noise(noise);
	 wfs2[i] = std::move(wf2);
      } // i
      decimation_row(ifkr, wf.info.qmid, wf.info.qver, 
		     dcut, rdm_vs_svd, wfs2,
		     rot, result.dwt, result.deff, fname);
      rot = rot.T(); // permute two lines for RCF

   } // superblock
}

// initial guess for next site within the bond
template <typename Km>
void twodot_guess_psi(const std::string superblock,
		      comb<Km>& icomb,
		      const directed_bond& dbond,
		      const linalg::matrix<typename Km::dtype>& vsol,
		      stensor4<typename Km::dtype>& wf,
		      const stensor2<typename Km::dtype>& rot){
   const bool debug = false
   if(debug) std::cout << "ctns::twodot_guess_psi superblock=" << superblock << std::endl;
   int nroots = vsol.cols();
   icomb.psi.clear();
   icomb.psi.resize(nroots);
   if(superblock == "lc1"){

      for(int i=0; i<nroots; i++){
         wf.from_array(vsol.col(i));
         //------------------------------------------
         // Two-dot case: simply use cwf[alpha,r,c2]
         //------------------------------------------
         // wf4[l,r,c1,c2] => wf2[lc1,c2r]
         auto wf2 = wf.merge_lc1_c2r();
         // rot.H()[alpha,lc1]*wf2[lc1,c2r] => cwf[alpha,c2r]
         auto cwf = rot.H().dot(wf2); 
         // cwf[alpha,c2r] => psi[alpha,r,c2]
         auto psi = cwf.split_cr(wf.info.qver, wf.info.qcol);
         //------------------------------------------
         icomb.psi[i] = std::move(psi);
      }

   }else if(superblock == "lr"){

      for(int i=0; i<nroots; i++){
         wf.from_array(vsol.col(i));
         //-------------------------------------------
         // Two-dot case: simply use cwf[alpha,c2,c1]
         //-------------------------------------------
	 // wf4[l,r,c1,c2] => wf2[lr,c1c2]
         wf.permCR_signed();
         auto wf2 = wf.merge_lr_c1c2();
         // rot.H()[alpha,lr]*wf3[lr,c1c2] => cwf[alpha,c1c2]
         auto cwf = rot.H().dot(wf2);
         // cwf[alpha,c1c2] => cwf[alpha,c2,c1] 
         auto psi = cwf.split_cr(wf.info.qmid, wf.info.qver);
	 //-------------------------------------------
         icomb.psi[i] = std::move(psi);
      }

   }else if(superblock == "c2r"){

      for(int i=0; i<vsol.cols(); i++){
         wf.from_array(vsol.col(i));
         //------------------------------------------
         // Two-dot case: simply use cwf[l,alpha,c1]
         //------------------------------------------
         // wf4[l,r,c1,c2] => wf2[lc1,c2r]
         auto wf2 = wf.merge_lc1_c2r();
         // wf2[lc1,c2r]*rot.H()[c2r,alpha] => cwf[lc1,alpha]
         auto cwf = wf2.dot(rot.H());
         // cwf[lc1,alpha] => cwf[l,alpha,c1]
         auto psi = cwf.split_lc(wf.info.qrow, wf.info.qmid);
         //------------------------------------------
         icomb.psi[i] = std::move(psi);
      }

   }else if(superblock == "c1c2"){

      for(int i=0; i<vsol.cols(); i++){
         wf.from_array(vsol.col(i));
         //----------------------------------------------
         // Two-dot case: simply use cwf[l,r,alpha]
         //----------------------------------------------
	 wf.permCR_signed();
         // wf4[l,c1,c2,r] => wf2[lr,c1c2]
         auto wf2 = wf.merge_lr_c1c2();
         // wf2[lr,c1c2]*rot.H()[c1c2,alpha] => cwf[lr,alpha]
         auto cwf = wf2.dot(rot.H());
         // cwf[lr,alpha] => psi[l,r,alpha]
         auto psi = cwf.split_lr(wf.info.qrow, wf.info.qcol);
         // revert ordering of the underlying basis
         psi.permCR_signed(); 
         //----------------------------------------------
         icomb.psi[i] = std::move(psi); // psi on backbone
      }

   } // superblock
}

template <typename Km>
void twodot_renorm(const input::schedule& schd,
		   sweep_data& sweeps,
		   const int isweep,
		   const int ibond, 
		   comb<Km>& icomb,
		   const linalg::matrix<typename Km::dtype>& vsol,
		   stensor4<typename Km::dtype>& wf,
	           const oper_dict<typename Km::dtype>& lqops,
	           const oper_dict<typename Km::dtype>& rqops,	
	           const oper_dict<typename Km::dtype>& c1qops,
	           const oper_dict<typename Km::dtype>& c2qops,
	           const integral::two_body<typename Km::dtype>& int2e, 
	           const integral::one_body<typename Km::dtype>& int1e,
	           const std::string scratch){
   using Tm = typename Km::dtype;
   int size = 1, rank = 0;
#ifndef SERIAL
   size = icomb.world.size();
   rank = icomb.world.rank();
#endif
   const bool ifkr = qkind::is_kramers<Km>();
   const auto& dbond = sweeps.seq[ibond];
   std::string superblock;
   if(dbond.forward){
      superblock = dbond.cturn? "lr" : "lc1";
   }else{
      superblock = dbond.cturn? "c1c2" : "c2r";
   }
   if(rank == 0) std::cout << "ctns::twodot_renorm superblock=" << superblock;
   auto& timing = sweeps.opt_timing[isweep][ibond];

   // build reduced density matrix & perform decimation
   stensor2<Tm> rot;
   if(rank == 0){
      twodot_decimation(sweeps, isweep, ibond, ifkr, 
		        superblock, vsol, wf, rot, scratch);
   }
#ifndef SERIAL
   if(size > 1) boost::mpi::broadcast(icomb.world, rot, 0); 
#endif
   timing.td = tools::get_time();

   // prepare guess for the next site
   if(rank == 0 && sweeps.guess){
      twodot_guess_psi(superblock, icomb, dbond, vsol, wf, rot);
   }
   timing.te = tools::get_time();

   // renorm operators	 
   const bool thresh = 1.e-10;
   const auto& p = dbond.p;
   const auto& pdx = icomb.topo.rindex.at(p); 
   oper_dict<Tm> qops;
   std::string fname;
   if(superblock == "lc1"){
      icomb.lsites[pdx] = rot.split_lc(wf.info.qrow, wf.info.qmid);
      //-------------------------------------------------------------------
      rot -= icomb.lsites[pdx].merge_lc();
      assert(rot.normF() < thresh);
      auto ovlp = contract_qt3_qt3("lc", icomb.lsites[pdx], icomb.lsites[pdx]);
      assert(ovlp.check_identityMatrix(thresh) < thresh);
      //-------------------------------------------------------------------
      oper_renorm_opAll("lc", icomb, p, int2e, int1e, 
		        lqops, c1qops, qops, schd.ctns.alg_renorm);
      fname = oper_fname(scratch, p, "l");
   }else if(superblock == "lr"){
      icomb.lsites[pdx]= rot.split_lr(wf.info.qrow, wf.info.qcol);
      //-------------------------------------------------------------------
      rot -= icomb.lsites[pdx].merge_lr();
      assert(rot.normF() < thresh);
      auto ovlp = contract_qt3_qt3("lr", icomb.lsites[pdx],icomb.lsites[pdx]);
      assert(ovlp.check_identityMatrix(thresh) < thresh);
      //-------------------------------------------------------------------
      oper_renorm_opAll("lr", icomb, p, int2e, int1e, 
		        lqops, rqops, qops, schd.ctns.alg_renorm);
      fname = oper_fname(scratch, p, "l");
   }else if(superblock == "c2r"){
      icomb.rsites[pdx] = rot.split_cr(wf.info.qver, wf.info.qcol);
      //-------------------------------------------------------------------
      rot -= icomb.rsites[pdx].merge_cr();
      assert(rot.normF() < thresh);
      auto ovlp = contract_qt3_qt3("cr", icomb.rsites[pdx],icomb.rsites[pdx]);
      assert(ovlp.check_identityMatrix(thresh) < thresh);
      //-------------------------------------------------------------------
      oper_renorm_opAll("cr", icomb, p, int2e, int1e, 
		        c2qops, rqops, qops, schd.ctns.alg_renorm);
      fname = oper_fname(scratch, p, "r");
   }else if(superblock == "c1c2"){
      icomb.rsites[pdx] = rot.split_cr(wf.info.qmid, wf.info.qver);
      //-------------------------------------------------------------------
      rot -= icomb.rsites[pdx].merge_cr();
      assert(rot.normF() < thresh);
      auto ovlp = contract_qt3_qt3("cr", icomb.rsites[pdx],icomb.rsites[pdx]);
      assert(ovlp.check_identityMatrix(thresh) < thresh);
      //-------------------------------------------------------------------
      oper_renorm_opAll("cr", icomb, p, int2e, int1e, 
		        c1qops, c2qops, qops, schd.ctns.alg_renorm);
      fname = oper_fname(scratch, p, "r");
   }
   timing.tf = tools::get_time();
   oper_save(fname, qops, rank);
}

} // ctns

#endif
