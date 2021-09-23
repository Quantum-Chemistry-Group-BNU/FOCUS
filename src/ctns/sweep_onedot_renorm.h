#ifndef SWEEP_ONEDOT_RENORM_H
#define SWEEP_ONEDOT_RENORM_H

#include "oper_io.h"
//#include "sweep_decimation.h"

namespace ctns{

const bool debug_renorm = true;
extern const bool debug_renorm;

const double thresh_noise = 1.e-10;
extern const double thresh_noise;

template <typename Tm>
void onedot_decimation(sweep_data& sweeps,
		       const int isweep,
		       const int ibond, 
		       const bool ifkr,
		       const std::string superblock,
		       const linalg::matrix<Tm>& vsol,
		       stensor3<Tm>& wf,
		       stensor2<Tm>& rot){ 
   const auto& dbond = sweeps.seq[ibond];
   const int& dbranch = sweeps.dbranch;
   const int dcut = (dbranch>0 && dbond.p1.second>0)? dbranch : sweeps.ctrls[isweep].dcut;
   const auto& noise = sweeps.ctrls[isweep].noise;
   const auto& rdm_vs_svd = sweeps.rdm_vs_svd;
   std::cout <<" (dbranch,dcut,noise,rdm_vs_svd)=" << dbranch << "," << dcut << ","
             << std::scientific << std::setprecision(1) << noise << "," << rdm_vs_svd
	     << std::endl;
   if(debug_renorm) std::cout << "> onedot_decimation" << std::endl;
   auto& result = sweeps.opt_result[isweep][ibond];

   std::vector<stensor2<Tm>> wfs2;
   if(superblock == "lc"){

      for(int i=0; i<vsol.cols(); i++){
         wf.from_array(vsol.col(i));
         //auto wf2 = wf.merge_lc();
	 //if(noise > thresh_noise) wf2.add_noise(noise);
	 //wfs2.push_back(wf2);
      }
/*
      decimation_row(ifkr, wf.qrow, wf.qmid, dcut, rdm_vs_svd, wfs2, 
		     rot, result.dwt, result.deff);

   }else if(superblock == "lr"){

      for(int i=0; i<vsol.cols(); i++){
         wf.from_array(vsol.col(i));
         // Need to first bring two dimensions adjacent to each other before merge!
   	 auto wf2 = wf.permCR_signed().merge_lr();
	 if(noise > thresh_noise) wf2.add_noise(noise);
	 wfs2.push_back(wf2);
      }
      decimation_row(ifkr, wf.qrow, wf.qcol, dcut, rdm_vs_svd, wfs2, 
		     rot, result.dwt, result.deff);

   }else if(superblock == "cr"){

      for(int i=0; i<vsol.cols(); i++){
         wf.from_array(vsol.col(i));
         auto wf2 = wf.merge_cr().T();
	 if(noise > thresh_noise) wf2.add_noise(noise);
	 wfs2.push_back(wf2);
      }
      decimation_row(ifkr, wf.qmid, wf.qcol, dcut, rdm_vs_svd, wfs2, 
		     rot, result.dwt, result.deff);
      rot = rot.T(); // rot[alpha,r] = (V^+)

*/
   } // superblock
}

// initial guess for next site within the bond
template <typename Km>
void onedot_guess_psi(const std::string superblock,
		      comb<Km>& icomb,
		      const directed_bond& dbond,
		      const linalg::matrix<typename Km::dtype>& vsol,
		      stensor3<typename Km::dtype>& wf,
		      const stensor2<typename Km::dtype>& rot){
   if(debug_renorm) std::cout << "> onedot_guess_psi" << std::endl;
   icomb.psi.clear();
   const auto& p0 = dbond.p0;
   const auto& p1 = dbond.p1;
/*
   if(superblock == "lc"){

      for(int i=0; i<vsol.cols(); i++){
         wf.from_array(vsol.col(i));
         auto cwf = rot.H().dot(wf.merge_lc()); // <-W[alpha,r]->
         auto psi = contract_qt3_qt2_l(icomb.rsites[p1],cwf);
         icomb.psi.push_back(psi);
      }

   }else if(superblock == "lr"){

      for(int i=0; i<vsol.cols(); i++){
         wf.from_array(vsol.col(i));
	 wf.permCR_signed();
         auto cwf = rot.H().dot(wf.merge_lr()); // <-W[alpha,r]->
         auto psi = contract_qt3_qt2_l(icomb.rsites[p1],cwf);
         icomb.psi.push_back(psi);
      }
      
   }else if(superblock == "cr"){

      const auto& cturn = sweeps.seq[ibond].cturn; 
      for(int i=0; i<vsol.cols(); i++){
         wf.from_array(vsol.col(i));
         auto cwf = wf.merge_cr().dot(rot.H()); // <-W[l,alpha]->
         if(!cturn){
            auto psi = contract_qt3_qt2_r(icomb.lsites[p0],cwf.T());
         }else{
            // special treatment of the propagation downside to backbone
            auto psi = contract_qt3_qt2_c(icomb.lsites[p0],cwf.T());
            psi.permCR_signed(); // |(lr)c> back to |lcr> order on backbone
            icomb.psi.push_back(psi);
         }
      }

   } // superblock
*/
}

template <typename Km>
void onedot_renorm(sweep_data& sweeps,
		   const int isweep,
		   const int ibond, 
		   comb<Km>& icomb,
		   const linalg::matrix<typename Km::dtype>& vsol,
		   stensor3<typename Km::dtype>& wf,
		   oper_dict<typename Km::dtype>& cqops,
		   oper_dict<typename Km::dtype>& lqops,
		   oper_dict<typename Km::dtype>& rqops,
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
   const auto& p = dbond.p;
   std::string superblock;
   if(dbond.forward){
      superblock = dbond.cturn? "lr" : "lc";
   }else{
      superblock = "cr";
   }
   if(rank == 0) std::cout << "ctns::onedot_renorm superblock=" << superblock;
   auto& timing = sweeps.opt_timing[isweep][ibond];

   // build reduced density matrix & perform decimation
   stensor2<Tm> rot;
   if(rank == 0){
      onedot_decimation(sweeps, isweep, ibond, ifkr, superblock, vsol, wf, rot);
   }
#ifndef SERIAL
   if(size > 1) boost::mpi::broadcast(icomb.world, rot, 0); 
#endif
   timing.td = tools::get_time();

   // prepare guess for the next site
   if(rank == 0 && sweeps.guess){
      onedot_guess_psi(superblock, icomb, dbond, vsol, wf, rot);
   }

   // renorm operators	 
   if(rank == 0 && debug_renorm) std::cout << "> renormalize operators" << std::endl;
   const bool thresh = 1.e-10;
   oper_dict<Tm> qops;
   std::string fname;
/*
   if(superblock == "lc"){
      icomb.lsites[p] = rot.split_lc(wf.qrow, wf.qmid);
      //-------------------------------------------------------------------	 
      assert((rot-icomb.lsites[p].merge_lc()).normF() < thresh);
      auto ovlp = contract_qt3_qt3_lc(icomb.lsites[p],icomb.lsites[p]);
      assert(ovlp.check_identityMatrix(thresh) < thresh);
      //-------------------------------------------------------------------
      oper_renorm_opAll("lc", icomb, p, int2e, int1e, lqops, cqops, qops);
      fname = oper_fname(scratch, p, "l");
   }else if(superblock == "lr"){
      icomb.lsites[p]= rot.split_lr(wf.qrow, wf.qcol);
      //-------------------------------------------------------------------	 
      assert((rot-icomb.lsites[p].merge_lr()).normF() < thresh);
      auto ovlp = contract_qt3_qt3_lr(icomb.lsites[p],icomb.lsites[p]);
      assert(ovlp.check_identityMatrix(thresh) < thresh);
      //-------------------------------------------------------------------	
      oper_renorm_opAll("lr", icomb, p, int2e, int1e, lqops, rqops, qops);
      fname = oper_fname(scratch, p, "l");
   }else if(superblock == "cr"){
      // update site tensor
      icomb.rsites[p] = rot.split_cr(wf.qmid, wf.qcol);
      //-------------------------------------------------------------------	
      assert((rot-icomb.rsites[p].merge_cr()).normF() < thresh);	 
      auto ovlp = contract_qt3_qt3_cr(icomb.rsites[p],icomb.rsites[p]);
      assert(ovlp.check_identityMatrix(thresh) < thresh);
      //-------------------------------------------------------------------
      oper_renorm_opAll("cr", icomb, p, int2e, int1e, cqops, rqops, qops);
      fname = oper_fname(scratch, p, "r");
   }
   timing.te = tools::get_time();
   oper_save(fname, qops);
*/
}

} // ctns

#endif
