#ifndef SWEEP_ONEDOT_RENORM_H
#define SWEEP_ONEDOT_RENORM_H

#include "oper_io.h"
#include "sweep_decimation.h"

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
   auto& result = sweeps.opt_result[isweep][ibond];
   int nroots = vsol.cols();
   std::vector<stensor2<Tm>> wfs2(nroots);
   if(superblock == "lc"){

      for(int i=0; i<nroots; i++){
         wf.from_array(vsol.col(i));
	 // wf3[l,r,c] => wf2[lc,r]
         auto wf2 = wf.merge_lc();
	 if(noise > thresh_noise) wf2.add_noise(noise);
	 wfs2[i] = std::move(wf2);
      }
      decimation_row(ifkr, wf.info.qrow, wf.info.qmid, 
		     dcut, rdm_vs_svd, wfs2, 
		     rot, result.dwt, result.deff);

   }else if(superblock == "lr"){

      for(int i=0; i<nroots; i++){
         wf.from_array(vsol.col(i));
         // Need to first bring two dimensions adjacent to each other before merge!
	 wf.permCR_signed();
	 // wf3[l,r,c] => wf2[lr,c]
   	 auto wf2 = wf.merge_lr();
	 if(noise > thresh_noise) wf2.add_noise(noise);
	 wfs2[i] = std::move(wf2);
      }
      decimation_row(ifkr, wf.info.qrow, wf.info.qcol, 
      		     dcut, rdm_vs_svd, wfs2, 
		     rot, result.dwt, result.deff);

   }else if(superblock == "cr"){

      for(int i=0; i<nroots; i++){
         wf.from_array(vsol.col(i));
	 // wf3[l,r,c] => wf2[l,cr]
         auto wf2 = wf.merge_cr().T();
	 if(noise > thresh_noise) wf2.add_noise(noise);
	 wfs2[i] = std::move(wf2);
      }
      decimation_row(ifkr, wf.info.qmid, wf.info.qcol, 
      		     dcut, rdm_vs_svd, wfs2, 
		     rot, result.dwt, result.deff);
      rot = rot.T(); // rot[alpha,r] = (V^+)

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
   if(debug_renorm) std::cout << "ctns::onedot_guess_psi" << std::endl;
   const auto& pdx0 = icomb.topo.rindex.at(dbond.p0);
   const auto& pdx1 = icomb.topo.rindex.at(dbond.p1);
   int nroots = vsol.cols();
   icomb.psi.clear();
   icomb.psi.resize(nroots);
   if(superblock == "lc"){

      for(int i=0; i<nroots; i++){
         wf.from_array(vsol.col(i));
         auto cwf = rot.H().dot(wf.merge_lc()); // <-W[alpha,r]->
         auto psi = contract_qt3_qt2_l(icomb.rsites[pdx1],cwf);
         icomb.psi[i] = std::move(psi);
      }

   }else if(superblock == "lr"){

      for(int i=0; i<nroots; i++){
         wf.from_array(vsol.col(i));
	 wf.permCR_signed();
         auto cwf = rot.H().dot(wf.merge_lr()); // <-W[alpha,r]->
         auto psi = contract_qt3_qt2_l(icomb.rsites[pdx1],cwf);
         icomb.psi[i] = std::move(psi);
      }
      
   }else if(superblock == "cr"){

      const auto& cturn = dbond.cturn; 
      for(int i=0; i<nroots; i++){
         wf.from_array(vsol.col(i));
         auto cwf = wf.merge_cr().dot(rot.H()); // <-W[l,alpha]->
         if(!cturn){
            auto psi = contract_qt3_qt2_r(icomb.lsites[pdx0],cwf.T());
	    icomb.psi[i] = std::move(psi);
         }else{
            // special treatment of the propagation downside to backbone
            auto psi = contract_qt3_qt2_c(icomb.lsites[pdx0],cwf.T());
            psi.permCR_signed(); // |(lr)c> back to |lcr> order on backbone
            icomb.psi[i] = std::move(psi);
         }
      }

   } // superblock
}

template <typename Km>
void onedot_renorm(const input::schedule& schd,
		   sweep_data& sweeps,
		   const int isweep,
		   const int ibond, 
		   comb<Km>& icomb,
		   const linalg::matrix<typename Km::dtype>& vsol,
		   stensor3<typename Km::dtype>& wf,
		   const oper_dict<typename Km::dtype>& lqops,
		   const oper_dict<typename Km::dtype>& rqops,
		   const oper_dict<typename Km::dtype>& cqops,
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
   timing.te = tools::get_time();

   // renorm operators	 
   const bool thresh = 1.e-10;
   const auto& p = dbond.p;
   const auto& pdx = icomb.topo.rindex.at(p); 
   oper_dict<Tm> qops;
   std::string fname;
   if(superblock == "lc"){
      icomb.lsites[pdx] = rot.split_lc(wf.info.qrow, wf.info.qmid);
      //-------------------------------------------------------------------
      rot -= icomb.lsites[pdx].merge_lc();
      assert(rot.normF() < thresh);
      auto ovlp = contract_qt3_qt3("lc", icomb.lsites[pdx], icomb.lsites[pdx]);
      assert(ovlp.check_identityMatrix(thresh) < thresh);
      //-------------------------------------------------------------------
      oper_renorm_opAll("lc", icomb, p, int2e, int1e, 
		        lqops, cqops, qops, schd.ctns.alg_renorm);
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
   }else if(superblock == "cr"){
      icomb.rsites[pdx] = rot.split_cr(wf.info.qmid, wf.info.qcol);
      //-------------------------------------------------------------------
      rot -= icomb.rsites[pdx].merge_cr();
      assert(rot.normF() < thresh);
      auto ovlp = contract_qt3_qt3("cr", icomb.rsites[pdx],icomb.rsites[pdx]);
      assert(ovlp.check_identityMatrix(thresh) < thresh);
      //-------------------------------------------------------------------
      oper_renorm_opAll("cr", icomb, p, int2e, int1e, 
		        cqops, rqops, qops, schd.ctns.alg_renorm);
      fname = oper_fname(scratch, p, "r");
   }
   timing.tf = tools::get_time();
   oper_save(fname, qops);
}

} // ctns

#endif
