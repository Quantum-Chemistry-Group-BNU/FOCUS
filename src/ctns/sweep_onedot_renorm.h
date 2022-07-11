#ifndef SWEEP_ONEDOT_RENORM_H
#define SWEEP_ONEDOT_RENORM_H

#include "oper_io.h"
#include "sweep_decimation.h"

namespace ctns{

const double thresh_noise = 1.e-10;
extern const double thresh_noise;
   
const double thresh_canon = 1.e-10;
extern const double thresh_canon;

template <typename Tm>
void onedot_decimation(const input::schedule& schd,
		       sweep_data& sweeps,
		       const int isweep,
		       const int ibond, 
		       const bool ifkr,
		       const std::string superblock,
		       const int ksupp,
		       const linalg::matrix<Tm>& vsol,
		       stensor3<Tm>& wf,
		       stensor2<Tm>& rot, 
	               const std::string fname){
   const bool debug = schd.ctns.verbose>0;
   const auto& rdm_vs_svd = schd.ctns.rdm_vs_svd;
   const auto& dbond = sweeps.seq[ibond];
   const int& dbranch = schd.ctns.dbranch;
   const int dcut = (dbranch>0 && dbond.p1.second>0)? dbranch : sweeps.ctrls[isweep].dcut;
   const bool iftrunc = 2*ksupp >= (int)std::log2(dcut); 
   const auto& noise = sweeps.ctrls[isweep].noise;
   if(debug){
      std::cout <<" (rdm_vs_svd,dbranch,dcut,iftrunc,noise)=" 
                << std::scientific << std::setprecision(1) << rdm_vs_svd << ","
                << dbranch << "," << dcut << "," << iftrunc << ","
                << noise << std::endl;
   }
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
		     iftrunc, dcut, rdm_vs_svd, wfs2, 
		     rot, result.dwt, result.deff, fname,
		     debug);

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
      		     iftrunc, dcut, rdm_vs_svd, wfs2, 
		     rot, result.dwt, result.deff, fname,
		     debug);

   }else if(superblock == "cr"){

      for(int i=0; i<nroots; i++){
         wf.from_array(vsol.col(i));
	 // wf3[l,r,c] => wf2[l,cr]
         auto wf2 = wf.merge_cr().T();
	 if(noise > thresh_noise) wf2.add_noise(noise);
	 wfs2[i] = std::move(wf2);
      }
      decimation_row(ifkr, wf.info.qmid, wf.info.qcol, 
      		     iftrunc, dcut, rdm_vs_svd, wfs2, 
		     rot, result.dwt, result.deff, fname,
		     debug);
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
   const bool debug = false;
   if(debug) std::cout << "ctns::onedot_guess_psi superblock=" << superblock << std::endl;
   const auto& pdx0 = icomb.topo.rindex.at(dbond.p0);
   const auto& pdx1 = icomb.topo.rindex.at(dbond.p1);
   int nroots = vsol.cols();
   icomb.psi.clear();
   icomb.psi.resize(nroots);
   if(superblock == "lc"){

      for(int i=0; i<nroots; i++){
         wf.from_array(vsol.col(i));
         auto cwf = rot.H().dot(wf.merge_lc()); // <-W[alpha,r]->
         auto psi = contract_qt3_qt2("l",icomb.rsites[pdx1],cwf);
         icomb.psi[i] = std::move(psi);
      }

   }else if(superblock == "lr"){

      for(int i=0; i<nroots; i++){
         wf.from_array(vsol.col(i));
	 wf.permCR_signed();
         auto cwf = rot.H().dot(wf.merge_lr()); // <-W[alpha,r]->
         auto psi = contract_qt3_qt2("l",icomb.rsites[pdx1],cwf);
         icomb.psi[i] = std::move(psi);
      }
      
   }else if(superblock == "cr"){

      auto cturn = dbond.is_cturn(); 
      for(int i=0; i<nroots; i++){
         wf.from_array(vsol.col(i));
         auto cwf = wf.merge_cr().dot(rot.H()); // <-W[l,alpha]->
         if(!cturn){
            auto psi = contract_qt3_qt2("r",icomb.lsites[pdx0],cwf.T());
	    icomb.psi[i] = std::move(psi);
         }else{
            // special treatment of the propagation downside to backbone
            auto psi = contract_qt3_qt2("c",icomb.lsites[pdx0],cwf.T());
            psi.permCR_signed(); // |(lr)c> back to |lcr> order on backbone
            icomb.psi[i] = std::move(psi);
         }
      }

   } // superblock
}

template <typename Km>
void onedot_renorm(comb<Km>& icomb,
	           const integral::two_body<typename Km::dtype>& int2e, 
	           const integral::one_body<typename Km::dtype>& int1e,
		   const input::schedule& schd,
	           const std::string scratch,
		   const linalg::matrix<typename Km::dtype>& vsol,
		   stensor3<typename Km::dtype>& wf,
		   const oper_dictmap<typename Km::dtype>& qops_dict,
		   oper_dict<typename Km::dtype>& qops,
		   sweep_data& sweeps,
		   const int isweep,
		   const int ibond){
   const auto& lqops = qops_dict.at("l");
   const auto& rqops = qops_dict.at("r");
   const auto& cqops = qops_dict.at("c");
   using Tm = typename Km::dtype;
   const bool ifkr = Km::ifkr;
   int size = 1, rank = 0;
#ifndef SERIAL
   size = icomb.world.size();
   rank = icomb.world.rank();
#endif
   const bool debug = (rank == 0);
   const auto& dbond = sweeps.seq[ibond];
   std::string superblock;
   if(dbond.forward){
      superblock = dbond.is_cturn()? "lr" : "lc";
   }else{
      superblock = "cr";
   }
   if(debug && schd.ctns.verbose>0){ 
      std::cout << "ctns::onedot_renorm superblock=" << superblock;
   }
   auto& timing = sweeps.opt_timing[isweep][ibond];

   // 1. build reduced density matrix & perform decimation
   stensor2<Tm> rot;
   if(rank == 0){
      auto dims = icomb.topo.check_partition(1, dbond, false);
      int ksupp;
      if(superblock == "lc"){
         ksupp = dims[0] + dims[2];
      }else if(superblock == "lr"){
         ksupp = dims[0] + dims[1];
      }else if(superblock == "cr"){
         ksupp = dims[1] + dims[2];
      }
      std::string fname = scratch+"/decimation"
	       	        + "_isweep"+std::to_string(isweep)
	                + "_ibond"+std::to_string(ibond)+".txt";
      onedot_decimation(schd, sweeps, isweep, ibond, ifkr, 
		        superblock, ksupp, vsol, wf, rot, fname);
   }
#ifndef SERIAL
   if(size > 1) boost::mpi::broadcast(icomb.world, rot, 0); 
#endif
   timing.td = tools::get_time();

   // 2. prepare guess for the next site
   if(rank == 0 && schd.ctns.guess){
      onedot_guess_psi(superblock, icomb, dbond, vsol, wf, rot);
   }
   timing.te = tools::get_time();

   // 3. renorm operators	 
   const auto p = dbond.get_current();
   const auto& pdx = icomb.topo.rindex.at(p); 
   std::string fname;
   if(schd.ctns.save_formulae) fname = scratch+"/rformulae"
	                             + "_isweep"+std::to_string(isweep)
	               		     + "_ibond"+std::to_string(ibond) + ".txt";
   if(superblock == "lc"){
      icomb.lsites[pdx] = rot.split_lc(wf.info.qrow, wf.info.qmid);
      //-------------------------------------------------------------------
      rot -= icomb.lsites[pdx].merge_lc();
      assert(rot.normF() < thresh_canon);
      auto ovlp = contract_qt3_qt3("lc", icomb.lsites[pdx], icomb.lsites[pdx]);
      assert(ovlp.check_identityMatrix(thresh_canon) < thresh_canon);
      //-------------------------------------------------------------------
      oper_renorm_opAll("lc", icomb, p, int2e, int1e, schd,
		        lqops, cqops, qops, fname); 
   }else if(superblock == "lr"){
      icomb.lsites[pdx]= rot.split_lr(wf.info.qrow, wf.info.qcol);
      //-------------------------------------------------------------------
      rot -= icomb.lsites[pdx].merge_lr();
      assert(rot.normF() < thresh_canon);
      auto ovlp = contract_qt3_qt3("lr", icomb.lsites[pdx],icomb.lsites[pdx]);
      assert(ovlp.check_identityMatrix(thresh_canon) < thresh_canon);
      //-------------------------------------------------------------------
      oper_renorm_opAll("lr", icomb, p, int2e, int1e, schd,
		        lqops, rqops, qops, fname); 
   }else if(superblock == "cr"){
      icomb.rsites[pdx] = rot.split_cr(wf.info.qmid, wf.info.qcol);
      //-------------------------------------------------------------------
      rot -= icomb.rsites[pdx].merge_cr();
      assert(rot.normF() < thresh_canon);
      auto ovlp = contract_qt3_qt3("cr", icomb.rsites[pdx],icomb.rsites[pdx]);
      assert(ovlp.check_identityMatrix(thresh_canon) < thresh_canon);
      //-------------------------------------------------------------------
      oper_renorm_opAll("cr", icomb, p, int2e, int1e, schd,
		        cqops, rqops, qops, fname); 
   }
}

} // ctns

#endif
