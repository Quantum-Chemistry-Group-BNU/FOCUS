#ifndef SWEEP_DECIMATION_H
#define SWEEP_DECIMATION_H

#include "sweep_prdm.h"

namespace ctns{

const double thresh_noise = 1.e-10;
extern const double thresh_noise;

const double thresh_sig2 = 1.e-20;
extern const double thresh_sig2;

const bool debug_decimation = true;
extern const bool debug_decimation;

// wf[L,R] = U[L,l]*sl*Vh[l,R]
template <typename Tm>
qtensor2<Tm> decimation_row_nkr(const qtensor2<Tm>& rdm,
			        const int dcut,
			        double& dwt,
			        int& deff){
   if(debug_decimation) std::cout << "ctns::decimation_row_nkr dcut=" << dcut << std::endl;
   const auto& qrow = rdm.qrow;
   // 0. normalize before diagonalization
   Tm rfac = 1.0/rdm.trace();
   // 1. compute reduced basis
   std::map<int,int> idx2sector; 
   std::vector<double> sig2all;
   std::map<int,linalg::matrix<Tm>> rbasis;
   int idx = 0;
   for(int br=0; br<rdm.rows(); br++){
      const auto& blk = rdm(br,br);
      if(blk.size() == 0) continue;
      // compute renormalized basis
      int rdim = qrow.get_dim(br);
      std::vector<double> sig2(rdim);
      linalg::matrix<Tm> rbas(rdim,rdim);
      auto rblk = rfac*blk;
      linalg::eig_solver(rblk, sig2, rbas, 1);
      // save
      std::copy(sig2.begin(), sig2.end(), std::back_inserter(sig2all));
      rbasis[br] = rbas;
      for(int i=0; i<rdim; i++){
	 idx2sector[idx] = br;
	 idx++;
      }
      if(debug_decimation){
	 if(br == 0) std::cout << "diagonalization of rdm for each symmetry sector:" << std::endl;
	 std::cout << " br=" << br << " qr=" << qrow.get_sym(br) << " rdim=" << rdim << " sig2=";
	 for(auto s : sig2) std::cout << s << " ";
	 std::cout << std::endl;
      }
   }
   // 2. select important sig2
   auto index = tools::sort_index(sig2all, 1);
   std::map<int,std::pair<int,double>> kept; // br->(dim,wt)
   deff = 0;
   double sum = 0.0;
   double SvN = 0.0;
   for(int i=0; i<sig2all.size(); i++){
      if(deff >= dcut) break; // discard rest
      int idx = index[i];
      if(sig2all[idx] < thresh_sig2) continue; // discard negative weights
      int br = idx2sector[idx];
      auto it = kept.find(br);
      if(it == kept.end()){
         kept[br].first = 1;
	 kept[br].second = sig2all[idx];
      }else{
	 kept[br].first += 1;
	 kept[br].second += sig2all[idx];
      }
      deff += 1;
      sum += sig2all[idx];
      SvN += -sig2all[idx]*std::log2(sig2all[idx]);
      if(debug_decimation){
	if(i == 0) std::cout << "sorted sig2:" << std::endl;     
	std::cout << " i=" << i << " (br,ith)=" << br << "," << kept[br].first-1 
             	  << " sig2=" << sig2all[idx] << " accum=" << sum << std::endl;
      }
   }
   dwt = 1.0-sum;
   std::cout << "decimation summary: " << qrow.get_dimAll() << "->" << deff
     	     << "  dwt=" << dwt << "  SvN=" << SvN << std::endl;
   // 3. construct qbond and qt2 by assembling blocks
   sum = 0.0;
   std::vector<int> br_matched;
   std::vector<std::pair<qsym,int>> dims;
   for(const auto& p : kept){
      const auto& br = p.first;
      const auto& dim = p.second.first;
      const auto& wt = p.second.second;
      const auto& qr = qrow.get_sym(br);
      br_matched.push_back(br);
      dims.push_back(std::make_pair(qr,dim));
      if(debug_decimation){
	 sum += wt;     
         std::cout << " br=" << p.first << " qr=" << qr << " dim=" 
		   << dim << " wt=" << wt << " accum=" << sum << std::endl;
      }
   }
   qbond qkept(dims);
   qtensor2<Tm> qt2(qsym(), qrow, qkept);
   for(int bc=0; bc<qkept.size(); bc++){
      int br = br_matched[bc];
      const auto& rbas = rbasis[br];
      auto& blk = qt2(br,bc); 
      std::copy(rbas.data(), rbas.data()+blk.size(), blk.data());
      if(debug_decimation){
         assert(qrow.get_sym(br) == qt2.qcol.get_sym(bc));
         if(bc == 0) std::cout << "reduced basis:" << std::endl;
         std::cout << " (br,bc)=" << br << "," << bc 
		   << " qsym=" << qt2.qcol.get_sym(bc)
		   << " shape=(" << blk.rows() << "," << blk.cols() << ")"
     	           << std::endl;
      }
   } // bc
   return qt2;
}

template <typename Tm>
qtensor2<Tm> decimation_row_kr(const qtensor2<Tm>& rdm,
			       const int dcut,
			       double& dwt,
			       int& deff,
 			       const qbond& qs1,
 			       const qbond& qs2,
 			       const qdpt& dpt){
   std::cout << "error: decimation_row_kr just work for complex<double>!" << std::endl;
   exit(1);
}

template <>
qtensor2<std::complex<double>> decimation_row_kr(const qtensor2<std::complex<double>>& rdm,
			       			 const int dcut,
			       			 double& dwt,
			       			 int& deff,
 			       			 const qbond& qs1,
 			       			 const qbond& qs2,
 			       			 const qdpt& dpt){
   if(debug_decimation) std::cout << "ctns::decimation_row_kr dcut=" << dcut << std::endl;
   const auto& qrow = rdm.qrow;
   // 0. normalize before diagonalization
   std::complex<double> rfac = 1.0/rdm.trace();
   // 1. compute reduced basis
   std::map<int,int> idx2sector; 
   std::vector<double> sig2all;
   std::map<int,linalg::matrix<std::complex<double>>> rbasis;
   int idx = 0;
   for(int br=0; br<rdm.rows(); br++){
      const auto& blk = rdm(br,br);
      if(blk.size() == 0) continue;
      // compute renormalized basis
      int rdim = qrow.get_dim(br);
      std::vector<double> sig2(rdim);
      linalg::matrix<std::complex<double>> rbas(rdim,rdim);
      auto rblk = rfac*blk;
      //------------------------
      // KRS-adapted decimation
      auto qr = qrow.get_sym(br);
      std::vector<int> idx_all;
      std::vector<double> phases;
      mapping2krbasis(qr,qs1,qs2,dpt,idx_all,phases);
      auto rhor = rblk.reorder_rowcol(idx_all,idx_all);
      eig_solver_kr<std::complex<double>>(qr, rhor, sig2, rbas, phases);
      rbas = rbas.reorder_row(idx_all,1);
      // save
      if(qr.parity() == 1){
	 int dim1 = phases.size();
	 assert(rdim == 2*dim1);
         std::copy(sig2.begin(), sig2.begin()+dim1, std::back_inserter(sig2all));
         for(int i=0; i<dim1; i++){
            idx2sector[idx] = br;
	    idx++;
         }
      }else{
         std::copy(sig2.begin(), sig2.end(), std::back_inserter(sig2all));
         for(int i=0; i<rdim; i++){
            idx2sector[idx] = br;
	    idx++;
         }
      }
      rbasis[br] = rbas;
      //------------------------
      if(debug_decimation){
	 if(br == 0) std::cout << "diagonalization of rdm for each symmetry sector:" << std::endl;
	 std::cout << " br=" << br << " qr=" << qr << " rdim=" << rdim << " sig2=";
	 for(auto s : sig2) std::cout << s << " ";
	 std::cout << std::endl;
      }
   }
   // 2. select important sig2
   auto index = tools::sort_index(sig2all, 1);
   std::map<int,std::pair<int,double>> kept; // br->(dim,wt)
   deff = 0;
   double sum = 0.0;
   double SvN = 0.0;
   for(int i=0; i<sig2all.size(); i++){
      if(deff >= dcut) break; // discard rest
      int idx = index[i];
      if(sig2all[idx] < thresh_sig2) continue; // discard negative weights
      int br = idx2sector[idx];
      auto qr = qrow.get_sym(br);
      int nfac = (qr.parity() == 1)? 2 : 1;
      auto it = kept.find(br);
      if(it == kept.end()){
         kept[br].first = nfac;
	 kept[br].second = nfac*sig2all[idx];
      }else{
	 kept[br].first += nfac;
	 kept[br].second += nfac*sig2all[idx];
      }
      deff += nfac;
      sum += nfac*sig2all[idx];
      SvN += -nfac*sig2all[idx]*std::log2(sig2all[idx]);
      if(debug_decimation){
         if(i == 0) std::cout << "sorted sig2:" << std::endl;     
	 std::cout << " i=" << i << " (br,ith)=" << br << "," << kept[br].first-1 
              	   << " sig2=" << sig2all[idx] << " accum=" << sum << std::endl;
      }
   }
   dwt = 1.0-sum;
   std::cout << "decimation summary: " << qrow.get_dimAll() << "->" << deff
     	     << "  dwt=" << dwt << "  SvN=" << SvN << std::endl;
   // 3. construct qbond and qt2 by assembling blocks
   sum = 0.0;
   std::vector<int> br_matched;
   std::vector<std::pair<qsym,int>> dims;
   for(const auto& p : kept){ // br->(dim,wt)
      const auto& br = p.first;
      const auto& dim = p.second.first;
      const auto& qr = qrow.get_sym(br);
      const auto& wt = p.second.second;
      br_matched.push_back(br);
      dims.push_back(std::make_pair(qr,dim));
      if(debug_decimation){
	 sum += wt;
         std::cout << " br=" << p.first << " qr=" << qr << " dim=" << dim 
		   << " wt=" << wt << " accum=" << sum << std::endl;
      }
   }
   qbond qkept(dims);
   qtensor2<std::complex<double>> qt2(qsym(), qrow, qkept);
   for(int bc=0; bc<qkept.size(); bc++){
      int br = br_matched[bc];
      const auto& rbas = rbasis[br];
      auto& blk = qt2(br,bc); 
      const auto& qr = qkept.get_sym(bc);
      int rdim = blk.rows();
      int cdim = blk.cols();
      if(qr.parity() == 1){
	 assert(rdim%2 == 0 && cdim%2 == 0);
	 int rdim1 = rdim/2;
	 int cdim1 = cdim/2;
         std::copy(rbas.col(0), rbas.col(0)+rdim*cdim1, blk.col(0));
	 std::copy(rbas.col(rdim1), rbas.col(rdim1)+rdim*cdim1, blk.col(cdim1));
      }else{
         std::copy(rbas.col(0), rbas.col(0)+rdim*cdim, blk.col(0));
      }
      if(debug_decimation){
         assert(qrow.get_sym(br) == qt2.qcol.get_sym(bc));
         if(bc == 0) std::cout << "reduced basis:" << std::endl;
         std::cout << " (br,bc)=" << br << "," << bc 
		   << " qsym=" << qt2.qcol.get_sym(bc)
		   << " shape=(" << blk.rows() << "," << blk.cols() << ")"
     	           << std::endl;
      }
   } // bc
   return qt2;
}

template <typename Tm>
qtensor2<Tm> decimation_row(const qtensor2<Tm>& rdm,
			    const int dcut,
			    double& dwt,
			    int& deff,
			    const bool ifkr,
			    const qbond& qs1,
			    const qbond& qs2,
			    const qdpt& dpt){
   qtensor2<Tm> qt2;
   if(!ifkr){
      qt2 = decimation_row_nkr(rdm, dcut, dwt, deff);
   }else{
      qt2 = decimation_row_kr(rdm, dcut, dwt, deff, qs1, qs2, dpt);
   }
   return qt2;
}

template <typename Km>
void decimation_onedot_lc(sweep_data& sweeps,
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
   const auto& dcut  = sweeps.ctrls[isweep].dcut;
   const auto& noise = sweeps.ctrls[isweep].noise; 
   std::cout << " renormalize |lc> : (dcut,noise)=" << dcut << "," 
	     << std::scientific << std::setprecision(1) << noise << std::endl;
   auto& timing = sweeps.opt_timing[isweep][ibond];
   auto& result = sweeps.opt_result[isweep][ibond];
   auto qprod = qmerge(wf.qrow, wf.qmid);
   auto qlc = qprod.first;
   auto dpt = qprod.second;
   qtensor2<typename Km::dtype> rdm(qsym(), qlc, qlc);
   // 1. build pRDM 
   for(int i=0; i<vsol.cols(); i++){
      wf.from_array(vsol.col(i));
      rdm += wf.merge_lc().get_rdm_row();
      if(noise > thresh_noise) get_prdm_lc(wf, lqops, cqops, noise, rdm);
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
      const auto& p1 = sweeps.seq[ibond].p1;	   
      icomb.psi.clear();
      for(int i=0; i<vsol.cols(); i++){
         wf.from_array(vsol.col(i));
         auto cwf = qt2.H().dot(wf.merge_lc()); // <-W[alpha,r]->
         auto psi = contract_qt3_qt2_l(icomb.rsites[p1],cwf);
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
void decimation_onedot_lr(sweep_data& sweeps,
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
   const auto& dcut  = sweeps.ctrls[isweep].dcut;
   const auto& noise = sweeps.ctrls[isweep].noise; 
   std::cout << " renormalize |lr> (comb) : (dcut,noise)=" << dcut << "," 
	     << std::scientific << std::setprecision(1) << noise << std::endl;
   auto& timing = sweeps.opt_timing[isweep][ibond];
   auto& result = sweeps.opt_result[isweep][ibond];
   auto qprod = qmerge(wf.qrow, wf.qcol);
   auto qlr = qprod.first;
   auto dpt = qprod.second;
   qtensor2<typename Km::dtype> rdm(qsym(), qlr, qlr);
   // 1. build pRDM 
   for(int i=0; i<vsol.cols(); i++){
      wf.from_array(vsol.col(i));
      // Note: need to first bring two dimensions adjacent to each other before merge!
      rdm += wf.permCR_signed().merge_lr().get_rdm_row();
      if(noise > thresh_noise) get_prdm_lr(wf, lqops, rqops, noise, rdm);
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
      const auto& p1 = sweeps.seq[ibond].p1;	   
      icomb.psi.clear();
      for(int i=0; i<vsol.cols(); i++){
         wf.from_array(vsol.col(i));
         auto cwf = qt2.H().dot(wf.permCR_signed().merge_lr()); // <-W[alpha,r]->
         auto psi = contract_qt3_qt2_l(icomb.rsites[p1],cwf);
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
void decimation_onedot_cr(sweep_data& sweeps,
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
   const auto& dcut  = sweeps.ctrls[isweep].dcut;
   const auto& noise = sweeps.ctrls[isweep].noise; 
   std::cout << " renormalize |cr> : (dcut,noise)=" << dcut << "," 
	     << std::scientific << std::setprecision(1) << noise << std::endl;
   auto& timing = sweeps.opt_timing[isweep][ibond];
   auto& result = sweeps.opt_result[isweep][ibond];
   auto qprod = qmerge(wf.qmid, wf.qcol);
   auto qcr = qprod.first;
   auto dpt = qprod.second;
   qtensor2<typename Km::dtype> rdm(qsym(), qcr, qcr);
   // 1. build pRDM 
   for(int i=0; i<vsol.cols(); i++){
      wf.from_array(vsol.col(i));
      rdm += wf.merge_cr().get_rdm_col();
      if(noise > thresh_noise) get_prdm_cr(wf, cqops, rqops, noise, rdm);
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
      const auto& p0 = sweeps.seq[ibond].p0;	  
      const auto& cturn = sweeps.seq[ibond].cturn; 
      icomb.psi.clear();
      for(int i=0; i<vsol.cols(); i++){
         wf.from_array(vsol.col(i));
         auto cwf = wf.merge_cr().dot(qt2.H()); // <-W[l,alpha]->
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
   oper_dict<typename Km::dtype> qops;
   oper_renorm_opAll("cr", icomb, p, int2e, int1e, cqops, rqops, qops);
   auto fname = oper_fname(scratch, p, "rop");
   oper_save(fname, qops);
}

template <typename Km>
void decimation_onedot(sweep_data& sweeps,
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
   const auto& dbond = sweeps.seq[ibond];
   std::cout << "ctns::decimation_onedot (forward,cturn)=" 
	     << dbond.forward << "," << dbond.cturn;
   // build reduced density matrix & perform renormalization
   if(dbond.forward){
      // update lsites & ql
      if(!dbond.cturn){
	 decimation_onedot_lc(sweeps, isweep, ibond, icomb, vsol, wf, 
			      lqops, cqops, int2e, int1e, scratch);
      }else{
	 // special for comb
         decimation_onedot_lr(sweeps, isweep, ibond, icomb, vsol, wf, 
			      lqops, rqops, int2e, int1e, scratch);
      } // cturn
   }else{
      // update rsites & qr
      decimation_onedot_cr(sweeps, isweep, ibond, icomb, vsol, wf, 
		           cqops, rqops, int2e, int1e, scratch); 
   }
}

/*
void tns::decimation_twodot(comb& icomb, 
		            const directed_bond& dbond,
		            const int dcut, 
			    const matrix& vsol,
			    qtensor4& wf,
		            double& dwt,
			    int& deff){
   auto p0 = get<0>(dbond);
   auto p1 = get<1>(dbond);
   auto forward = get<2>(dbond);
   auto p = forward? p0 : p1;
   bool cturn = (icomb.type[p0] == 3 && p1.second == 1);
   cout << "tns::decimation_twodot (fw,ct,dcut)=(" 
	<< forward << "," << cturn << "," << dcut << ") "; 
   qtensor2 rdm;
   if(forward){
      // update lsites & ql
      if(!cturn){
         cout << "renormalize |lc1>" << endl;
	 for(int i=0; i<vsol.cols(); i++){
	    wf.from_array(vsol.col(i));
            auto wf3 = wf.merge_c2r();
	    if(i == 0){
	       rdm  = wf3.merge_lc().get_rdm_row();
	    }else{
	       rdm += wf3.merge_lc().get_rdm_row();
	    }
	 }
	 auto qt2 = decimation_row(rdm, dcut, dwt, deff);
         icomb.lsites[p] = qt2.split_lc(wf.qrow, wf.qmid, wf.dpt_lc1().second);
 	 //-------------------------------------------------------------------	 
	 assert((qt2-icomb.lsites[p].merge_lc()).normF() < 1.e-10);
	 auto ovlp = contract_qt3_qt3_lc(icomb.lsites[p],icomb.lsites[p]);
	 assert(ovlp.check_identityMatrix(1.e-10,false)<1.e-10);
 	 //-------------------------------------------------------------------	
	 // initial guess for next site within the bond
	 icomb.psi.clear();
	 for(int i=0; i<vsol.cols(); i++){
	    wf.from_array(vsol.col(i));
	    auto wf3 = wf.merge_c2r();
	    auto cwf = qt2.T().dot(wf3.merge_lc()); // <-W[alpha,r]->
	    auto psi = cwf.split_cr(wf.qver, wf.qcol, wf.dpt_c2r().second);
	    icomb.psi.push_back(psi);
	 }
      }else{
         cout << "renormalize |lr> (comb)" << endl;
	 for(int i=0; i<vsol.cols(); i++){
            wf.from_array(vsol.col(i));
	    if(i == 0){
	       rdm  = wf.permCR_signed().merge_lr_c1c2().get_rdm_row();
	    }else{
	       rdm += wf.permCR_signed().merge_lr_c1c2().get_rdm_row();
	    }
	 }
	 auto qt2 = decimation_row(rdm, dcut, dwt, deff);
	 icomb.lsites[p]= qt2.split_lr(wf.qrow, wf.qcol, wf.dpt_lr().second);
 	 //-------------------------------------------------------------------	 
	 assert((qt2-icomb.lsites[p].merge_lr()).normF() < 1.e-10);
	 auto ovlp = contract_qt3_qt3_lr(icomb.lsites[p],icomb.lsites[p]);
	 assert(ovlp.check_identityMatrix(1.e-10,false)<1.e-10);
 	 //-------------------------------------------------------------------	
	 // initial guess for next site within the bond
	 icomb.psi.clear();
	 for(int i=0; i<vsol.cols(); i++){
	    wf.from_array(vsol.col(i));
	    auto cwf = qt2.T().dot(wf.permCR_signed().merge_lr_c1c2());
	    auto psi = cwf.split_cr(wf.qmid, wf.qver, wf.dpt_c1c2().second);
	    icomb.psi.push_back(psi); // psi on branch
	 }
      }
   }else{
      // update rsites & qr
      if(!cturn){
         cout << "renormalize |c2r>" << endl;
	 for(int i=0; i<vsol.cols(); i++){
	    wf.from_array(vsol.col(i));
	    auto wf3 = wf.merge_lc1();
            if(i == 0){
               rdm  = wf3.merge_cr().get_rdm_col();
            }else{
               rdm += wf3.merge_cr().get_rdm_col();
            }
         }
         auto qt2 = decimation_row(rdm, dcut, dwt, deff, true);
   
	 if(permute) qt2 = qt2.P();
	 
	 
	 
	 icomb.rsites[p] = qt2.split_cr(wf.qver, wf.qcol, wf.dpt_c2r().second);
         //-------------------------------------------------------------------	
         assert((qt2-icomb.rsites[p].merge_cr()).normF() < 1.e-10);	 
         auto ovlp = contract_qt3_qt3_cr(icomb.rsites[p],icomb.rsites[p]);
         assert(ovlp.check_identityMatrix(1.e-10,false)<1.e-10);
         //-------------------------------------------------------------------	
         // initial guess for next site within the bond
	 icomb.psi.clear();
	 for(int i=0; i<vsol.cols(); i++){
	    wf.from_array(vsol.col(i));
	    auto wf3 = wf.merge_lc1();
	    auto cwf = wf3.merge_cr().dot(qt2.T()); // <-W[l,alpha]->
	    auto psi = cwf.split_lc(wf.qrow, wf.qmid, wf.dpt_lc1().second);
            icomb.psi.push_back(psi);
	 }
      }else{
         cout << "renormalize |c1c2> (comb)" << endl;
	 for(int i=0; i<vsol.cols(); i++){
            wf.from_array(vsol.col(i));
	    if(i == 0){
	       rdm  = wf.permCR_signed().merge_lr_c1c2().get_rdm_col();
	    }else{
	       rdm += wf.permCR_signed().merge_lr_c1c2().get_rdm_col();
	    }
	 }
	 auto qt2 = decimation_row(rdm, dcut, dwt, deff, true);
	 icomb.rsites[p]= qt2.split_cr(wf.qmid, wf.qver, wf.dpt_c1c2().second);
         //-------------------------------------------------------------------	
         assert((qt2-icomb.rsites[p].merge_cr()).normF() < 1.e-10);	 
         auto ovlp = contract_qt3_qt3_cr(icomb.rsites[p],icomb.rsites[p]);
         assert(ovlp.check_identityMatrix(1.e-10,false)<1.e-10);
         //-------------------------------------------------------------------	
         // initial guess for next site within the bond
	 icomb.psi.clear();
	 for(int i=0; i<vsol.cols(); i++){
	    wf.from_array(vsol.col(i));
	    auto cwf = wf.permCR_signed().merge_lr_c1c2().dot(qt2.T()); 
	    auto psi = cwf.split_lr(wf.qrow, wf.qcol, wf.dpt_lr().second);
	    psi = psi.permCR_signed();
            icomb.psi.push_back(psi); // psi on backbone
	 }
      }
   }
}
*/

/*
template <typename Km>
void renorm_twodot(const directed_bond& dbond,
		   const comb<Km>& icomb, 
	           oper_dict<typename Km::dtype>& c1qops,
	           oper_dict<typename Km::dtype>& c2qops,
	           oper_dict<typename Km::dtype>& lqops,
	           oper_dict<typename Km::dtype>& rqops,	
	           const integral::two_body<typename Km::dtype>& int2e, 
	           const integral::one_body<typename Km::dtype>& int1e,
	           const std::string scratch){
   std::cout << "ctns::renorm_twodot superblock=";
   const auto& p = dbond.p;
   const auto& forward = dbond.forward;
   const auto& cturn = dbond.cturn;
   oper_dict<typename Km::dtype> qops;
   if(forward){
      if(!cturn){
	 std::cout << "lc1" << std::endl;
	 oper_renorm_opAll("lc", icomb, p, int2e, int1e, lqops, c1qops, qops);
      }else{
	 std::cout << "lr" << std::endl;
	 oper_renorm_opAll("lr", icomb, p, int2e, int1e, lqops, rqops, qops);
      }
      std::string fname = oper_fname(scratch, p, "lop");
      oper_save(fname, qops);
   }else{
      if(!cturn){
         std::cout << "c2r" << std::endl;
         oper_renorm_opAll("cr", icomb, p, int2e, int1e, c2qops, rqops, qops);
      }else{
	 //     
	 //        c2      
	 //        |
	 //   c1---p
	 //        |
	 //    l---*---r
	 //
         std::cout << "c1c2" << std::endl;
         oper_renorm_opAll("cr", icomb, p, int2e, int1e, c1qops, c2qops, qops);
      }
      std::string fname = oper_fname(scratch, p, "rop");
      oper_save(fname, qops);
   }
}
*/

} // ctns

#endif
