#ifndef SWEEP_DECIMATION_H
#define SWEEP_DECIMATION_H

namespace ctns{

const double thresh_sig2 = 1.e-14;
extern const double thresh_sig2;

const double thresh_sig2accum = 0.99;
extern const double thresh_sig2accum;

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
   
   
   int idx = 0, nqr = rdm.rows();
   for(int br=0; br<nqr; br++){
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
	 if(br == 0) std::cout << " diagonalization of rdm for each symmetry sector:" << std::endl;
	 std::cout << " br=" << br << " qr=" << qrow.get_sym(br) << " rdim=" << rdim << " sig2=";
	 for(auto s : sig2) std::cout << s << " ";
	 std::cout << std::endl;
      }
   }


   // 2. select important sig2
   auto index = tools::sort_index(sig2all, 1);
   std::vector<int> kept_dim(nqr,0);
   std::vector<double> kept_wts(nqr,0.0);
   deff = 0;
   double sum = 0.0, SvN = 0.0;
   for(int i=0; i<sig2all.size(); i++){
      if(dcut > -1 && deff >= dcut) break; // discard rest
      int idx = index[i];
      if(sig2all[idx] < thresh_sig2) continue; // discard negative weights
      int br = idx2sector[idx];
      kept_dim[br] += 1;
      kept_wts[br] += sig2all[idx];
      deff += 1;
      sum += sig2all[idx];
      SvN += -sig2all[idx]*std::log2(sig2all[idx]);
      if(sum <= thresh_sig2accum){
	 if(i == 0) std::cout << " important sig2: thresh_sig2accum=" << thresh_sig2accum << std::endl;
	 std::cout << "  i=" << i << " br=" << br << " qr=" << qrow.get_sym(br) << "[" << kept_dim[br]-1 << "]"
                   << " sig2=" << sig2all[idx] << " accum=" << sum << std::endl;
      }
   }
   dwt = 1.0-sum;
   std::cout << " decimation summary: " << qrow.get_dimAll() << "->" << deff
     	     << "  dwt=" << dwt << "  SvN=" << SvN << std::endl;
   // 3. construct qbond and qt2 by assembling blocks
   sum = 0.0;
   std::vector<int> br_kept;
   std::vector<std::pair<qsym,int>> dims;
   auto index2 = tools::sort_index(kept_wts, 1);
   for(int i=0; i<nqr; i++){
      int br = index2[i];
      if(kept_dim[br] == 0) continue;
      const auto& qr = qrow.get_sym(br);
      const auto& dim = kept_dim[br];
      const auto& wts = kept_wts[br];
      br_kept.push_back( br );
      dims.push_back( std::make_pair(qr,dim) );
      sum += wts;     
      std::cout << "  i=" << i << " br=" << br << " qr=" << qr << " dim=" 
		<< dim << " wts=" << wts << " accum=" << sum << std::endl;
   }
   qbond qkept(dims);
   qtensor2<Tm> qt2(qsym(), qrow, qkept);
   for(int bc=0; bc<qkept.size(); bc++){
      int br = br_kept[bc];
      const auto& rbas = rbasis[br];
      auto& blk = qt2(br,bc); 
      std::copy(rbas.data(), rbas.data()+blk.size(), blk.data());
      if(debug_decimation){
         assert(qrow.get_sym(br) == qt2.qcol.get_sym(bc));
         if(bc == 0) std::cout << " reduced basis:" << std::endl;
         std::cout << "  (br,bc)=" << br << "," << bc 
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
   tools::exit("error: decimation_row_kr only works for complex<double>!");
   return qtensor2<Tm>(); // return a fake object to avoid warning
}
template <>
inline qtensor2<std::complex<double>> decimation_row_kr(const qtensor2<std::complex<double>>& rdm,
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
   int idx = 0, nqr = rdm.rows();
   for(int br=0; br<nqr; br++){
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
      std::vector<int> pos_new;
      std::vector<double> phases;
      // mapping product basis to kramers paired basis
      mapping2krbasis(qr,qs1,qs2,dpt,pos_new,phases);
      assert(pos_new.size() == qrow.get_dim(br)); 
      auto rhor = rblk.reorder_rowcol(pos_new,pos_new);
      kramers::eig_solver_kr<std::complex<double>>(qr, phases, rhor, sig2, rbas);
      // save (for odd-electron subspace, only save half of sig2 for later sorting)
      rbasis[br] = rbas.reorder_row(pos_new,1);
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
      //------------------------
      if(debug_decimation){
	 if(br == 0) std::cout << " diagonalization of rdm for each symmetry sector:" << std::endl;
	 std::cout << " br=" << br << " qr=" << qr << " rdim=" << rdim << " sig2=";
	 for(auto s : sig2) std::cout << s << " ";
	 std::cout << std::endl;
      }
   }


   // 2. select important sig2
   auto index = tools::sort_index(sig2all, 1);
   std::vector<int> kept_dim(nqr,0);   
   std::vector<double> kept_wts(nqr,0.0);
   deff = 0;
   double sum = 0.0, SvN = 0.0;
   for(int i=0; i<sig2all.size(); i++){
      if(dcut > -1 && deff >= dcut) break; // discard rest
      int idx = index[i];
      if(sig2all[idx] < thresh_sig2) continue; // discard negative weights
      int br = idx2sector[idx];
      auto qr = qrow.get_sym(br);
      int nfac = (qr.parity() == 1)? 2 : 1;
      kept_dim[br] += nfac;
      kept_wts[br] += nfac*sig2all[idx];
      deff += nfac;
      sum += nfac*sig2all[idx];
      SvN += -nfac*sig2all[idx]*std::log2(sig2all[idx]);
      if(sum <= thresh_sig2accum){
	 if(i == 0) std::cout << " important sig2: thresh_sig2accum=" << thresh_sig2accum << std::endl;
	 std::cout << "  i=" << i << " br=" << br << " qr=" << qr << "[" << kept_dim[br]-1 << "]"
                   << " sig2=" << sig2all[idx] << " accum=" << sum << std::endl;
      }
   }
   dwt = 1.0-sum;
   std::cout << " decimation summary: " << qrow.get_dimAll() << "->" << deff
     	     << "  dwt=" << dwt << "  SvN=" << SvN << std::endl;
   // 3. construct qbond and qt2 by assembling blocks
   sum = 0.0;
   std::vector<int> br_kept;
   std::vector<std::pair<qsym,int>> dims;
   auto index2 = tools::sort_index(kept_wts, 1);
   for(int i=0; i<nqr; i++){
      int br = index2[i];
      if(kept_dim[br] == 0) continue;
      const auto& qr = qrow.get_sym(br);
      const auto& dim = kept_dim[br];
      const auto& wts = kept_wts[br];
      br_kept.push_back( br );
      dims.push_back( std::make_pair(qr,dim) );
      sum += wts;
      std::cout << "  i=" << i << " br=" << br << " qr=" << qr << " dim=" 
		<< dim << " wts=" << wts << " accum=" << sum << std::endl;
   }
   qbond qkept(dims);
   qtensor2<std::complex<double>> qt2(qsym(), qrow, qkept);
   for(int bc=0; bc<qkept.size(); bc++){
      int br = br_kept[bc];
      const auto& rbas = rbasis[br];
      auto& blk = qt2(br,bc); 
      const auto& qr = qkept.get_sym(bc);
      int rdim = blk.rows();
      int cdim = blk.cols();
      assert(qrow.get_sym(br) == qkept.get_sym(bc));
      assert(rbas.rows() == blk.rows());
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
         if(bc == 0) std::cout << " reduced basis:" << std::endl;
         std::cout << "  (br,bc)=" << br << "," << bc 
		   << " qsym=" << qt2.qcol.get_sym(bc)
		   << " shape=(" << blk.rows() << "," << blk.cols() << ")"
     	           << std::endl;
      }
   } // bc
   return qt2;
}

// if dcut=-1, no truncation is performed except for sig2 < thresh_sig2
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

} // ctns

#endif
