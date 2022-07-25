#ifndef SWEEP_DECIMATION_H
#define SWEEP_DECIMATION_H

#ifdef _OPENMP
#include <omp.h>
#endif

#include <numeric>

namespace ctns{

const double thresh_sig2 = 1.e-14;
extern const double thresh_sig2;

const bool debug_decimation = false;
extern const bool debug_decimation;

// select important sigs
inline void decimation_selection(const bool ifkr,
			         const qbond& qrow,
				 const std::vector<bool>& ifmatched,
			         const std::vector<double>& sig2all,
			         const std::map<int,int>& idx2sector,
		                 const int& dcut,
			         double& dwt,
		                 int& deff,
   			         std::vector<int>& br_kept,
   			         std::vector<std::pair<qsym,int>>& dims,
		    		 const std::string fname){
   std::ofstream fout(fname);
   auto index = tools::sort_index(sig2all, 1); // sort all sigs
   const int nqr = qrow.size();
   std::vector<int> kept_dim(nqr,0); // no. of states kept in each symmetry sector
   std::vector<double> kept_wts(nqr,0.0); // weights kept in each symmetry sector
   deff = 0; // bond dimension kept (including additional for symmetry)
   double accum = 0.0, SvN = 0.0;
   fout << "sorted renormalized states: total=" << sig2all.size()
        << " dcut=" << dcut << " thresh_sig2=" << thresh_sig2 
	<< std::endl;
   for(int i=0; i<sig2all.size(); i++){
      if(dcut > -1 && deff >= dcut) break; // discard rest
      int idx = index[i];
      if(sig2all[idx] < thresh_sig2) continue; // discard negative weights
      int br = idx2sector.at(idx);
      auto qr = qrow.get_sym(br);
      int nfac = (ifkr && qr.parity()==1)? 2 : 1; // odd case: kept KR-pair
      deff += nfac;
      kept_dim[br] += nfac;
      kept_wts[br] += nfac*sig2all[idx];
      accum += nfac*sig2all[idx];
      SvN += -nfac*sig2all[idx]*std::log2(sig2all[idx]);
      fout << " i=" << i << " qr=" << qr 
	   << " " << kept_dim[br]/nfac-1 << "-th"
           << " sig2=" << sig2all[idx] 
	   << " accum=" << accum << std::endl;
   } // i
   dwt = 1.0-accum;
   // construct qbond & recompute deff including additional states 
   deff = 0;
   accum = 0.0;
   auto index2 = tools::sort_index(kept_wts, 1); // order symmetry sectors by kept weights
   fout << "select renormalized states per symmetry sector: nqr=" << nqr << std::endl;
   for(int iqr=0; iqr<nqr; iqr++){
      int br = index2[iqr];
      const auto& qr = qrow.get_sym(br);
      const auto& dim0 = qrow.get_dim(br);
      const auto& dim = kept_dim[br];
      const auto& wts = kept_wts[br];
      if(dim != 0){
         br_kept.push_back(br);
         dims.emplace_back(qr,dim);
         accum += wts;    
         deff += dim;
         // save information
         fout << " iqr=" << iqr << " qr=" << qr
  	      << " dim[full,kept]=" << dim0 << "," << dim 
              << " wts=" << wts << " accum=" << accum << " deff=" << deff 
  	      << std::endl;
      }else{
// ZL@20220517 disable such choice, since it will create many sector with dim=1 
/*
	 // additional: kept at least one state per sector
	 if(!ifmatched[br]) continue;
	 br_kept.push_back(br);
	 int dmin = (ifkr && qr.parity()==1)? 2 : 1;
         dims.emplace_back(qr,dmin);
         deff += dmin;
         // save information
         fout << " iqr=" << iqr << " qr=" << qr
	      << " dim[full,kept]=" << dim0 << "," << dmin 
              << " wts=" << wts << " accum=" << accum << " deff=" << deff
	      << " (additional)" << std::endl;
*/
      }
   } // iqr
   fout << "decimation summary: " << qrow.get_dimAll() << "->" << deff  
        << " dwt=" << std::showpos << std::scientific << std::setprecision(3) << dwt 
	<< " SvN=" << std::noshowpos << SvN << std::endl;
   fout.close();
   std::cout << "decimation summary: " << qrow.get_dimAll() << "->" << deff  
	     << " dwt=" << std::showpos << std::scientific << std::setprecision(3) << dwt 
             << " SvN=" << std::noshowpos << SvN << std::endl;
}

// generate renormalized basis from wfs2[row,col] for row
template <typename Tm>
void decimation_row_nkr(const qbond& qs1,
		        const qbond& qs2,
			const bool iftrunc,
		        const int dcut,
			const double rdm_vs_svd,
		        const std::vector<stensor2<Tm>>& wfs2,
		        stensor2<Tm>& rot,
		        double& dwt,
		        int& deff,
		        const std::string fname){
   const auto qprod = qmerge(qs1, qs2);
   const auto& qrow = qprod.first;
   const auto& dpt = qprod.second;
   assert(qrow == wfs2[0].info.qrow);
   const auto& qcol = wfs2[0].info.qcol;
   int nroots = wfs2.size();
   int nqr = qrow.size();
   if(debug_decimation){
      std::cout << "ctns::decimation_row_nkr"
	        << " dcut=" << dcut << " nqr=" << nqr
		<< std::endl;
   }
   qrow.print("qsuper");
   
   // 0. untruncated case
   if(!iftrunc){
      auto isym = qrow.get_sym(0).isym();
      stensor2<Tm> qt2(qsym(isym), qrow, qrow); // identity matrix
      for(int br=0; br<nqr; br++){
         const auto& qr = qrow.get_sym(br);
         const int rdim = qrow.get_dim(br);
	 auto blk = qt2(br,br);
	 for(int r=0; r<rdim; r++){
            blk(r,r) = 1.0;
	 } // r
      }
      rot = std::move(qt2);
      dwt = 0.0;
      deff = qrow.get_dimAll();
      std::cout << "decimation summary: keep all " 
	        << deff << " states" << std::endl;
      return;
   }

   // 1. compute reduced basis
   std::map<int,std::pair<std::vector<double>,linalg::matrix<Tm>>> results;
//#ifdef _OPENMP
//   #pragma omp parallel for schedule(dynamic)
//#endif
   for(int br=0; br<nqr; br++){
      const auto& qr = qrow.get_sym(br);
      const int rdim = qrow.get_dim(br);
      if(debug_decimation){ 
         if(br == 0) std::cout << "decimation for each symmetry sector:" << std::endl;
	 std::cout << ">br=" << br << " qr=" << qr << " rdim=" << rdim << std::endl;
      }
      // search for matched block 
      std::vector<double> sigs2;
      linalg::matrix<Tm> U;
      int matched = 0;
      for(int bc=0; bc<qcol.size(); bc++){
	 if(wfs2[0](br,bc).empty()) continue;
	 const auto& qc = qcol.get_sym(bc);     
	 if(debug_decimation) std::cout << " find matched qc =" << qc << std::endl;
	 matched += 1;
	 if(matched > 1) tools::exit("multiple matched qc is not supported!"); 
         // compute renormalized basis
	 std::vector<linalg::matrix<Tm>> blks(nroots);
         for(int iroot=0; iroot<nroots; iroot++){
	    blks[iroot] = wfs2[iroot](br,bc).to_matrix().T();
	 }
	 kramers::get_renorm_states_nkr(blks, sigs2, U, rdm_vs_svd, debug_decimation);
      } // qc
//#ifdef _OPENMP
//      #pragma omp critical
//#endif
      results[br] = std::make_pair(sigs2, U);
   } // br
   int idx = 0;
   double sig2sum = 0.0;
   std::vector<bool> ifmatched(nqr);
   std::vector<double> sig2all;
   std::map<int,int> idx2sector;
   for(int br=0; br<nqr; br++){
      const auto& sigs2 = results[br].first;
      ifmatched[br] = (sigs2.size() > 0);
      if(!ifmatched[br]) continue;
      std::copy(sigs2.begin(), sigs2.end(), std::back_inserter(sig2all));
      sig2sum += std::accumulate(sigs2.begin(), sigs2.end(), 0.0);
      for(int k=0; k<sigs2.size(); k++){
         idx2sector[idx] = br;
         idx++;
      }
   } // br
   sig2sum = 1.0/sig2sum;
   std::transform(sig2all.begin(), sig2all.end(), sig2all.begin(),
		  [sig2sum](const double& x){ return x*sig2sum; });
   sig2sum = std::accumulate(sig2all.begin(), sig2all.end(), 0.0);
   assert(std::abs(sig2sum - 1.0) < 1.e-10);
   
   // 2. select important sig2 & form rot
   std::vector<int> br_kept;
   std::vector<std::pair<qsym,int>> dims;
   decimation_selection(false, qrow, ifmatched, sig2all, idx2sector, dcut, 
		        dwt, deff, br_kept, dims, fname);
   qbond qkept(dims);
   auto isym = qkept.get_sym(0).isym();
   stensor2<Tm> qt2(qsym(isym), qrow, qkept);
   for(int bc=0; bc<qkept.size(); bc++){
      int br = br_kept[bc];
      // copy the kept portion of rbas into blk
      const auto& rbas = results[br].second;
      auto blk = qt2(br,bc); 
      linalg::xcopy(blk.size(), rbas.data(), blk.data());
      if(debug_decimation){
         assert(qrow.get_sym(br) == qt2.info.qcol.get_sym(bc));
         if(bc == 0) std::cout << "reduced basis:" << std::endl;
         std::cout << " (br,bc)=" << br << "," << bc 
		   << " qsym=" << qt2.info.qcol.get_sym(bc)
		   << " shape=(" << blk.rows() << "," << blk.cols() << ")"
     	           << std::endl;
      }
   } // bc
   rot = std::move(qt2);
}

template <typename Tm>
void decimation_row_kr(const qbond& qs1,
		       const qbond& qs2,
		       const bool iftrunc,
		       const int dcut,
		       const double rdm_vs_svd,
		       const std::vector<stensor2<Tm>>& wfs2,
		       stensor2<Tm>& rot,
		       double& dwt,
		       int& deff,
		       const std::string fname){
   tools::exit("error: decimation_row_kr only works for complex<double>!");
}
template <>
inline void decimation_row_kr(const qbond& qs1,
		              const qbond& qs2,
			      const bool iftrunc,
		              const int dcut,
		              const double rdm_vs_svd,
		              const std::vector<stensor2<std::complex<double>>>& wfs2,
		              stensor2<std::complex<double>>& rot,
		              double& dwt,
		              int& deff,
		              const std::string fname){
   using Tm = std::complex<double>;
   const auto qprod = qmerge(qs1, qs2);
   const auto& qrow = qprod.first;
   const auto& dpt = qprod.second;
   assert(qrow == wfs2[0].info.qrow);
   const auto& qcol = wfs2[0].info.qcol;
   int nroots = wfs2.size();
   int nqr = qrow.size();
   int dim12 = qrow.get_dimAll(); 
   if(debug_decimation){ 
      std::cout << "ctns::decimation_row_kr"
	        << " dcut=" << dcut << " nqr=" << nqr 
		<< std::endl;
   }
   qrow.print("qsuper");

   // 0. untruncated case
   if(!iftrunc){
      auto isym = qrow.get_sym(0).isym();
      stensor2<Tm> qt2(qsym(isym), qrow, qrow); // identity matrix
      for(int br=0; br<nqr; br++){
         const auto& qr = qrow.get_sym(br);
         const int rdim = qrow.get_dim(br);
	 auto blk = qt2(br,br);
         std::vector<double> sigs2(rdim);
         linalg::matrix<Tm> U;
	 // mapping product basis to kramers paired basis
         std::vector<int> pos_new;
         std::vector<double> phases;
         mapping2krbasis(qr, qs1, qs2, dpt, pos_new, phases);
	 // compute KRS-adapted renormalized basis (from a fake rho = Iden)
         auto rhor = linalg::identity_matrix<Tm>(rdim);
	 kramers::eig_solver_kr<std::complex<double>>(qr, phases, rhor, sigs2, U);
	 // convert back to the original product basis
         U = U.reorder_row(pos_new,1);
	 linalg::xcopy(rdim*rdim,U.data(),blk.data()); 
	 /*
	 std::cout << "br=" << br << " qr=" << qr << std::endl;
	 U.print("U");
	 */
      }
      rot = std::move(qt2);
      dwt = 0.0;
      deff = qrow.get_dimAll();
      std::cout << "decimation summary: keep all " 
	        << deff << " states" << std::endl;
      return;
   }

   // 1. compute reduced basis
   std::map<int,std::pair<std::vector<double>,linalg::matrix<Tm>>> results;
//#ifdef _OPENMP
//   #pragma omp parallel for schedule(dynamic)
//#endif
   for(int br=0; br<nqr; br++){
      const auto& qr = qrow.get_sym(br);
      const int rdim = qrow.get_dim(br);
      if(debug_decimation){ 
         if(br == 0) std::cout << "decimation for each symmetry sector:" << std::endl;
	 std::cout << ">br=" << br << " qr=" << qr << " rdim=" << rdim << std::endl;
      }
      // search for matched block 
      std::vector<double> sigs2;
      linalg::matrix<Tm> U;
      int matched = 0;
      for(int bc=0; bc<qcol.size(); bc++){
	 if(wfs2[0](br,bc).empty()) continue;
	 const auto& qc = qcol.get_sym(bc);     
	 if(debug_decimation) std::cout << " find matched qc =" << qc << std::endl;
	 matched += 1;
	 if(matched > 1) tools::exit("multiple matched qc is not supported!"); 
         // mapping product basis to kramers paired basis
         std::vector<int> pos_new;
         std::vector<double> phases;
         mapping2krbasis(qr, qs1, qs2, dpt, pos_new, phases);
         assert(pos_new.size() == rdim);
	 // compute KRS-adapted renormalized basis
	 std::vector<linalg::matrix<Tm>> blks(nroots);
         for(int iroot=0; iroot<nroots; iroot++){
	    blks[iroot] = wfs2[iroot](br,bc).to_matrix().reorder_row(pos_new).T();
	 }
	 kramers::get_renorm_states_kr(qr, phases, blks, sigs2, U, rdm_vs_svd, debug_decimation);
	 // convert back to the original product basis
         U = U.reorder_row(pos_new,1);
      } // qc
//#ifdef _OPENMP
//      #pragma omp critical
//#endif
      results[br] = std::make_pair(sigs2, U);
   } // br
   int idx = 0;
   double sig2sum = 0.0;
   std::vector<bool> ifmatched(nqr);
   std::vector<double> sig2all;
   std::map<int,int> idx2sector;
   for(int br=0; br<nqr; br++){
      const auto& qr = qrow.get_sym(br);
      const auto& sigs2 = results[br].first;
      ifmatched[br] = (sigs2.size() > 0);
      if(!ifmatched[br]) continue; 
      int nkept = results[br].second.cols();
      assert(nkept == sigs2.size());
      if(qr.parity() == 0){
         std::copy(sigs2.begin(), sigs2.end(), std::back_inserter(sig2all));
         sig2sum += std::accumulate(sigs2.begin(), sigs2.end(), 0.0);
         for(int i=0; i<nkept; i++){
            idx2sector[idx] = br;
            idx++;
         }
      }else{
         // for odd-electron subspace, only save half of sig2 for later sorting
         assert(nkept%2 == 0);
         int nkept2 = nkept/2;
         std::copy(sigs2.begin(), sigs2.begin()+nkept2, std::back_inserter(sig2all));
         sig2sum += 2.0*std::accumulate(sigs2.begin(), sigs2.begin()+nkept2, 0.0);
         for(int i=0; i<nkept2; i++){
            idx2sector[idx] = br;
            idx++;
         }
      } // parity
   } // br
   sig2sum = 1.0/sig2sum;
   std::transform(sig2all.begin(), sig2all.end(), sig2all.begin(),
		  [sig2sum](const double& x){ return x*sig2sum; });
   //NOTE: in kr case, sig2all only contain partial sigs2, thus no check is applied
   //sig2sum = std::accumulate(sig2all.begin(), sig2all.end(), 0.0);
   //assert(std::abs(sig2sum - 1.0) < 1.e-10); 
   
   // 2. select important sig2 & form rot
   std::vector<int> br_kept;
   std::vector<std::pair<qsym,int>> dims;
   decimation_selection(true, qrow, ifmatched, sig2all, idx2sector, dcut, 
		        dwt, deff, br_kept, dims, fname);
   qbond qkept(dims);
   auto isym = qkept.get_sym(0).isym();
   stensor2<Tm> qt2(qsym(isym), qrow, qkept);
   for(int bc=0; bc<qkept.size(); bc++){
      int br = br_kept[bc];
      // copy rbas into blk
      const auto& rbas = results[br].second;
      auto blk = qt2(br,bc); 
      int rdim = blk.rows();
      int cdim = blk.cols();
      const auto& qr = qkept.get_sym(bc);
      assert(qrow.get_sym(br) == qkept.get_sym(bc));
      assert(rbas.rows() == blk.rows());
      if(qr.parity() == 0){
         linalg::xcopy(rdim*cdim, rbas.col(0), blk.col(0));
      }else{
	 assert(rdim%2 == 0 && cdim%2 == 0 && rbas.cols()%2==0);
	 int cdim0 = rbas.cols()/2;
	 int cdim1 = cdim/2;
	 linalg::xcopy(rdim*cdim1, rbas.col(0), blk.col(0));
	 linalg::xcopy(rdim*cdim1, rbas.col(cdim0), blk.col(cdim1));
      }
      if(debug_decimation){
         assert(qrow.get_sym(br) == qt2.info.qcol.get_sym(bc));
         if(bc == 0) std::cout << "reduced basis:" << std::endl;
         std::cout << " (br,bc)=" << br << "," << bc 
		   << " qsym=" << qt2.info.qcol.get_sym(bc)
		   << " shape=(" << blk.rows() << "," << blk.cols() << ")"
     	           << std::endl;
      }
   } // bc
   rot = std::move(qt2);
}

// if dcut=-1, no truncation is performed except for sig2 < thresh_sig2
template <typename Tm>
void decimation_row(const bool ifkr,
                    const qbond& qs1,
		    const qbond& qs2,
		    const bool iftrunc,
		    const int dcut,
		    const double rdm_vs_svd,
		    const std::vector<stensor2<Tm>>& wfs2,
		    stensor2<Tm>& rot,
		    double& dwt,
		    int& deff,
		    const std::string fname,
		    const bool debug){
   if(debug){
      std::cout << "ctns::decimation_row: ";
      if(iftrunc) std::cout << "fname=" << fname;
      std::cout << std::endl;
   }
   if(!ifkr){
      decimation_row_nkr(qs1, qs2, iftrunc, dcut, rdm_vs_svd, 
		         wfs2, rot, dwt, deff, fname);
   }else{
      decimation_row_kr(qs1, qs2, iftrunc, dcut, rdm_vs_svd, 
		        wfs2, rot, dwt, deff, fname);
   }
}

} // ctns

#endif
