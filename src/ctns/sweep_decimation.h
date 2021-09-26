#ifndef SWEEP_DECIMATION_H
#define SWEEP_DECIMATION_H

#ifdef _OPENMP
#include <omp.h>
#endif

#include <numeric>

namespace ctns{

const double thresh_sig2 = 1.e-14;
extern const double thresh_sig2;

const double thresh_sig2accum = 0.99;
extern const double thresh_sig2accum;

const bool debug_decimation = true; //false;
extern const bool debug_decimation;

const bool debug_sig2 = true; //false;
extern const bool debug_sig2;

// select important sigs
inline void decimation_selection(const bool ifkr,
			         const qbond& qrow,
			         const std::vector<double>& sig2all,
			         const std::map<int,int>& idx2sector,
		                 const int& dcut,
			         double& dwt,
		                 int& deff,
   			         std::vector<int>& br_kept,
   			         std::vector<std::pair<qsym,int>>& dims){
   // sort all sigs
   const int nqr = qrow.size();
   auto index = tools::sort_index(sig2all, 1);
   std::vector<int> kept_dim(nqr,0); // no. of states kept in each symmetry sector
   std::vector<double> kept_wts(nqr,0.0); // weights kept in each symmetry sector
   deff = 0;
   double sum = 0.0, SvN = 0.0;
   for(int i=0; i<sig2all.size(); i++){
      if(dcut > -1 && deff >= dcut) break; // discard rest
      int idx = index[i];
      if(sig2all[idx] < thresh_sig2) continue; // discard negative weights
      int br = idx2sector.at(idx);
      auto qr = qrow.get_sym(br);
      int nfac = (ifkr && qr.parity() == 1)? 2 : 1; // odd case: kept KR-pair
      kept_dim[br] += nfac;
      kept_wts[br] += nfac*sig2all[idx];
      deff += nfac;
      sum += nfac*sig2all[idx];
      SvN += -nfac*sig2all[idx]*std::log2(sig2all[idx]);
      if(debug_sig2 && sum <= thresh_sig2accum){
	 if(i == 0) std::cout << "important sig2: thresh_sig2accum=" << thresh_sig2accum << std::endl;
	 std::cout << " i=" << i << " br=" << br << " qr=" << qr << "[" << kept_dim[br]-1 << "]"
                   << " sig2=" << sig2all[idx] << " accum=" << sum << std::endl;
      }
   }
   dwt = 1.0-sum;
   std::cout << "decimation summary: " << qrow.get_dimAll() << "->" << deff
     	     << "  dwt=" << dwt << "  SvN=" << SvN << std::endl;
   // construct qbond 
   sum = 0.0;
   auto index2 = tools::sort_index(kept_wts, 1); // order symmetry sectors by kept weights
   for(int i=0; i<nqr; i++){
      int br = index2[i];
      if(kept_dim[br] == 0) continue;
      const auto& qr = qrow.get_sym(br);
      const auto& dim = kept_dim[br];
      const auto& wts = kept_wts[br];
      br_kept.push_back(br);
      dims.emplace_back(qr,dim);
      sum += wts;    
      if(debug_sig2){ 
         std::cout << " i=" << i << " br=" << br << " qr=" << qr << " dim=" 
                   << dim << " wts=" << wts << " accum=" << sum << std::endl;
      }
   }
}

// generate renormalized basis from wfs2[row,col] for row
template <typename Tm>
void decimation_row_nkr(const qbond& qs1,
		        const qbond& qs2,
		        const int dcut,
			const double rdm_vs_svd,
		        const std::vector<stensor2<Tm>>& wfs2,
		        stensor2<Tm>& rot,
		        double& dwt,
		        int& deff){
   auto qprod = qmerge(qs1, qs2);
   auto qrow = qprod.first;
   auto dpt = qprod.second;
   assert(qrow == wfs2[0].info.qrow);
   auto qcol = wfs2[0].info.qcol;
   int nroots = wfs2.size();
   const int nqr = qrow.size();
#ifdef _OPENMP
   int maxthreads = omp_get_max_threads();
#else
   int maxthreads = 1;
#endif
   if(debug_decimation){
      std::cout << "ctns::decimation_row_nkr dcut=" << dcut 
	        << " nqr=" << nqr << " maxthreads=" << maxthreads
		<< std::endl;
   }
   // 1. compute reduced basis
   std::map<int,std::pair<std::vector<double>,linalg::matrix<Tm>>> results;
#ifdef _OPENMP
   #pragma omp parallel for schedule(dynamic)
#endif
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
	 const auto& qc = qcol.get_sym(bc);     
	 const auto& blk = wfs2[0](br,bc);
	 if(blk.size() == 0) continue;
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
      // save
      if(matched == 1){
#ifdef _OPENMP
         #pragma omp critical
#endif
         results[br] = std::make_pair(sigs2, U);
      }
   } // br
   int idx = 0;
   double sig2sum = 0.0;
   std::vector<double> sig2all;
   std::map<int,int> idx2sector;
   for(int br=0; br<nqr; br++){
      const auto& sigs2 = results[br].first;
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
   decimation_selection(false, qrow, sig2all, idx2sector, dcut, dwt, deff, br_kept, dims);
   qbond qkept(dims);
   auto isym = qkept.get_sym(0).isym();
   stensor2<Tm> qt2(qsym(isym), qrow, qkept);
   for(int bc=0; bc<qkept.size(); bc++){
      int br = br_kept[bc];
      // copy the kept portion of rbas into blk
      const auto& rbas = results[br].second;
      auto& blk = qt2(br,bc); 
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
		       const int dcut,
		       const double rdm_vs_svd,
		       const std::vector<stensor2<Tm>>& wfs2,
		       stensor2<Tm>& rot,
		       double& dwt,
		       int& deff){
   tools::exit("error: decimation_row_kr only works for complex<double>!");
}
template <>
inline void decimation_row_kr(const qbond& qs1,
		              const qbond& qs2,
		              const int dcut,
		              const double rdm_vs_svd,
		              const std::vector<stensor2<std::complex<double>>>& wfs2,
		              stensor2<std::complex<double>>& rot,
		              double& dwt,
		              int& deff){
   using Tm = std::complex<double>;
   auto qprod = qmerge(qs1, qs2);
   auto qrow = qprod.first;
   auto dpt = qprod.second;
   assert(qrow == wfs2[0].info.qrow);
   auto qcol = wfs2[0].info.qcol;
   int nroots = wfs2.size();
   const int nqr = qrow.size();
#ifdef _OPENMP
   int maxthreads = omp_get_max_threads();
#else
   int maxthreads = 1;
#endif
   if(debug_decimation){ 
      std::cout << "ctns::decimation_row_kr dcut=" << dcut 
                << " nqr=" << nqr << " maxthreads=" << maxthreads
		<< std::endl;
   }
   // 1. compute reduced basis
   std::map<int,std::pair<std::vector<double>,linalg::matrix<Tm>>> results;
#ifdef _OPENMP
   #pragma omp parallel for schedule(dynamic)
#endif
   for(int br=0; br<nqr; br++){
      const auto& qr = qrow.get_sym(br);
      const int rdim = qrow.get_dim(br);
      if(debug_decimation){ 
         if(br == 0) std::cout << "decimation for each symmetry sector:" << std::endl;
	 std::cout << ">br=" << br << " qr=" << qr << " rdim=" << rdim << std::endl;
      }
      // mapping product basis to kramers paired basis
      std::vector<int> pos_new;
      std::vector<double> phases;
      mapping2krbasis(qr, qs1, qs2, dpt, pos_new, phases);
      assert(pos_new.size() == rdim);
      // search for matched block 
      std::vector<double> sigs2;
      linalg::matrix<Tm> U;
      int matched = 0;
      for(int bc=0; bc<qcol.size(); bc++){
	 const auto& qc = qcol.get_sym(bc);     
	 const auto& blk = wfs2[0](br,bc);
	 if(blk.size() == 0) continue;
	 if(debug_decimation) std::cout << " find matched qc =" << qc << std::endl;
	 matched += 1;
	 if(matched > 1) tools::exit("multiple matched qc is not supported!"); 
	 // compute KRS-adapted renormalized basis
	 std::vector<linalg::matrix<Tm>> blks(nroots);
         for(int iroot=0; iroot<nroots; iroot++){
	    blks[iroot] = wfs2[iroot](br,bc).to_matrix().reorder_row(pos_new).T();
	 }
	 kramers::get_renorm_states_kr(qr, phases, blks, sigs2, U, rdm_vs_svd, debug_decimation);
      } // qc
      // save 
      int nkept = U.cols();
      if(nkept > 0){
#ifdef _OPENMP
         #pragma omp critical
#endif
         results[br] = std::make_pair(sigs2, U.reorder_row(pos_new,1));
      }
   } // br
   int idx = 0;
   double sig2sum = 0.0;
   std::vector<double> sig2all;
   std::map<int,int> idx2sector;
   for(int br=0; br<nqr; br++){
      const auto& qr = qrow.get_sym(br);
      const auto& sigs2 = results[br].first;
      int nkept = results[br].second.cols();
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
   decimation_selection(true, qrow, sig2all, idx2sector, dcut, dwt, deff, br_kept, dims);
   qbond qkept(dims);
   auto isym = qkept.get_sym(0).isym();
   stensor2<Tm> qt2(qsym(isym), qrow, qkept);
   for(int bc=0; bc<qkept.size(); bc++){
      int br = br_kept[bc];
      // copy rbas into blk
      const auto& rbas = results[br].second;
      auto& blk = qt2(br,bc); 
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
		    const int dcut,
		    const double rdm_vs_svd,
		    const std::vector<stensor2<Tm>>& wfs2,
		    stensor2<Tm>& rot,
		    double& dwt,
		    int& deff){
   if(!ifkr){
      decimation_row_nkr(qs1, qs2, dcut, rdm_vs_svd, wfs2, rot, dwt, deff);
   }else{
      decimation_row_kr(qs1, qs2, dcut, rdm_vs_svd, wfs2, rot, dwt, deff);
   }
}

} // ctns

#endif
