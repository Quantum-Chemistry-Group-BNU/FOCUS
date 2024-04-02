#ifndef SWEEP_DECIM_H
#define SWEEP_DECIM_H

#include "sweep_decim_nkr.h"
#include "sweep_decim_kr.h"
#include "sadmrg/sweep_decim_su2.h"

namespace ctns{

   const double thresh_sig2 = 1.e-14;
   extern const double thresh_sig2;
   
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

      std::streambuf *psbuf, *backup;
      std::ofstream file;
      bool ifsave = !fname.empty();
      if(ifsave){
         // http://www.cplusplus.com/reference/ios/ios/rdbuf/
         file.open(fname);
         backup = std::cout.rdbuf(); // back up cout's streambuf
         psbuf = file.rdbuf(); // get file's streambuf
         std::cout.rdbuf(psbuf); // assign streambuf to cout
      }

      auto index = tools::sort_index(sig2all, 1); // sort all sigs
      const int nqr = qrow.size();
      std::vector<int> kept_dim(nqr,0); // no. of states kept in each symmetry sector
      std::vector<double> kept_wts(nqr,0.0); // weights kept in each symmetry sector
      deff = 0; // bond dimension kept (including additional for symmetry)
      double accum = 0.0, SvN = 0.0;
      std::cout << "sorted renormalized states: total=" << sig2all.size()
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
         std::cout << " i=" << i << " qr=" << qr 
            << " " << kept_dim[br]/nfac-1 << "-th"
            << " sig2=" << sig2all[idx] 
            << " accum=" << accum << std::endl;
      } // i
      dwt = 1.0-accum;
      // construct qbond & recompute deff including additional states 
      deff = 0;
      accum = 0.0;
      // order symmetry sectors by kept weights / dimensions
      std::vector<int> index2;
      if(sort_by_dim){
         index2 = tools::sort_index(kept_dim, 1);
      }else{
         index2 = tools::sort_index(kept_wts, 1); 
      }
      std::cout << "select renormalized states per symmetry sector: nqr=" << nqr << std::endl;
      for(int iqr=0; iqr<nqr; iqr++){
         int br = index2[iqr];
         const auto& qr = qrow.get_sym(br);
         const auto& dim0 = qrow.get_dim(br);
         const auto& dim = kept_dim[br];
         const auto& wts = kept_wts[br];
         if(dim == 0) continue;
         br_kept.push_back(br);
         dims.emplace_back(qr,dim);
         accum += wts;    
         deff += dim;
         // save information
         std::cout << " iqr=" << iqr << " qr=" << qr
            << " dim[full,kept]=" << dim0 << "," << dim 
            << " wts=" << wts << " accum=" << accum << " deff=" << deff 
            << std::endl;
      } // iqr

      if(ifsave){
         std::cout << "decimation summary: " << qrow.get_dimAll() << "->" << deff  
            << " dwt=" << std::showpos << std::scientific << std::setprecision(3) << dwt 
            << " SvN=" << std::noshowpos << SvN << std::endl;
         std::cout.rdbuf(backup); // restore cout's original streambuf
         file.close();
      }
      std::cout << "decimation summary: " << qrow.get_dimAll() << "->" << deff  
         << " dwt=" << std::showpos << std::scientific << std::setprecision(3) << dwt 
         << " SvN=" << std::noshowpos << SvN << std::endl;
   }

   // generate renormalized basis from wfs2[row,col] for row
   // if dcut=-1, no truncation is performed except for sig2 < thresh_sig2
   template <typename Qm, typename Tm>
      void decimation_row(const comb<Qm,Tm>& icomb,
            const qbond& qs1,
            const qbond& qs2,
            const bool iftrunc,
            const int dcut,
            const double rdm_svd,
            const int alg_decim,
            const std::vector<qtensor2<Qm::ifabelian,Tm>>& wfs2,
            qtensor2<Qm::ifabelian,Tm>& rot,
            double& dwt,
            int& deff,
            const std::string fname,
            const bool debug){
         int rank = 0, size = 1;
#ifndef SERIAL
         rank = icomb.world.rank();
         size = icomb.world.size();
#endif   
         auto t0 = tools::get_time();
         const auto qprod = qmerge_spaces(Qm::ifabelian, qs1, qs2);
         const auto& qrow = qprod.first;
         const auto& dpt = qprod.second;
         assert(qrow == wfs2[0].info.qrow);
         const auto& qcol = wfs2[0].info.qcol;
         int nroots = wfs2.size();
         int nqr = qrow.size();
         if(debug){
            std::cout << "ctns::decimation_row"
               << " ifab=" << Qm::ifabelian
               << " iftrunc=" << iftrunc
               << " dcut=" << dcut << " nqr=" << nqr
               << " alg_decim=" << alg_decim
               << " mpisize=" << size;
            if(iftrunc && !fname.empty()) std::cout << " fname=" << fname;
            std::cout << std::endl;
            qrow.print("qsuper");
         }

         // 0. untruncated case
         if(!iftrunc){
            if(rank == 0){
               auto isym = qrow.get_sym(0).isym();
               qtensor2<Qm::ifabelian,Tm> qt2(qsym(isym), qrow, qrow); // identity matrix
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
               if(debug) std::cout << "decimation summary: keep all " 
                  << deff << " states" << std::endl;
            } // rank-0
            return;
         }

         // 1. compute reduced basis
         std::map<int,std::pair<std::vector<double>,linalg::matrix<Tm>>> results;
         decimation_genbasis(icomb, qs1, qs2, qrow, qcol, dpt, rdm_svd, alg_decim, wfs2, results);
         auto t1 = tools::get_time();

         if(rank == 0){
            // 1.5 merge all sig2 and normalize
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
            linalg::xscal(sig2all.size(), 1.0/sig2sum, sig2all.data());
            sig2sum = std::accumulate(sig2all.begin(), sig2all.end(), 0.0);
            assert(std::abs(sig2sum - 1.0) < 1.e-10);

            // 2. select important sig2 
            std::vector<int> br_kept;
            std::vector<std::pair<qsym,int>> dims;
            decimation_selection(false, qrow, ifmatched, sig2all, idx2sector, dcut, 
                  dwt, deff, br_kept, dims, fname);
            auto t2 = tools::get_time();

            // 3. form rot
            qbond qkept(dims);
            auto isym = qkept.get_sym(0).isym();
            qtensor2<Qm::ifabelian,Tm> qt2(qsym(isym), qrow, qkept);
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
            qkept.print("qkept");

            if(debug){
               auto t3 = tools::get_time();
               std::cout << "----- TIMING for decimation_row: "
                  << tools::get_duration(t3-t0) << " S"
                  << " T(decim/select/formRot)="
                  << tools::get_duration(t1-t0) << ","
                  << tools::get_duration(t2-t1) << ","
                  << tools::get_duration(t3-t2) << " -----"
                  << std::endl;
            }
         } // rank=0
      }

   template <>
      inline void decimation_row(const comb<qkind::qNK,std::complex<double>>& icomb,
            const qbond& qs1,
            const qbond& qs2,
            const bool iftrunc,
            const int dcut,
            const double rdm_svd,
            const int alg_decim,
            const std::vector<stensor2<std::complex<double>>>& wfs2,
            stensor2<std::complex<double>>& rot,
            double& dwt,
            int& deff,
            const std::string fname,
            const bool debug){
         using Tm = std::complex<double>;
         int rank = 0, size = 1;
#ifndef SERIAL
         rank = icomb.world.rank();
         size = icomb.world.size();
#endif   
         auto t0 = tools::get_time();
         const auto qprod = qmerge(qs1, qs2);
         const auto& qrow = qprod.first;
         const auto& dpt = qprod.second;
         assert(qrow == wfs2[0].info.qrow);
         const auto& qcol = wfs2[0].info.qcol;
         int nqr = qrow.size();
         int dim12 = qrow.get_dimAll(); 
         if(debug){ 
            std::cout << "ctns::decimation_row_kr"
               << " iftrunc=" << iftrunc
               << " dcut=" << dcut << " nqr=" << nqr
               << " alg_decim=" << alg_decim
               << " mpisize=" << size;
            if(iftrunc && !fname.empty()) std::cout << " fname=" << fname;
            std::cout << std::endl;
            qrow.print("qsuper");
         }

         // 0. untruncated case
         if(!iftrunc){
            if(rank == 0){
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
               }
               rot = std::move(qt2);
               dwt = 0.0;
               deff = qrow.get_dimAll();
               if(debug) std::cout << "decimation summary: keep all " 
                  << deff << " states" << std::endl;
            } // rank-0
            return;
         }

         // 1. compute reduced basis
         std::map<int,std::pair<std::vector<double>,linalg::matrix<Tm>>> results;
         decimation_genbasis(icomb, qs1, qs2, qrow, qcol, dpt, rdm_svd, alg_decim, wfs2, results);
         auto t1 = tools::get_time();

         if(rank == 0){
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
            linalg::xscal(sig2all.size(), 1.0/sig2sum, sig2all.data());
            //NOTE: in kr case, sig2all only contain partial sigs2, thus no check is applied
            //sig2sum = std::accumulate(sig2all.begin(), sig2all.end(), 0.0);
            //assert(std::abs(sig2sum - 1.0) < 1.e-10); 

            // 2. select important sig2
            std::vector<int> br_kept;
            std::vector<std::pair<qsym,int>> dims;
            decimation_selection(true, qrow, ifmatched, sig2all, idx2sector, dcut, 
                  dwt, deff, br_kept, dims, fname);
            auto t2 = tools::get_time();
            
            // 3. form rot
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
            qkept.print("qkept");

            if(debug){
               auto t3 = tools::get_time();
               std::cout << "----- TIMING for decimation_row: "
                  << tools::get_duration(t3-t0) << " S"
                  << " T(decim/select/formRot)="
                  << tools::get_duration(t1-t0) << ","
                  << tools::get_duration(t2-t1) << ","
                  << tools::get_duration(t3-t2) << " -----"
                  << std::endl;
            }
         } // rank-0
      }

} // ctns

#endif
