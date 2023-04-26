#ifndef SWEEP_DECIM_H
#define SWEEP_DECIM_H

#include "sweep_decim_util.h"

namespace ctns{

   // generate renormalized basis from wfs2[row,col] for row
   // if dcut=-1, no truncation is performed except for sig2 < thresh_sig2
   template <typename Km>
      void decimation_row(const comb<Km>& icomb,
            const qbond& qs1,
            const qbond& qs2,
            const bool iftrunc,
            const int dcut,
            const double rdm_svd,
            const int alg_decim,
            const std::vector<stensor2<typename Km::dtype>>& wfs2,
            stensor2<typename Km::dtype>& rot,
            double& dwt,
            int& deff,
            const std::string fname,
            const bool debug){
         using Tm = typename Km::dtype;
         int rank = 0, size = 1, maxthreads = 1;
#ifndef SERIAL
         rank = icomb.world.rank();
         size = icomb.world.size();
#endif   
#ifdef _OPENMP
         maxthreads = omp_get_max_threads();
#endif
         auto t0 = tools::get_time();
         const auto qprod = qmerge(qs1, qs2);
         const auto& qrow = qprod.first;
         const auto& dpt = qprod.second;
         assert(qrow == wfs2[0].info.qrow);
         const auto& qcol = wfs2[0].info.qcol;
         int nroots = wfs2.size();
         int nqr = qrow.size();
         if(debug){
            std::cout << "ctns::decimation_row_nkr"
               << " dcut=" << dcut << " nqr=" << nqr
               << " alg_decim=" << alg_decim
               << " mpisize=" << size
               << " maxthreads=" << maxthreads;
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
            qkept.print("qkept");

            if(debug){
               auto t2 = tools::get_time();
               std::cout << "----- TIMING for decimation_row: "
                  << tools::get_duration(t2-t0) << " S"
                  << " T(decim/formRot)="
                  << tools::get_duration(t1-t0) << ","
                  << tools::get_duration(t2-t1) << " -----"
                  << std::endl;
            }
         } // rank=0
      }

   template <>
      inline void decimation_row(const comb<qkind::cNK>& icomb,
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
         int rank = 0, size = 1, maxthreads = 1;
#ifndef SERIAL
         rank = icomb.world.rank();
         size = icomb.world.size();
#endif   
#ifdef _OPENMP
         maxthreads = omp_get_max_threads();
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
               << " dcut=" << dcut << " nqr=" << nqr
               << " alg_decim=" << alg_decim
               << " maxthreads=" << maxthreads 
               << std::endl;
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
            qkept.print("qkept");

            if(debug){
               auto t2 = tools::get_time();
               std::cout << "----- TIMING for decimation_row: "
                  << tools::get_duration(t2-t0) << " S"
                  << " T(decim/formRot)="
                  << tools::get_duration(t1-t0) << ","
                  << tools::get_duration(t2-t1) << " -----"
                  << std::endl;
            }
         } // rank-0
      }

} // ctns

#endif
