#ifndef SWEEP_DECIM_SU2_H
#define SWEEP_DECIM_SU2_H

namespace ctns{

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
            const std::vector<stensor2su2<Tm>>& wfs2,
            stensor2su2<Tm>& rot,
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
         const auto qprod = qmerge_su2(qs1, qs2);
         const auto& qrow = qprod.first;
         const auto& dpt = qprod.second;
         assert(qrow == wfs2[0].info.qrow);
         const auto& qcol = wfs2[0].info.qcol;
         int nroots = wfs2.size();
         int nqr = qrow.size();
         if(debug){
            std::cout << "ctns::decimation_row(su2)"
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
               stensor2su2<Tm> qt2(qsym(isym), qrow, qrow); // identity matrix
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
/*
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
*/
      }

} // ctns

#endif
