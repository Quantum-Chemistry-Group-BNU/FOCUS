#ifndef SWEEP_DECIM_SU2_H
#define SWEEP_DECIM_SU2_H

#include "../sweep_decim_nkr.h"

namespace ctns{

   //
   // NOTE: this part has not been revised (Only OpenMP is supported)
   //
   template <typename Qm, typename Tm>
      void decimation_genbasis(const comb<Qm,Tm>& icomb,
            const qbond& qs1,
            const qbond& qs2,
            const qbond& qrow,
            const qbond& qcol,
            const qdpt& dpt,
            const double rdm_svd,
            const int alg_decim,
            const std::vector<stensor2su2<Tm>>& wfs2,
            decim_map<Tm>& results,
            const bool debug){
         int rank = 0, size = 1;
#ifndef SERIAL
         rank = icomb.world.rank();
         size = icomb.world.size();
#endif   
         auto t0 = tools::get_time();
         
         int nroots = wfs2.size();
         int nqr = qrow.size();
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
            // 1. search for matched block 
            std::vector<int> matched_bc;
            int dim = 0; 
            for(int bc=0; bc<qcol.size(); bc++){
               if(wfs2[0](br,bc).empty()) continue;
               const auto& qc = qcol.get_sym(bc);     
               const int cdim = qcol.get_dim(bc);
               if(debug_decimation) std::cout << " find matched qc =" << qc << std::endl;
               matched_bc.push_back(bc);
               dim += cdim;
            } // qc
            if(dim == 0) continue;
            // 2. merge matrix into large blocks
            std::vector<linalg::matrix<Tm>> blks(nroots);
            // compute KRS-adapted renormalized basis
            for(int iroot=0; iroot<nroots; iroot++){
               linalg::matrix<Tm> clr(rdim,dim);
               int off = 0;
               for(const auto& bc : matched_bc){
                  const auto blk = wfs2[iroot](br,bc);
                  linalg::xcopy(blk.size(), blk.data(), clr.data()+off);
                  off += blk.size();
               }
               blks[iroot] = clr.T(); // to be used in get_renorm_states_nkr 
            }
            // 3. decimation
            std::vector<double> sigs2;
            linalg::matrix<Tm> U;
            kramers::get_renorm_states_nkr(blks, sigs2, U, rdm_svd, debug_decimation);
#ifdef _OPENMP
#pragma omp critical
#endif
            results[br] = std::make_pair(sigs2, U);
         } // br

         if(debug and rank == 0){
            auto t1 = tools::get_time();
            std::cout << "----- TMING FOR decimation_genbasis(su2): "
               << tools::get_duration(t1-t0) << " S -----"
               << std::endl;
         }
      }

} // ctns

#endif
