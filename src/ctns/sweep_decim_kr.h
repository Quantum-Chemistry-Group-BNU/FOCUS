#ifndef SWEEP_DECIM_KR_H
#define SWEEP_DECIM_KR_H

#include "sweep_decim_nkr.h"

namespace ctns{

   // NOTE: this part has not been revised (Only OpenMP is supported)
   template <>
      inline void decimation_genbasis(const comb<qkind::qNK,std::complex<double>>& icomb,
            const qbond& qs1,
            const qbond& qs2,
            const qbond& qrow,
            const qbond& qcol,
            const qdpt& dpt,
            const double rdm_svd,
            const int alg_decim,
            const std::vector<stensor2<std::complex<double>>>& wfs2,
            decim_map<std::complex<double>>& results){
         using Tm = std::complex<double>;
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
               kramers::get_renorm_states_kr(qr, phases, blks, sigs2, U, rdm_svd, debug_decimation);
               // convert back to the original product basis
               U = U.reorder_row(pos_new,1);
            } // qc
#ifdef _OPENMP
#pragma omp critical
#endif
            results[br] = std::make_pair(sigs2, U);
         } // br
         
         if(rank == 0){
            auto t1 = tools::get_time();
            std::cout << "----- TMING FOR decimation_genbasis(kr): "
               << tools::get_duration(t1-t0) << " S -----"
               << std::endl;
         }
      }

} // ctns

#endif
