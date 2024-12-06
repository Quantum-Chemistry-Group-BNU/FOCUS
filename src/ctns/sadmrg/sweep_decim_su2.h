#ifndef SWEEP_DECIM_SU2_H
#define SWEEP_DECIM_SU2_H

#include "../sweep_decim_nkr.h"
#ifdef GPU
#include "../../gpu/gpu_linalg.h"
#endif

namespace ctns{
                  
   const int thrdgpu_eig = 150;
   extern const int thrdgpu_eig;

   const int thrdgpu_svd = 350*350;
   extern const int thrdgpu_svd;

   // SU2 case
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

         double tcpu=0.0, tgpu=0.0;
         int nroots = wfs2.size();
         int nqr = qrow.size();
         if(alg_decim == 0){

            // raw: openmpi version without MPI
            if(rank == 0){
               auto ti = tools::get_time();
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
               auto tf = tools::get_time();
               tcpu = tools::get_duration(tf-ti);
            }

#ifdef GPU
         }else if(alg_decim == 2){

            // mixed cpu-gpu version using cusolver
            if(rank == 0){
               std::vector<std::pair<int,std::vector<int>>> brbc(nqr);
               std::vector<std::pair<int,int>> drdc(nqr);
               std::vector<std::vector<int>> idx_cpugpu(2);
               int nqr_eff = 0;
               for(int br=0; br<nqr; br++){
                  const auto& qr = qrow.get_sym(br);
                  const int rdim = qrow.get_dim(br);
                  if(debug_decimation){ 
                     if(br == 0) std::cout << "decimation for each symmetry sector:" << std::endl;
                     std::cout << ">br=" << br << " qr=" << qr << " rdim=" << rdim << std::endl;
                  }
                  // 1. search for matched block 
                  std::vector<int> matched_bc;
                  int cdim = 0; 
                  for(int bc=0; bc<qcol.size(); bc++){
                     if(wfs2[0](br,bc).empty()) continue;
                     const auto& qc = qcol.get_sym(bc);     
                     const int dim = qcol.get_dim(bc);
                     if(debug_decimation) std::cout << " find matched qc =" << qc << std::endl;
                     matched_bc.push_back(bc);
                     cdim += dim;
                  } // bc
                  if(cdim == 0) continue;
                  brbc[nqr_eff] = std::make_pair(br,matched_bc);
                  drdc[nqr_eff] = std::make_pair(rdim,cdim);
                  // partition the task
                  if(rdim == cdim){
                     idx_cpugpu[rdim>thrdgpu_eig].push_back(nqr_eff);
                  }else{
                     idx_cpugpu[rdim*cdim>thrdgpu_svd].push_back(nqr_eff);
                  }
                  nqr_eff += 1;
               } // br
               brbc.resize(nqr_eff);
               drdc.resize(nqr_eff);
               tools::print_vector(idx_cpugpu[0],"idx_cpu");
               tools::print_vector(idx_cpugpu[1],"idx_gpu");

               // decimation on cpu and gpu
               auto ta = tools::get_time(); 
               // cpu
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
               for(int i=0; i<idx_cpugpu[0].size(); i++){
                  int idx = idx_cpugpu[0][i];
                  const auto& br = brbc[idx].first;
                  const auto& matched_bc = brbc[idx].second;
                  const auto& rdim = drdc[idx].first;
                  const auto& cdim = drdc[idx].second;
                  // 2. merge matrix into large blocks
                  std::vector<linalg::matrix<Tm>> blks(nroots);
                  // compute KRS-adapted renormalized basis
                  for(int iroot=0; iroot<nroots; iroot++){
                     linalg::matrix<Tm> clr(rdim,cdim);
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
               } // i
               auto tb = tools::get_time(); 
               tcpu = tools::get_duration(tb-ta);

               // gpu
               for(int i=0; i<idx_cpugpu[1].size(); i++){
                  int idx = idx_cpugpu[1][i];
                  const auto& br = brbc[idx].first;
                  const auto& matched_bc = brbc[idx].second;
                  const auto& rdim = drdc[idx].first;
                  const auto& cdim = drdc[idx].second;
                  // 2. merge matrix into large blocks
                  std::vector<linalg::matrix<Tm>> blks(nroots);
                  // compute KRS-adapted renormalized basis
                  for(int iroot=0; iroot<nroots; iroot++){
                     linalg::matrix<Tm> clr(rdim,cdim);
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
                  linalg::get_renorm_states_nkr_gpu(blks, sigs2, U, rdm_svd, debug_decimation);
                  results[br] = std::make_pair(sigs2, U);
               } // i
               auto tc = tools::get_time();
               tgpu = tools::get_duration(tc-tb);
            } // rank-0

#endif 
         }else{
            std::cout << "error: no such option for decimation_genbasis(su2): alg_decim=" << alg_decim << std::endl;
            exit(1);  
         } // alg_decim

         if(debug and rank == 0){
            auto t1 = tools::get_time();
            double total = tools::get_duration(t1-t0);
            double trest = total - tcpu - tgpu;
            std::cout << "----- TMING FOR decimation_genbasis(su2): " << total << " S"
               << " T(cpu/gpu/rest)=" << tcpu << "," << tgpu << "," << trest << " -----"
               << std::endl;
         }
      }

} // ctns

#endif
