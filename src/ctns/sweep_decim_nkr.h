#ifndef SWEEP_DECIM_NKR_H
#define SWEEP_DECIM_NKR_H

#include "sweep_decim_util.h"

namespace ctns{

   // nonsu2 case: qNSz
   template <typename Qm, typename Tm>
      void decimation_genbasis(const comb<Qm,Tm>& icomb,
            const qbond& qs1,
            const qbond& qs2,
            const qbond& qrow,
            const qbond& qcol,
            const qdpt& dpt,
            const double rdm_svd,
            const int svd_iop,
            const int alg_decim,
            const std::vector<stensor2<Tm>>& wfs2,
            decim_map<Tm>& results,
            const bool debug){
         int rank = 0, size = 1;
#ifndef SERIAL
         rank = icomb.world.rank();
         size = icomb.world.size();
#endif   
         // determine local_brbc for each rank
         auto t0 = tools::get_time();
         std::vector<std::vector<brbctype>> local_brbc(size);
         decimation_divide(icomb, qrow, qcol, alg_decim, wfs2, local_brbc);
         auto t1 = tools::get_time();
         if(debug and rank == 0){
            std::cout << "timing for divide=" << tools::get_duration(t1-t0) << std::endl;
         }

         // start decimation
         int local_size = local_brbc[rank].size();
         std::vector<std::pair<int,decim_item<Tm>>> local_results(local_size);
         double tcpu = 0.0, tgpu = 0.0;
         int nroots = wfs2.size();
         if(alg_decim == 0 || alg_decim == 1){

            // OpenMP or MPI + OpenMP
            auto ti = tools::get_time();
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
            for(int ibr=0; ibr<local_size; ibr++){
               const auto& brbc = local_brbc[rank][ibr];
               const auto& br = std::get<0>(brbc); 
               const auto& matched_bc = std::get<2>(brbc);
               assert(matched_bc.size() == 1);
               int bc = matched_bc[0];
               // compute renormalized basis
               std::vector<linalg::matrix<Tm>> blks(nroots);
               for(int iroot=0; iroot<nroots; iroot++){
                  blks[iroot] = wfs2[iroot](br,bc).to_matrix().T();
               }
               // decimation
               std::vector<double> sigs2;
               linalg::matrix<Tm> U;
               kramers::get_renorm_states_nkr(blks, sigs2, U, rdm_svd, svd_iop, debug_decimation);
#ifdef _OPENMP
#pragma omp critical
#endif
               local_results[ibr] = std::make_pair(br,std::make_pair(sigs2, U));
            } // br
            auto tf = tools::get_time();
            tcpu = tools::get_duration(tf-ti);

#ifdef GPU 
         }else if(alg_decim == 2 || alg_decim == 3){

            // ZL@2024/12/06 support decimation using cusolver
            std::vector<std::vector<int>> ibr_cpugpu(2);
            for(int ibr=0; ibr<local_size; ibr++){
               const auto& brbc = local_brbc[rank][ibr];
               const auto& br = std::get<0>(brbc); 
               const auto& cdim = std::get<1>(brbc);
               const auto& matched_bc = std::get<2>(brbc);
               assert(matched_bc.size() == 1);
               int rdim = qrow.get_dim(br);
               // partition the task
               if(rdim == cdim){
                  ibr_cpugpu[rdim>thrdgpu_eig].push_back(ibr);
               }else{
                  ibr_cpugpu[rdim*cdim>thrdgpu_svd].push_back(ibr);
               } 
            } // ibr
            if(rank == 0){
               tools::print_vector(ibr_cpugpu[0],"ibr_cpu");
               tools::print_vector(ibr_cpugpu[1],"ibr_gpu");
            }

            // decimation on cpu and gpu
            auto ta = tools::get_time(); 
            // cpu
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
            for(int i=0; i<ibr_cpugpu[0].size(); i++){
               int ibr = ibr_cpugpu[0][i];
               const auto& brbc = local_brbc[rank][ibr];
               const auto& br = std::get<0>(brbc);
               const auto& bc = std::get<2>(brbc)[0];
               // compute renormalized basis
               std::vector<linalg::matrix<Tm>> blks(nroots);
               for(int iroot=0; iroot<nroots; iroot++){
                  blks[iroot] = wfs2[iroot](br,bc).to_matrix().T();
               }
               // decimation
               std::vector<double> sigs2;
               linalg::matrix<Tm> U;
               kramers::get_renorm_states_nkr(blks, sigs2, U, rdm_svd, svd_iop, debug_decimation);
#ifdef _OPENMP
#pragma omp critical
#endif
               local_results[ibr] = std::make_pair(br,std::make_pair(sigs2, U));
            } // i
            auto tb = tools::get_time();
            tcpu = tools::get_duration(tb-ta);

            // gpu
            for(int i=0; i<ibr_cpugpu[1].size(); i++){
               int ibr = ibr_cpugpu[1][i];
               const auto& brbc = local_brbc[rank][ibr];
               const auto& br = std::get<0>(brbc);
               const auto& bc = std::get<2>(brbc)[0];
               // compute renormalized basis
               std::vector<linalg::matrix<Tm>> blks(nroots);
               for(int iroot=0; iroot<nroots; iroot++){
                  blks[iroot] = wfs2[iroot](br,bc).to_matrix().T();
               }
               // decimation
               std::vector<double> sigs2;
               linalg::matrix<Tm> U;
               linalg::get_renorm_states_nkr_gpu(blks, sigs2, U, rdm_svd, svd_iop, debug_decimation);
               local_results[ibr] = std::make_pair(br,std::make_pair(sigs2, U));
            } // i
            auto tc = tools::get_time();
            tgpu = tools::get_duration(tc-tb);

#endif
         }else{
            std::cout << "error: no such option for decimation_genbasis(nkr): alg_decim=" << alg_decim << std::endl;
            exit(1);
         } // alg_decim
         auto t2 = tools::get_time();
         if(debug and rank == 0){
            std::cout << "timing for compute=" << tools::get_duration(t2-t1) << std::endl;
         }

         // collect local_results to results in rank-0
         decimation_collect(icomb, alg_decim, local_brbc, local_results, results, size, rank);
         auto t3 = tools::get_time();
         if(debug and rank == 0){
            std::cout << "timing for collect=" << tools::get_duration(t3-t2) << std::endl;
         }

         if(debug and rank == 0){
            double tcompute = tools::get_duration(t2-t1);
            double trest = tcompute - tcpu - tgpu; 
            std::cout << "----- TMING FOR decimation_genbasis(nkr): " 
               << tools::get_duration(t3-t0) << " S"
               << " T(divide/compute[cpu/gpu/rest]/collect)="
               << tools::get_duration(t1-t0) << ","
               << tcpu << "," << tgpu << "," << trest << ","
               << tools::get_duration(t3-t2) << " -----" 
               << std::endl;
         }
      }

} // ctns

#endif
