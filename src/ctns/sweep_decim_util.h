#ifndef SWEEP_DECIM_UTIL_H
#define SWEEP_DECIM_UTIL_H

#include <numeric>
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef GPU
#include "../../gpu/gpu_linalg.h"
#endif

namespace ctns{

#ifdef GPU
   const int thrdgpu_eig = 150;
   extern const int thrdgpu_eig;

   const int thrdgpu_svd = 350*350;
   extern const int thrdgpu_svd;
#endif

   const bool debug_decimation = false;
   extern const bool debug_decimation;

   template <typename Tm>
      using decim_item = std::pair<std::vector<double>,linalg::matrix<Tm>>; // (sigs,Umat)
   template <typename Tm>
      using decim_map = std::map<int,decim_item<Tm>>; // br->(sigs,Umat)

   using brbctype = std::tuple<int,int,std::vector<int>>; // {br,cdim,{bc}}

   // determine whether MPI is used
   inline bool decim_serial(const int alg_decim, 
         const int size){
      return alg_decim==0 || alg_decim==2 || (alg_decim>0 && size==1);
   }

   template <typename Qm, typename Tm>
      void decimation_divide(const comb<Qm,Tm>& icomb,
            const qbond& qrow,
            const qbond& qcol,
            const int alg_decim,
            const std::vector<qtensor2<Qm::ifabelian,Tm>>& wfs2,
            std::vector<std::vector<brbctype>>& local_brbc){
         const bool debug = false;
         int rank = 0, size = 1;
#ifndef SERIAL
         rank = icomb.world.rank();
         size = icomb.world.size();
#endif   
         int nqr = qrow.size();

         // 1. preprocess wfs2 to find contributing blocks
         std::vector<brbctype> brbc(nqr);
         std::vector<double> decim_cost(nqr);
         std::vector<int> index;
         int nqr_eff = 0;
         if(rank == 0){
            for(int br=0; br<nqr; br++){
               const auto& qr = qrow.get_sym(br);
               int rdim = qrow.get_dim(br);
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
               brbc[nqr_eff] = std::make_tuple(br,cdim,matched_bc);
               int m = std::max(rdim,cdim);
               int n = std::min(rdim,cdim);
               // estimator for FLOP of SVD following
               // https://en.wikipedia.org/wiki/Singular_value_decomposition
               decim_cost[nqr_eff] = double(n)*n*m;
               if(debug){
                  std::cout << "nqr_eff=" << nqr_eff
                     << " rdim,cdim=" << rdim << "," << cdim 
                     << " cost=" << decim_cost[nqr_eff]
                     << std::endl;
               }
               nqr_eff += 1;
            } // br
            brbc.resize(nqr_eff);
            decim_cost.resize(nqr_eff);
            index = tools::sort_index(decim_cost, 1);
         } // rank-0

         // 2. from {brbc,decim_cost,index} to contruct local_brbc
         if(decim_serial(alg_decim,size)){
            if(rank == 0) local_brbc[rank] = std::move(brbc);
#ifndef SERIAL
         }else{
            // partition the task set using a greedy algorithm to ensure load balance 
            if(rank == 0){
               std::vector<double> local_cost(size,0);
               for(int i=0; i<nqr_eff; i++){
                  int idx = index[i];
                  auto ptr = std::min_element(local_cost.begin(), local_cost.end());
                  int pos = std::distance(local_cost.begin(), ptr);
                  if(debug){
                     std::cout << "br=" << std::get<0>(brbc[idx])
                        << " cost=" << decim_cost[idx] << " pos=" << pos 
                        << std::endl;
                     tools::print_vector(local_cost, "local_cost");
                  }
                  local_cost[pos] += decim_cost[idx];
                  local_brbc[pos].push_back(brbc[idx]);
               } // i
               if(debug){
                  for(int i=0; i<size; i++){
                     std::cout << "rank=" << rank << " i=" << i 
                        << " size(local_brbc)=" << local_brbc[i].size() 
                        << " cost=" << local_cost[i] << std::endl;
                     for(int j=0; j<local_brbc[i].size(); j++){
                        std::cout << "j=" << j << " br=" << std::get<0>(local_brbc[i][j]) << std::endl;
                     }
                  }
               }
            } // rank-0
            // broadcast the partition into each processes
            boost::mpi::broadcast(icomb.world, local_brbc, 0);
#endif
         } // alg_decim
      }

   template <typename Qm, typename Tm>
      void decimation_collect(const comb<Qm,Tm>& icomb,
            const int alg_decim,
            const std::vector<std::vector<brbctype>>& local_brbc,
            const std::vector<std::pair<int,decim_item<Tm>>>& local_results,
            decim_map<Tm>& results,
            const int size,
            const int rank){
         if(alg_decim==0 || alg_decim==2 || (alg_decim>0 && size==1)){
            // reconstruct results
            if(rank == 0){
               for(int ibr=0; ibr<local_results.size(); ibr++){
                  int br = local_results[ibr].first;
                  results[br] = std::move(local_results[ibr].second);
               }
            }
#ifndef SERIAL
         }else{
            //--- algorithm 1: gather: there is problem in boost::mpi ---
            /*
               std::vector<std::vector<std::pair<int,decim_item<Tm>>>> full_results;
               boost::mpi::gather(icomb.world, local_results, full_results, 0);
            // reconstruct results
            if(rank == 0){
            assert(full_results.size() == size);
            for(int i=0; i<full_results.size(); i++){
            for(int j=0; j<full_results[i].size(); j++){
            int br = full_results[i][j].first;
            if(br == -1) continue;
            results[br] = std::move(full_results[i][j].second);
            }
            }
            }
            */
            //--- algorithm 2: isend & irecv: should work for rdim<16000 --- 
            if(rank != 0){
               // send results to rank-0 
               std::vector<boost::mpi::request> request(local_results.size());
               for(int i=0; i<local_results.size(); i++){
                  const int br = local_results[i].first;
                  const auto& item = local_results[i].second; 
                  request[i] = icomb.world.isend(0, br, item); // send results
               }
               boost::mpi::wait_all(&request[0], &request[0]+local_results.size());
            }else{
               // save local results first
               for(int ibr=0; ibr<local_results.size(); ibr++){
                  int br = local_results[ibr].first;
                  results[br] = std::move(local_results[ibr].second);
               }
               // recv results from other ranks
               int nrecv = 0;
               for(int i=1; i<size; i++){
                  for(int j=0; j<local_brbc[i].size(); j++){
                     nrecv += 1;
                  }
               }
               std::vector<boost::mpi::request> request(nrecv);
               int idx = 0;
               for(int i=1; i<size; i++){
                  for(int j=0; j<local_brbc[i].size(); j++){
                     int br = std::get<0>(local_brbc[i][j]);
                     request[idx] = icomb.world.irecv(i, br, results[br]); // recieve results
                     idx += 1;      
                  }
               }
               boost::mpi::wait_all(&request[0], &request[0]+nrecv);
            } // rank
#endif
         } // alg_decim
      }

} // ctns

#endif
