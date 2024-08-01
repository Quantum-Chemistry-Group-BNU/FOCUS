#ifndef SWEEP_DECIM_NKR_H
#define SWEEP_DECIM_NKR_H

#ifdef _OPENMP
#include <omp.h>
#endif

#include <numeric>

namespace ctns{

   const bool debug_decimation = false;
   extern const bool debug_decimation;

   const bool debug_auxbasis = false; // whether include random basis for no matching col case
   extern const bool debug_auxbasis;

   template <typename Tm>
      using decim_item = std::pair<std::vector<double>,linalg::matrix<Tm>>; // (sigs,Umat)
   template <typename Tm>
      using decim_map = std::map<int,decim_item<Tm>>; // br->(sigs,Umat)

   template <typename Qm, typename Tm>
      void decimation_divide(const comb<Qm,Tm>& icomb,
            const qbond& qrow,
            const qbond& qcol,
            const int alg_decim,
            const std::vector<stensor2<Tm>>& wfs2,
            std::vector<std::vector<std::pair<int,int>>>& local_brbc){
         const bool debug = false;
         int rank = 0, size = 1;
#ifndef SERIAL
         rank = icomb.world.rank();
         size = icomb.world.size();
#endif   
         int nqr = qrow.size();
         
         // 1. preprocess wfs2 to find contributing blocks
         std::vector<std::pair<int,int>> brbc(nqr);
         std::vector<double> decim_cost(nqr);
         std::vector<int> index;
         int nqr_eff = 0;
         if(rank == 0){
            for(int br=0; br<nqr; br++){
               int rdim = qrow.get_dim(br);
               // find matched column
               int matched = 0;
               for(int bc=0; bc<qcol.size(); bc++){
                  if(wfs2[0](br,bc).empty()) continue;
                  matched += 1;
                  if(matched > 1) tools::exit("multiple matched qc is not supported!");
                  brbc[nqr_eff] = std::make_pair(br,bc);
                  int cdim = qcol.get_dim(bc);
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
               } // bc
               // no matched column  
               if(debug_auxbasis && matched == 0){
                  brbc[nqr_eff] = std::make_pair(br,-1);
                  decim_cost[nqr_eff] = double(rdim)*rdim*rdim;
                  nqr_eff += 1;
               }
            } // br
            brbc.resize(nqr_eff);
            decim_cost.resize(nqr_eff);
            index = tools::sort_index(decim_cost, 1);
         } // rank-0
         
         // 2. from {brbc,decim_cost,index} to contruct local_brbc
         if(alg_decim==0 || (alg_decim>0 && size==1)){
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
                     std::cout << "br,bc=" << brbc[idx].first << "," << brbc[idx].second
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
                        std::cout << "j=" << j << " br=" << local_brbc[i][j].first << std::endl;
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
            const std::vector<std::vector<std::pair<int,int>>>& local_brbc,
            const std::vector<std::pair<int,decim_item<Tm>>>& local_results,
            decim_map<Tm>& results,
            const int size,
            const int rank){
         if(alg_decim==0 || (alg_decim>0 && size==1)){
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
                  request[i] = icomb.world.isend(0, br, item);
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
                     int br = local_brbc[i][j].first;
                     request[idx] = icomb.world.irecv(i, br, results[br]);
                     idx += 1;      
                  }
               }
               boost::mpi::wait_all(&request[0], &request[0]+nrecv);
            } // rank
#endif
         } // alg_decim
      }

   template <typename Qm, typename Tm>
      void decimation_genbasis(const comb<Qm,Tm>& icomb,
            const qbond& qs1,
            const qbond& qs2,
            const qbond& qrow,
            const qbond& qcol,
            const qdpt& dpt,
            const double rdm_svd,
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
         std::vector<std::vector<std::pair<int,int>>> local_brbc(size);
         decimation_divide(icomb, qrow, qcol, alg_decim, wfs2, local_brbc);
         auto t1 = tools::get_time();
         if(debug and rank == 0){
            std::cout << "timing for divide=" << tools::get_duration(t1-t0) << std::endl;
         }

         // start decimation
         int local_size = local_brbc[rank].size();
         std::vector<std::pair<int,decim_item<Tm>>> local_results(local_size);
         int nroots = wfs2.size();
         if(alg_decim == 0){
            // MPI + OpenMP + serial SVD
#ifdef _OPENMP
            #pragma omp parallel for schedule(dynamic)
#endif
            for(int ibr=0; ibr<local_size; ibr++){
               int br = local_brbc[rank][ibr].first;
               int bc = local_brbc[rank][ibr].second;
               std::vector<double> sigs2;
               linalg::matrix<Tm> U;
               if(debug_auxbasis && bc == -1){
                  // generate a random unitary
                  int rdim = qrow.get_dim(br);
                  std::vector<linalg::matrix<Tm>> blks(1);
                  blks[0] = linalg::random_matrix<Tm>(rdim,rdim);
                  kramers::get_renorm_states_nkr(blks, sigs2, U, rdm_svd, debug_decimation);
                  std::transform(sigs2.begin(), sigs2.end(), sigs2.begin(),
                        [](const double& x){ return 1.e-14*x; });
                  local_results[ibr] = std::make_pair(br,std::make_pair(sigs2, U));
               }else{
                  // compute renormalized basis
                  std::vector<linalg::matrix<Tm>> blks(nroots);
                  for(int iroot=0; iroot<nroots; iroot++){
                     blks[iroot] = wfs2[iroot](br,bc).to_matrix().T();
                  }
                  kramers::get_renorm_states_nkr(blks, sigs2, U, rdm_svd, debug_decimation);
                  local_results[ibr] = std::make_pair(br,std::make_pair(sigs2, U));
               }
            } // br
         }else{
            // MPI + parallel SVD
            for(int ibr=0; ibr<local_size; ibr++){
               int br = local_brbc[rank][ibr].first;
               int bc = local_brbc[rank][ibr].second;
               std::vector<double> sigs2;
               linalg::matrix<Tm> U;
               if(debug_auxbasis && bc == -1){
                  // generate a random unitary
                  int rdim = qrow.get_dim(br);
                  std::vector<linalg::matrix<Tm>> blks(1);
                  blks[0] = linalg::random_matrix<Tm>(rdim,rdim);
                  kramers::get_renorm_states_nkr(blks, sigs2, U, rdm_svd, debug_decimation);
                  std::transform(sigs2.begin(), sigs2.end(), sigs2.begin(),
                        [](const double& x){ return 1.e-14*x; });
                  local_results[ibr] = std::make_pair(br,std::make_pair(sigs2, U));
               }else{
                  // compute renormalized basis
                  std::vector<linalg::matrix<Tm>> blks(nroots);
                  for(int iroot=0; iroot<nroots; iroot++){
                     blks[iroot] = wfs2[iroot](br,bc).to_matrix().T();
                  }
                  kramers::get_renorm_states_nkr(blks, sigs2, U, rdm_svd, debug_decimation);
                  local_results[ibr] = std::make_pair(br,std::make_pair(sigs2, U));
               }
            } // br
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
            std::cout << "----- TMING FOR decimation_genbasis(nkr): " 
               << tools::get_duration(t3-t0) << " S"
               << " T(divide/compute/collect)="
               << tools::get_duration(t1-t0) << ","
               << tools::get_duration(t2-t1) << "," 
               << tools::get_duration(t3-t2) << " -----" 
               << std::endl;
         }
      }
   
} // ctns

#endif
