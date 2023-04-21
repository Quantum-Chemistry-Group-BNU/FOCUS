#ifndef SWEEP_DECIM_UTIL_H
#define SWEEP_DECIM_UTIL_H

#ifdef _OPENMP
#include <omp.h>
#endif

#include <numeric>

namespace ctns{

   const double thresh_sig2 = 1.e-14;
   extern const double thresh_sig2;

   const bool debug_decimation = false;
   extern const bool debug_decimation;

   template <typename Tm>
   using decim_item = std::pair<std::vector<double>,linalg::matrix<Tm>>;
   template <typename Tm>
   using decim_map = std::map<int,decim_item<Tm>>;

   template <typename Km>
      void decimation_genbasis(const comb<Km>& icomb,
            const qbond& qs1,
            const qbond& qs2,
            const qbond& qrow,
            const qbond& qcol,
            const qdpt& dpt,
            const double rdm_svd,
            const int alg_decim,
            const std::vector<stensor2<typename Km::dtype>>& wfs2,
            decim_map<typename Km::dtype>& results){
         auto t0 = tools::get_time();
         using Tm = typename Km::dtype;
         int rank = 0, size = 1;
#ifndef SERIAL
         rank = icomb.world.rank();
         size = icomb.world.size();
#endif   
         int nroots = wfs2.size();
         int nqr = qrow.size();
         std::vector<int> local_br;
         if(alg_decim == 0){
            if(rank == 0){
               for(int br=rank; br<nqr; br+=1){
                  local_br.push_back(br);
               }
            }
         }else{
//            for(int br=rank; br<nqr; br+=size){
//               local_br.push_back(br);
//            }
            std::vector<std::vector<int>> local_brs(size);
            if(rank == 0){
               std::vector<double> local_cost(size,0);
               for(int br=0; br<nqr; br++){
                  const auto& qr = qrow.get_sym(br);
                  const int rdim = qrow.get_dim(br);

                  int matched=0, bc1=0, cdim1=0;
                  for(int bc=0; bc<qcol.size(); bc++){
                     if(wfs2[0](br,bc).empty()) continue;
                     const auto& qc = qcol.get_sym(bc);     
                     if(debug_decimation) std::cout << " find matched qc =" << qc << std::endl;
                     matched += 1;
                     if(matched > 1) tools::exit("multiple matched qc is not supported!");
                     bc1 = bc;
                     cdim1 = qcol.get_dim(bc);
                  } // qc
      
                  if(matched == 1){ 
                     int m = std::max(rdim,cdim1);
                     int n = std::min(rdim,cdim1);
                     // https://en.wikipedia.org/wiki/Singular_value_decomposition
                     double cost = double(n)*n*m;
                     auto ptr = std::min_element(local_cost.begin(), local_cost.end());
                     int pos = std::distance(local_cost.begin(), ptr);
                     std::cout << "br,rdim,cdim=" << br << "," << rdim << "," << cdim1 
                        << " m,n=" << m << "," << n
                        << " cost=" << cost 
                        << " pos=" << pos 
                        << std::endl;
                     tools::print_vector(local_cost, "local_cost");
                     local_cost[pos] += cost;
                     local_brs[pos].push_back(br);
                  }
               }

                for(int i=0; i<size; i++){
                   std::cout << "rank=" << rank << " i=" << i << " sz=" << local_brs[i].size() 
                      << " cost=" << local_cost[i] << std::endl;
                   tools::print_vector(local_brs[i],"vec");
                }
            }

            boost::mpi::scatter(icomb.world, local_brs, local_br, 0);
         
         }
         int local_size = local_br.size();
         std::vector<std::pair<int,decim_item<Tm>>> local_results(local_size);
         std::vector<double> timing1(local_size);
         std::vector<double> timing2(local_size);
         std::vector<double> timing3(local_size);

         std::cout << "size=" << size << " rank=" << rank
           << " local_size=" << local_size << std::endl;
         icomb.world.barrier();

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
         for(int ibr=0; ibr<local_size; ibr++){
            int br = local_br[ibr];
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
               // compute renormalized basis
               auto t0x = tools::get_time();
               std::vector<linalg::matrix<Tm>> blks(nroots);
               for(int iroot=0; iroot<nroots; iroot++){
                  blks[iroot] = wfs2[iroot](br,bc).to_matrix().T();
               }
               auto t1x = tools::get_time();
               kramers::get_renorm_states_nkr(blks, sigs2, U, rdm_svd, debug_decimation);
               auto t2x = tools::get_time();
               timing1[ibr] = tools::get_duration(t1x-t0x);
               timing2[ibr] = tools::get_duration(t2x-t1x);
            } // qc
            auto t0x = tools::get_time();
            local_results[ibr] = std::make_pair(br,std::make_pair(sigs2, U));
            auto t1x = tools::get_time();
            timing3[ibr] = tools::get_duration(t1x-t0x);
         } // br
         auto t1 = tools::get_time();

         double time1 =0;
         double time2 =0;
         double time3 =0;
         for(int i=0; i<local_size; i++){
            time1 += timing1[i];
            time2 += timing2[i];
            time3 += timing3[i];
         }
         std::cout << "rank=" << rank << " t1,t2,t3="
            << time1 << " " << time2 << " " << time3
            << " total=" << (time1+time2+time3)
            << std::endl;

         // collection
         if(alg_decim == 0){
            // reconstruct results
            if(rank == 0){
               for(int ibr=0; ibr<local_size; ibr++){
                  int br = local_results[ibr].first;
                  results[br] = std::move(local_results[ibr].second);
               }
            }
         }else{
            std::vector<std::vector<std::pair<int,decim_item<Tm>>>> full_results;
            boost::mpi::gather(icomb.world, local_results, full_results, 0);
            // reconstruct results
            if(rank == 0){
               assert(full_results.size() == size);
               for(int i=0; i<full_results.size(); i++){
                  for(int j=0; j<full_results[i].size(); j++){
                     int br = full_results[i][j].first;
                     results[br] = std::move(full_results[i][j].second);
                  }
               }
            }
         }
         auto t2 = tools::get_time();
         if(rank == 0){
            std::cout << "----- TMING FOR decimation_genbasis: " 
               << tools::get_duration(t2-t0) << " S"
               << " T(compute/gather)="
               << tools::get_duration(t1-t0) << ","
               << tools::get_duration(t2-t1) << " -----" 
               << std::endl;
         }
      }

   template <>
   inline void decimation_genbasis(const comb<qkind::cNK>& icomb,
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
      }

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
         if(dim != 0){
            br_kept.push_back(br);
            dims.emplace_back(qr,dim);
            accum += wts;    
            deff += dim;
            // save information
            std::cout << " iqr=" << iqr << " qr=" << qr
               << " dim[full,kept]=" << dim0 << "," << dim 
               << " wts=" << wts << " accum=" << accum << " deff=" << deff 
               << std::endl;
         }else{
            // additional: kept at least one state per sector
            // ZL@20220517 disable such choice, since it will create many sector with dim=1 
            /*
               if(!ifmatched[br]) continue;
               br_kept.push_back(br);
               int dmin = (ifkr && qr.parity()==1)? 2 : 1;
               dims.emplace_back(qr,dmin);
               deff += dmin;
            // save information
            std::cout << " iqr=" << iqr << " qr=" << qr
            << " dim[full,kept]=" << dim0 << "," << dmin 
            << " wts=" << wts << " accum=" << accum << " deff=" << deff
            << " (additional)" << std::endl;
            */
         }
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

} // ctns

#endif
