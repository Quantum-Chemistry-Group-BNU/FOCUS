#ifndef SWEEP_FINAL_H
#define SWEEP_FINAL_H

#include "../core/tools.h"
#include "../core/linalg.h"

namespace ctns{

   template <typename Qm, typename Tm>
      void sweep_final_LCR2CRR(comb<Qm,Tm>& icomb,
            const double rdm_svd,
            const std::string fname,
            const bool debug){
         if(debug) std::cout << "\nctns::sweep_final_LCR2CRR: convert [LC]R => [CR]R" << std::endl;
         const auto& pdx0 = icomb.topo.rindex.at(std::make_pair(0,0));
         const auto& pdx1 = icomb.topo.rindex.at(std::make_pair(1,0));
         int nroots = icomb.cpsi.size();
         // 1. setup 
         const auto& wfinfo = icomb.cpsi[0].info;
         qtensor2<Qm::ifabelian,Tm> rot;
         std::vector<qtensor2<Qm::ifabelian,Tm>> wfs2(nroots);
         for(int i=0; i<nroots; i++){
            auto wf2 = icomb.cpsi[i].recouple_cr().merge_cr().P();
            wfs2[i] = std::move(wf2);
         }
         const int dcut = 4*nroots; // psi[l,n,r,i] => U[l,i,a]sigma[a]Vt[a,n,r]
         double dwt; 
         int deff;
         const bool iftrunc = true;
         const int alg_decim = 0;
         std::vector<double> sigs2full;
         // 2. decimation
         decimation_row(icomb, wfinfo.qmid, wfinfo.qcol, 
               iftrunc, dcut, rdm_svd, alg_decim, 
               wfs2, sigs2full, rot, dwt, deff, fname, 
               debug);
         rot = rot.P();
         // 3. save site1
         icomb.sites[pdx1] = rot.split_cr(wfinfo.qmid, wfinfo.qcol);
         // 4. compute C0 [following twodot_guess_psi] 
         for(int i=0; i<nroots; i++){
            auto cwf = wfs2[i].P().dot(rot.H()); // <-W[l,alpha]->
            auto psi = contract_qt3_qt2("r",icomb.sites[pdx0],cwf.P()); // A0(1,n,r)
            icomb.cpsi[i] = std::move(psi);
         }
      }

   template <typename Qm, typename Tm>
      void sweep_final_CR2cRR(comb<Qm,Tm>& icomb,
            const double rdm_svd,
            const std::string fname,
            const bool debug){
         if(debug) std::cout << "\nctns::sweep_final_CR2cRR: convert [C]R => [cR]R" << std::endl;
         const auto& pdx0 = icomb.topo.rindex.at(std::make_pair(0,0));
         const auto& pdx1 = icomb.topo.rindex.at(std::make_pair(1,0));
         int nroots = icomb.cpsi.size();
         // 1. setup 
         const auto& wfinfo = icomb.cpsi[0].info;
         qtensor2<Qm::ifabelian,Tm> rot;
         std::vector<qtensor2<Qm::ifabelian,Tm>> wfs2(nroots);
         for(int i=0; i<nroots; i++){
            auto wf2 = icomb.cpsi[i].recouple_cr().merge_cr().P();
            wfs2[i] = std::move(wf2);
         }
         const int dcut = nroots; // psi[1,n,r,i] => U[1,i,a]sigma[a]Vt[a,n,r]
         double dwt;
         int deff;
         const bool iftrunc = true;
         const int alg_decim = 0;
         std::vector<double> sigs2full;
         // 2. decimation
         decimation_row(icomb, wfinfo.qmid, wfinfo.qcol,
               iftrunc, dcut, rdm_svd, alg_decim,
               wfs2, sigs2full, rot, dwt, deff, fname,
               debug);
         rot = rot.P();
         // 3. save site0
         icomb.sites[pdx0] = rot.split_cr(wfinfo.qmid, wfinfo.qcol);
         // 4. form rwfuns(iroot,irbas)
         const auto& sym_state = icomb.get_qsym_state(); // not wfinfo.sym in singlet embedding (wfinfo.sym=0) 
         qbond qrow({{sym_state, 1}});
         const auto& qcol = rot.info.qrow;
         icomb.rwfuns.resize(nroots);
         for(int i=0; i<nroots; i++){
            auto cwf = wfs2[i].P().dot(rot.H()); // <-W[1,alpha]->
            // change the carrier of sym_state from center to left
            qtensor2<Qm::ifabelian,Tm> rwfun(qsym(Qm::isym), qrow, qcol, dir_RWF);
            assert(cwf.size() == rwfun.size());
            linalg::xcopy(cwf.size(), cwf.data(), rwfun.data());
            icomb.rwfuns[i] = std::move(rwfun);
         } // iroot
      }

   // cpsi1_to_rwfuns: generate right canonical form (RCF) for later usage: LCRR => cRRRR
   template <typename Qm, typename Tm>
      void sweep_final(comb<Qm,Tm>& icomb,
            const input::schedule& schd,
            const std::string scratch,
            const int isweep,
            const std::string rcfprefix){
         int size = 1, rank = 0;
#ifndef SERIAL
         size = icomb.world.size();
         rank = icomb.world.rank();
#endif  

         // only perform canonicalization at rank-0
         if(rank == 0){ 
            std::cout << "\nctns::sweep_final: convert into RCF & save into ";
            
            // 0. rcf file to be saved on disk
            auto rcanon_file = schd.scratch+"/"+rcfprefix+"rcanon_isweep"+std::to_string(isweep);
            if(!Qm::ifabelian) rcanon_file += "_su2";
            std::cout << rcanon_file << std::endl;
            std::cout << tools::line_separator << std::endl;
   
            // 1. compute C0 & R1 from cpsi via decimation: LCRR => CRRR
            std::string fname;
            // reorthogonalize {cpsi} in case there is truncation in the last sweep
            // such that they are not orthonormal any more, which can happens for
            // small bond dimension.
            icomb.orthonormalize_cpsi();
            fname = scratch+"/decimation_isweep"+std::to_string(isweep)+"_C0R1.txt";
            sweep_final_LCR2CRR(icomb, schd.ctns.rdm_svd, fname, schd.ctns.verbose>0);
   
            // 2. compute rwfuns & R0 from C0: CRRR => cRRRR
            icomb.orthonormalize_cpsi();
            fname = scratch+"/decimation_isweep"+std::to_string(isweep)+"_cR0.txt";
            sweep_final_CR2cRR(icomb, schd.ctns.rdm_svd, fname, schd.ctns.verbose>0);
   
            // 3. save & check
            rcanon_save(icomb, rcanon_file);
            rcanon_check(icomb, schd.ctns.thresh_ortho);
	    icomb.display_size();
         }

         // however, the result needs to be broadcast, in case icomb will be used later
#ifndef SERIAL
         if(size > 1){
            const auto& pdx0 = icomb.topo.rindex.at(std::make_pair(0,0));
            const auto& pdx1 = icomb.topo.rindex.at(std::make_pair(1,0));
            mpi_wrapper::broadcast(icomb.world, icomb.sites[pdx1], 0);
            mpi_wrapper::broadcast(icomb.world, icomb.sites[pdx0], 0);
            boost::mpi::broadcast(icomb.world, icomb.rwfuns, 0);
         } 
#endif
      }

} // ctns

#endif
