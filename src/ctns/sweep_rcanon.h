#ifndef SWEEP_RCANON_H
#define SWEEP_RCANON_H

#include "../core/tools.h"
#include "../core/linalg.h"

namespace ctns{

   // rwfuns_to_cpsi: generate initial guess for 
   // the initial sweep optimization at p=(1,0): cRRRR => LCRR (L=Id)
   template <typename Qm, typename Tm>
      void sweep_init(comb<Qm,Tm>& icomb, const int nroots){
         std::cout << "\nctns::sweep_init: nroots=" << nroots << std::endl;
         if(icomb.get_nroots() < nroots){
            std::cout << "dim(psi0)=" << icomb.get_nroots() << " nroots=" << nroots << std::endl;
            tools::exit("error in sweep_init: requested nroots exceed!");
         }
         const auto& rindex = icomb.topo.rindex;
         auto& site0 = icomb.sites[rindex.at(std::make_pair(0,0))];
         auto& site1 = icomb.sites[rindex.at(std::make_pair(1,0))]; // const
         auto sym_state = icomb.get_sym_state();
         icomb.cpsi.resize(nroots);
         for(int iroot=0; iroot<nroots; iroot++){
            // qt2(1,r): ->-*->-
            auto qt2 = icomb.rwfuns[iroot];
            // qt2(1,r)*site0(r,r0,n0) = qt3(1,r0,n0)
            auto qt3 = contract_qt3_qt2("l",site0,qt2);
            // qt3(1,r0,n0) -> cwf(n0,r0)
            //     n0
            //    /|\       ->  n0-<-*->-r0
            // q->-*->-r0            |
            //                       q
            stensor2<Tm> cwf(sym_state, site0.info.qmid, site0.info.qcol, dir_WF2);
            for(int br=0; br<cwf.rows(); br++){
               for(int bc=0; bc<cwf.cols(); bc++){
                  auto blk = cwf(br,bc);
                  if(blk.empty()) continue;
                  const auto blk0 = qt3(0,bc,br);
                  int rdim = cwf.info.qrow.get_dim(br);
                  int cdim = cwf.info.qcol.get_dim(bc);
                  for(int ir=0; ir<rdim; ir++){
                     for(int ic=0; ic<cdim; ic++){
                        blk(ir,ic) = blk0(0,ic,ir);
                     } // ic
                  } // ir
               } // bc
            } // br
            // cwf(n0,r0)*site1(r0,r1,n1) = psi(n0,r1,n1)
            icomb.cpsi[iroot] = contract_qt3_qt2("l",site1,cwf);
         } // iroot
         site0 = get_left_bsite<Tm>(Qm::isym); // C[0]R[1] => L[0]C[1] (L[0]=Id) 
         icomb.display_size();
      }

   // cpsi1_to_rwfuns: generate right canonical form (RCF) for later usage: LCRR => cRRRR
   template <typename Qm, typename Tm>
      void sweep_final(comb<Qm,Tm>& icomb,
            const input::schedule& schd,
            const std::string scratch,
            const int isweep){
         auto rcanon_file = schd.scratch+"/rcanon_isweep"+std::to_string(isweep)+".info";
         std::cout << "\nctns::sweep_final: convert into RCF & save into "
            << rcanon_file << std::endl;
         std::cout << tools::line_separator << std::endl;

         // 1. reorthogonalize {cpsi} in case there is truncation in the last sweep
         // such that they are not orthonormal any more, which can happens for
         // small bond dimension. 
         size_t ndim = icomb.cpsi[0].size();
         int nroots = icomb.cpsi.size();
         std::vector<Tm> v0(ndim*nroots);
         for(int i=0; i<nroots; i++){
            icomb.cpsi[i].to_array(&v0[ndim*i]);
         }
         int nindp = linalg::get_ortho_basis(ndim, nroots, v0.data()); // reorthogonalization
         assert(nindp == nroots);
         for(int i=0; i<nroots; i++){
            icomb.cpsi[i].from_array(&v0[ndim*i]);
         }
         v0.clear();
         const auto& pdx0 = icomb.topo.rindex.at(std::make_pair(0,0));
         const auto& pdx1 = icomb.topo.rindex.at(std::make_pair(1,0));

         // 2. compute C0 & R1 from cpsi via decimation: LCRR => CRRR
         {
            std::cout << "Convert [LC]RR => [CR]RR:" << std::endl;
            const auto& wfinfo = icomb.cpsi[0].info;
            stensor2<Tm> rot;
            std::vector<stensor2<Tm>> wfs2(nroots);
            for(int i=0; i<nroots; i++){
               auto wf2 = icomb.cpsi[i].merge_cr().T();
               wfs2[i] = std::move(wf2);
            }
            const int dcut = 4*nroots; // psi[l,n,r,i] => U[l,i,a]sigma[a]Vt[a,n,r]
            double dwt; 
            int deff;
            std::string fname = scratch+"/decimation"
               + "_isweep"+std::to_string(isweep) + "_C0R1.txt";
            const bool iftrunc = true;
            const int alg_decim = 0;
            decimation_row(icomb, wfinfo.qmid, wfinfo.qcol, 
                  iftrunc, dcut, schd.ctns.rdm_svd, alg_decim, 
                  wfs2, rot, dwt, deff, fname, 
                  schd.ctns.verbose>0);
            rot = rot.T();
            icomb.sites[pdx1] = rot.split_cr(wfinfo.qmid, wfinfo.qcol);
            // compute C0 
            for(int i=0; i<nroots; i++){
               auto cwf = icomb.cpsi[i].merge_cr().dot(rot.H()); // <-W[l,alpha]->
               auto psi = contract_qt3_qt2("r",icomb.sites[pdx0],cwf.T()); // A0(1,n,r)
               icomb.cpsi[i] = std::move(psi);
            }
         }

         // 3. compute rwfuns & R0 from C0: CRRR => cRRRR
         {
            std::cout << "\nConvert [C]RRR => [cR]RRR:" << std::endl;
            const auto& wfinfo = icomb.cpsi[0].info;
            stensor2<Tm> rot;
            std::vector<stensor2<Tm>> wfs2(nroots);
            for(int i=0; i<nroots; i++){
               auto wf2 = icomb.cpsi[i].merge_cr().T();
               wfs2[i] = std::move(wf2);
            }
            const int dcut = nroots; // psi[1,n,r,i] => U[1,i,a]sigma[a]Vt[a,n,r]
            double dwt;
            int deff;
            std::string fname = scratch+"/decimation"
               + "_isweep"+std::to_string(isweep) + "_cR0.txt";
            const bool iftrunc = true;
            const int alg_decim = 0;
            decimation_row(icomb, wfinfo.qmid, wfinfo.qcol,
                  iftrunc, dcut, schd.ctns.rdm_svd, alg_decim,
                  wfs2, rot, dwt, deff, fname,
                  schd.ctns.verbose>0);
            rot = rot.T();
            icomb.sites[pdx0] = rot.split_cr(wfinfo.qmid, wfinfo.qcol);
            // form rwfuns(iroot,irbas)
            const auto& sym_state = wfinfo.sym;
            qbond qrow({{sym_state, 1}});
            const auto& qcol = rot.info.qrow;
            icomb.rwfuns.resize(nroots);
            for(int iroot=0; iroot<nroots; iroot++){
               auto cwf = icomb.cpsi[iroot].merge_cr().dot(rot.H()); // <-W[1,alpha]->
               // change the carrier of sym_state from center to left
               stensor2<Tm> rwfun(qsym(Qm::isym), qrow, qcol, dir_RWF);
               for(int ic=0; ic<qcol.get_dim(0); ic++){
                  rwfun(0,0)(0,ic) = cwf(0,0)(0,ic);
               }
               icomb.rwfuns[iroot] = std::move(rwfun);
            } // iroot
         }

         // 4. save & check
         ctns::rcanon_save(icomb, rcanon_file);
         ctns::rcanon_check(icomb, schd.ctns.thresh_ortho);

         std::cout << "..... end of isweep = " << isweep << " .....\n" << std::endl;
      }

} // ctns

#endif
