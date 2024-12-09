#ifndef SWEEP_INIT_H
#define SWEEP_INIT_H

#include "../core/tools.h"
#include "../core/linalg.h"
#include "ctns_rcanon.h"

namespace ctns{

   // rwfuns_to_cpsi: generate initial guess from RCF for 
   // the initial sweep optimization at p=(1,0) require 
   // configuration: cR0R1R2R3 => L0[C1]R2R3 (L=Id)

   template <typename Qm, typename Tm>
      void sweep_init(comb<Qm,Tm>& icomb, const int nroots, const bool singlet=false){
         int size = 1, rank = 0;
#ifndef SERIAL
         size = icomb.world.size();
         rank = icomb.world.rank();
#endif  
         const auto& rindex = icomb.topo.rindex;
         int rdx1 = rindex.at(std::make_pair(1,0));
         int rdx0 = rindex.at(std::make_pair(0,0));
         if(rank == 0){ 
            auto sym_state = icomb.get_qsym_state();
            std::cout << "\nctns::sweep_init: nroots=" << nroots 
               << " sym_state=" << sym_state
               << " singlet=" << singlet
               << std::endl;
            if(icomb.get_nroots() < nroots){
               std::cout << "dim(psi0)=" << icomb.get_nroots() << " nroots=" << nroots << std::endl;
               tools::exit("error in sweep_init: requested nroots exceed!");
            }
            const auto wf2 = get_boundary_coupling<Qm,Tm>(sym_state, singlet); // env*C[0]
            auto site0new = get_left_bsite<Qm,Tm>(sym_state, singlet); // C[0]R[1] => L[0]C[1] (L[0]=Id) 
            site0new.print("site0new");
            icomb.cpsi.resize(nroots);
            for(int iroot=0; iroot<nroots; iroot++){
               // qt2(1,r): ->-*->-
               auto qt2 = contract_qt2_qt2(wf2,icomb.rwfuns[iroot]);
               // qt2(1,r)*site0(r,r0,n0) = qt3(1,r0,n0)[CRcouple]
               auto qt3 = contract_qt3_qt2("l",icomb.sites[rdx0],qt2);
               // recouple to qt3(1,r0,n0)[LCcouple] => cwf(1*n0,r0)
               auto cwf = qt3.recouple_lc().merge_lc();
               // must be consistent, as merge_lc may change the order
               cwf = cwf.align_qrow(site0new.info.qcol);
               // cwf(n0,r0)*site1(r0,r1,n1) = psi(n0,r1,n1)
               icomb.cpsi[iroot] = contract_qt3_qt2("l",icomb.sites[rdx1],cwf);
            } // iroot
            icomb.sites[rdx0] = std::move(site0new);
            icomb.display_size();
         }
#ifndef SERIAL
         if(size > 1){
            boost::mpi::broadcast(icomb.world, icomb.cpsi, 0);
            mpi_wrapper::broadcast(icomb.world, icomb.sites[rdx0], 0);
         }
#endif
      }

   template <typename Qm, typename Tm>
      void sweep_init_single(comb<Qm,Tm>& icomb, const int iroot, const bool singlet=true){
         int size = 1, rank = 0;
#ifndef SERIAL
         size = icomb.world.size();
         rank = icomb.world.rank();
#endif  
         const auto& rindex = icomb.topo.rindex;
         int rdx1 = rindex.at(std::make_pair(1,0));
         int rdx0 = rindex.at(std::make_pair(0,0));
         if(rank == 0){ 
            auto sym_state = icomb.get_qsym_state();
            std::cout << "\nctns::sweep_init_single: iroot=" << iroot
               << " sym_state=" << sym_state
               << " singlet=" << singlet
               << std::endl;
            const auto& site1 = icomb.sites[rdx1]; // const
            auto& site0 = icomb.sites[rdx0]; // will be updated
            const auto wf2 = get_boundary_coupling<Qm,Tm>(sym_state, singlet); // env*C[0]
            auto site0new = get_left_bsite<Qm,Tm>(sym_state, singlet); // C[0]R[1] => L[0]C[1] (L[0]=Id) 
            icomb.rwfuns[0] = std::move(icomb.rwfuns[iroot]);
            icomb.rwfuns.resize(1); 
            // qt2(1,r): ->-*->-
            auto qt2 = contract_qt2_qt2(wf2,icomb.rwfuns[0]);
            // qt2(1,r)*site0(r,r0,n0) = qt3(1,r0,n0)[CRcouple]
            auto qt3 = contract_qt3_qt2("l",site0,qt2);
            // recouple to qt3(1,r0,n0)[LCcouple] => cwf(1*n0,r0)
            auto cwf = qt3.recouple_lc().merge_lc();
            // must be consistent, as merge_lc may change the order
            cwf = cwf.align_qrow(site0new.info.qcol);
            // cwf(n0,r0)*site1(r0,r1,n1) = psi(n0,r1,n1)
            icomb.cpsi.resize(1);
            icomb.cpsi[0] = contract_qt3_qt2("l",site1,cwf);
            site0new.print("site0new");
            site0 = std::move(site0new);
            icomb.display_size();
         }
#ifndef SERIAL
         if(size > 1){
            boost::mpi::broadcast(icomb.world, icomb.cpsi, 0);
            boost::mpi::broadcast(icomb.world, icomb.rwfuns, 0);
            mpi_wrapper::broadcast(icomb.world, icomb.sites[rdx0], 0);
         }
#endif
      }

} // ctns

#endif
