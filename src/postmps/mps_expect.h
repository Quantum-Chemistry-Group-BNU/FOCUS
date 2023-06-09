#ifndef MPS_EXPECT_H
#define MPS_EXPECT_H

#include "../core/integral_util.h"

namespace ctns{

   // main for sweep optimizations for CTNS
   template <typename Km>
      void mps_expect_H(const mps<Km>& imps, // initial comb wavefunction
            const int iroot,
            const int nsample,
            const integral::two_body<typename Km::dtype>& int2e,
            const integral::one_body<typename Km::dtype>& int1e,
            const double ecore,
            sci::heatbath_table<typename Km::dtype> hbtab,
            const bool eps2,
            const bool ifdouble,
            const bool ifsingle,
            const bool debug){
         using Tm = typename Km::dtype;
         auto t0 = tools::get_time();
         if(debug){
            std::cout << "\nctns::mps_expect_H nsample=" << nsample << std::endl;
         }

         int noff = nsample/10;
         int k = imps.nphysical*2;
         int no = imps.get_sym_state().ne();
         int nv = k - no;
         int nsingles = no*nv;
         std::vector<int> olst(no), vlst(nv);
         // generate samples 
         double ene = 0.0, ene2 = 0.0, std = 0.0;
         for(int iter=0; iter<nsample; iter++){
            auto pr = mps_random(imps,iroot);
            auto state = pr.first;
            Tm psi_i = pr.second;
            // given state |i>, loop over <i|H|j> psi(j)/psi(i)
            state.get_olst(olst.data());
            state.get_vlst(vlst.data());
            double v0i = std::abs(psi_i);
            Tm eloc = ecore + fock::get_Hii(state,int2e,int1e);
            // singles
            if(ifsingle){
               for(int ia=0; ia<nsingles; ia++){
                  int ix = ia%no, ax = ia/no;
                  int i = olst[ix], a = vlst[ax];
                  fock::onstate state1(state);
                  state1[i] = 0;
                  state1[a] = 1;
                  auto pr = fock::get_HijS(state,state1,int2e,int1e);
                  Tm psi_j = mps_CIcoeff(imps, iroot, state1);
                  eloc += pr.first * psi_j/psi_i;
               } // ia 
            }
            // doubles
            if(ifdouble){
               for(int ijdx=0; ijdx<no*(no-1)/2; ijdx++){
                  auto pr = tools::inverse_pair0(ijdx);
                  int i = olst[pr.first], j = olst[pr.second];
                  int ij = tools::canonical_pair0(i,j);
                  for(const auto& p : hbtab.eri4.at(ij)){
                     if(p.first*v0i < eps2) break; // avoid searching all doubles
                     auto ab = tools::inverse_pair0(p.second);
                     int a = ab.first, b = ab.second;
                     if(state[a]==0 && state[b]==0){ // if true double excitations
                        fock::onstate state2(state);
                        state2[i] = 0;
                        state2[j] = 0;
                        state2[a] = 1;
                        state2[b] = 1;
                        auto pr = fock::get_HijD(state,state2,int2e,int1e);
                        Tm psi_j = mps_CIcoeff(imps, iroot, state2);
                        eloc += pr.first * psi_j/psi_i;
                     }
                  } // ab
               } // ij
            }
            // accumulate
            double fac = 1.0/(iter+1.0);
            ene = (ene*iter + std::real(eloc))*fac;
            ene2 = (ene2*iter + std::norm(eloc))*fac; 
            if((iter+1)%noff == 0){
               // Note: <psi|H-E|P><P|H-E|psi> is not <psi|(H-E)^2|psi>,
               // which can be simply seen by taking |psi> as a single determinant!
               // Thus, it is not the variance of the wavefunction.
               std = std::sqrt(std::abs(ene2-ene*ene)/(iter+1.e-10));
               std::cout << " iter=" << iter 
                  << " <H>=" << std::defaultfloat << std::setprecision(12) << ene 
                  << " std=" << std::scientific << std::setprecision(3) << std
                  << " range=(" << std::defaultfloat << std::setprecision(12) 
                  << ene-std << "," << ene+std << ")" 
                  << std::endl;
            }
         } // sample
         if(debug){
            auto t1 = tools::get_time();
            tools::timing("ctns::mps_expect_H", t0, t1);
         }
      }

   template <typename Km>
      void mps_expect(const input::schedule& schd){
         using Tm = typename Km::dtype;
         int rank = 0, size = 1;
#ifndef SERIAL
         rank = schd.world.rank();
         size = schd.world.size();
#endif
         const bool debug = (rank==0);
         if(debug) std::cout << "\nctns::mps_expect" << std::endl;

         // read integral for H
         integral::two_body<Tm> int2e;
         integral::one_body<Tm> int1e;
         double ecore;
         if(rank == 0) integral::load(int2e, int1e, ecore, schd.integral_file);
#ifndef SERIAL
         if(size > 1){
            boost::mpi::broadcast(schd.world, ecore, 0);
            boost::mpi::broadcast(schd.world, int1e, 0);
            mpi_wrapper::broadcast(schd.world, int2e, 0);
         }
#endif
         sci::heatbath_table<Tm> hbtab(int2e, int1e);

         // generate integral for N
         integral::two_body<Tm> int2e_n(int1e.sorb);
         integral::one_body<Tm> int1e_n(int1e.sorb);
         double ecore_n = 0.0;
         //integral::generate_N(int1e_n);
#ifndef SERIAL
         if(size > 1){
            boost::mpi::broadcast(schd.world, ecore_n, 0);
            boost::mpi::broadcast(schd.world, int1e_n, 0);
            mpi_wrapper::broadcast(schd.world, int2e_n, 0);
         }
#endif
         sci::heatbath_table<Tm> hbtab_n(int2e_n, int1e_n);

         // generate integral for S2
         integral::two_body<Tm> int2e_s2(int1e.sorb);
         integral::one_body<Tm> int1e_s2(int1e.sorb);
         double ecore_s2 = 0.0;
         //integral::generate_S2(int2e_s2, int1e_s2);
#ifndef SERIAL
         if(size > 1){
            boost::mpi::broadcast(schd.world, ecore_s2, 0);
            boost::mpi::broadcast(schd.world, int1e_s2, 0);
            mpi_wrapper::broadcast(schd.world, int2e_s2, 0);
         }
#endif
         sci::heatbath_table<Tm> hbtab_s2(int2e_s2, int1e_s2);

         topology topo;
         topo.read(schd.postmps.topology_file);
         //topo.print();
         int nket = schd.postmps.ket.size();
         for(int j=0; j<nket; j++){
            std::cout << "\n### jket=" << j << " ###" << std::endl;
            mps<Km> kmps;
            auto kmps_file = schd.scratch+"/rcanon_isweep"+std::to_string(schd.postmps.ket[j])+".info"; 
            kmps.nphysical = topo.nphysical;
            kmps.image2 = topo.image2;
            kmps.load(kmps_file);
            // compute expectation value via sampling
            // <H>
            mps_expect_H(kmps, schd.postmps.iroot, schd.postmps.nsample,
                  int2e, int1e, ecore, hbtab, schd.postmps.eps2, 
                  true, true, debug);
            // <N>
            mps_expect_H(kmps, schd.postmps.iroot, schd.postmps.nsample,
                  int2e_n, int1e_n, ecore_n, hbtab_n, schd.postmps.eps2, 
                  false, true, debug);
            /*
            // <S2>
            mps_expect_H(kmps, schd.postmps.iroot, schd.postmps.nsample,
                  int2e_s2, int1e_s2, ecore_s2, hbtab_s2, schd.postmps.eps2, 
                  true, true, debug);
            */
         }
      }

} // ctns

#endif
