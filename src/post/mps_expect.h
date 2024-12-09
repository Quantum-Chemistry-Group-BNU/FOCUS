#ifndef MPS_EXPECT_H
#define MPS_EXPECT_H
#include "mps.h"

namespace ctns{

   // expectation value of a Hermitian operator
   template <typename Qm, typename Tm>
      std::pair<double,double> mps_expect_op(const mps<Qm,Tm>& imps, 
            const int iroot,
            const std::string opname,
            const int nsample,
            const integral::two_body<Tm>& int2e,
            const integral::one_body<Tm>& int1e,
            const double& ecore,
            const sci::heatbath_table<Tm>& hbtab,
            const double eps2,
            const bool debug){
         auto t0 = tools::get_time();
         if(debug){
            std::cout << "\nctns::mps_expect_op iroot=" << iroot
               << " opname=" << opname 
               << " nsample=" << nsample
               << " eps2=" << std::scientific << eps2 
               << std::endl;
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
            auto t0 = tools::get_time();
            auto pr = mps_random(imps,iroot);
            auto state = pr.first;
            Tm psi_i = pr.second;
            // given state |i>, loop over <i|H|j> psi(j)/psi(i)
            state.get_olst(olst.data());
            state.get_vlst(vlst.data());
            double v0i = std::abs(psi_i);
            Tm eloc = ecore + fock::get_Hii(state,int2e,int1e);
            auto t1 = tools::get_time();
            std::cout << "iter=" << iter << " t1=" << tools::get_duration(t1-t0) << std::endl;
            // singles
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
            auto t2 = tools::get_time();
            std::cout << "iter=" << iter << " t2=" << tools::get_duration(t2-t1) 
               << " nsingles=" << nsingles << " t2av=" << tools::get_duration(t2-t1)/nsingles 
               << std::endl;
            // doubles
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
            auto t3 = tools::get_time();
            std::cout << "iter=" << iter << " t3=" << tools::get_duration(t3-t2) << std::endl;
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
                  << " <O>=" << std::defaultfloat << std::setprecision(12) << ene 
                  << " std=" << std::scientific << std::setprecision(3) << std
                  << " range=(" << std::defaultfloat << std::setprecision(12) 
                  << ene-std << "," << ene+std << ")" 
                  << std::endl;
            }
         } // sample
         if(debug){
            auto t1 = tools::get_time();
            tools::timing("ctns::mps_expect_op", t0, t1);
         }
         return std::make_pair(ene,std);
      }

   // expectation value of a Hermitian operator
   template <typename Qm, typename Tm>
      std::pair<double,double> mps_expect_op2(const mps<Qm,Tm>& imps, 
            const int iroot,
            const std::string opname,
            const int nsample,
            const integral::two_body<Tm>& int2e,
            const integral::one_body<Tm>& int1e,
            const double& ecore,
            const sci::heatbath_table<Tm>& hbtab,
            const double eps2,
            const bool debug){
         auto t0 = tools::get_time();
         if(debug){
            std::cout << "\nctns::mps_expect_op iroot=" << iroot
               << " opname=" << opname 
               << " nsample=" << nsample
               << " eps2=" << std::scientific << eps2 
               << std::endl;
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
            auto t0x = tools::get_time();
            auto pr = mps_random(imps,iroot);
            auto state = pr.first;
            Tm psi_i = pr.second;
            // given state |i>, loop over <i|H|j> psi(j)/psi(i)
            state.get_olst(olst.data());
            state.get_vlst(vlst.data());
            double v0i = std::abs(psi_i);
            Tm eloc = ecore + fock::get_Hii(state,int2e,int1e);
            auto t1x = tools::get_time();
            std::cout << "iter=" << iter << " t1=" << tools::get_duration(t1x-t0x) << std::endl;
            std::vector<std::pair<fock::onstate,Tm>> sdspace;
            // singles
            for(int ia=0; ia<nsingles; ia++){
               int ix = ia%no, ax = ia/no;
               int i = olst[ix], a = vlst[ax];
               fock::onstate state1(state);
               state1[i] = 0;
               state1[a] = 1;
               auto pr = fock::get_HijS(state,state1,int2e,int1e);
               if(abs(pr.first)>eps2) sdspace.emplace_back(state1, pr.first);
            } // ia 
            auto t2x = tools::get_time();
            std::cout << "iter=" << iter << " t2=" << tools::get_duration(t2x-t1x) 
               << " sdsize=" << sdspace.size() << " t2av=" << tools::get_duration(t2x-t1x)/nsingles 
               << std::endl;
            // doubles
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
                     if(abs(pr.first)>eps2) sdspace.emplace_back(state2, pr.first);
                  }
               } // ab
            } // ij
            auto t3x = tools::get_time();
            size_t sdsize = sdspace.size();
            std::cout << "iter=" << iter << " t3=" << tools::get_duration(t3x-t2x) 
               << " sdsize=" << sdsize << std::endl;
            std::vector<Tm> elocs(sdsize);
#ifdef _OPENMP
            #pragma omp parallel for schedule(static)
#endif
            for(int i=0; i<sdsize; i++){
               const auto& pr = sdspace[i];
               const auto& sdstate = pr.first;
               const auto& Hnm = pr.second;
               Tm psi_j = mps_CIcoeff(imps, iroot, sdstate);;
               elocs[i] = Hnm * psi_j/psi_i;
            }
            eloc += std::accumulate(elocs.begin(), elocs.end(), Tm(0.0));
            auto t4x = tools::get_time();
            std::cout << "iter=" << iter << " t4=" << tools::get_duration(t4x-t3x) 
               << " t4av=" << tools::get_duration(t4x-t3x)/sdsize
               << std::endl;
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
                  << " <O>=" << std::defaultfloat << std::setprecision(12) << ene 
                  << " std=" << std::scientific << std::setprecision(3) << std
                  << " range=(" << std::defaultfloat << std::setprecision(12) 
                  << ene-std << "," << ene+std << ")" 
                  << std::endl;
            }
         } // sample
         if(debug){
            auto t1 = tools::get_time();
            tools::timing("ctns::mps_expect_op", t0, t1);
         }
         return std::make_pair(ene,std);
      }

   template <typename Qm, typename Tm>
      void mps_expect(const input::schedule& schd){
         int rank = 0, size = 1;
#ifndef SERIAL
         rank = schd.world.rank();
         size = schd.world.size();
#endif
         const bool debug = (rank==0);
         if(debug) std::cout << "\nctns::mps_expect" << std::endl;

         // read integral for O
         integral::two_body<Tm> int2e;
         integral::one_body<Tm> int1e;
         double ecore;
         if(rank == 0) integral::load(int2e, int1e, ecore, schd.post.integral_file);
#ifndef SERIAL
         if(size > 1){
            boost::mpi::broadcast(schd.world, ecore, 0);
            mpi_wrapper::broadcast(schd.world, int1e, 0);
            mpi_wrapper::broadcast(schd.world, int2e, 0);
         }
#endif
         sci::heatbath_table<Tm> hbtab(int2e, int1e);
  
         //------------------------
         // Use functions for comb for debug 
         const bool debug_vmc = false;
         if(debug_vmc){
            comb<Qm,Tm> icomb;
            icomb.topo.read(schd.post.topology_file);
            auto rcanon_file = schd.scratch+"/rcanon_isweep"+std::to_string(schd.post.ket[0]);
            rcanon_load(icomb, rcanon_file);
            auto scratch = schd.scratch+"/sweep";
            io::create_scratch(scratch, (rank == 0));
            auto Oij = get_Hmat(icomb, int2e, int1e, ecore, schd, scratch);
            if(rank == 0) Oij.print("Oij",8);
            //vmc_estimate(icomb, int2e, int1e, ecore, schd, scratch);
         }
         //------------------------

         topology topo;
         topo.read(schd.post.topology_file);
         //topo.print();
         int nket = schd.post.ket.size();
         for(int j=0; j<nket; j++){
            std::cout << "\n### jket=" << j << " ###" << std::endl;
            mps<Qm,Tm> kmps;
            auto kmps_file = schd.scratch+"/rcanon_isweep"+std::to_string(schd.post.ket[j]); 
            kmps.nphysical = topo.nphysical;
            kmps.image2 = topo.image2;
            kmps.load(kmps_file);
            // compute expectation value via sampling
            // <O>
            if(schd.post.opname=="s2"){
               double sz = kmps.get_sym_state().tm()*0.5; 
               ecore = sz + sz*sz;
            }
            auto epair = mps_expect_op2(kmps, schd.post.iroot, 
                  schd.post.opname, schd.post.nsample, 
                  int2e, int1e, ecore, hbtab, schd.post.eps2, debug);
         }
      }

} // ctns

#endif
