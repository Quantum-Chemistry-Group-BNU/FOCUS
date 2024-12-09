#ifndef MPS_ES2PROJ_H
#define MPS_ES2PROJ_H
#include "mps.h"

namespace ctns{

   // expectation value of a Hermitian operator
   template <typename Qm1, typename Qm2, typename Tm>
      std::pair<double,double> mps_expect_es2proj(const mps<Qm1,Tm>& imps,
            const mps<Qm2,Tm> & imps_low, 
            const int iroot,
            const int ne,
            const int tm,
            const int ts,
            const double ps,
            const int nsample,
            const integral::two_body<Tm>& int2e,
            const integral::one_body<Tm>& int1e,
            const double& ecore,
            const sci::heatbath_table<Tm>& hbtab,
            const double eps2,
            const bool debug){
         auto t0 = tools::get_time();
         if(debug){
            std::cout << "\nctns::mps_expect_es2proj:" 
               << " Qm1=" << qkind::get_name<Qm1>()
               << " Qm2=" << qkind::get_name<Qm2>()
               << " iroot=" << iroot
               << " ts=" << ts
               << " ps=" << ps
               << " nsample=" << nsample
               << " eps2=" << std::scientific << eps2 
               << std::endl;
         }
         
         // generate quadrature
         std::vector<double> xts, wts;
         fock::gen_s2quad(imps.nphysical, ne, ts/2.0, tm/2.0, xts, wts);
 
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
            // compute <n|P|psi>
            Tm psi2_i = 0.0;
            for(int i=0; i<xts.size(); i++){
               auto rmps = mps_ryrotation(imps_low, xts[i]);
               psi2_i += wts[i]*mps_CIcoeff(rmps,iroot,state);
            }
            double v0i = std::abs(psi2_i);
            Tm eloc = ecore + fock::get_Hii(state,int2e,int1e)*psi2_i/psi_i;
            // singles
            for(int ia=0; ia<nsingles; ia++){
               int ix = ia%no, ax = ia/no;
               int i = olst[ix], a = vlst[ax];
               fock::onstate state1(state);
               state1[i] = 0;
               state1[a] = 1;
               auto pr = fock::get_HijS(state,state1,int2e,int1e);
               // compute <n|P|psi>
               Tm psi2_j = 0.0;
               for(int i=0; i<xts.size(); i++){
                  auto rmps = mps_ryrotation(imps_low, xts[i]);
                  psi2_j += wts[i]*mps_CIcoeff(rmps,iroot,state1);
               }
               eloc += pr.first * psi2_j/psi_i;
            } // ia 
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
                     // compute <n|P|psi>
                     Tm psi2_j = 0.0;
                     for(int i=0; i<xts.size(); i++){
                        auto rmps = mps_ryrotation(imps_low, xts[i]);
                        psi2_j += wts[i]*mps_CIcoeff(rmps,iroot,state2);
                     }
                     eloc += pr.first * psi2_j/psi_i;
                  }
               } // ab
            } // ij
            // accumulate
            double fac = 1.0/(iter+1.0)/ps;
            ene = (ene*iter + std::real(eloc))*fac;
            ene2 = (ene2*iter + std::norm(eloc))*fac; 
            if((iter+1)%noff == 0){
               // Note: <psi|H-E|P><P|H-E|psi> is not <psi|(H-E)^2|psi>,
               // which can be simply seen by taking |psi> as a single determinant!
               // Thus, it is not the variance of the wavefunction.
               std = std::sqrt(std::abs(ene2-ene*ene)/(iter+1.e-10));
               std::cout << " iter=" << iter 
                  << " <HP>/<P>=" << std::defaultfloat << std::setprecision(12) << ene 
                  << " std=" << std::scientific << std::setprecision(3) << std
                  << " range=(" << std::defaultfloat << std::setprecision(12) 
                  << ene-std << "," << ene+std << ")" 
                  << std::endl;
            }
         } // sample
         if(debug){
            auto t1 = tools::get_time();
            tools::timing("ctns::mps_expect_es2proj", t0, t1);
         }
         return std::make_pair(ene,std);
      }

   template <typename Qm, typename Tm>
      void mps_es2proj(const input::schedule& schd){
         int rank = 0, size = 1;
#ifndef SERIAL
         rank = schd.world.rank();
         size = schd.world.size();
#endif
         const bool debug = (rank==0);
         if(debug) std::cout << "\nctns::mps_es2proj" << std::endl;

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
            auto sym = kmps.get_sym_state();
            int ne = sym.ne();
            int tm = sym.tm();
            if(ne%2 != schd.post.twos%2){
               std::cout << "error: inconsistent ne and twos! (ne,twos)=" << ne << "," << schd.post.twos << std::endl;
               exit(1);
            }
            if(qkind::is_qNSz<Qm>()){
               mps<qkind::qN,Tm> kmps_low;
               lowerSym(kmps, kmps_low);
               double ps = mps_expect_s2proj(kmps_low, schd.post.iroot, ne, tm, schd.post.twos);
               auto epair = mps_expect_es2proj(kmps, kmps_low, schd.post.iroot, 
                     ne, tm, schd.post.twos, ps, schd.post.nsample, 
                     int2e, int1e, ecore, hbtab, schd.post.eps2, debug);
            }
         }
      }

} // ctns

#endif
