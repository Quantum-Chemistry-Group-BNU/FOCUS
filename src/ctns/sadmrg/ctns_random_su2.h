#ifndef CTNS_RANDOM_SU2_H
#define CTNS_RANDOM_SU2_H

/*
   Algorithms for CTNS:

   3. rcanon_random: random sampling from distribution p(n)=|<n|CTNS>|^2
*/

#include "../../core/onspace.h"
#include "../../core/analysis.h"
#include "../ctns_comb.h"

namespace ctns{

   // --- Non-Abelian MPS ---

   // Sampling SA-MPS to get {|csf>,coeff(csf)=<csf|Psi[i]>} 
   // In case that CTNS is unnormalized, p(csf) is also unnormalized. 
   template <typename Qm, typename Tm, std::enable_if_t<!Qm::ifabelian,int> = 0>
      std::pair<fock::csfstate,Tm> rcanon_random(const comb<Qm,Tm>& icomb, 
            const int iroot,
            const bool debug=false){
         // only correct for MPS, because csf is linearly coupled.
         assert(icomb.topo.ifmps);
         if(debug) std::cout << "\nctns::rcanon_random iroot=" << iroot << std::endl; 
         fock::csfstate state(icomb.get_nphysical());
         // initialize boundary wf for i-th state
         auto wf = icomb.rwfuns[iroot];
         const auto& nodes = icomb.topo.nodes; 
         const auto& rindex = icomb.topo.rindex;
         auto sym = icomb.get_qsym_state();
         int ne = sym.ne();
         int ts = sym.ts();
         // loop from left to right
         for(int i=0; i<icomb.topo.nbackbone; i++){
            const auto& node = nodes[i][0];
            assert(node.lindex == i); 
            int tp = node.type;
            if(tp == 0 || tp == 1){
               const auto& site = icomb.sites[rindex.at(std::make_pair(i,0))];
               
               // 0. setup probability for (bc,bm)
               auto qt3 = contract_qt3_qt2("l",site,wf);
               const auto& qrow = qt3.info.qrow;
               const auto& qcol = qt3.info.qcol;
               const auto& qmid = qt3.info.qmid;
               assert(qrow.size() == 1);
               int size = qmid.size()*qcol.size();
               std::vector<double> weights(size,0.0);
               std::vector<std::pair<int,int>> indices(size);
               for(int bm=0; bm<qmid.size(); bm++){
                  for(int bc=0; bc<qcol.size(); bc++){
                     int iaddr = bm*qcol.size()+bc;
                     indices[iaddr] = std::make_pair(bc,bm);
                     auto qr = qrow.get_sym(0);
                     auto qc = qcol.get_sym(bc);
                     auto qm = qmid.get_sym(bm);
                     auto blk3 = qt3.get_rcf_symblk(qr,qc,qm);
                     if(blk3.empty()) continue;
                     weights[iaddr] = std::pow(linalg::xnrm2(blk3.size(), blk3.data()),2);
                  }
               }

               // 1. sample
               std::discrete_distribution<> dist(weights.begin(), weights.end());
               int idx = dist(tools::generator);
               auto key = indices[idx];
               int bc = key.first;
               int bm = key.second;
               //std::cout << "\nsite i=" << i << " bc=" << bc << " bm=" << bm << std::endl;
               //qt3.print("qt3",2);
               
               // 2. construct wf
               qbond qleft({{qcol.get_sym(bc),1}}); // because dr*dm=1
               wf.init(qsym(3,0,0),qleft,qcol,{1-std::get<1>(qt3.info.dir),std::get<1>(qt3.info.dir)});
               auto blk3 = qt3(0,bc,bm,qrow.get_sym(0).ts());
               auto blk2 = wf(0,bc);
               //wf.print("wf",2);
               assert(!blk3.empty() && !blk2.empty() && blk3.size()==blk2.size());
               linalg::xcopy(blk3.size(), blk3.data(), blk2.data());
               
               // 3. setup state
               auto sym = qcol.get_sym(bc);
               int dne = ne - sym.ne();
               int dts = ts - sym.ts();
               state.setocc(i, dne, dts);
               ne = sym.ne();
               ts = sym.ts();

            } // tp
         }
         // finally wf should be the corresponding CI coefficients
         assert(wf.rows() == 1 && wf.cols() == 1); 
         Tm coeff0 = wf(0,0)(0,0);
         if(debug){
            auto coeff1 = rcanon_CIcoeff(icomb, state)[iroot];
            std::cout << " state=" << state 
               << " coeff0=" << coeff0 
               << " coeff1=" << coeff1 
               << " diff=" << coeff0-coeff1 
               << std::endl;
            assert(std::abs(coeff0-coeff1)<1.e-10);
         }
         return std::make_pair(state,coeff0);
      }

} // ctns

#endif
