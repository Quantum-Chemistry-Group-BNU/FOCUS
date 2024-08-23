#ifndef CTNS_RCANON_H
#define CTNS_RCANON_H

namespace ctns{

   // --<--*-->--
   //     /|\
   // boundary_coupling: |tot>=|vaccum>*|physical>
   template <typename Qm, typename Tm>
      qtensor2<Qm::ifabelian,Tm> get_boundary_coupling(const qsym& sym_state, const bool singlet){
         const int isym = sym_state.isym();
         qbond qcol({{sym_state,1}});
         qtensor2<Qm::ifabelian,Tm> wf2;
         if(isym == 3 && singlet){
            qbond qrow({{qsym(3,sym_state.ts(),sym_state.ts()),1}}); // couple to fictious site
            wf2.init(qsym(3,sym_state.ne()+sym_state.ts(),0),qrow,qcol,dir_WF2);
         }else{ 
            qbond qrow({{qsym(isym,0,0),1}}); // couple to vacuum
            wf2.init(sym_state,qrow,qcol,dir_WF2);
         }
         assert(wf2.size() == 1);
         wf2._data[0] = 1.0;
         return wf2;
      }

   // r*R0*R1*R2 to C0*R1*R2 
   template <typename Qm, typename Tm>
      void init_cpsi_dot0(comb<Qm,Tm>& icomb,
            const int iroot=-1,
            const bool singlet=true){
         const auto& rindex = icomb.topo.rindex;
         const auto& site0 = icomb.sites[rindex.at(std::make_pair(0,0))]; // will be updated
         const auto sym_state = icomb.get_qsym_state();
         const auto wf2 = get_boundary_coupling<Qm,Tm>(sym_state, singlet); // env*C[0]
         int nroots = (iroot==-1)? icomb.get_nroots() : 1;
         icomb.cpsi.resize(nroots);
         if(iroot == -1){
            for(int iroot=0; iroot<nroots; iroot++){
               // qt2(1,r): ->-*->-
               auto qt2 = contract_qt2_qt2(wf2,icomb.rwfuns[iroot]);
               // qt2(1,r)*site0(r,r0,n0) = qt3(1,r0,n0)[CRcouple]
               auto qt3 = contract_qt3_qt2("l",site0,qt2);
               // recouple to qt3(1,r0,n0)[LCcouple]
               icomb.cpsi[iroot] = qt3.recouple_lc();
            } // iroot
         }else{
            // get an MPS for a single state
            if(iroot != 0) icomb.rwfuns[0] = std::move(icomb.rwfuns[iroot]);
            icomb.rwfuns.resize(1);
            // qt2(1,r): ->-*->-
            auto qt2 = contract_qt2_qt2(wf2,icomb.rwfuns[0]);
            // qt2(1,r)*site0(r,r0,n0) = qt3(1,r0,n0)[CRcouple]
            auto qt3 = contract_qt3_qt2("l",site0,qt2);
            // recouple to qt3(1,r0,n0)[LCcouple]
            icomb.cpsi[0] = qt3.recouple_lc();
         }
      }

   //     | qs |
   //  ---*----*---(0,0)
   //    r1  r0
   template <typename Qm, typename Tm>
   void rcanon_lastdots(comb<Qm,Tm>& icomb){
      int nphysical = icomb.get_nphysical();
      auto& rsite0 = icomb.sites[0];
      auto& rsite1 = icomb.sites[1];
      auto& qs = rsite0.info.qrow;
      auto qmid = get_qbond_phys(Qm::isym);
      if(qs == qmid) return; // no need to do anything
      assert(qs.size() <= qmid.size());
      // We simply use computation to avoid explicitly deal
      // with the formation of the last two sites.
      auto rmat = rsite0.merge_cr();
      // P is due to definition in contract_qt3_qt2
      rsite1 = contract_qt3_qt2("r",rsite1,rmat.P()); 
      rsite0 = get_right_bsite<Qm,Tm>();
   }

} // ctns

#endif
