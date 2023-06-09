#ifndef MPS_OVLP_H
#define MPS_OVLP_H

namespace ctns{

   // <MPS[i]|MPS[j]>
   template <typename Km>
      linalg::matrix<typename Km::dtype> get_Smat(const mps<Km>& bmps,
            const mps<Km>& kmps){
         auto t0 = tools::get_time();
         assert(bmps.nphysical == kmps.nphysical);
         int nphysical = bmps.nphysical;
         // loop over sites on backbone
         stensor2<typename Km::dtype> qt2_r;
         for(int i=nphysical-1; i>0; i--){
            const auto& bsite = bmps.sites[i];
            const auto& ksite = kmps.sites[i];
            if(i == nphysical-1){
               qt2_r = contract_qt3_qt3("cr",bsite,ksite);
            }else{
               auto qtmp = contract_qt3_qt2("r",ksite,qt2_r);
               qt2_r = contract_qt3_qt3("cr",bsite,qtmp);
            }
         }
         // first merge: sum_l rwfuns[j,l]*site0[l,r,n] => site[j,r,n]
         auto bsite = bmps.get_site0();
         auto ksite = kmps.get_site0();
         auto qtmp = contract_qt3_qt2("r",ksite,qt2_r);
         qt2_r = contract_qt3_qt3("cr",bsite,qtmp);
         auto Smat = qt2_r.to_matrix();
         auto t1 = tools::get_time();
         tools::timing("ctns::mps::get_Smat", t0, t1);
         return Smat;
      }

   template <typename Km>
      void mps_ovlp(const input::schedule& schd){
         using Tm = typename Km::dtype;
         std::cout << "\nctns::mps_ovlp" << std::endl;
         topology topo;
         topo.read(schd.postmps.topology_file);
         //topo.print();
         // <bra|ket>
         int nket = schd.postmps.ket.size();
         for(int j=0; j<nket; j++){
            std::cout << "\n### jket=" << j << " ###" << std::endl;
            mps<Km> kmps;
            auto kmps_file = schd.scratch+"/rcanon_isweep"+std::to_string(schd.postmps.ket[j])+".info"; 
            kmps.nphysical = topo.nphysical;
            kmps.image2 = topo.image2;
            kmps.load(kmps_file);
            kmps.print();
            // compute column
            std::vector<linalg::matrix<Tm>> ovlp_i;
            std::vector<int> bra(schd.postmps.ket[j]+1);
            std::iota(bra.begin(), bra.end(), 0);
            if(schd.postmps.bra[0] != -1) bra = schd.postmps.bra; 
            int nbra = bra.size();
            ovlp_i.resize(nbra);
            for(int i=0; i<nbra; i++){
               std::cout << "\n### jket=" << j << " ibra=" << i << " ###" << std::endl;
               mps<Km> bmps;
               auto bmps_file = schd.scratch+"/rcanon_isweep"+std::to_string(bra[i])+".info";
               bmps.nphysical = topo.nphysical;
               bmps.image2 = topo.image2;
               bmps.load(bmps_file);
               // compute overlap
               ovlp_i[i] = get_Smat(bmps, kmps);
               ovlp_i[i].print("Smat",8);
            }
            // print summary
            int bstate = ovlp_i[0].rows();
            int kstate = ovlp_i[0].cols();
            for(int b=0; b<bstate; b++){
               for(int k=0; k<kstate; k++){
                  std::cout << "\nSUMMARY: jket=" << j
                     << " bstate=" << b 
                     << " kstate=" << k 
                     << std::endl;
                  std::cout << "    i    idx        ovlp     " << std::endl;
                  std::cout << "-----------------------------" << std::endl;
                  for(int i=0; i<nbra; i++){
                     std::cout << std::setw(5) << i
                        << " " << std::setw(5) << bra[i]
                        << " " << std::setw(16) << std::scientific << std::setprecision(6) << ovlp_i[i](b,k)
                        << std::endl;
                  } // i
               } // k
            } // b
         }
      }

} // ctns

#endif
