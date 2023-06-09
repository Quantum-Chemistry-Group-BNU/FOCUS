#ifndef MPS_CICOEFF_H
#define MPS_CICOEFF_H

namespace ctns{

   // Algorithm 2: <n|MPS> by contraction
   template <typename Km>
      typename Km::dtype mps_CIcoeff(const mps<Km>& imps,
            const int iroot,
            const fock::onstate& state){
         using Tm = typename Km::dtype;
         // compute <n|MPS> by contracting all sites
         stensor2<Tm> qt2_r;
         for(int i=imps.nphysical-1; i>=0; i--){
            // site on backbone with physical index
            const auto& site = imps.sites[i];
            // given occ pattern, extract the corresponding qblock
            int pindex = imps.get_pindex(i);
            auto qt2 = site.fix_mid( occ2mdx(Km::isym, state, pindex) ); 
            if(i == imps.nphysical-1){
               qt2_r = std::move(qt2);
            }else{
               qt2_r = qt2.dot(qt2_r); // (out,x)*(x,in)->(out,in)
            }
         } // i
         const auto& wfcoeff = imps.rwfuns[iroot].dot(qt2_r);
         assert(wfcoeff.rows() == 1 && wfcoeff.cols() == 1);
         const auto blk = wfcoeff(0,0);
         // in case this CTNS does not encode this det, no such block 
         if(blk.empty()) return 0.0; 
         // compute fermionic sign changes to match ordering of orbitals
         double sgn = state.permute_sgn(imps.image2);
         return sgn*blk(0,0);
      }

   template <typename Km>
      void mps_cicoeff_check(const mps<Km>& imps,
            const int iroot,
            const double thresh_print=1.e-8){
         using Tm = typename Km::dtype;
         std::cout << "\nctns::mps_cicoeff_check iroot=" << iroot << std::endl;
         auto fci_space = imps.get_fcispace();
         int dim = fci_space.size();
         std::vector<Tm> coeff(dim,0.0);
         // cmat[j,i] = <D[i]|CTNS[j]>
         for(int i=0; i<dim; i++){
            coeff[i] = mps_CIcoeff(imps, iroot, fci_space[i]);
            if(std::abs(coeff[i]) < thresh_print) continue;
            std::cout << " i=" << i << " state=" << fci_space[i] 
               << " <n|MPS>=" << coeff[i]
               << std::endl; 
         }
         double Sdiag = fock::coeff_entropy(coeff);
         double ovlp = std::pow(linalg::xnrm2(dim,&coeff[0]),2);
         std::cout << "ovlp=" << ovlp << " Sdiag(exact)=" << Sdiag << std::endl;
      }

   template <typename Km>
      void mps_cicoeff(const input::schedule& schd){
         using Tm = typename Km::dtype;
         std::cout << "\nctns::mps_cicoeff" << std::endl;
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
            // compute cicoeff 
            mps_cicoeff_check(kmps, schd.postmps.iroot);
         }
      }

} // ctns

#endif
