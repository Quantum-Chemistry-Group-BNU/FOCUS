#ifndef CTNS_TOSU2_ENV_H
#define CTNS_TOSU2_ENV_H

namespace ctns{

   // Density matrix environement:
   //      *---*---*---*-
   //     /    |   |   |
   //    |     |   |
   //     \    |   |   |
   //      *---*---*---*-
   template <typename Tm>
      std::vector<stensor2<Tm>> buildDMLeftEnv(const comb<qkind::qNSz,Tm>& icomb_NSz,
            const bool debug=true){
         if(debug) std::cout << "\nctns::buildDMLeftEnv" << std::endl;
         int nsite = icomb_NSz.get_nphysical();
         int nroot = icomb_NSz.get_nroots(); 
         std::vector<stensor2<Tm>> dmenv(nsite);
         // left most rhol = \sum_i rwfuns[i].T * rwfuns[i].conj()
         //                = \sum_i (rwfuns[i].H * rwfuns[i]).conj()
         auto rhol = icomb_NSz.rwfuns[0].H().dot(icomb_NSz.rwfuns[0]);
         for(int i=1; i<nroot; i++){
            rhol += icomb_NSz.rwfuns[i].H().dot(icomb_NSz.rwfuns[i]);
         }
         rhol *= 1.0/nroot;
         rhol._conj();
         dmenv[nsite-1] = std::move(rhol);
         // update environment 
         // /--*--
         // *  |
         // \--*--
         for(int i=nsite-1; i>=1; i--){
            const auto& site = icomb_NSz.sites[i];
            auto qtmp = contract_qt3_qt2("l",site,dmenv[i]);
            dmenv[i-1] = contract_qt3_qt3("lc",site,qtmp);
         }

         // check
         for(int i=0; i<dmenv.size(); i++){
            const auto& dm = dmenv[i];
            double tr = std::real(dm.to_matrix().trace());
            std::cout << "i=" << i << " tr(rhol)=" << tr << std::endl; 
         }

         return dmenv;
      }

} // ctns

#endif
