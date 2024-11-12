#ifndef RDM_CHECK_H
#define RDM_CHECK_H

namespace ctns{

   template <typename Qm, typename Tm>
      void rdm_checkinput(const input::schedule& schd,
            const ctns::comb<Qm,Tm>& icomb, 
            const ctns::comb<Qm,Tm>& icomb2, 
            const bool is_same){

         // available key 
         const std::set<std::string> keys_avail = {"ova",
            "1p1h",
            "2p2h",
            "1p0h", "0p1h",
            "2p0h", "0p2h",
            "2p1h", "1p2h",
            "3p2h", "2p3h",
            "3p3h",
            "mrpt2"};
         std::cout << "\n" << tools::line_separator2 << std::endl;
         std::cout << "task_prop:";
         for(const auto& key : schd.ctns.task_prop){
            std::cout << " " << key;
            if(keys_avail.find(key) == keys_avail.end()){
               std::cout << "\nerror: keys_avail does not contain key=" << key << std::endl;
               exit(1);
            }
         }
         std::cout << std::endl;

         // information about mps
         std::cout << " is_same=" << is_same << std::endl;
         std::cout << " MPS1:" << " nroot=" << icomb.get_nroots() 
            << " iroot=" << schd.ctns.iroot << " file=" << schd.ctns.rcanon_file 
            << std::endl;
         assert(schd.ctns.iroot <= icomb.get_nroots());
         std::cout << " MPS2:" << " nroot=" << icomb2.get_nroots() 
            << " jroot=" << schd.ctns.jroot << " file=" << schd.ctns.rcanon2_file
            << std::endl;
         assert(schd.ctns.jroot <= icomb2.get_nroots());

         // size of rdm
         const size_t k = schd.sorb;
         const size_t k2 = k*(k-1)/2;
         const size_t k3 = k*(k-1)*(k-2)/6;
         std::cout << std::scientific << std::setprecision(2)
            << "size of rdms:" << std::endl;
         size_t rdm1size = k*k;
         std::cout << " size(rdm1)=" << rdm1size
            << ":" << tools::sizeMB<Tm>(rdm1size) << "MB"
            << ":" << tools::sizeGB<Tm>(rdm1size) << "GB" 
            << std::endl;
         size_t rdm2size = k2*k2;
         std::cout << " size(rdm2)=" << rdm2size
            << ":" << tools::sizeMB<Tm>(rdm2size) << "MB"
            << ":" << tools::sizeGB<Tm>(rdm2size) << "GB" 
            << std::endl;
         size_t rdm3size = k3*k3;
         std::cout << " size(rdm3)=" << rdm3size
            << ":" << tools::sizeMB<Tm>(rdm3size) << "MB"
            << ":" << tools::sizeGB<Tm>(rdm3size) << "GB" 
            << std::endl;
         std::cout << tools::line_separator2 << std::endl;
      }

} // ctns

#endif
