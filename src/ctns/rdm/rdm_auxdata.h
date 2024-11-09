#ifndef RDM_AUXDATA_H
#define RDM_AUXDATA_H

namespace ctns{

   // for other purpose such as debug, dsrg-mrpt2, etc.
   template <typename Tm>
      struct rdmaux{
         public:
            void dsrg_load(const int nv2, const int nc2, const int no2);
         public:
            // for debug
            linalg::matrix<Tm> rdm; 
            // dsrg-mrpt2
            linalg::matrix<Tm> dsrg_hv; // hv[ew,(xy)] 
            linalg::matrix<Tm> dsrg_tv; // tv[ez,(uv)]
            linalg::matrix<Tm> dsrg_hc; // hc[mz,(uv)]
            linalg::matrix<Tm> dsrg_tc; // tc[mw,(xy)]
      };

   template <typename Tm>
      void rdmaux<Tm>::dsrg_load(const int nv2, const int nc2, const int no2){
         std::cout << "rdmaux::dsrg_load (nv2,nc2,no2)=" << nv2 << "," << nc2 << "," << no2 << std::endl;
         auto t0 = tools::get_time();
         // load 
         size_t nopair = no2*(no2-1)/2;
         {
            dsrg_hv.resize(nv2*no2,nopair);
            std::ifstream ifs("dsrg.hv", std::ios::binary);
            ifs.read((char*)(dsrg_hv.data()), dsrg_hv.size()*sizeof(Tm));
            ifs.close();
            std::cout << "hv" << std::endl;
            for(int i=0; i<nv2*no2; i++){
               std::cout << " i=" << i << " " << dsrg_hv(i,1) << std::endl; 
            }
         }
         {         
            dsrg_tv.resize(nv2*no2,nopair);
            std::ifstream ifs("dsrg.tv", std::ios::binary);
            ifs.read((char*)(dsrg_tv.data()), dsrg_tv.size()*sizeof(Tm));
            ifs.close();
            std::cout << "tv" << std::endl;
            for(int i=0; i<nv2*no2; i++){
               std::cout << " i=" << i << " " << dsrg_tv(i,1) << std::endl; 
            }
         }
         {
            dsrg_hc.resize(nc2*no2,nopair);
            std::ifstream ifs("dsrg.hc", std::ios::binary);
            ifs.read((char*)(dsrg_hc.data()), dsrg_hc.size()*sizeof(Tm));
            ifs.close();
            std::cout << "hc" << std::endl;
            for(int i=0; i<nc2*no2; i++){
               std::cout << " i=" << i << " " << dsrg_hc(i,1) << std::endl; 
            }
         }
         {
            dsrg_tc.resize(nc2*no2,nopair); 
            std::ifstream ifs("dsrg.tc", std::ios::binary);
            ifs.read((char*)(dsrg_tc.data()), dsrg_tc.size()*sizeof(Tm));
            ifs.close();
            std::cout << "tc" << std::endl;
            for(int i=0; i<nc2*no2; i++){
               std::cout << " i=" << i << " " << dsrg_tc(i,1) << std::endl; 
            }
         }
         auto t1 = tools::get_time();
         tools::timing("rdmaux::dsrg_load", t0, t1);
      }

}

#endif
