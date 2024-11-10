#ifndef RDM_AUXDATA_H
#define RDM_AUXDATA_H

namespace ctns{

   // for other purpose such as debug, dsrg-mrpt2, etc.
   template <typename Tm>
      struct rdmaux{
         public:
            void dsrg_load(const int nv2, const int nc2, const int no2);
            void dsrg_contract(const size_t idx, const size_t jdx, const Tm rdm32ij);
         public:
            // for debug
            linalg::matrix<Tm> rdm; 
            // dsrg-mrpt2
            size_t nv2 = 0, nc2 = 0, no2 = 0; // no. of spin-orbitals for virtual/core/active
            int alg_mrpt2 = -1; // algorithm for mprt2: =2 on-the-fly contraction
            int pw = -1; // index for <0|x+y+z+vu|w>
            double ept2v = 0.0, ept2c = 0.0;
            linalg::matrix<Tm> dsrg_hv; // hv[ew,(xy)] 
            linalg::matrix<Tm> dsrg_tv; // tv[ez,(uv)]
            linalg::matrix<Tm> dsrg_hc; // hc[mz,(uv)]
            linalg::matrix<Tm> dsrg_tc; // tc[mw,(xy)]
      };

   // load h and t for dsrg
   template <typename Tm>
      void rdmaux<Tm>::dsrg_load(const int _nv2, const int _nc2, const int _no2){
         const bool debug = false;
         std::cout << "\nrdmaux::dsrg_load (nv2,nc2,no2)=" << nv2 << "," << nc2 << "," << no2 << std::endl;
         auto t0 = tools::get_time();

         // setup 
         nv2 = _nv2;
         nc2 = _nc2;
         no2 = _no2;
         size_t nopair = no2*(no2-1)/2;

         // hv
         {
            dsrg_hv.resize(nv2*no2,nopair);
            std::ifstream ifs("dsrg.hv", std::ios::binary);
            ifs.read((char*)(dsrg_hv.data()), dsrg_hv.size()*sizeof(Tm));
            ifs.close();
            if(debug){
               std::cout << "hv" << std::endl;
               for(int i=0; i<nv2*no2; i++){
                  std::cout << " i=" << i << " " << dsrg_hv(i,1) << std::endl; 
               }
            }
         }

         // tv
         {         
            dsrg_tv.resize(nv2*no2,nopair);
            std::ifstream ifs("dsrg.tv", std::ios::binary);
            ifs.read((char*)(dsrg_tv.data()), dsrg_tv.size()*sizeof(Tm));
            ifs.close();
            if(debug){
               std::cout << "tv" << std::endl;
               for(int i=0; i<nv2*no2; i++){
                  std::cout << " i=" << i << " " << dsrg_tv(i,1) << std::endl; 
               }
            }
         }

         // hc
         {
            dsrg_hc.resize(nc2*no2,nopair);
            std::ifstream ifs("dsrg.hc", std::ios::binary);
            ifs.read((char*)(dsrg_hc.data()), dsrg_hc.size()*sizeof(Tm));
            ifs.close();
            if(debug){
               std::cout << "hc" << std::endl;
               for(int i=0; i<nc2*no2; i++){
                  std::cout << " i=" << i << " " << dsrg_hc(i,1) << std::endl; 
               }
            }
         }

         // tc
         {
            dsrg_tc.resize(nc2*no2,nopair); 
            std::ifstream ifs("dsrg.tc", std::ios::binary);
            ifs.read((char*)(dsrg_tc.data()), dsrg_tc.size()*sizeof(Tm));
            ifs.close();
            if(debug){
               std::cout << "tc" << std::endl;
               for(int i=0; i<nc2*no2; i++){
                  std::cout << " i=" << i << " " << dsrg_tc(i,1) << std::endl; 
               }
            }
         }

         auto t1 = tools::get_time();
         tools::timing("rdmaux::dsrg_load", t0, t1);
      }

   // accumulate ept2v and ept2c
   template <typename Tm>
      void rdmaux<Tm>::dsrg_contract(const size_t idx, const size_t jdx, const Tm rdm32ij){
         auto pxyz_triple = tools::inverse_triple0(idx);
         auto px = std::get<0>(pxyz_triple);
         auto py = std::get<1>(pxyz_triple);
         auto pz = std::get<2>(pxyz_triple);
         auto puv_pair = tools::inverse_pair0(jdx);
         auto pu = puv_pair.first;
         auto pv = puv_pair.second;
         auto puv = jdx;
         auto pxy = tools::canonical_pair0(px,py);
         auto pxz = tools::canonical_pair0(px,pz);
         auto pyz = tools::canonical_pair0(py,pz);
         // -hv[ew,xy]*tv[uv,ez]*<0|x+y+z+vu|w> (x>y>z)
         auto vterm0 = linalg::xdot(nv2, &dsrg_hv(pw*nv2,pxy), &dsrg_tv(pz*nv2,puv));
         auto vterm1 = linalg::xdot(nv2, &dsrg_hv(pw*nv2,pxz), &dsrg_tv(py*nv2,puv));
         auto vterm2 = linalg::xdot(nv2, &dsrg_hv(pw*nv2,pyz), &dsrg_tv(px*nv2,puv));
         ept2v += std::real((-vterm0 + vterm1 - vterm2)*rdm32ij); 
         // tc[mw,xy]*hc[uv,mz]*<0|x+y+z+vu|w> (x>y>z)
         auto cterm0 = linalg::xdot(nc2, &dsrg_tc(pw*nc2,pxy), &dsrg_hc(pz*nc2,puv));
         auto cterm1 = linalg::xdot(nc2, &dsrg_tc(pw*nc2,pxz), &dsrg_hc(py*nc2,puv));
         auto cterm2 = linalg::xdot(nc2, &dsrg_tc(pw*nc2,pyz), &dsrg_hc(px*nc2,puv));
         ept2c += std::real((cterm0 - cterm1 + cterm2)*rdm32ij);
      }

}

#endif
