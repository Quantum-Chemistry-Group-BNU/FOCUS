#ifndef RDM_MRPT2_H
#define RDM_MRPT2_H

#include "rdm_auxdata.h"

namespace ctns{

   template <typename Qm, typename Tm>
      double rdm_mrpt2(const comb<Qm,Tm>& icomb,
            const input::schedule& schd,
            const std::string scratch){
         int size = 1, rank = 0;
#ifndef SERIAL
         size = icomb.world.size();
         rank = icomb.world.rank();
#endif
         
         const auto& nv2 = schd.ctns.nv2;
         const auto& nc2 = schd.ctns.nc2;
         const size_t k = schd.sorb;
         const size_t k2 = k*(k-1)/2;
         const size_t k3 = k*(k-1)*(k-2)/6;
         linalg::matrix<Tm> rdm3;

         // load dsrg information
         rdmaux<Tm> aux;
         aux.dsrg_load(nv2, nc2, k);

         auto image1 = icomb.topo.get_image1();

         double ept2v = 0.0, ept2c = 0.0, ept2 = 0.0;
         auto t0 = tools::get_time();
         if(schd.ctns.alg_mrpt2 == 0){

            // compute rdm3 = <p+q+r+stu> (p>q>r,s<t<u)
            rdm3.resize(k3,k3);
            for(int i=0; i<k; i++){
               auto tx = tools::get_time();
               int ki = i/2, spin_i = i%2;
               auto icomb_i = apply_opC(icomb, ki, spin_i, 0); // i|psi> (u=i)
               linalg::matrix<Tm> rdm32(k3,k2);
               ctns::rdm_sweep("3p2h", false, icomb, icomb_i, schd, scratch, rdm32, aux);
               // copy data to rdm3 <Psi_0|p+q+r+st|Psi_i>
               int pi = 2*image1[ki] + spin_i; // map to the orbital index
               for(int pt=0; pt<pi; pt++){ 
                  for(int ps=0; ps<pt; ps++){
                     auto psti = tools::canonical_triple0(pi,pt,ps);
                     auto pst = tools::canonical_pair0(pt,ps); 
                     Tm sgn1 = tools::sgn_triple0(pi,pt,ps);
                     Tm sgn2 = tools::sgn_pair0(pt,ps);
                     linalg::xaxpy(k3, sgn1*sgn2, rdm32.col(pst), rdm3.col(psti)); 
                  }
               }
               auto ty = tools::get_time();
               if(rank == 0) std::cout << " i=" << i << " time=" << tools::get_duration(ty-tx) << " S" << std::endl;
            } // i 
            auto tz = tools::get_time();
            if(rank == 0) std::cout << "total time for 3-RDM: " << tools::get_duration(tz-t0) << " S" << std::endl;

            // assemble e2 by contracting <x+y+z+wvu>
            for(int px=0; px<k; px++){
               for(int py=0; py<px; py++){
                  for(int pz=0; pz<py; pz++){
                     for(int pu=0; pu<k; pu++){
                        for(int pv=0; pv<pu; pv++){
                           for(int pw=0; pw<pv; pw++){
                              auto pxyz = tools::canonical_triple0(px,py,pz);
                              auto puvw = tools::canonical_triple0(pu,pv,pw);
                              const auto& val = rdm3(pxyz,puvw);
                              auto pxy = tools::canonical_pair0(px,py);
                              auto pxz = tools::canonical_pair0(px,pz);
                              auto pyz = tools::canonical_pair0(py,pz);
                              auto puv = tools::canonical_pair0(pu,pv);
                              auto puw = tools::canonical_pair0(pu,pw);
                              auto pvw = tools::canonical_pair0(pv,pw);
                              // -hv[ew,xy]*tv[uv,ez]*<0|x+y+z+vu|w> (x>y>z)
                              auto vterm0 = linalg::xdot(nv2, &aux.dsrg_hv(pw*nv2,pxy), &aux.dsrg_tv(pz*nv2,puv));
                              auto vterm1 = linalg::xdot(nv2, &aux.dsrg_hv(pw*nv2,pxz), &aux.dsrg_tv(py*nv2,puv));
                              auto vterm2 = linalg::xdot(nv2, &aux.dsrg_hv(pw*nv2,pyz), &aux.dsrg_tv(px*nv2,puv));
                              auto vterm3 = linalg::xdot(nv2, &aux.dsrg_hv(pv*nv2,pxy), &aux.dsrg_tv(pz*nv2,puw));
                              auto vterm4 = linalg::xdot(nv2, &aux.dsrg_hv(pv*nv2,pxz), &aux.dsrg_tv(py*nv2,puw));
                              auto vterm5 = linalg::xdot(nv2, &aux.dsrg_hv(pv*nv2,pyz), &aux.dsrg_tv(px*nv2,puw));
                              auto vterm6 = linalg::xdot(nv2, &aux.dsrg_hv(pu*nv2,pxy), &aux.dsrg_tv(pz*nv2,pvw));
                              auto vterm7 = linalg::xdot(nv2, &aux.dsrg_hv(pu*nv2,pxz), &aux.dsrg_tv(py*nv2,pvw));
                              auto vterm8 = linalg::xdot(nv2, &aux.dsrg_hv(pu*nv2,pyz), &aux.dsrg_tv(px*nv2,pvw));
                              ept2v += std::real((-vterm0 + vterm1 - vterm2 
                                       +vterm3 - vterm4 + vterm5
                                       -vterm6 + vterm7 - vterm8)*val); 
                              // tc[mw,xy]*hc[uv,mz]*<0|x+y+z+vu|w> (x>y>z)
                              auto cterm0 = linalg::xdot(nc2, &aux.dsrg_tc(pw*nc2,pxy), &aux.dsrg_hc(pz*nc2,puv));
                              auto cterm1 = linalg::xdot(nc2, &aux.dsrg_tc(pw*nc2,pxz), &aux.dsrg_hc(py*nc2,puv));
                              auto cterm2 = linalg::xdot(nc2, &aux.dsrg_tc(pw*nc2,pyz), &aux.dsrg_hc(px*nc2,puv));
                              auto cterm3 = linalg::xdot(nc2, &aux.dsrg_tc(pv*nc2,pxy), &aux.dsrg_hc(pz*nc2,puw));
                              auto cterm4 = linalg::xdot(nc2, &aux.dsrg_tc(pv*nc2,pxz), &aux.dsrg_hc(py*nc2,puw));
                              auto cterm5 = linalg::xdot(nc2, &aux.dsrg_tc(pv*nc2,pyz), &aux.dsrg_hc(px*nc2,puw));
                              auto cterm6 = linalg::xdot(nc2, &aux.dsrg_tc(pu*nc2,pxy), &aux.dsrg_hc(pz*nc2,pvw));
                              auto cterm7 = linalg::xdot(nc2, &aux.dsrg_tc(pu*nc2,pxz), &aux.dsrg_hc(py*nc2,pvw));
                              auto cterm8 = linalg::xdot(nc2, &aux.dsrg_tc(pu*nc2,pyz), &aux.dsrg_hc(px*nc2,pvw));
                              ept2c += std::real((cterm0 - cterm1 + cterm2
                                       -cterm3 + cterm4 - cterm5
                                       +cterm6 - cterm7 + cterm8)*val);
                           } // w
                        } // v
                     } // u
                  } // z
               } // y
            } // x 

         }else if(schd.ctns.alg_mrpt2 == 1){

            for(int i=0; i<k; i++){
               auto tx = tools::get_time();
               int ki = i/2, spin_i = i%2;
               auto icomb_i = apply_opC(icomb, ki, spin_i, 0); // i|psi> (u=i)
               linalg::matrix<Tm> rdm32(k3,k2);
               ctns::rdm_sweep("3p2h", false, icomb, icomb_i, schd, scratch, rdm32, aux);
               // copy data to rdm3 <Psi_0|p+q+r+st|Psi_i>
               int pw = 2*image1[ki] + spin_i; // map to the orbital index
               for(int px=0; px<k; px++){
                  for(int py=0; py<px; py++){
                     for(int pz=0; pz<py; pz++){
                        for(int pu=0; pu<k; pu++){
                           for(int pv=0; pv<pu; pv++){
                              auto pxyz = tools::canonical_triple0(px,py,pz);
                              auto puv = tools::canonical_pair0(pu,pv);
                              const auto& val = rdm32(pxyz,puv);
                              auto pxy = tools::canonical_pair0(px,py);
                              auto pxz = tools::canonical_pair0(px,pz);
                              auto pyz = tools::canonical_pair0(py,pz);
                              // -hv[ew,xy]*tv[uv,ez]*<0|x+y+z+vu|w> (x>y>z)
                              auto vterm0 = linalg::xdot(nv2, &aux.dsrg_hv(pw*nv2,pxy), &aux.dsrg_tv(pz*nv2,puv));
                              auto vterm1 = linalg::xdot(nv2, &aux.dsrg_hv(pw*nv2,pxz), &aux.dsrg_tv(py*nv2,puv));
                              auto vterm2 = linalg::xdot(nv2, &aux.dsrg_hv(pw*nv2,pyz), &aux.dsrg_tv(px*nv2,puv));
                              ept2v += std::real((-vterm0 + vterm1 - vterm2)*val); 
                              // tc[mw,xy]*hc[uv,mz]*<0|x+y+z+vu|w> (x>y>z)
                              auto cterm0 = linalg::xdot(nc2, &aux.dsrg_tc(pw*nc2,pxy), &aux.dsrg_hc(pz*nc2,puv));
                              auto cterm1 = linalg::xdot(nc2, &aux.dsrg_tc(pw*nc2,pxz), &aux.dsrg_hc(py*nc2,puv));
                              auto cterm2 = linalg::xdot(nc2, &aux.dsrg_tc(pw*nc2,pyz), &aux.dsrg_hc(px*nc2,puv));
                              ept2c += std::real((cterm0 - cterm1 + cterm2)*val);
                           } // v
                        } // u
                     } // z
                  } // y
               } // x
               auto ty = tools::get_time();
               if(rank == 0) std::cout << " i=" << i << " time=" << tools::get_duration(ty-tx) << " S" << std::endl;
            } // i

         }else{

            std::cout << "error: no such option for alg_mrpt2=" << schd.ctns.alg_mrpt2 << std::endl;
            exit(1);

         } // alg_mrpt2
         auto t1 = tools::get_time();

         ept2 = ept2v + ept2c;
         if(rank == 0){
            std::cout << "\nept2v = " << std::fixed << std::setprecision(schd.ctns.outprec) << ept2v << std::endl; 
            std::cout << "ept2c = " << std::fixed << std::setprecision(schd.ctns.outprec) << ept2c << std::endl; 
            std::cout << "ept2  = " << std::fixed << std::setprecision(schd.ctns.outprec) << ept2 << std::endl; 
            std::cout << "total time for PT2: " << tools::get_duration(t1-t0) << " S" << std::endl;
         }
         return ept2;
      }

} // ctns

#endif
