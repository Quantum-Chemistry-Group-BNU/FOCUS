#ifndef MPS_S2PROJ_H
#define MPS_S2PROJ_H

#include "core/special.h"
#include "mps_cicoeff.h"

namespace ctns{

   // merge symmetry and get offset
   qbond lowerSym(const qbond& qs, 
         std::vector<int>& brmap,
         std::vector<int>& broff){
      int nsym = qs.size();
      brmap.resize(nsym); // map br to br2 in qs2
      broff.resize(nsym); // offset in the block 
      // start processing qs 
      std::vector<qsym> qsec2;
      std::map<qsym,int> qidx;
      std::map<qsym,int> qoff;
      for(int i=0; i<nsym; i++){
         auto sym = qs.get_sym(i);
         auto dim = qs.get_dim(i);
         qsym sym2(1, sym.ne());
         if(qoff.find(sym2) == qoff.end()){
            brmap[i] = qsec2.size();
            qidx[sym2] = qsec2.size();
            qsec2.push_back(sym2);
            broff[i] = 0; 
            qoff[sym2] = dim; 
         }else{
            brmap[i] = qidx[sym2];
            broff[i] = qoff[sym2];
            qoff[sym2] += dim;
         }
      } // i
      qbond qs2;
      nsym = qsec2.size();
      qs2.dims.resize(nsym);
      for(int i=0; i<nsym; i++){
         qs2.dims[i] = std::make_pair(qsec2[i],qoff[qsec2[i]]); 
      }
      assert(qs.get_dimAll() == qs2.get_dimAll());
      return qs2;
   }

   // lower symmetry for stensor3
   template <typename Tm, typename Tm2>
      void lowerSym(const stensor3<Tm>& isite, stensor3<Tm2>& rsite){
         std::cout << "error: inconsistent data type in lowerSym[stensor3]!" << std::endl;
         exit(1);
      }
   template <typename Tm>
      void lowerSym(const stensor3<Tm>& isite, stensor3<Tm>& rsite){
         std::vector<int> brmap, bcmap, bmmap;
         std::vector<int> broff, bcoff, bmoff;
         auto rqrow = lowerSym(isite.info.qrow, brmap, broff);
         auto cqcol = lowerSym(isite.info.qcol, bcmap, bcoff);
         auto mqmid = lowerSym(isite.info.qmid, bmmap, bmoff);
         rsite.init(qsym(1,isite.info.sym.ne()), rqrow, cqcol, mqmid, isite.info.dir);
         int br, bc, bm;
         for(int i=0; i<isite.info._nnzaddr.size(); i++){
            int idx = isite.info._nnzaddr[i];
            isite.info._addr_unpack(idx,br,bc,bm);
            const auto iblk = isite(br,bc,bm); // (l,r,n);
            auto rblk = rsite(brmap[br],bcmap[bc],bmmap[bm]); // (l,r,n);
            int roff = broff[br];
            int coff = bcoff[bc];
            int moff = bmoff[bm];
            for(int im=0; im<iblk.dim2; im++){
               for(int ic=0; ic<iblk.dim1; ic++){
                  for(int ir=0; ir<iblk.dim0; ir++){
                     rblk(roff+ir,coff+ic,moff+im) = iblk(ir,ic,im); 
                  } // ir
               } // ic
            } // im
         } // i
      }

   // lower symmetry for stensor2
   template <typename Tm, typename Tm2>
      void lowerSym(const stensor2<Tm>& isite, stensor2<Tm2>& rsite){
         std::cout << "error: inconsistent data type in lowerSym[stensor2]!" << std::endl;
         exit(1);
      }
   template <typename Tm>
      void lowerSym(const stensor2<Tm>& isite, stensor2<Tm>& rsite){
         std::vector<int> brmap, bcmap;
         std::vector<int> broff, bcoff;
         auto rqrow = lowerSym(isite.info.qrow, brmap, broff);
         auto cqcol = lowerSym(isite.info.qcol, bcmap, bcoff);
         rsite.init(qsym(1,isite.info.sym.ne()), rqrow, cqcol, isite.info.dir);
         int br, bc;
         for(int i=0; i<isite.info._nnzaddr.size(); i++){
            int idx = isite.info._nnzaddr[i];
            isite.info._addr_unpack(idx,br,bc);
            const auto iblk = isite(br,bc); // (l,r);
            auto rblk = rsite(brmap[br],bcmap[bc]); // (l,r);
            int roff = broff[br];
            int coff = bcoff[bc];
            for(int ic=0; ic<iblk.dim1; ic++){
               for(int ir=0; ir<iblk.dim0; ir++){
                  rblk(roff+ir,coff+ic) = iblk(ir,ic); 
               } // ir
            } // ic
         } // i
      }

   // lower symmetry for MPS
   template <typename Qm1, typename Qm2, typename Tm>
      void lowerSym(const mps<Qm1,Tm>& imps, mps<Qm2,Tm>& rmps){
         std::cout << "lowerSym for MPS:"
            << " Qm1=" << qkind::get_name<Qm1>()
            << " Qm2=" << qkind::get_name<Qm2>()
            << std::endl;
         const bool debug = false; 
         // copy basic information
         rmps.nphysical = imps.nphysical;
         rmps.image2 = imps.image2;
         rmps.sites.resize(imps.nphysical);
         // lower symmetry for each site
         for(int k=0; k<imps.nphysical; k++){
             lowerSym(imps.sites[k], rmps.sites[k]);
         }
         // lower symmetry for rwfuns
         int nroots = imps.rwfuns.size();
         rmps.rwfuns.resize(nroots);
         for(int k=0; k<nroots; k++){
            lowerSym(imps.rwfuns[k], rmps.rwfuns[k]);
         }
         // debug
         if(debug){
            mps_cicoeff_check(imps, 0);
            mps_cicoeff_check(rmps, 0);
         }
      }

   // apply ry rotation exp(-i*x*Sy) to MPS
   template <typename Qm, typename Tm>
      mps<Qm,Tm> mps_ryrotation(const mps<Qm,Tm>& imps, const double x){
         auto rmps = imps;
         for(int k=0; k<rmps.nphysical; k++){
            // A[n](l,r) = A(l,r,n)
            const auto& isite = imps.sites[k];
            const auto& rsite = rmps.sites[k];
            int br, bc, bm;
            for(int i=0; i<rsite.info._nnzaddr.size(); i++){
               int idx = rsite.info._nnzaddr[i];
               rsite.info._addr_unpack(idx,br,bc,bm);
               const auto iblk = isite(br,bc,bm); // (l,r,n);
               auto rblk = rsite(br,bc,bm); // (l,r,n);
               if(bm != 2) continue;
               assert(rblk.dim2 == 2);
               // zmpo_dmrg/source/mpo_dmrg_opers.py
               // [c   s] [ A[a=|01>] ] => [  c*A[a] + s*A[b] ]
               // [-s  c] [ A[b=|10>] ] => [ -s*A[a] + c*A[b] ]
               double c = cos(0.5*x);
               double s = sin(0.5*x);
               size_t N = rblk.dim0*rblk.dim1;
               linalg::xscal(N, c, rblk.get(0).data());
               linalg::xaxpy(N, s, iblk.get(1).data(), rblk.get(0).data());
               linalg::xscal(N, c, rblk.get(1).data());
               linalg::xaxpy(N, -s, iblk.get(0).data(), rblk.get(1).data());
            }
         }
         return rmps;
      }

   // compute <Psi|Ps|Psi>
   template <typename Qm, typename Tm>
      double mps_expect_s2proj(const mps<Qm,Tm>& imps, 
            const int iroot,
            const int ne,
            const int tm,
            const int ts){
         std::vector<double> xts, wts;
         special::gen_s2quad(imps.nphysical, ne, ts/2.0, tm/2.0, xts, wts);
         double ps = 0.0; 
         for(int i=0; i<xts.size(); i++){
            // apply |rmps> = exp(-i*beta*Sy)|imps>
            auto rmps = mps_ryrotation(imps, xts[i]);
            // compute <imps|rmps>
            Tm ova = get_Smat(imps, rmps)(iroot, iroot);
            ps += wts[i]*std::real(ova);
         }
         return ps;
      }

   template <typename Qm, typename Tm>
      void mps_s2proj(const input::schedule& schd){
         int rank = 0, size = 1;
#ifndef SERIAL
         rank = schd.world.rank();
         size = schd.world.size();
#endif
         const bool debug = (rank==0);
         if(debug) std::cout << "\nctns::mps_s2proj" << std::endl;
         topology topo;
         topo.read(schd.post.topology_file);
         //topo.print();
         int nket = schd.post.ket.size();
         for(int j=0; j<nket; j++){
            std::cout << "\n### jket=" << j << " ###" << std::endl;
            mps<Qm,Tm> kmps;
            auto kmps_file = schd.scratch+"/rcanon_isweep"+std::to_string(schd.post.ket[j])+".info"; 
            kmps.nphysical = topo.nphysical;
            kmps.image2 = topo.image2;
            kmps.load(kmps_file);
            // compute expectation value via sampling
            auto sym = kmps.get_sym_state();
            if(sym.isym() != 2){
               std::cout << "error: other isym also requires Sz proj! isym=" << sym.isym() << std::endl;
               exit(1);
            }
            int k = kmps.nphysical;
            int ne = sym.ne();
            int tm = sym.tm();
            int na_max = min(ne,k);
            int nb_min = ne - na_max;
            int ts_max = na_max - nb_min;
            int ts_min = ne%2;
            std::cout << "qsym=" << sym << " ts_min/max=" << ts_min << "," << ts_max << std::endl;
            double sum = 0.0;
            for(int ts=ts_min; ts<=ts_max; ts+=2){
               std::cout << "\nts=" << ts << std::endl;
               double ps = 0.0;
               if(qkind::is_qNSz<Qm>()){
                  mps<qkind::qN,Tm> kmps_low;
                  lowerSym(kmps, kmps_low);
                  ps = mps_expect_s2proj(kmps_low, schd.post.iroot, ne, tm, ts);
               }
               sum += ps;
               std::cout << " ts=" << ts << " ps=" << ps << " sum=" << sum << std::endl;
            } // ts
            std::cout << "\nfinal sum=" << sum << " 1-sum=" << (1.0-sum) << std::endl;
            if(abs(1.0-sum)>1.e-10){
               std::cout << "error: deviation from 1!" << std::endl;
               exit(1);
            }
         } // j
      }

} // ctns

#endif
