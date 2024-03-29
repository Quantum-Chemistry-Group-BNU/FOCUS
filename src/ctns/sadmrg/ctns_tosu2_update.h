#ifndef CTNS_TOSU2_UPDATE_H
#define CTNS_TOSU2_UPDATE_H

#include "ctns_tosu2_wmat.h"
#include "ctns_tosu2_csite.h"

namespace ctns{

   template <typename Tm>
      Wmatrix<Tm> updateWmat(const Wmatrix<Tm>& csite, 
            const std::map<qsym,linalg::matrix<Tm>>& Yinfo,
            const bool debug=true){
         const Tm alpha = 1.0, beta = 0.0; 
         if(debug) std::cout << "\nctns::updateWmat" << std::endl;
         const auto& qrow = csite.qrow;
         const auto& qcol = csite.qcol;
         // form new qcol
         //display_qbond3(qcol,"qcol");
         qbond3 qcol2;
         for(int i=0; i<qcol.size(); i++){
            auto sym3 = qcol[i].first;
            qsym sym(3,std::get<0>(sym3),std::get<1>(sym3));
            /*
               std::cout << "i=" << i
               << " sym=" << sym
               << " sym3=" << std::get<2>(sym3)
               << " qcoli=" << qcol[i].second
               << std::endl;
               */
            // some symmetry sector may not present after decimation
            if(Yinfo.find(sym) != Yinfo.end()){
               assert(qcol[i].second == Yinfo.at(sym).cols()); // since Y=U^T
               qcol2.push_back(std::make_pair(sym3,Yinfo.at(sym).rows()));
            }
         }
         // form Wmatrix
         Wmatrix<Tm> wmat(qrow,qcol2);
         //display_qbond3(qcol2,"qcol2");
         int jdx = 0;
         for(int j=0; j<qcol.size(); j++){
            auto sym3 = qcol[j].first;
            qsym sym(3,std::get<0>(sym3),std::get<1>(sym3));
            // some symmetry sector may not present after decimation
            if(Yinfo.find(sym) == Yinfo.end()) continue;
            const auto& ymat = Yinfo.at(sym);
            for(int i=0; i<qrow.size(); i++){
               size_t offc = csite._offset.at(std::make_pair(i,j));
               size_t offw = wmat._offset[std::make_pair(i,jdx)];
               if(offc == 0 || offw == 0) continue;
               const Tm* cblk = csite._data.data() + offc-1;
               Tm* wblk = wmat._data.data() + offw-1;
               int di = qrow.get_dim(i);
               int dj = qcol2[jdx].second;
               int dk = qcol[j].second;

               // W[i,j] = C[i,k] Y*[j,k]
               linalg::xgemm("N", "C", di, dj, dk, alpha,
                     cblk, di, ymat.data(), dj, beta,
                     wblk, di); 

               /*
                  std::cout << "# i,j,jdx=" << i << "," << j << "," << jdx
                  << " symi=" << qrow.get_sym(i)
                  << " symj=(" << std::get<0>(qcol[j].first) << ","
                  << std::get<1>(qcol[j].first) << ","
                  << std::get<2>(qcol[j].first) << ")"
                  << " symjdx=(" << std::get<0>(qcol2[jdx].first) << ","
                  << std::get<1>(qcol2[jdx].first) << ","
                  << std::get<2>(qcol2[jdx].first) << ")"
                  << " di,dj,dk=" << di << "," << dj << "," << dk 
                  << std::endl;
                  linalg::matrix<Tm> cmat(di,dk,cblk);
                  cmat.print("cmat");
                  ymat.print("ymat");
                  linalg::matrix<Tm> wmat(di,dj,wblk);
                  wmat.print("wmat");
                  */

            } // i
            jdx += 1;
         } // j
         return wmat;
      }

   // expand Yinfo into MPS site
   template <typename Tm>
      stensor3su2<Tm> updateSite(const std::map<qsym,linalg::matrix<Tm>>& Yinfo,
            const qproduct3& qprod,
            const qbond3& qs1,
            const qbond3& qs2,
            const bool debug=true){
         int debug_level = 0;
         if(debug) std::cout << "\nctns::updateSite" << std::endl;
         const auto& dpt = qprod.second;
         auto qmidInfo = qbond3_to_qbond(qs1);
         auto qcolInfo = qbond3_to_qbond(qs2);
         const auto& qmid = qmidInfo.first;
         const auto& pmid = qmidInfo.second;
         const auto& qcol = qcolInfo.first;
         const auto& pcol = qcolInfo.second;

         // debug: dpt
         if(debug_level>0){
            for(const auto& pr : dpt){
               const auto& sym3 = pr.first;
               const auto& comp = pr.second;
               std::cout << "sym3=(" << std::get<0>(sym3) << "," << std::get<1>(sym3)
                  << "," << std::get<2>(sym3) << ")"
                  << std::endl;
               for(int i=0; i<comp.size(); i++){
                  int i1 = std::get<0>(comp[i]);
                  int i2 = std::get<1>(comp[i]);
                  auto q1 = qs1[i1].first;
                  auto q2 = qs2[i2].first;
                  std::cout << " i1,i2=" << i1 << "," << i2 
                     << " q1=(" << std::get<0>(q1) << "," << std::get<1>(q1) 
                     << "," << std::get<2>(q1) << ")"
                     << " q2=(" << std::get<0>(q2) << "," << std::get<1>(q2)
                     << "," << std::get<2>(q2) << ")"
                     << std::endl;
               } // i
            }
         }

         // 1. form qrow
         qbond qrow;
         std::map<qsym,int> prow;
         int idx = 0;
         for(const auto& pr : Yinfo){

            if(debug_level>0){
               std::cout << "idx=" << idx << " sym=" << pr.first << std::endl;
               const auto & ymat = pr.second;
               auto ovlp = linalg::xgemm("N","N",ymat,ymat.H());
               ymat.print("Ymat");
               ovlp.print("ovlp");
            }

            qrow.dims.push_back(std::make_pair(pr.first,pr.second.rows()));
            prow[pr.first] = idx;
            idx += 1;
         }
         // 2. assemble site
         stensor3su2<Tm> site(qsym(3,0,0),qrow,qcol,qmid,dir_RCF,CRcouple);
         for(const auto& pr : Yinfo){
            const auto& sym = pr.first;
            const auto& ymat = Yinfo.at(sym);
            int n = sym.ne();
            int ts = sym.ts();
            qsym3 sym3(n,ts,ts); // find the high-spin case
            const auto& comp = dpt.at(sym3);
            // loop over combination
            std::vector<int> unique_ns(1,0);
            int i1 = std::get<0>(comp[0]);
            int i2 = std::get<1>(comp[0]);
            auto q1 = qs1[i1].first;
            auto q2 = qs2[i2].first;
            qsym sym1(3,std::get<0>(q1),std::get<1>(q1));
            qsym sym2(3,std::get<0>(q2),std::get<1>(q2));
            for(int i=0; i<comp.size(); i++){
               int i1 = std::get<0>(comp[i]);
               int i2 = std::get<1>(comp[i]);
               auto q1 = qs1[i1].first;
               auto q2 = qs2[i2].first;
               qsym sym1new(3,std::get<0>(q1),std::get<1>(q1));
               qsym sym2new(3,std::get<0>(q2),std::get<1>(q2));
               // new combination is found
               if(sym1 != sym1new || sym2 != sym2new){
                  sym1 = sym1new;
                  sym2 = sym2new;
                  unique_ns.push_back(i);
               }
            }
            if(debug_level>0){ 
               std::cout << "\nsym=" << sym << std::endl;
               tools::print_vector(unique_ns,"unique_ns");
            }

            // loop over unique (N1,S1),(N2,S2) combinations
            for(int i=0; i<unique_ns.size(); i++){
               int idx = unique_ns[i];
               int i1 = std::get<0>(comp[idx]);
               int i2 = std::get<1>(comp[idx]);
               auto q1 = qs1[i1].first;
               auto q2 = qs2[i2].first;
               qsym sym1(3,std::get<0>(q1),std::get<1>(q1));
               qsym sym2(3,std::get<0>(q2),std::get<1>(q2));
               int d1 = qs1[i1].second;
               int d2 = qs2[i2].second;
               // locate the block
               int brow = prow.at(sym);
               int bmid = pmid.at(sym1);
               int bcol = pcol.at(sym2);
               int tsi = ts; // intermediate spin is ts because site sym has S=0. 
               auto blk = site(brow,bcol,bmid,tsi);
               size_t offcr = std::get<2>(comp[idx]); 
               int drow = qrow.get_dim(brow);
               assert(drow == ymat.rows());
               int dmid = d1;
               int dcol = d2;
               size_t N = drow*dcol*dmid;
               const Tm* xptr = ymat.data() + offcr*drow;
               linalg::xcopy(N, xptr, blk.data());

               if(debug_level>0){
                  std::cout << "i=" << i << " sym1=" << sym1 << " sym2=" << sym2 << std::endl;
                  std::cout << "br,bc,bm=" << brow << "," << bcol << "," << bmid
                     << " dr,dc,dm=" << drow << "," << dcol << "," << dmid
                     << " symr=" << qrow.get_sym(brow)
                     << " symc=" << qcol.get_sym(bcol)
                     << " symm=" << qmid.get_sym(bmid)
                     << " N=" << N << " offcr=" << offcr 
                     << std::endl;
                  std::cout << ymat.rows() << "," << ymat.cols() << std::endl;
                  blk.print("blk");
               }
            }
         }
         
         if(debug_level>0){ 
            std::cout << "\nfinal site:" << std::endl;
            site.print("site",2);
         }

         return site;
      }

   // deal with the final rwfun
   template <typename Tm>
      std::vector<stensor2su2<Tm>> updateRWFuns(const comb<qkind::qNSz,Tm>& icomb_NSz,
            const Wmatrix<Tm>& wmat,
            const int twos,
            const bool debug=true){
         const Tm alpha = 1.0, beta = 0.0;
         if(debug) std::cout << "\nctns::updateRWFuns twos=" << twos << std::endl;

         // 1. just analyze the population
         // rwfunsW[i,b] = rwfuns[i,a]*wmat[a,b] 
         const auto& rwfuns = icomb_NSz.rwfuns;
         int nroot = rwfuns.size();
         for(int iroot=0; iroot<nroot; iroot++){
            if(debug) std::cout << "\niroot=" << iroot << std::endl;
            assert(rwfuns[iroot].info.qcol == wmat.qrow);
            assert(rwfuns[iroot].rows()==1);
            // find the block
            int b = 0;
            for(b=0; b<rwfuns[iroot].cols(); b++){
               if(rwfuns[iroot].info.qrow.get_sym(0) == 
                     rwfuns[iroot].info.qcol.get_sym(b)) break;
            }
            // rwfuns[iroot].dot(wmat);
            double pop = 0.0;
            auto blkr = rwfuns[iroot](0,b);
            for(int j=0; j<wmat.qcol.size(); j++){
               size_t offw = wmat._offset.at(std::make_pair(b,j));
               if(offw == 0) continue;
               const Tm* blkw = wmat._data.data() + offw-1;
               int d0 = 1;
               int di = wmat.qrow.get_dim(b);
               int dj = wmat.qcol[j].second;
               linalg::matrix<Tm> rwfunW(d0,dj);
               linalg::xgemm("N", "N", d0, dj, di, alpha,
                     blkr.data(), d0, blkw, di, beta,
                     rwfunW.data(), d0);
               // check
               auto pop_j = std::pow(rwfunW.normF(),2); 
               pop += pop_j; 
               if(debug){
                  auto symj = wmat.qcol[j].first;
                  std::cout << "# j=" << j << " sym=(" << std::get<0>(symj) << ","
                     << std::get<1>(symj) << "," << std::get<2>(symj) << ")";
                  if(std::get<1>(symj) == twos) std::cout << " target sector!";
                  std::cout << std::endl;
                  rwfunW.print("rwfunW");
                  std::cout << " pop[j]=" << std::setprecision(10) << pop_j << std::endl;
               }
            } // j
            if(debug) std::cout << "total pop=" << std::setprecision(10) << pop 
               << " diff(1-pop)=" << 1.0-pop << std::endl;
         } // iroot

         // 2. projection
         auto wf2 = icomb_NSz.get_wf2();
         auto sym_state = icomb_NSz.get_sym_state();
         int n = sym_state.ne();
         int tm = sym_state.tm();
         int nstate;
         linalg::matrix<Tm> wf2new;
         for(int j=0; j<wmat.qcol.size(); j++){
            const auto& sym3 = wmat.qcol[j].first;
            size_t offw = wmat._offset.at(std::make_pair(0,j));
            if(offw == 0) continue;
            const Tm* blkw = wmat._data.data() + offw-1;
            // check target symmetry (N,S,M)
            if(n == std::get<0>(sym3) && 
                  twos == std::get<1>(sym3) && 
                  tm == std::get<2>(sym3)){
               int di = wmat.qrow.get_dim(0); 
               int dj = wmat.qcol[j].second;
               // possible number of states, ns can be smaller than nroot,
               // when the truncation threshold is very large! 
               nstate = std::min(nroot,dj);
               linalg::matrix<Tm> rwfunW(nstate,dj);
               // rW(n,j) = rwfun(n,i)*W(i,j)
               linalg::xgemm("N", "N", nstate, dj, di, alpha,
                     wf2.data(), nroot, blkw, di, beta,
                     rwfunW.data(), nstate);
               // SVD
               std::vector<double> s;
               linalg::matrix<Tm> U, Vt;
               linalg::svd_solver(rwfunW, s, U, Vt, 13);
               // lowdin orthonormalization: wf2new = U*Vt 
               wf2new = linalg::xgemm("N","N",U,Vt);
            }
         } // j
         if(wf2new.size() == 0){
            std::cout << "error: no such wavefunction with target twos=" << twos 
               << " change twos instead!" << std::endl;
            exit(1); 
         }

         // 3. assemble vector form of rwfuns
         qsym vac_sym(3,0,0);
         qsym state_sym(3,n,twos);
         qbond qrow({{state_sym,1}});
         auto qcolInfo = qbond3_to_qbond(wmat.qcol);
         const auto& qcol = qcolInfo.first;
         const auto& pcol = qcolInfo.second;
         int jdx = pcol.at(state_sym);
         std::vector<stensor2su2<Tm>> rwfuns_new(nstate);
         for(int i=0; i<nstate; i++){
            stensor2su2<Tm> state(vac_sym, qrow, qcol, dir_RWF);
            for(int ic=0; ic<wf2new.cols(); ic++){
               state(0,jdx)(0,ic) = wf2new(i,ic); 
            }
            rwfuns_new[i] = std::move(state);
         } // i

         return rwfuns_new;
      }

} // ctns

#endif
