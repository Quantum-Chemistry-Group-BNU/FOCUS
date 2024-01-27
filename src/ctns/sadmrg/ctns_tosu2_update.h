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
            qsym sym({3,std::get<0>(sym3),std::get<1>(sym3)});
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
            qsym sym({3,std::get<0>(sym3),std::get<1>(sym3)});
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
         if(debug) std::cout << "\nctns::updateSite" << std::endl;
         const auto& dpt = qprod.second;
         auto qmidInfo = qbond3_to_qbond(qs1);
         auto qcolInfo = qbond3_to_qbond(qs2);
         const auto& qmid = qmidInfo.first;
         const auto& pmid = qmidInfo.second;
         const auto& qcol = qcolInfo.first;
         const auto& pcol = qcolInfo.second;
         qmid.print("qmid"); 
         qcol.print("qcol");
         // 1. form qrow
         qbond qrow;
         std::map<qsym,int> prow;
         int idx = 0;
         for(const auto& pr : Yinfo){
            qrow.dims.push_back(std::make_pair(pr.first,pr.second.rows()));
            prow[pr.first] = idx;
            idx += 1;
         }
         qrow.print("qrow");
         // 2. assemble site
         stensor3su2<Tm> site(qsym({3,0,0}),qrow,qcol,qmid,dir_RCF,CRcouple);
         site.print("site",2);
         for(const auto& pr : Yinfo){
            const auto& sym = pr.first;
            const auto& ymat = Yinfo.at(sym);
            std::cout << "sym=" << sym << std::endl;
            int n = sym.ne();
            int ts = sym.ts();
            qsym3 sym3({n,ts,ts}); // find the high-spin case
            const auto& comp = dpt.at(sym3);
            // loop over combination
            std::vector<int> unique_ns(1,0);
            int i1 = std::get<0>(comp[0]);
            int i2 = std::get<1>(comp[0]);
            auto q1 = qs1[i1].first;
            auto q2 = qs2[i2].first;
            qsym sym1({3,std::get<0>(q1),std::get<1>(q1)});
            qsym sym2({3,std::get<0>(q2),std::get<1>(q2)});
            for(int i=0; i<comp.size(); i++){
               int i1 = std::get<0>(comp[i]);
               int i2 = std::get<1>(comp[i]);
               auto q1 = qs1[i1].first;
               auto q2 = qs2[i2].first;
               qsym sym1new({3,std::get<0>(q1),std::get<1>(q1)});
               qsym sym2new({3,std::get<0>(q2),std::get<1>(q2)});
               // new combination is found
               if(sym1 != sym1new || sym2 != sym2new){
                  sym1 = sym1new;
                  sym2 = sym2new;
                  unique_ns.push_back(i);
               }
            }
            tools::print_vector(unique_ns,"unique_ns");
            // loop over unique (N1,S1),(N2,S2) combinations
            for(int i=0; i<unique_ns.size(); i++){
               std::cout << "j=" << i << std::endl;
               int i1 = std::get<0>(comp[i]);
               int i2 = std::get<1>(comp[i]);
               auto q1 = qs1[i1].first;
               auto q2 = qs2[i2].first;
               qsym sym1({3,std::get<0>(q1),std::get<1>(q1)});
               qsym sym2({3,std::get<0>(q2),std::get<1>(q2)});
               int d1 = qs1[i1].second;
               int d2 = qs2[i2].second;
               // locate the block
               std::cout << "sym=" << sym << std::endl;
               int brow = prow.at(sym);
               std::cout << "brow=" << brow << std::endl;
               std::cout << "sym1=" << sym1 << std::endl;
               int bmid = pmid.at(sym1);
               std::cout << "bmid=" << bmid << std::endl;
               std::cout << "sym2=" << sym2 << std::endl;
               int bcol = pcol.at(sym2);
               std::cout << "bcol=" << bcol << std::endl;
               int tsi = ts; // intermediate spin is ts because site sym has S=0. 
               auto blk = site(brow,bcol,bmid,tsi);
               blk.print("blk");
               size_t offcr = std::get<2>(comp[i]); 
               int drow = qrow.get_dim(brow);
               assert(drow == ymat.rows());
               int dmid = d1;
               int dcol = d2;
               size_t N = drow*dcol*dmid;
               const Tm* xptr = ymat.data() + offcr*drow;
               linalg::xcopy(N, xptr, blk.data());
               std::cout << "br,bc,bm=" << brow << "," << bcol << "," << bmid
                  << " dr,dc,dm=" << drow << "," << dcol << "," << dmid
                  << " N=" << N << " offcr=" << offcr 
                  << std::endl;
               std::cout << ymat.rows() << "," << ymat.cols() << std::endl;

            }
         }
         return site;
      }

   // deal with the final rwfun
   template <typename Tm>
      std::vector<stensor2su2<Tm>> updateRWFuns(const comb<qkind::qNSz,Tm>& icomb_NSz,
            const Wmatrix<Tm>& wmat,
            const int twos){
         const Tm alpha = 1.0, beta = 0.0;
         std::cout << "\nctns::updateRWFuns twos=" << twos << std::endl;

         wmat.qrow.print("wmat_qrow");
         display_qbond3(wmat.qcol,"wmat_qcol");

         // 1. analyze the population
         const auto& rwfuns = icomb_NSz.rwfuns;
         int nroot = rwfuns.size();
         for(int iroot=0; iroot<nroot; iroot++){
            std::cout << "\niroot=" << iroot << std::endl;
            assert(rwfuns[iroot].rows()==1 && rwfuns[iroot].cols()==1);
            assert(rwfuns[iroot].info.qcol == wmat.qrow);
            // rwfuns[iroot].dot(wmat);
            double pop = 0.0;
            auto blkr = rwfuns[iroot](0,0);
            for(int j=0; j<wmat.qcol.size(); j++){
               size_t offw = wmat._offset.at(std::make_pair(0,j));
               if(offw == 0) continue;
               const Tm* blkw = wmat._data.data() + offw-1;
               int d0 = 1;
               int di = wmat.qrow.get_dim(0);
               int dj = wmat.qcol[j].second;
               linalg::matrix<Tm> rwfunW(d0,dj);
               linalg::xgemm("N", "N", d0, dj, di, alpha,
                     blkr.data(), d0, blkw, di, beta,
                     rwfunW.data(), d0);
               // check
               auto symj = wmat.qcol[j].first;
               std::cout << "> j=" << j << " sym=(" << std::get<0>(symj) << ","
                  << std::get<1>(symj) << "," << std::get<2>(symj) << ")"
                  << std::endl;
               rwfunW.print("rwfunW");
               auto pop_j = std::pow(rwfunW.normF(),2); 
               std::cout << "pop[j] = " << std::setprecision(10) << pop_j << std::endl;
               pop += pop_j; 
            } // j
            std::cout << "total pop = " << std::setprecision(10) << pop << std::endl;
         } // iroot

         // projection
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
               // rW = rwfun(n,i)*W(i,j)
               linalg::xgemm("N", "N", nstate, dj, di, alpha,
                     wf2.data(), nroot, blkw, di, beta,
                     rwfunW.data(), nstate);
               // SVD
               std::vector<double> s;
               linalg::matrix<Tm> U, Vt;
               linalg::svd_solver(rwfunW, s, U, Vt, 13);
               // lowdin orthonormalization 
               wf2new = linalg::xgemm("N","N",U,Vt);
            }
         } // j

         // 3. assemble rwfuns
         qsym vac_sym({3,0,0});
         qsym state_sym({3,n,twos});
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
