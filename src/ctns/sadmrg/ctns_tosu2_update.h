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
         display_qbond3(qcol,"qcol");
         qbond3 qcol2;
         for(int i=0; i<qcol.size(); i++){
            auto sym3 = qcol[i].first;
            qsym sym({3,std::get<0>(sym3),std::get<1>(sym3)});
            std::cout << "i=" << i
               << " sym=" << sym
               << " sym3=" << std::get<2>(sym3)
               << " qcoli=" << qcol[i].second
               << std::endl;
            // some symmetry sector may not present after decimation
            if(Yinfo.find(sym) != Yinfo.end()){
               assert(qcol[i].second == Yinfo.at(sym).cols()); // since Y=U^T
               qcol2.push_back(std::make_pair(sym3,Yinfo.at(sym).rows()));
            }
         }
         // form Wmatrix
         Wmatrix<Tm> wmat(qrow,qcol2);
         display_qbond3(qcol2,"qcol2");
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

            } // i
            jdx += 1;
         } // j
         return wmat;
      }

   // deal with the final rwfun
   template <typename Tm>
      void finalWaveFunction(const std::vector<stensor2<Tm>>& rwfuns,
            const Wmatrix<Tm>& wmat){
         const Tm alpha = 1.0, beta = 0.0;
         std::cout << "\nctns::finalWaveFunction" << std::endl;
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
               std::cout << "j=" << j << " sym=(" << std::get<0>(symj) << ","
                  << std::get<1>(symj) << "," << std::get<2>(symj) << ")"
                  << std::endl;
               rwfunW.print("rwfunW");
               pop += std::pow(rwfunW.normF(),2); 
            } // j
            std::cout << "pop=" << std::setprecision(10) << pop << std::endl;
         } // iroot
      }

} // ctns

#endif
