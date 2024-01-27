#ifndef CTNS_TOSU2_DM_H
#define CTNS_TOSU2_DM_H

#include "ctns_tosu2_csite.h"

namespace ctns{

   // The general DM has offdiagonal blocks, e.g., considering
   // |psi[S=1]> = |L[s=1]>*(|R[s=0]> + |R[s=1]> + |R[s=2]>).
   // When considering the DM in the right space, DM is not block diagonal.
   // However, for decimation only the block diagonal part is necessary!
   template <typename Tm>
      struct CoupledDM{
         public:
            CoupledDM(const qbond3& _qrow, const qbond3& _qcol);
         public:
            qbond3 qrow;
            qbond3 qcol;
            // derived
            size_t _size;
            std::map<std::pair<int,int>,size_t> _offset;
            std::vector<Tm> _data;
      };

   template <typename Tm>
      CoupledDM<Tm>::CoupledDM(const qbond3& _qrow, const qbond3& _qcol){
         qrow = _qrow;
         qcol = _qcol;
         // nonzero blocks
         _size = 1;
         for(int i=0; i<qrow.size(); i++){
            auto symi = qrow[i].first;
            int di = qrow[i].second;
            for(int j=0; j<qcol.size(); j++){
               auto symj = qcol[j].first;
               int dj = qcol[j].second;
               // symmetry conservation
               if(std::get<0>(symi) == std::get<0>(symj) &&
                     std::get<1>(symi) == std::get<1>(symj) &&
                     std::get<2>(symi) == std::get<2>(symj)){
                  _offset[std::make_pair(i,j)] = _size;
                  _size += di*dj;
               }else{
                  _offset[std::make_pair(i,j)] = 0;
               }
            }
         }
         _size -= 1;
         _data.resize(_size);
         memset(_data.data(), 0, _size*sizeof(Tm));
      }

   template <typename Tm>
      CoupledDM<Tm> formCoupledDM(const Wmatrix<Tm>& csite,
            const stensor2<Tm>& rhol,
            const bool debug=true){
         if(debug) std::cout << "\nctns::formCoupledDM" << std::endl;
         const Tm alpha = 1.0, beta0 = 0.0, beta1=1.0; 
         const auto& qrow = csite.qrow;
         const auto& qcol = csite.qcol;
         CoupledDM<Tm> cdm(qcol, qcol);
         // DM = (B*rhol*B^+), B=A^T => DM = A^TA^* 
         double tr = 0.0;
         for(int j=0; j<qcol.size(); j++){
            int dj = qcol[j].second;
            size_t off = cdm._offset[std::make_pair(j,j)];
            if(off == 0) continue;
            Tm* blkDM = cdm._data.data() + off-1;
            // DM[J,J] = \sum_I (A^+[J,I] * rho[I,I]^* * A[I,J])^* at the symmetry block level
            //         = \sum_I (A^+[J,I] * rho[I,I]^T * A[I,J])^* as rho is Hermitian
            for(int i=0; i<qrow.size(); i++){
               int di = qrow.get_dim(i);
               size_t offA = csite._offset.at(std::make_pair(i,j));
               if(offA == 0) continue;
               auto blkD = rhol(i,i);
               if(blkD.empty()) continue;
               const Tm* blkA = csite._data.data() + offA-1; 
               linalg::matrix<Tm> rhoA(di,dj);
               // rho[I,I]^T * A[I,J]
               linalg::xgemm("T", "N", di, dj, di, alpha,
                     blkD.data(), di, blkA, di, beta0,
                     rhoA.data(), di);
               // A^+[J,I] * rhoA[I,J] 
               linalg::xgemm("C", "N", dj, dj, di, alpha,
                     blkA, di, rhoA.data(), di, beta1,
                     blkDM, dj);

               /*
               std::cout << "> i=" << i << std::endl;
               blkD.print("blkD");
               linalg::matrix<Tm> amat(di,dj,blkA);
               amat.print("amat");
               rhoA.print("rhoA");
               linalg::matrix<Tm> mat(dj,dj,blkDM);
               mat.print("blkDM");
               */
            } // i

            /*
            linalg::matrix<Tm> mat(dj,dj,blkDM);
            std::cout << "final j=" << j << " off=" << off << std::endl;
            mat.print("blkDM");
            tr += std::real(mat.trace());
            std::cout << "tr=" << tr << std::endl;
            */

         } // j
         linalg::xconj(cdm._size, cdm._data.data());

         /*
         std::cout << "tr(rho)=" << tr << std::endl;
         if(std::abs(tr-1.0)>1.e-6){
            std::cout << "error: tr(rho) != 1, diff=" << tr-1.0 << std::endl;
            //exit(1);
         }
         */

         return cdm;
      }

   template <typename Tm>
      std::map<qsym,linalg::matrix<Tm>> decimQuasiDM(const CoupledDM<Tm>& cdm,
            const double thresh_tosu2,
            const bool debug=true){
         if(debug) std::cout << "\nctns::decimQuasiDM thresh_tosu2=" << thresh_tosu2 << std::endl;
         // preprocess qrow, use qmap store indices of symmetry sectors (N,S,M)
         // sharing the same (N,S), which will be state-averaged in quasi-DM 
         const auto& qrow = cdm.qrow;
         std::vector<qsym> syms;
         std::map<qsym,std::vector<int>> qmap;
         for(int i=0; i<qrow.size(); i++){
            auto qr = qrow[i].first;
            qsym sym({3,std::get<0>(qr),std::get<1>(qr)});
            qmap[sym].push_back(i);
         }
         // loop over symmetry sectors
         std::map<qsym,linalg::matrix<Tm>> Yinfo;
         qbond ys;
         double sigs2sum = 0.0;
         for(const auto& pr : qmap){
            const auto& sym = pr.first;
            const auto& comp = pr.second;
            int dim = qrow[comp[0]].second;
            
            //std::cout << "\nsym=" << sym << " dim=" << dim << std::endl;
            
            // form quasi-RDM by averaging different M
            linalg::matrix<Tm> qdm(dim,dim);
            int N = dim*dim;
            for(int i=0; i<comp.size(); i++){
               int idx = comp[i];
               
               //std::cout << " i=" << i << " comp[i]=" << idx << std::endl;
               size_t off = cdm._offset.at(std::make_pair(idx,idx));
               if(off == 0) continue;
               const Tm* cptr = cdm._data.data() + off-1; 

               /*
               std::cout << "off=" << off << std::endl;
               linalg::matrix<Tm> cdmBLK(dim,dim,cptr);
               cdmBLK.print("cdmBLK");
               */

               linalg::xaxpy(N, 1.0, cptr, qdm.data());
            }
            // decimation
            std::vector<double> sigs2(dim);
            linalg::matrix<Tm> U;
            //qdm.print("qdm");
            linalg::eig_solver(qdm, sigs2, U, 1);
            //tools::print_vector(sigs2,"sigs2");
            
            // selection of important states 
            int nkept = 0;
            for(int k=0; k<sigs2.size(); k++){
               if(sigs2[k]<thresh_tosu2){
                  break;
               }else{
                  nkept += 1;
               }
            }
            //std::cout << "nkept=" << nkept << std::endl;
            if(nkept == 0) continue;
            sigs2sum += std::accumulate(sigs2.begin(), sigs2.begin()+nkept, 0.0);
            // save
            linalg::matrix<Tm> Ukept(dim, nkept, U.data());
            Yinfo[sym] = Ukept.T();
            ys.dims.push_back(std::make_pair(sym,nkept));
         }
         ys.print("ys");
         std::cout << "sigs2sum=" << sigs2sum << std::endl;
         if(std::abs(sigs2sum-1.0)>1.e-6){
            std::cout << "sigs2sum is not 1! diff=" << sigs2sum-1.0 << std::endl;
            //exit(1);
         }
         return Yinfo;
      }

} // ctns

#endif
