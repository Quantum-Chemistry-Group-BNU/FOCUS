#ifndef PREPROCESS_RMMTASK_H
#define PREPROCESS_RMMTASK_H

#include "preprocess_rlist.h"
#include "preprocess_mmbatch.h"
#include "preprocess_mmreduce.h"

#include "time.h"
#include "sys/time.h"
#include "oper_timer.h"

namespace ctns{

   // Interface to Rlist
   template <typename Tm>
      struct RMMtask{
         public:
            void init(Rlist<Tm>& Rlst, 
                  const int hdxorder,
                  const int _batchblas,
                  const size_t _batchsize,
                  const size_t offset,
                  const size_t offset0=0);
            // form intermeidate operators
            void inter(const int k, Tm** opaddr){
               struct timeval t0, t1;
               // perform GEMV_BATCH
               Tm* ptrs[6];
               ptrs[0] = opaddr[0];
               ptrs[1] = opaddr[1];
               ptrs[2] = opaddr[2];
               ptrs[3] = opaddr[3];
               ptrs[4] = opaddr[4];
               ptrs[5] = alphavec[k].data(); 
               gettimeofday(&t0, NULL);
               imvbatch[k].kernel(batchblas, ptrs);
#ifdef GPU
#ifdef USE_HIP
               hipDeviceSynchronize();
#else
               cudaDeviceSynchronize();
#endif
#endif
               gettimeofday(&t1, NULL);
               oper_timer.renorm.tInter += ((double)(t1.tv_sec - t0.tv_sec) 
                     + (double)(t1.tv_usec - t0.tv_usec)/1000000.0);
            }
            // perform GEMMs [c2,c1,r,l]
            void kernel(const int k, Tm** ptrs){
               struct timeval t0, t1;
               for(int i=0; i<mmbatch2[k].size(); i++){
                  gettimeofday(&t0, NULL);
                  mmbatch2[k][i].kernel(batchblas, ptrs);
#ifdef GPU
#ifdef USE_HIP
                  hipDeviceSynchronize();
#else
                  cudaDeviceSynchronize();
#endif
#endif
                  gettimeofday(&t1, NULL);
                  oper_timer.renorm.tHx[i] += ((double)(t1.tv_sec - t0.tv_sec) 
                        + (double)(t1.tv_usec - t0.tv_usec)/1000000.0);
                  oper_timer.renorm.cHx[i] += mmbatch2[k][i].cost; 
               } // i
            }
            // reduction
            void reduction(const int k, Tm* workspace, Tm* y){
               struct timeval t0, t1;
               gettimeofday(&t0, NULL);

               // 1. collect O(r,r') += \sum_c O(r,r',c) [axpy_batch]
               if(icase == 1){
                  const Tm alpha = 1.0;
                  if(batchblas == 0 || batchblas == 1){
                     for(size_t j=0; j<collectc[k].size(); j++){
                        int dimc = collectc[k][j].first;
                        size_t size = collectc[k][j].second;
                        Tm* ptr = workspace+j*offset; 
                        for(int i=1; i<dimc; i++){
                           linalg::xaxpy(size, alpha, ptr+i*size, ptr);
                        }
                     }
#ifdef GPU
                  }else if(batchblas == 2){
                     std::cout << "NOT IMPLEMENTED YET!" << std::endl;
                     exit(1);
#endif
                  } // batchblas
               } // icase
               
               // 2. reduction by GEMV
               Tm* ptrs[3];
               ptrs[0] = workspace;
               ptrs[1] = const_cast<Tm*>(coefflst[k].data());
               ptrs[2] = y;
               mvbatch[k].kernel(batchblas, ptrs);
#ifdef GPU
#ifdef USE_HIP
               hipDeviceSynchronize();
#else
               cudaDeviceSynchronize();
#endif
#endif
               gettimeofday(&t1, NULL);
               oper_timer.renorm.tHx[7] += ((double)(t1.tv_sec - t0.tv_sec) 
                     + (double)(t1.tv_usec - t0.tv_usec)/1000000.0);
            }
         public:
            int icase = -1, batchblas = -1;
            size_t totsize = 0, batchsize = 0, offset = 0, nbatch = 0;
            double cost = 0.0;
            std::vector<std::vector<MMbatch<Tm>>> mmbatch2; // mmbatch2[ibatch][icase]
            std::vector<std::vector<std::pair<int,size_t>>> collectc; // icase=1 
            std::vector<std::vector<Tm>> coefflst;
            std::vector<MVbatch<Tm>> mvbatch;
            // --- intermediates [Direct] --- 
            std::vector<std::vector<Tm>> alphavec; 
            std::vector<MVbatch<Tm>> imvbatch;
      };

   template <typename Tm>
      void RMMtask<Tm>::init(Rlist<Tm>& Rlst,
            const int hxorder,
            const int _batchblas,
            const size_t _batchsize,
            const size_t _offset,
            const size_t offset0){
         // init
         batchblas = _batchblas;
         batchsize = _batchsize;
         totsize = Rlst.size();
         offset = _offset;
         if(batchsize == 0 && totsize !=0){
            std::cout << "error: inconsistent batchsize & totsize!" << std::endl;
            std::cout << "batchsize=" << batchsize << " totsize=" << totsize << std::endl;
            exit(1);
         }
         if(batchsize == 0 || totsize == 0) return;
         nbatch = totsize/batchsize;
         if(totsize%batchsize != 0) nbatch += 1; // thus, this works even for totsize < batchsize
         
         // start process Rlst
         icase = Rlst[0].icase; 
         mmbatch2.resize(nbatch);
         collectc.resize(nbatch);
         coefflst.resize(nbatch);
         mvbatch.resize(nbatch);
         alphavec.resize(nbatch);
         imvbatch.resize(nbatch);
         for(int k=0; k<nbatch; k++){
            size_t off = k*batchsize;
            size_t jlen = std::min(totsize-off, batchsize);
         
            // 1. setup imvbatch for inter
            if(offset0 != 0){
               size_t nInter = 0;
               size_t dalpha = 0;
               for(size_t j=0; j<jlen; j++){
                  size_t jdx = off+j;
                  const auto& Rblk = Rlst[jdx];
                  if(Rblk.posInter == -1) continue;
                  nInter += 1;
                  dalpha += Rblk.alpha_vec.size();
               }
               if(nInter > 0){
                  MVlist<Tm> mvlst(nInter); 
                  alphavec[k].resize(dalpha);
                  size_t idx = 0, adx = 0;
                  for(size_t j=0; j<jlen; j++){
                     size_t jdx = off+j;
                     auto& Rblk = Rlst[jdx];
                     if(Rblk.posInter == -1) continue;
                     int ipos = Rblk.posInter;
                     int len = Rblk.alpha_vec.size();
                     linalg::xcopy(len, Rblk.alpha_vec.data(), &alphavec[k][adx]);
                     size_t opsize = Rblk.dimout[ipos]*Rblk.dimin[ipos];
                     MVinfo<Tm> mv;
                     mv.transA = 'N';
                     mv.M = opsize;
                     mv.N = len;
                     mv.LDA = Rblk.ldaInter; 
                     mv.locA = ipos;
                     mv.offA = Rblk.off[ipos];
                     mv.locx = 5;
                     mv.offx = adx;
                     mv.locy = locInter;
                     mv.offy = j*offset0;
                     mvlst[idx] = mv; 
                     adx += len;
                     idx += 1;
                     Rblk.off[ipos] = j*offset0; // overwrite old position
                  } // j
                  imvbatch[k].init(mvlst);
               }
            }

            // 2. setup mmbatch2[k]
            const int nd = 7;
            std::vector<size_t> dims(nd,0);
            // count how many gemms in each case 
            for(size_t j=0; j<jlen; j++){
               size_t jdx = off+j;
               const auto& Rblk = Rlst[jdx];
               int pos[3];
               pos[0] = Rblk.dagger[2]? 0 : 1;
               pos[1] = Rblk.dagger[1]? 2 : 3;
               pos[2] = Rblk.dagger[0]? 4 : 5;
               dims[pos[0]] += Rblk.identity(2)? 0 : 1; // c
               dims[pos[1]] += Rblk.identity(1)? 0 : Rblk.dimout[2]; // r
               dims[pos[2]] += Rblk.identity(0)? 0 : 1; // l
               dims[6] += (icase == 1)? Rblk.dimin2[2] : 1; 
            }

            // generation of mmlst2
            MMlist2<Tm> mmlst2(nd);
            for(int i=0; i<nd; i++){
               mmlst2[i].resize(dims[i]); // c,r,l,psi
            }
            std::vector<size_t> idx(nd,0);
            for(size_t j=0; j<jlen; j++){
               size_t jdx = off+j;
               const auto& Rblk = Rlst[jdx];
               int pos[4];
               pos[0] = Rblk.dagger[2]? 0 : 1;
               pos[1] = Rblk.dagger[1]? 2 : 3;
               pos[2] = Rblk.dagger[0]? 4 : 5;
               pos[3] = 6;
               MMlist2<Tm> mmtmp2(4);
               Rblk.get_MMlist_onedot(mmtmp2, j*offset, true);
               for(int i=0; i<mmtmp2.size(); i++){
                  int ipos = pos[i];
                  // copy the mmlst to the correct place
                  for(int k=0; k<mmtmp2[i].size(); k++){
                     mmlst2[ipos][idx[ipos]] = mmtmp2[i][k];
                     idx[ipos]++;
                  } //k
               } // i
            } // j

            // convert to batch list
            mmbatch2[k].resize(nd);
            for(int i=0; i<nd; i++){
               if(hxorder == 1){ // sort by cost
                  std::stable_sort(mmlst2[i].begin(), mmlst2[i].end(),
                        [](const MMinfo<Tm>& mm1, const MMinfo<Tm>& mm2){
                        return mm1.cost() > mm2.cost();
                        });
               }
               mmbatch2[k][i].init(mmlst2[i]);
               cost += mmbatch2[k][i].cost; // accumulate MM cost
            } // i

            /*
            std::cout << "mmbatch:" << std::endl;
            for(int i=0; i<nd; i++){
               std::cout << "I=" << i << " size=" << mmbatch2[k][i].size << std::endl;
               for(int j=0; j<mmbatch2[k][i].size; j++){
                  std::cout << " j=" << j << " M,N,K="
                     << mmbatch2[k][i].M[j] << ","
                     << mmbatch2[k][i].N[j] << ","
                     << mmbatch2[k][i].K[j] << std::endl;
               }
            }
            */

            // 3. setup mvbatch[k]
            // information for collect O(r,r') += \sum_c O(r,r',c)
            if(icase == 1){
               collectc[k].resize(jlen);
               for(size_t j=0; j<jlen; j++){
                  size_t jdx = off+j;
                  const auto& Rblk = Rlst[jdx];
                  collectc[k][j] = std::make_pair(Rblk.dimin2[2], Rblk.size);
               } // j
            }

            coefflst[k].resize(jlen);
            MVlist<Tm> mvlst; 
            int nmu = 0;
            size_t offrop = Rlst[off].offrop;
            for(size_t j=0; j<jlen; j++){
               size_t jdx = off+j;
               const auto& Rblk = Rlst[jdx];
               coefflst[k][j] = Rblk.coeff;
               /*
               std::cout << "j=" << j 
                  << " Rblk.size=" << Rblk.size
                  << " Rblk.offrop=" << Rblk.offrop 
                  << " offset=" << offset
                  << " nmu=" << nmu
                  << " coeff=" << coefflst[k][j]
                  << std::endl;
               */ 
               if(Rblk.offrop != offrop){
                  // append into mvbatch
                  MVinfo<Tm> mv;
                  mv.transA = 'N';
                  mv.M = Rlst[jdx-1].size;
                  mv.N = nmu;
                  mv.LDA = offset;
                  mv.locA = 0;
                  mv.offA = (j-nmu)*offset;
                  mv.locx = 1;
                  mv.offx = (j-nmu);
                  mv.locy = 2;
                  mv.offy = Rlst[jdx-1].offrop;
                  mvlst.push_back(mv);
                  // new 
                  nmu = 1;
                  offrop = Rblk.offrop;
               }else{
                  nmu += 1;
               }
            } // j
            // append into mvbatch
            MVinfo<Tm> mv;
            mv.transA = 'N';
            mv.M = Rlst[off+jlen-1].size;
            mv.N = nmu;
            mv.LDA = offset;
            mv.locA = 0;
            mv.offA = (jlen-nmu)*offset;
            mv.locx = 1;
            mv.offx = (jlen-nmu);
            mv.locy = 2;
            mv.offy = Rlst[off+jlen-1].offrop;
            mvlst.push_back(mv);
            const Tm beta = 1.0;
            mvbatch[k].init(mvlst, beta);
            /*
            // debug:
            std::cout << "mvbatch:" << mvbatch[k].size << std::endl;
            for(int i=0; i<mvbatch[k].size; i++){
               std::cout << "i=" << i << " M,N="
                  << mvbatch[k].M[i] << ","
                  << mvbatch[k].N[i] << " Aloc,off="
                  << mvbatch[k].locA[i] << "," << mvbatch[k].offA[i] << " xloc,off="
                  << mvbatch[k].locx[i] << "," << mvbatch[k].offx[i] << " yloc,off="
                  << mvbatch[k].locy[i] << "," << mvbatch[k].offy[i] 
                  << std::endl;
            }
            */
         } // k
      }

   template <typename Tm>
      using RMMtasks = std::vector<RMMtask<Tm>>;

} // ctns

#endif
