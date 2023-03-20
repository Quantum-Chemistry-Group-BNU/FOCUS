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
            void init(const Rlist<Tm>& Rlst, 
                  const int _batchgemm,
                  const size_t _batchsize,
                  const size_t offset,
                  const int hdxorder);
            // perform GEMMs [c2,c1,r,l]
            void kernel(const int k, Tm** ptrs){
/*
               struct timeval t0, t1;
               for(int i=0; i<mmbatch2[k].size(); i++){
                  gettimeofday(&t0, NULL);
                  mmbatch2[k][i].kernel(batchgemm, ptrs);
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
*/
            }
            // reduction
            void reduction(const int k, Tm* workspace, Tm* y, const int iop) const{
/*
               struct timeval t0, t1;
               gettimeofday(&t0, NULL);
               // 1. collect O(r,r') += \sum_c O(r,r',c)
               if(icase == 1){
                  if(iop == 0){
                     for(size_t j=0; j<jlen; j++){
                        int dimc = collect[k][j].first;
                        size_t size = collect[k][j].second;
                        Tm* ptr = workspace+j*offset; 
                        for(int i=1; i<dimc; i++){
                           linalg::xaxpy(size, ptr+i*size, ptr);
                        }
                     }
#ifdef GPU
                  }else if(iop == 1){
                     std::cout << "NOT IMPLEMENTED YET!" << std::endl;
                     exit(1);
                  } // iop
#endif
               } // icase
               // 2. reduction by GEMV
               int batchgemv = iop+1;
               Tm* ptrs[3];
               ptrs[0] = workspace;
               ptrs[1] = &coefflst[k];
               ptrs[2] = y;
               mvbatch[k].kernel(batchgemv, ptrs);
#ifdef GPU
#ifdef USE_HIP
               hipDeviceSynchronize();
#else
               cudaDeviceSynchronize();
#endif
#endif
               gettimeofday(&t1, NULL);
               oper_timer.renorm.tHx[8] += ((double)(t1.tv_sec - t0.tv_sec) 
                     + (double)(t1.tv_usec - t0.tv_usec)/1000000.0);
*/
            }
         public:
            int icase, batchgemm;
            size_t totsize, batchsize, offset, nbatch;
            double cost = 0.0;
            std::vector<std::vector<MMbatch<Tm>>> mmbatch2; // mmbatch2[ibatch][icase]
            std::vector<std::vector<std::pair<int,size_t>>> collectc; // icase=1 
            std::vector<std::vector<Tm>> coefflst;
            std::vector<MVbatch<Tm>> mvbatch;
      };

   template <typename Tm>
      void RMMtask<Tm>::init(const Rlist<Tm>& Rlst,
            const int _batchgemm,
            const size_t _batchsize,
            const size_t _offset,
            const int hxorder){
         // init
         batchgemm = _batchgemm;
         totsize = Rlst.size();
         batchsize = _batchsize;
         offset = _offset;
         if(batchsize == 0 && totsize !=0){
            std::cout << "error: inconsistent batchsize & totsize!" << std::endl;
            std::cout << "batchsize=" << batchsize << " totsize=" << totsize << std::endl;
            exit(1);
         }
         if(batchsize == 0) return;
         nbatch = totsize/batchsize;
         if(totsize%batchsize != 0) nbatch += 1; // thus, this works even for totsize < batchsize
         icase = Rlst[0].icase; 

         // start process Rlst
         mmbatch2.resize(nbatch);
         collectc.resize(nbatch);
         coefflst.resize(nbatch);
         mvbatch.resize(nbatch);
         for(int k=0; k<nbatch; k++){
            size_t off = k*batchsize;
            size_t jlen = std::min(totsize-off, batchsize);

            // initialization
            int nd = 7;
            int pos[4] = {0,1,2};
            std::vector<size_t> dims(nd,0);
            // count how many gemms in each case 
            for(size_t j=0; j<jlen; j++){
               size_t jdx = off+j;
               const auto& Rblk = Rlst[jdx];
               pos[0] = Rblk.dagger[2]? 0 : 1;
               pos[1] = Rblk.dagger[1]? 2 : 3;
               pos[2] = Rblk.dagger[0]? 4 : 5;
               dims[pos[0]] += Rblk.identity(2)? 0 : 1; // c
               dims[pos[1]] += Rblk.identity(1)? 0 : Rblk.dimout[2]; // r
               dims[pos[2]] += Rblk.identity(0)? 0 : 1; // l
               dims[6] += (icase == 1)? Rblk.dimin2[2] : 1; 
            }

            // setup mmbatch2[k]
            // generation of mmlst2
            MMlist2<Tm> mmlst2(nd);
            for(int i=0; i<nd; i++){
               mmlst2[i].resize(dims[i]); // c,r,l,psi
            }
            std::vector<size_t> idx(nd,0);
            for(size_t j=0; j<jlen; j++){
               size_t jdx = off+j;
               const auto& Rblk = Rlst[jdx];
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

            // information for collect O(r,r') += \sum_c O(r,r',c)
            if(icase == 1){
               collectc[k].resize(jlen);
               for(size_t j=0; j<jlen; j++){
                  size_t jdx = off+j;
                  const auto& Rblk = Rlst[jdx];
                  collectc[k][j] = std::make_pair(Rblk.dimin2[2], Rblk.size);
               } // j
            }

            MVlist<Tm> mvlst; 
            int nmu = 0;
            size_t offrop = Rlst[off].offrop;
            for(size_t j=0; j<jlen; j++){
               size_t jdx = off+j;
               const auto& Rblk = Rlst[jdx];
               coefflst[k][j] = Rblk.coeff;
               if(Rblk.offrop == offrop){
                  nmu += 1;
               }else{
                  // append into mvbatch
                  MVinfo<Tm> mv;
                  mv.transA = 'N';
                  mv.M = Rblk.size;
                  mv.N = nmu;
                  mv.LDA = offset;
                  mv.locA = 0;
                  mv.offA = (j-nmu)*offset;
                  mv.locx = 1;
                  mv.offx = (j-nmu);
                  mv.locy = 2;
                  mv.offy = offrop;
                  mvlst.push_back(mv);
                  // new 
                  nmu = 0;
                  offrop = Rblk.offrop;
               }
            } // j
            // append into mvbatch
            const auto& Rblk = Rlst[jlen-1];
            MVinfo<Tm> mv;
            mv.transA = 'N';
            mv.M = Rblk.size;
            mv.N = nmu;
            mv.LDA = offset;
            mv.locA = 0;
            mv.offA = (jlen-nmu)*offset;
            mv.locx = 1;
            mv.offx = (jlen-nmu);
            mv.locy = 2;
            mv.offy = offrop;
            mvlst.push_back(mv);
            mvbatch[k].init(mvlst);
         } // k
      }

   template <typename Tm>
      using RMMtasks = std::vector<RMMtask<Tm>>;

} // ctns

#endif
