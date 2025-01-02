#ifndef PREPROCESS_RMMTASK_H
#define PREPROCESS_RMMTASK_H

#include "preprocess_rlist.h"
#include "preprocess_mmbatch.h"
#include "preprocess_mvbatch.h"

#include "time.h"
#include "sys/time.h"
#include "oper_timer.h"

#ifdef GPU
#include "../gpu/gpu_blas.h"
#endif

namespace ctns{

   // Interface to Rlist
   template <typename Tm>
      struct RMMtask{
         public:
            void clear(){
               mmbatch2.clear();
               collectc.clear();
               collectc.clear();
               mvbatch.clear();
               imvbatch.clear();
            }
            void init(Rlist<Tm>& Rlst, 
                  const int _batchblas,
                  const std::tuple<int,int,int>& _batchrenorm,
                  const size_t _batchsize,
                  const size_t offset,
                  const size_t offset0=0);
            // save dimensions for optimization
            void save(const std::string fmmtask) const{
               // inter: 
               for(int k=0; k<imvbatch.size(); k++){
                  std::string fmmtask_k = fmmtask+"_inter_"+std::to_string(k)+".txt";
                  imvbatch[k].save(fmmtask_k);
               }
               // gemm: no. of batch 
               for(int k=0; k<mmbatch2.size(); k++){
                  // 0,1,2,3,4,5,6
                  for(int i=0; i<mmbatch2[k].size(); i++){
                     std::string fmmtask_k_i = fmmtask+"_"+std::to_string(k)+"."
                        +std::to_string(i)+".txt";
                     mmbatch2[k][i].save(fmmtask_k_i);
                  }
               }
               // red:
               for(int k=0; k<mvbatch.size(); k++){
                  std::string fmmtask_k = fmmtask+"_red_"+std::to_string(k)+".txt";
                  mvbatch[k].save(fmmtask_k);
               }
            }
            void deviceSync() const{
#ifdef GPU
               GPUmem.sync();
#endif
            }
            // form intermeidate operators
            void inter(const int k, Tm** opaddr, const Tm* alphas){
               // perform GEMV_BATCH
               Tm* ptrs[6];
               ptrs[0] = opaddr[0]; // l
               ptrs[1] = opaddr[1]; // r
               ptrs[2] = opaddr[2]; // c1
               ptrs[3] = opaddr[3]; // c2
               ptrs[4] = opaddr[4]; // inter
               ptrs[5] = const_cast<Tm*>(alphas);
               auto t0 = tools::get_time();
               imvbatch[k].kernel(batchinter, ptrs);
               this->deviceSync();
               auto t1 = tools::get_time();
               oper_timer.renorm.t_inter += tools::get_duration(t1-t0);
               oper_timer.renorm.c_inter += imvbatch[k].cost;
            }
            // perform GEMMs [c2,c1,r,l]
            void kernel(const int k, Tm** ptrs){
               assert(mmbatch2[k].size() == 7);
               for(int i=0; i<mmbatch2[k].size(); i++){
                  auto t0 = tools::get_time();
                  mmbatch2[k][i].kernel(batchgemm, ptrs);
                  this->deviceSync();
                  auto t1 = tools::get_time();
                  oper_timer.renorm.tHx[i] += tools::get_duration(t1-t0);
                  oper_timer.renorm.cHx[i] += mmbatch2[k][i].cost; 
               } // i
            }
            // reduction
            void reduction(const int k, Tm* workspace, Tm* y, Tm* dev_red=nullptr){
               auto t0 = tools::get_time();

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
                     for(size_t j=0; j<collectc[k].size(); j++){
                        int dimc = collectc[k][j].first;
                        size_t size = collectc[k][j].second;
                        Tm* ptr = workspace+j*offset; 
                        for(int i=1; i<dimc; i++){
                           linalg::xaxpy_gpu(size, alpha, ptr+i*size, ptr);
                        }
                     }
#endif
                  } // batchblas
               } // icase

               // 2. reduction by GEMV
               Tm* pcoeff = const_cast<Tm*>(coefflst[k].data());  
#ifdef GPU
               if(dev_red != nullptr){
                  GPUmem.to_gpu(dev_red, pcoeff, coefflst[k].size()*sizeof(Tm));
                  pcoeff = dev_red;
               }
#endif
               Tm* ptrs[3];
               ptrs[0] = workspace;
               ptrs[1] = pcoeff;
               ptrs[2] = y;
               mvbatch[k].kernel(batchred, ptrs);
               this->deviceSync();

               auto t1 = tools::get_time();
               oper_timer.renorm.t_red += tools::get_duration(t1-t0);
               oper_timer.renorm.c_red += mvbatch[k].cost;
            }
         public:
            int icase = -1, batchblas = -1;
            int batchinter = -1,  batchgemm = -1, batchred = -1;
            size_t totsize = 0, batchsize = 0, offset = 0, nbatch = 0;
            double cost = 0.0;
            // --- GEMM ---
            std::vector<std::vector<MMbatch<Tm>>> mmbatch2; // mmbatch2[ibatch][icase]
            // --- reduction --- 
            std::vector<std::vector<std::pair<int,size_t>>> collectc; // icase=1 
            std::vector<std::vector<Tm>> coefflst;
            std::vector<MVbatch<Tm>> mvbatch;
            // --- intermediates [Direct] --- 
            std::vector<MVbatch<Tm>> imvbatch;
      };

   template <typename Tm>
      void RMMtask<Tm>::init(Rlist<Tm>& Rlst,
            const int _batchblas,
            const std::tuple<int,int,int>& _batchrenorm,
            const size_t _batchsize,
            const size_t _offset,
            const size_t offset0){
         // init
         batchblas = _batchblas;
         batchinter= std::get<0>(_batchrenorm);
         batchgemm = std::get<1>(_batchrenorm);
         batchred  = std::get<2>(_batchrenorm);
         batchsize = _batchsize;
         totsize = Rlst.size();
         offset = _offset;
         if(batchsize == 0 && totsize !=0){
            std::cout << "error: inconsistent batchsize & totsize!" << std::endl;
            std::cout << "batchsize=" << batchsize << " totsize=" << totsize << std::endl;
            exit(1);
         }
         if(batchsize == 0 || totsize == 0) return;
         nbatch = (totsize+batchsize-1)/batchsize;

         // start process Rlst
         icase = Rlst[0].icase; 
         mmbatch2.resize(nbatch);
         collectc.resize(nbatch);
         coefflst.resize(nbatch);
         mvbatch.resize(nbatch);
         imvbatch.resize(nbatch);
#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif	
         for(int k=0; k<nbatch; k++){
            size_t off = k*batchsize;
            size_t jlen = std::min(totsize-off, batchsize);

            // 1. setup imvbatch for inter
            if(offset0 != 0){
               size_t nInter = 0;
               for(size_t j=0; j<jlen; j++){
                  size_t jdx = off+j;
                  const auto& Rblk = Rlst[jdx];
                  if(Rblk.posInter == -1) continue;
                  nInter += 1;
               }
               if(nInter > 0){
                  MVlist<Tm> mvlst(nInter); 
                  size_t idx = 0, adx = 0;
                  for(size_t j=0; j<jlen; j++){
                     size_t jdx = off+j;
                     auto& Rblk = Rlst[jdx];
                     if(Rblk.posInter == -1) continue;
                     int ipos = Rblk.posInter;
                     size_t opsize = Rblk.dimout[ipos]*Rblk.dimin[ipos];
                     MVinfo<Tm> mv;
                     mv.transA = 'N';
                     mv.M = opsize;
                     mv.N = Rblk.lenInter;
                     mv.LDA = Rblk.ldaInter; 
                     mv.locA = ipos;
                     mv.offA = Rblk.off[ipos];
                     mv.locx = 5;
                     mv.offx = Rblk.offInter;
                     mv.locy = locInter;
                     mv.offy = j*offset0;
                     mvlst[idx] = mv; 
                     idx += 1;
                     Rblk.off[ipos] = j*offset0; // overwrite old position
                  } // j
                  // sort
                  std::stable_sort(mvlst.begin(), mvlst.end(),
                        [](const MVinfo<Tm>& mv1, const MVinfo<Tm>& mv2){
                        return mv1 > mv2;
                        });
                  imvbatch[k].init(mvlst);
               } // nInter
            }

            // 2. setup mmbatch2[k]
            const int nd = 7; // c,ct,r,rt,l,lt,psi
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
               Rblk.get_MMlist2_onedot(mmtmp2, j*offset, true);
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
               // sort
               std::stable_sort(mmlst2[i].begin(), mmlst2[i].end(),
                     [](const MMinfo<Tm>& mm1, const MMinfo<Tm>& mm2){
                     return mm1 > mm2; 
                     });
               mmbatch2[k][i].init(mmlst2[i]);
#ifdef _OPENMP
#pragma omp critical
#endif	
               cost += mmbatch2[k][i].cost; // accumulate MM cost
            } // i

            // 3. reduction
            // 3.1 information for collect O(r,r') += \sum_c O(r,r',c)
            if(icase == 1){
               collectc[k].resize(jlen);
               for(size_t j=0; j<jlen; j++){
                  size_t jdx = off+j;
                  const auto& Rblk = Rlst[jdx];
                  collectc[k][j] = std::make_pair(Rblk.dimin2[2], Rblk.size);
               } // j
            }
            // 3.2 setup mvbatch[k]
            coefflst[k].resize(jlen);
            MVlist<Tm> mvlst; 
            int nmu = 0;
            size_t offrop = Rlst[off].offrop;
            for(size_t j=0; j<jlen; j++){
               size_t jdx = off+j;
               const auto& Rblk = Rlst[jdx];
               coefflst[k][j] = Rblk.coeff;
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
            // sort
            std::stable_sort(mvlst.begin(), mvlst.end(),
                  [](const MVinfo<Tm>& mv1, const MVinfo<Tm>& mv2){
                  return mv1 > mv2;
                  });
            mvbatch[k].init(mvlst, beta);
         } // k
      }

   template <typename Tm>
      using RMMtasks = std::vector<RMMtask<Tm>>;

   template <typename Tm>
      void save_mmtask(const RMMtasks<Tm>& rmmtasks, const std::string fmmtask){
         std::cout << "save_mmtask fmmtask = " << fmmtask << std::endl;
         for(int i=0; i<rmmtasks.size(); i++){
            std::string fmmtask_i = fmmtask+"_iblk"+std::to_string(i);
            rmmtasks[i].save(fmmtask_i);
         }
      }
   template <typename Tm>
      void save_mmtask(const RMMtask<Tm>& rmmtask, const std::string fmmtask){
         std::cout << "save_mmtask fmmtask = " << fmmtask << std::endl;
         rmmtask.save(fmmtask);
      }

} // ctns

#endif
