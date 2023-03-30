#ifndef PREPROCESS_HMMTASK_H
#define PREPROCESS_HMMTASK_H

#include "preprocess_hxlist.h"
#include "preprocess_mmbatch.h"
#include "preprocess_mmreduce.h"

#include "time.h"
#include "sys/time.h"
#include "oper_timer.h"

namespace ctns{

   // Interface to Hxlist
   template <typename Tm>
      struct HMMtask{
         public:
            void init(Hxlist<Tm>& Hxlst, 
                  const int hdxorder,
                  const int _batchblas,
                  const size_t _batchsize,
                  const size_t offset,
                  const size_t offset0=0);
            // save dimensions for optimization
            void save(const std::string fgemm) const{
               for(int k=0; k<mmbatch2.size(); k++){
                  for(int i=0; i<mmbatch2[k].size(); i++){
                     std::string fgemmki = fgemm+"_"+std::to_string(k)+"."
                        +std::to_string(i)+".txt";
                     mmbatch2[k][i].save(fgemmki);
                  }
               }
            }
            // form intermeidate operators
            void inter(const int k, Tm** opaddr, const Tm* alphas){
               struct timeval t0, t1;
               // perform GEMV_BATCH
               Tm* ptrs[6];
               ptrs[0] = opaddr[0];
               ptrs[1] = opaddr[1];
               ptrs[2] = opaddr[2];
               ptrs[3] = opaddr[3];
               ptrs[4] = opaddr[4];
               ptrs[5] = const_cast<Tm*>(alphas);
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
               oper_timer.sigma.t_inter += ((double)(t1.tv_sec - t0.tv_sec) 
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
                  oper_timer.sigma.tHx[i] += ((double)(t1.tv_sec - t0.tv_sec) 
                        + (double)(t1.tv_usec - t0.tv_usec)/1000000.0);
                  oper_timer.sigma.cHx[i] += mmbatch2[k][i].cost; 
               }
            }
            // reduction of y[:] = \sum_i ai*yi[:]
            void reduction(const int k, Tm* workspace, Tm* y) const{
               struct timeval t0, t1;
               gettimeofday(&t0, NULL);
               mmreduce[k].reduction(batchblas, workspace, y);
#ifdef GPU
#ifdef USE_HIP
               hipDeviceSynchronize();
#else
               cudaDeviceSynchronize();
#endif
#endif
               gettimeofday(&t1, NULL);
               oper_timer.sigma.t_red += ((double)(t1.tv_sec - t0.tv_sec) 
                     + (double)(t1.tv_usec - t0.tv_usec)/1000000.0);
            }
         public:
            int batchblas = -1;
            size_t totsize = 0, batchsize = 0, nbatch = 0;
            double cost = 0.0;
            std::vector<std::vector<MMbatch<Tm>>> mmbatch2; // mmbatch2[ibatch][icase]
            std::vector<MMreduce<Tm>> mmreduce; // mmreduce[ibatch]
            // --- intermediates [Direct] ---
            std::vector<MVbatch<Tm>> imvbatch;
      };

   template <typename Tm>
      void HMMtask<Tm>::init(Hxlist<Tm>& Hxlst,
            const int hxorder,
            const int _batchblas,
            const size_t _batchsize,
            const size_t offset,
            const size_t offset0){
         // init
         batchblas = _batchblas;
         batchsize = _batchsize;
         totsize = Hxlst.size();
         if(batchsize == 0 && totsize !=0){
            std::cout << "error: inconsistent batchsize & totsize!" << std::endl;
            std::cout << "batchsize=" << batchsize << " totsize=" << totsize << std::endl;
            exit(1);
         }
         if(batchsize == 0 || totsize == 0) return;
         nbatch = totsize/batchsize;
         if(totsize%batchsize != 0) nbatch += 1; // thus, this works even for totsize < batchsize

         // start process Hxlst
         mmbatch2.resize(nbatch);
         mmreduce.resize(nbatch);
         imvbatch.resize(nbatch);
         for(int k=0; k<nbatch; k++){
            size_t off = k*batchsize;
            size_t jlen = std::min(totsize-off, batchsize);

            // 1. setup imvbatch for inter
            if(offset0 != 0){
               size_t nInter = 0;
               for(size_t j=0; j<jlen; j++){
                  size_t jdx = off+j;
                  const auto& Hxblk = Hxlst[jdx];
                  if(Hxblk.posInter == -1) continue;
                  nInter += 1;
               }
               if(nInter > 0){
                  MVlist<Tm> mvlst(nInter); 
                  size_t idx = 0;
                  for(size_t j=0; j<jlen; j++){
                     size_t jdx = off+j;
                     auto& Hxblk = Hxlst[jdx];
                     if(Hxblk.posInter == -1) continue;
                     int ipos = Hxblk.posInter;
                     size_t opsize = Hxblk.dimout[ipos]*Hxblk.dimin[ipos];
                     MVinfo<Tm> mv;
                     mv.transA = 'N';
                     mv.M = opsize;
                     mv.N = Hxblk.lenInter;
                     mv.LDA = Hxblk.ldaInter; 
                     mv.locA = ipos;
                     mv.offA = Hxblk.off[ipos];
                     mv.locx = 5;
                     mv.offx = Hxblk.offInter;
                     mv.locy = locInter;
                     mv.offy = j*offset0;
                     mvlst[idx] = mv; 
                     idx += 1;
                     Hxblk.off[ipos] = j*offset0; // overwrite old position
                  } // j
                  imvbatch[k].init(mvlst);
               }
            }

            // 2. setup mmbatch2[k]
            const int nd = 8;
            std::vector<size_t> dims(nd,0);
            // count how many gemms in each case 
            for(size_t j=0; j<jlen; j++){
               size_t jdx = off+j;
               const auto& Hxblk = Hxlst[jdx];
               int pos[4];
               pos[0] = Hxblk.dagger[3]? 0 : 1;
               pos[1] = Hxblk.dagger[2]? 2 : 3;
               pos[2] = Hxblk.dagger[1]? 4 : 5;
               pos[3] = Hxblk.dagger[0]? 6 : 7;
               dims[pos[0]] += Hxblk.identity(3)? 0 : 1; 
               dims[pos[1]] += Hxblk.identity(2)? 0 : Hxblk.dimout[3]; // c1
               dims[pos[2]] += Hxblk.identity(1)? 0 : Hxblk.dimout[3]*Hxblk.dimout[2]; // r
               dims[pos[3]] += Hxblk.identity(0)? 0 : 1; // l
            }

            // generation of mmlst2
            MMlist2<Tm> mmlst2(nd);
            for(int i=0; i<nd; i++){
               mmlst2[i].resize(dims[i]); // c2,c1,r,l
            }
            std::vector<size_t> idx(nd,0);
            for(size_t j=0; j<jlen; j++){
               size_t jdx = off+j;
               const auto& Hxblk = Hxlst[jdx];
               int pos[4];
               pos[0] = Hxblk.dagger[3]? 0 : 1;
               pos[1] = Hxblk.dagger[2]? 2 : 3;
               pos[2] = Hxblk.dagger[1]? 4 : 5;
               pos[3] = Hxblk.dagger[0]? 6 : 7;
               MMlist2<Tm> mmtmp2(4);
               Hxblk.get_MMlist_twodot(mmtmp2, j*offset);
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

            // 3. setup mmreduce[k]
            mmreduce[k].batchsize = jlen;
            mmreduce[k].ndim = Hxlst[off].size;
            mmreduce[k].offset = offset;
            mmreduce[k].offout = Hxlst[off].offout;
            mmreduce[k].coeff.resize(jlen);
            for(size_t j=0; j<jlen; j++){
               size_t jdx = off+j;
               const auto& Hxblk = Hxlst[jdx];
               mmreduce[k].coeff[j] = Hxblk.coeff;
            }

         } // k
      }

   template <typename Tm>
      using HMMtasks = std::vector<HMMtask<Tm>>;

   template <typename Tm>
      void save_hmmtasks(const HMMtasks<Tm>& hmmtasks, const int isweep, const int ibond){
         std::string fgemm = "hmmtasks_gemm";
         fgemm += "_isweep"+std::to_string(isweep) + "_ibond"+std::to_string(ibond);
         for(int i=0; i<hmmtasks.size(); i++){
            std::string fgemmi = fgemm+"_iblk"+std::to_string(i);
            hmmtasks[i].save(fgemmi);
         }
         std::string freduction = "hmmtasks_reduction";
         freduction += "_isweep"+std::to_string(isweep) + "_ibond"+std::to_string(ibond)+".txt";
         std::ofstream fout(freduction);
         for(int i=0; i<hmmtasks.size(); i++){
            for(int j=0; j<hmmtasks[i].mmreduce.size(); j++){
               const auto& red = hmmtasks[i].mmreduce[j];
               fout << "iblk=" << i << " ibatch=" << j 
                  << " batchsize=" << red.batchsize << " ndim=" << red.ndim
                  << std::endl;      
            }
         }
         fout.close();
      }

} // ctns

#endif
