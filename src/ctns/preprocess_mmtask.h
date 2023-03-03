#ifndef PREPROCESS_MMTASK_H
#define PREPROCESS_MMTASK_H

#include "preprocess_hxlist.h"
#include "preprocess_mmbatch.h"
#include "preprocess_mmreduce.h"

#include "time.h"
#include "sys/time.h"
#include "oper_timer.h"

namespace ctns{

   // Interface to Hxlist
   template <typename Tm>
      struct MMtask{
         public:
            void init(const Hxlist<Tm>& Hxlst, 
                  const int _batchgemm,
                  const size_t _batchsize,
                  const size_t offset,
                  const int hdxorder,
                  const int icase=0);
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
            // perform GEMMs [c2,c1,r,l]
            void kernel(const int k, Tm** ptrs){
               struct timeval t0, t1;
               for(int i=0; i<mmbatch2[k].size(); i++){
                  gettimeofday(&t0, NULL);
                  mmbatch2[k][i].kernel(batchgemm, ptrs);
                  gettimeofday(&t1, NULL);
                  oper_timer.tHx[i] += ((double)(t1.tv_sec - t0.tv_sec) 
                        + (double)(t1.tv_usec - t0.tv_usec)/1000000.0);
               }
            }
            // reduction of y[:] = \sum_i ai*yi[:]
            void reduction(const int k, Tm* workspace, Tm* y, const int iop) const{
               struct timeval t0, t1;
               gettimeofday(&t0, NULL);
               mmreduce[k].reduction(workspace, y, iop);
               gettimeofday(&t1, NULL);
               oper_timer.tHx[8] += ((double)(t1.tv_sec - t0.tv_sec) 
                     + (double)(t1.tv_usec - t0.tv_usec)/1000000.0);
            }
         public:
            int batchgemm;
            size_t totsize, batchsize, nbatch;
            double cost = 0.0;
            std::vector<std::vector<MMbatch<Tm>>> mmbatch2; // mmbatch2[ibatch][icase]
            std::vector<MMreduce<Tm>> mmreduce; // mmreduce[ibatch]
      };
   template <typename Tm>
      using MMtasks = std::vector<MMtask<Tm>>;

   template <typename Tm>
      void MMtask<Tm>::init(const Hxlist<Tm>& Hxlst,
            const int _batchgemm,
            const size_t _batchsize,
            const size_t offset,
            const int hxorder,
            const int _batchcase){
         // init
         batchgemm = _batchgemm;
         totsize = Hxlst.size();
         batchsize = _batchsize;
         if(batchsize == 0 && totsize !=0){
            std::cout << "error: inconsistent batchsize & totsize!" << std::endl;
            std::cout << "batchsize=" << batchsize << " totsize=" << totsize << std::endl;
            exit(1);
         }
         if(batchsize == 0) return;
         nbatch = totsize/batchsize;
         if(totsize%batchsize != 0) nbatch += 1; // thus, this works even for totsize < batchsize

         // start process Hxlst
         mmbatch2.resize(nbatch);
         mmreduce.resize(nbatch);
         for(int k=0; k<nbatch; k++){
            size_t off = k*batchsize;
            size_t jlen = std::min(totsize-off, batchsize);

            // initialization
            int nd = (_batchcase==0)? 4 : 8;
            int pos[4] = {0,1,2,3};
            std::vector<size_t> dims(nd,0);
            // count how many gemms in each case 
            for(int j=0; j<jlen; j++){
               size_t jdx = off+j;
               const auto& Hxblk = Hxlst[jdx];
               if(_batchcase == 1){
                  pos[0] = Hxblk.dagger[3]? 0 : 1;
                  pos[1] = Hxblk.dagger[2]? 2 : 3;
                  pos[2] = Hxblk.dagger[1]? 4 : 5;
                  pos[3] = Hxblk.dagger[0]? 6 : 7;
               }
               dims[pos[0]] += Hxblk.identity(3)? 0 : 1; 
               dims[pos[1]] += Hxblk.identity(2)? 0 : Hxblk.dimout[3]; // c1
               dims[pos[2]] += Hxblk.identity(1)? 0 : Hxblk.dimout[3]*Hxblk.dimout[2]; // r
               dims[pos[3]] += Hxblk.identity(0)? 0 : 1; // l
            }

            // setup mmbatch2[k]
            // generation of mmlst2
            MMlist2<Tm> mmlst2(nd);
            for(int i=0; i<nd; i++){
               mmlst2[i].resize(dims[i]); // c2,c1,r,l
            }
            std::vector<size_t> idx(nd,0);
            for(int j=0; j<jlen; j++){
               size_t jdx = off+j;
               const auto& Hxblk = Hxlst[jdx];
               if(_batchcase == 1){
                  pos[0] = Hxblk.dagger[3]? 0 : 1;
                  pos[1] = Hxblk.dagger[2]? 2 : 3;
                  pos[2] = Hxblk.dagger[1]? 4 : 5;
                  pos[3] = Hxblk.dagger[0]? 6 : 7;
               }
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

            // setup mmreduce[k]
            mmreduce[k].size = jlen;
            mmreduce[k].ndim = Hxlst[off].size;
            mmreduce[k].offset = offset;
            mmreduce[k].offout = Hxlst[off].offout;
            mmreduce[k].coeff.resize(jlen);
            for(int j=0; j<jlen; j++){
               size_t jdx = off+j;
               const auto& Hxblk = Hxlst[jdx];
               mmreduce[k].coeff[j] = Hxblk.coeff;
            }

         } // k
      }

} // ctns

#endif
