#if defined(GPU) && defined(NCCL)

#ifndef OPER_AB2PQ_HCGPU_H
#define OPER_AB2PQ_HCGPU_H

#include "gpu_kernel/batched_Hermitian_Conjugate.h"

namespace ctns{

   template <bool ifab, typename Tm>
      void batchedHermitianConjugateGPU(const qoper_dict<ifab,Tm>& qops1,
            const char type1,
            qoper_dict<ifab,Tm>& qops2,
            const char type2,
            const bool adjoint,
            const Tm* dev_data1,
            Tm* dev_data2,
            const bool order=true){
         // count the number of task
         size_t nblks = 0;
         for(const auto& pr : qops2(type2)){
            const auto& index = pr.first;
            const auto& qt2 = pr.second;
            nblks += qt2.info._nnzaddr.size();
         }
         if(nblks == 0) return;
         // setup tasks
         std::vector<size_t> offs(nblks*2);
         std::vector<int> dims(nblks*2);
         std::vector<Tm> facs(nblks);
         size_t iblk = 0;
         for(const auto& pr : qops2(type2)){
            const auto& index = pr.first;
            const auto& qt2 = pr.second;
            const auto& qt1 = qops1(type1).at(index);
            for(int i=0; i<qt2.info._nnzaddr.size(); i++){
               auto key = qt2.info._nnzaddr[i];
               int br, bc;
               qt2.info._addr_unpack(key,br,bc); // works for both su2 and nonsu2
               size_t loff2 = qt2.info.get_offset(br,bc);
               assert(loff2 > 0);
               size_t goff2 = qops2._offset.at(std::make_pair(type2,index)) + loff2-1;
               size_t loff1 = qt1.info.get_offset(bc,br);
               assert(loff1 > 0);
               size_t goff1 = qops1._offset.at(std::make_pair(type1,index)) + loff1-1;
               offs[2*iblk] = goff2;
               offs[2*iblk+1] = goff1;
               dims[2*iblk] = qt2.info.qrow.get_dim(br);
               dims[2*iblk+1] = qt2.info.qcol.get_dim(bc);
               if(!adjoint){
                  facs[iblk] = 1.0;
               }else{
                  // <br||Tk_bar||bc> = (-1)^{k-jc+jr}sqrt{[jc]/[jr]}<bc||Tk||br>*
                  int tsr = qt2.info.qrow.get_sym(br).ts();
                  int tsc = qt2.info.qcol.get_sym(bc).ts();
                  int deltats = (qt2.info.sym.ts() + tsr - tsc);
                  assert(deltats%2 == 0);
                  Tm fac = (deltats/2)%2==0? 1.0 : -1.0;
                  fac *= std::sqrt((tsc+1.0)/(tsr+1.0));
                  facs[iblk] = fac;
               }
               iblk += 1;
            }
         }
         if(order){
            auto offs_old = offs;
            auto dims_old = dims;
            auto facs_old = facs;
            std::vector<size_t> sizes(nblks);
            for(int i=0; i<nblks; i++){
               sizes[i] = dims[2*i]*dims[2*i+1];
            }
            auto index = tools::sort_index(sizes, 1);
            for(int i=0; i<nblks; i++){
               int idx = index[i];
               offs[2*i] = offs_old[2*idx];
               offs[2*i+1] = offs_old[2*idx+1];
               dims[2*i] = dims_old[2*idx];
               dims[2*i+1] = dims_old[2*idx+1];
               facs[i] = facs_old[idx];
            }
         }
         // allocate memory
         size_t* dev_offs = (size_t*)GPUmem.allocate(nblks*2*sizeof(size_t));
         GPUmem.to_gpu(dev_offs, offs.data(), nblks*2*sizeof(size_t));
         int* dev_dims = (int*)GPUmem.allocate(nblks*2*sizeof(int));
         GPUmem.to_gpu(dev_dims, dims.data(), nblks*2*sizeof(int));
         Tm* dev_facs = (Tm*)GPUmem.allocate(nblks*sizeof(Tm));
         GPUmem.to_gpu(dev_facs, facs.data(), nblks*sizeof(Tm));
         // invoke kernel
         batched_Hermitian_Conjugate(nblks, dev_offs, dev_dims, dev_facs, dev_data1, dev_data2);
         // deallocate
         GPUmem.deallocate(dev_offs, nblks*2*sizeof(size_t));
         GPUmem.deallocate(dev_dims, nblks*2*sizeof(int));
         GPUmem.deallocate(dev_facs, nblks*sizeof(Tm));
      }

}

#endif

#endif
