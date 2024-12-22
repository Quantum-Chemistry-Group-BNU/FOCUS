#ifndef OPER_REDUCE_H
#define OPER_REDUCE_H

#ifndef SERIAL
#include "../core/mpi_wrapper.h"
#endif

namespace ctns{

#ifndef SERIAL
   // ifdist1=true & ifnccl=true: Use NCCL to perform reduction for opS and opH on GPU directly
   template <bool ifab, typename Tm>
      void reduce_opSH_gpu(const bool ifdists,
            qoper_dict<ifab,Tm>& qops,
            const int alg_renorm,
            const bool ifkr,
            const int size,
            const int rank){
#ifdef GPU
         if(alg_renorm>10){
#ifndef NCCL
            std::cout << "error: NCCL must be used for comm[opS,opH] for ifnccl=true!" << std::endl;
            exit(1);
#else
            if(!ifdists){
               // Sp[iproc] += \sum_i Sp[i]
               auto opS_index = qops.oper_index_op('S');
               for(int p : opS_index){
                  int iproc = distribute1(ifkr,size,p);
                  auto& opS = qops('S')[p];
                  size_t opsize = opS.size();
                  size_t off = qops._offset[std::make_pair('S',p)];
                  Tm* dev_ptr = qops._dev_data+off;
                  nccl_comm.reduce(dev_ptr, opsize, iproc);
                  if(iproc != rank) GPUmem.memset(dev_ptr, opsize*sizeof(Tm));
               }
            }
            // H[0] += \sum_i H[i]
            auto& opH = qops('H')[0];
            size_t opsize = opH.size();
            size_t off = qops._offset[std::make_pair('H',0)];
            Tm* dev_ptr = qops._dev_data+off;
            nccl_comm.reduce(dev_ptr, opsize, 0);
            if(rank != 0) GPUmem.memset(dev_ptr, opsize*sizeof(Tm));
#endif
         }
#endif // GPU
      }

   // ifdist1=true and ifnccl=false: reduction of opS and opH on CPU and send back to GPU 
   template <typename Qm, typename Tm>
      void reduce_opSH_cpu(const bool ifdists,
            qoper_dict<Qm::ifabelian,Tm>& qops,
            const comb<Qm,Tm>& icomb,
            const int alg_renorm,
            const int size,
            const int rank){
         if(!ifdists){
            // Sp[iproc] += \sum_i Sp[i]
            auto opS_index = qops.oper_index_op('S');
            size_t opS_totsize = 0;
            for(int p : opS_index){
               int iproc = distribute1(Qm::ifkr,size,p);
               auto& opS = qops('S')[p];
               size_t opsize = opS.size();
               opS_totsize += opsize;
               size_t off = qops._offset[std::make_pair('S',p)];
               mpi_wrapper::reduce(icomb.world, opS.data(), opsize, iproc);
               if(iproc != rank) opS.set_zero();
            }
         }
         // H[0] += \sum_i H[i]
         auto& opH = qops('H')[0];
         size_t opsize = opH.size();
         size_t off = qops._offset[std::make_pair('H',0)];
         mpi_wrapper::reduce(icomb.world, opH.data(), opsize, 0);
         if(rank != 0) opH.set_zero();
#ifdef GPU
         // send the collected opS and opH backto GPU
         if(alg_renorm>10){
            if(!ifdists and opS_index.size()>0){
               int p0 = opS_index[0];
               size_t off = qops._offset[std::make_pair('S',p0)];
               Tm* ptr0 = qops('S')[p0].data();
               GPUmem.to_gpu(qops._dev_data+off, ptr0, opS_totsize*sizeof(Tm));
            }
            GPUmem.to_gpu(qops._dev_data+off, opH.data(), opsize*sizeof(Tm));
         }
#endif
      }
#endif // serial

} // ctns

#endif
