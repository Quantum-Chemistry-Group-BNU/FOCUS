#ifndef OPER_AB2PQ_UTIL_H
#define OPER_AB2PQ_UTIL_H

#include "sadmrg/symbolic_compxwf_opS_su2.h"
#if defined(GPU) && defined(NCCL)
#include "gpu_kernel/batched_Hermitian_Conjugate.h"
#endif

namespace ctns{

   template <typename Tm>
      linalg::matrix<Tm> get_A2Pmat(const std::vector<int>& aindex, 
            const std::vector<int>& pindex, 
            const integral::two_body<Tm>& int2e){
         int rows = aindex.size();
         int cols = pindex.size();
         linalg::matrix<Tm> cmat(rows,cols);
         for(int icol=0; icol<cols; icol++){
            int ipq = pindex[icol];
            auto pq = oper_unpack(ipq);
            int p = pq.first;
            int q = pq.second;
            for(int irow=0; irow<rows; irow++){
               int isr = aindex[irow];
               auto sr = oper_unpack(isr);
               int s = sr.first;
               int r = sr.second;
               cmat(irow,icol) = int2e.get(p,q,s,r);
            } // irow
         } // icol
         return cmat;
      } 

   template <typename Tm>
      linalg::matrix<Tm> get_A2Pmat_su2(const std::vector<int>& aindex, 
            const std::vector<int>& pindex, 
            const integral::two_body<Tm>& int2e,
            const int ts){
         int rows = aindex.size();
         int cols = pindex.size();
         linalg::matrix<Tm> cmat(rows,cols);
         for(int icol=0; icol<cols; icol++){
            int ipq = pindex[icol];
            auto pq = oper_unpack(ipq);
            int p2 = pq.first, kp = p2/2;
            int q2 = pq.second, kq = q2/2;
            for(int irow=0; irow<rows; irow++){
               int isr = aindex[irow];
               auto sr = oper_unpack(isr);
               int s2 = sr.first, ks = s2/2;
               int r2 = sr.second, kr = r2/2;
               cmat(irow,icol) = get_xint2e_su2(int2e,ts,kp,kq,ks,kr);
            } // irow
         } // icol
         return cmat;
      } 

   template <typename Tm>
      linalg::matrix<Tm> get_B2Qmat(const std::vector<int>& bindex, 
            const std::vector<int>& qindex, 
            const integral::two_body<Tm>& int2e,
            const bool swap_qr){
         int rows = bindex.size();
         int cols = qindex.size();
         linalg::matrix<Tm> cmat(rows,cols);
         for(int icol=0; icol<cols; icol++){
            int ips = qindex[icol];
            auto ps = oper_unpack(ips);
            int p = ps.first;
            int s = ps.second;
            for(int irow=0; irow<rows; irow++){
               int iqr = bindex[irow];
               auto qr = oper_unpack(iqr);
               int q = qr.first;
               int r = qr.second;
               double wqr = (q==r)? 0.5 : 1.0;
               if(!swap_qr){
                  cmat(irow,icol) = wqr*int2e.get(p,q,s,r);
               }else{
                  cmat(irow,icol) = wqr*int2e.get(p,r,s,q);
               }
            } // irow
         } // icol
         return cmat;
      }

   template <typename Tm>
      linalg::matrix<Tm> get_B2Qmat_su2(const std::vector<int>& bindex, 
            const std::vector<int>& qindex, 
            const integral::two_body<Tm>& int2e,
            const int ts, 
            const bool swap_qr){
         int rows = bindex.size();
         int cols = qindex.size();
         linalg::matrix<Tm> cmat(rows,cols);
         for(int icol=0; icol<cols; icol++){
            int ips = qindex[icol];
            auto ps = oper_unpack(ips);
            int p2 = ps.first, kp = p2/2;
            int s2 = ps.second, ks = s2/2;
            for(int irow=0; irow<rows; irow++){
               int iqr = bindex[irow];
               auto qr = oper_unpack(iqr);
               int q2 = qr.first, kq = q2/2;
               int r2 = qr.second, kr = r2/2;
               double wqr = (kq==kr)? 0.5 : 1.0;
               if(!swap_qr){
                  cmat(irow,icol) = wqr*get_vint2e_su2(int2e,ts,kp,kq,ks,kr);
               }else{
                  if(ts == 2) wqr = -wqr; // (-1)^k in my note for Qps^k
                  cmat(irow,icol) = wqr*get_vint2e_su2(int2e,ts,kp,kr,ks,kq);
               }
            } // irow
         } // icol
         return cmat;
      }

#if defined(GPU) && defined(NCCL)
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
               if(!adjoint or ifab){
                  facs[iblk] = 1.0; // no phase for abeilian case
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
#endif // GPU & NCCL

} // ctns

#endif
