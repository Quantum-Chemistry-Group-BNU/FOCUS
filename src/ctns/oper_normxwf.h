#ifndef OPER_NORMXWF_H
#define OPER_NORMXWF_H

#ifdef _OPENMP
#include <omp.h>
#endif

#include "oper_kernel.h"
#include "oper_timer.h"

namespace ctns{

// kernel for computing Cp|ket>
template <typename Tm>
stensor3<Tm> oper_normxwf_opC(const std::string superblock,
		              const stensor3<Tm>& site,
		              const oper_dict<Tm>& qops1,
		              const oper_dict<Tm>& qops2,
			      const int index,
			      const int iformula,
		              const bool ifdagger=false){
   auto t0 = tools::get_time();
   std::cout << "oper_normxwf_opC index=" << index << " iformula=" << iformula << std::endl;
 
   stensor3<Tm> opwf;
   if(iformula == 1){
      const auto& op1 = qops1('C').at(index);

      site.print("site",2);
      op1.print("op1",2);
      //opwf = oper_kernel_OIwf(superblock, site, op1, ifdagger);
      //opwf.print("opwf",2);

   }else if(iformula == 2){
      const auto& op2 = qops2('C').at(index);
      
      site.print("site",2);
      op2.print("op2",2);
      //opwf = oper_kernel_IOwf(superblock, site, op2, 1, ifdagger);
      //opwf.print("opwf",2);

   } // iformula

   auto t1 = tools::get_time();
#ifdef _OPENMP
   #pragma omp critical
#endif
   { 
      oper_timer.nC += 1;
      oper_timer.tC += tools::get_duration(t1-t0);
   }
   return opwf; 
}

// kernel for computing Apq|ket> 
template <typename Tm>
stensor3<Tm> oper_normxwf_opA(const std::string superblock,
		              const stensor3<Tm>& site,
		              const oper_dict<Tm>& qops1,
		              const oper_dict<Tm>& qops2,
		              const int index,
			      const int iformula,
			      const bool ifdagger=false){
   auto t0 = tools::get_time();
   std::cout << "oper_normxwf_opA index=" << index << " iformula=" << iformula << std::endl;

   const bool ifkr = qops1.ifkr;
   stensor3<Tm> opwf;
   if(iformula == 1){
      const auto& op1 = qops1('A').at(index);
      opwf = oper_kernel_OIwf(superblock, site, op1, ifdagger);
   }else if(iformula == 2){
      const auto& op2 = qops2('A').at(index);
      opwf = oper_kernel_IOwf(superblock, site, op2, 0, ifdagger);
   }else if(iformula == 3){
      auto pq = oper_unpack(index);	
      int p = pq.first, sp = p%2;
      int q = pq.second, sq = q%2;
      // kr opposite spin case: <a1A^+a2B^+> = [a1A^+]*[a2B^+]
      const bool ifnot_kros = !(ifkr && sp != sq);
      const auto& op1 = qops1('C').at(p);
      const auto& op2 = ifnot_kros? qops2('C').at(q) : qops2('C').at(q-1).K(1);
      opwf = oper_kernel_OOwf(superblock, site, op1, op2, 1, ifdagger);
      if(ifdagger) opwf *= -1.0; // (c1*c2)^d = c2d*c1d = -c1d*c2d
   }else if(iformula == 4){
      auto qp = oper_unpack(index);	
      int p = qp.second, sp = p%2;
      int q = qp.first, sq = q%2;
      // kr opposite spin case: <a2A^+a1B^+> = -[a1B^+]*[a2A^+]
      const bool ifnot_kros = !(ifkr && sp != sq);
      const auto& op1 = ifnot_kros? -qops1('C').at(p) : -qops1('C').at(p-1).K(1);
      const auto& op2 = qops2('C').at(q);
      opwf = oper_kernel_OOwf(superblock, site, op1, op2, 1, ifdagger);
      if(ifdagger) opwf *= -1.0;
   } // iformula

   auto t1 = tools::get_time();
#ifdef _OPENMP
   #pragma omp critical
#endif
   { 
      oper_timer.nA += 1;
      oper_timer.tA += tools::get_duration(t1-t0);
   }
   return opwf;
}

// kernel for computing Bps|ket> 
template <typename Tm>
stensor3<Tm> oper_normxwf_opB(const std::string superblock,
		              const stensor3<Tm>& site,
		              const oper_dict<Tm>& qops1,
		              const oper_dict<Tm>& qops2,
		              const int index,
			      const int iformula,
			      const bool ifdagger=false){
   auto t0 = tools::get_time();
   std::cout << "oper_normxwf_opB index=" << index << " iformula=" << iformula << std::endl;

   const bool ifkr = qops1.ifkr;
   stensor3<Tm> opwf;
   if(iformula == 1){
      const auto& op1 = qops1('B').at(index);
      opwf = oper_kernel_OIwf(superblock, site, op1, ifdagger);
   }else if(iformula == 2){
      const auto& op2 = qops2('B').at(index);
      opwf = oper_kernel_IOwf(superblock, site, op2, 0, ifdagger);
   }else if(iformula == 3){
      auto pq = oper_unpack(index);	
      int p = pq.first, sp = p%2;
      int q = pq.second, sq = q%2;
      // kr opposite spin case: <a1A^+a2B> = [a1A^+]*[a2B]
      const bool ifnot_kros = !(ifkr && sp != sq);
      const auto& op1 = qops1('C').at(p);
      const auto& op2 = ifnot_kros? qops2('C').at(q).H() : qops2('C').at(q-1).H().K(1);
      opwf = oper_kernel_OOwf(superblock, site, op1, op2, 1, ifdagger);
      if(ifdagger) opwf *= -1.0;
   }else if(iformula == 4){
      auto qp = oper_unpack(index);	
      int p = qp.second, sp = p%2;
      int q = qp.first, sq = q%2;
      // kr opposite spin case: <a2A^+a1B> = -[a1B]*[a2A^+]
      const bool ifnot_kros = !(ifkr && sp != sq);
      const auto& op1 = ifnot_kros? -qops1('C').at(p).H() : -qops1('C').at(p-1).H().K(1);
      const auto& op2 = qops2('C').at(q);
      opwf = oper_kernel_OOwf(superblock, site, op1, op2, 1, ifdagger);
      if(ifdagger) opwf *= -1.0;
   } // iformula

   auto t1 = tools::get_time();
#ifdef _OPENMP
   #pragma omp critical
#endif
   {
      oper_timer.nB += 1;
      oper_timer.tB += tools::get_duration(t1-t0);
   }
   return opwf;
}

} // ctns

#endif
