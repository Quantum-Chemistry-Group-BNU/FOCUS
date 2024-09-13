#ifndef RDM_ASSEMBLE_SIMPLE_H
#define RDM_ASSEMBLE_SIMPLE_H

#include "rdm_string.h"
#include "rdm_distribute.h"
#include "rdm_oputil.h"

namespace ctns{

   template <bool ifab, typename Tm, std::enable_if_t<!ifab,int> = 0>
      void rdm_assemble_simple(const bool is_same,
            const type_pattern& pattern,
            const char lkey, 
            const char ckey, 
            const char rkey,
            const bool ldagger, 
            const bool cdagger, 
            const bool rdagger, 
            const qoper_map<ifab,Tm>& lops, 
            const qoper_map<ifab,Tm>& cops, 
            const qoper_map<ifab,Tm>& rops, 
            const qtensor3<ifab,Tm>& wf3bra,
            const qtensor3<ifab,Tm>& wf3ket,
            const std::vector<int>& leval,
            const std::vector<int>& reval,
            linalg::matrix<Tm>& rdm,
            const linalg::matrix<Tm>& tdm,
            const int rank){
         std::cout << "error: rdm_assemble_simple does not support su2 case!" << std::endl;
         exit(1);
      }
   template <bool ifab, typename Tm, std::enable_if_t<ifab,int> = 0>
      void rdm_assemble_simple(const bool is_same,
            const type_pattern& pattern,
            const char lkey, 
            const char ckey, 
            const char rkey,
            const bool ldagger, 
            const bool cdagger, 
            const bool rdagger, 
            const qoper_map<ifab,Tm>& lops, 
            const qoper_map<ifab,Tm>& cops, 
            const qoper_map<ifab,Tm>& rops, 
            const qtensor3<ifab,Tm>& wf3bra,
            const qtensor3<ifab,Tm>& wf3ket,
            const std::vector<int>& leval,
            const std::vector<int>& reval,
            linalg::matrix<Tm>& rdm,
            const linalg::matrix<Tm>& tdm,
            const int rank){
         // assemble rdms
         int lparity = op2parity.at(lkey);
         int cparity = op2parity.at(ckey);
         int rparity = op2parity.at(rkey);
         for(const auto& rdx : reval){
            const auto& rop = rops.at(rdx);
            std::cout << "rop: key=" << rkey << " rdx=" << rdx << " normF()=" << rop.normF() << std::endl;
            //rop.print("rop");
            auto rstr = get_calst(rkey, rdx, rdagger);
            auto opxwf1 = oper_kernel_IOwf("cr", wf3ket, rop, rparity, rdagger);
            for(const auto& cpr : cops){
               const auto& cdx = cpr.first;
               const auto& cop = cpr.second;
               std::cout << "cop: key=" << ckey << " cdx=" << cdx << " normF()=" << cop.normF() << std::endl;
               //cop.print("cop");
               //cop.to_matrix().print("copmat");
               auto cstr = get_calst(ckey, cdx, cdagger);
               auto opxwf2 = oper_kernel_OIwf("cr", opxwf1, cop, cdagger);
               if((cparity+rparity)%2 == 1) opxwf2.row_signed();
               auto op2 = contract_qt3_qt3("cr", wf3bra, opxwf2); 
               for(const auto& ldx : leval){
                  auto lop = ldagger? lops.at(ldx).H() : lops.at(ldx);
                  std::cout << "lop: key=" << lkey << " ldx=" << ldx << " normF()=" << lop.normF() << std::endl;
                  //lop.print("lop");
                  auto lstr = get_calst(lkey, ldx, ldagger);
                  if(tools::is_complex<Tm>()) lop = lop.conj();
                  Tm val = contract_qt2_qt2_full(lop, op2); 
                  // assign val to rdm
                  rdmstring rdmstr(lstr, cstr, rstr);
                  auto rdmstr2 = rdmstr;
                  Tm sgn = rdmstr2.sort();
                  auto ijdx = rdmstr2.get_ijdx();
                  size_t idx = ijdx.first;
                  size_t jdx = ijdx.second;
                  rdm(idx,jdx) = sgn*val;
                  double diff = std::abs(sgn*val - tdm(idx,jdx));
                  std::cout << "rank=" << rank
                     << " pattern=" << pattern.to_string() 
                     << " ldx,cdx,rdx=" << ldx << "," << cdx << "," << rdx 
                     << " rdmstr=" << rdmstr.to_string1()
                     << " rdmstr2=" << rdmstr2.to_string1()
                     << " rdmstr=" << rdmstr.to_string()
                     << " rdmstr2=" << rdmstr2.to_string()
                     << " sgn=" << sgn 
                     << " val=" << std::setprecision(12) << sgn*val
                     << " idx,jdx=" << idx << "," << jdx
                     << " tdm=" << tdm(idx,jdx)
                     << " diff=" << diff
                     << std::endl;
                  assert(diff < 1.e-8);
                  if(is_same) rdm(jdx,idx) = tools::conjugate(rdm(idx,jdx));
               }
            }
         }
      }

   template <bool ifab, typename Tm, std::enable_if_t<!ifab,int> = 0>
      void rdm_assemble_simple_parallel(const bool is_same,
            const char lkey, 
            const char ckey, 
            const char rkey,
            const bool ldagger, 
            const bool cdagger, 
            const bool rdagger, 
            const qoper_map<ifab,Tm>& lops, 
            const qoper_map<ifab,Tm>& cops, 
            const qoper_map<ifab,Tm>& rops, 
            const qtensor3<ifab,Tm>& wf3bra,
            const qtensor3<ifab,Tm>& wf3ket,
            const std::vector<int>& leval,
            const std::vector<int>& reval,
            linalg::matrix<Tm>& rdm){
         std::cout << "error: rdm_assemble_simple_parallel does not support su2 case!" << std::endl;
         exit(1);
      }
   template <bool ifab, typename Tm, std::enable_if_t<ifab,int> = 0>
      void rdm_assemble_simple_parallel(const bool is_same,
            const char lkey, 
            const char ckey, 
            const char rkey,
            const bool ldagger, 
            const bool cdagger, 
            const bool rdagger, 
            const qoper_map<ifab,Tm>& lops, 
            const qoper_map<ifab,Tm>& cops, 
            const qoper_map<ifab,Tm>& rops, 
            const qtensor3<ifab,Tm>& wf3bra,
            const qtensor3<ifab,Tm>& wf3ket,
            const std::vector<int>& leval,
            const std::vector<int>& reval,
            linalg::matrix<Tm>& rdm){
         // assemble rdms
         int lparity = op2parity.at(lkey);
         int cparity = op2parity.at(ckey);
         int rparity = op2parity.at(rkey);
         for(const auto& rdx : reval){
            const auto& rop = rops.at(rdx);
            auto rstr = get_calst(rkey, rdx, rdagger);
            auto opxwf1 = oper_kernel_IOwf("cr", wf3ket, rop, rparity, rdagger);
            for(const auto& cpr : cops){
               const auto& cdx = cpr.first;
               const auto& cop = cpr.second;
               auto cstr = get_calst(ckey, cdx, cdagger);
               auto opxwf2 = oper_kernel_OIwf("cr", opxwf1, cop, cdagger);
               if((cparity+rparity)%2 == 1) opxwf2.row_signed();
               auto op2 = contract_qt3_qt3("cr", wf3bra, opxwf2);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
               for(int l=0; l<leval.size(); l++){
                  const auto& ldx = leval[l];
                  auto lop = ldagger? lops.at(ldx).H() : lops.at(ldx);
                  auto lstr = get_calst(lkey, ldx, ldagger);
                  if(tools::is_complex<Tm>()) lop = lop.conj();
                  Tm val = contract_qt2_qt2_full(lop, op2); 
                  // assign val to rdm
                  rdmstring rdmstr(lstr, cstr, rstr);
                  Tm sgn = rdmstr.sort();
                  auto ijdx = rdmstr.get_ijdx();
                  size_t idx = ijdx.first;
                  size_t jdx = ijdx.second;
#ifdef _OPENMP
#pragma omp critical
#endif
                  {
                     rdm(idx,jdx) = sgn*val;
                     if(is_same) rdm(jdx,idx) = tools::conjugate(rdm(idx,jdx));
                  }
               }
            }
         }
      }

} // ctns

#endif
