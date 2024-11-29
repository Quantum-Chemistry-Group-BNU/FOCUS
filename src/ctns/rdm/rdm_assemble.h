#ifndef RDM_ASSEMBLE_H
#define RDM_ASSEMBLE_H

#include "rdm_string.h"
#include "rdm_distribute.h"
#include "rdm_oputil.h"
#include "rdm_auxdata.h"
#include "../sadmrg/oper_dot_su2.h"

namespace ctns{

   template <typename Qm, typename Tm, std::enable_if_t<Qm::ifabelian,int> = 0>
      void rdm_assemble(const bool is_same,
            const comb<Qm,Tm>& icomb,
            const qoper_dictmap<Qm::ifabelian,Tm>& qops_dict, 
            const qtensor3<Qm::ifabelian,Tm>& wf3bra,
            const qtensor3<Qm::ifabelian,Tm>& wf3ket,
            const std::vector<type_pattern>& allpatterns,
            const input::schedule& schd,
            const std::string scratch,
            linalg::matrix<Tm>& rdm,
            rdmaux<Tm>& aux){
         int size = 1, rank = 0;
#ifndef SERIAL
         size = icomb.world.size();
         rank = icomb.world.rank();
#endif
         const bool ifab = Qm::ifabelian;
         const int alg_rdm = schd.ctns.alg_rdm;
         const bool debug = (rank==0); 
         if(debug){ 
            std::cout << "ctns::rdm_assemble"
               << " ifab=" << ifab
               << " alg_rdm=" << alg_rdm
               << std::endl;
         }
         auto t0 = tools::get_time();

         const auto& str2optype = is_same? str2optype_same : str2optype_diff;
         const auto& lqops = qops_dict.at("l");
         const auto& rqops = qops_dict.at("r");
         const auto& cqops = qops_dict.at("c");
         // Check identity operator
         if(is_same){
            const auto& lop = lqops('I').at(0);
            const auto& cop = cqops('I').at(0);
            const auto& rop = rqops('I').at(0);
            auto ldiff = linalg::check_identityMatrix(lop.to_matrix());
            auto cdiff = linalg::check_identityMatrix(cop.to_matrix());
            auto rdiff = linalg::check_identityMatrix(rop.to_matrix());
            if(debug){
               std::cout << "rank=" << rank << " ldiff,cdiff,rdiff=" 
                  << ldiff << "," << cdiff << "," << rdiff << std::endl;
            }
            assert(ldiff < 1.e-8);
            assert(cdiff < 1.e-8);
            assert(rdiff < 1.e-8);
         }
               
         // assemble RDMs by pattern
         for(int i=0; i<allpatterns.size(); i++){
            const auto& pattern = allpatterns[i];
            const auto& loptype  = str2optype.at(pattern.left);
            const auto& coptype = str2optype.at(pattern.center);
            const auto& roptype = str2optype.at(pattern.right);
            const auto& lkey = loptype.first; 
            const auto& ldagger = loptype.second;
            const auto& ckey = coptype.first; 
            const auto& cdagger = coptype.second;
            const auto& rkey = roptype.first; 
            const auto& rdagger = roptype.second;
            const auto& lops = lqops(lkey);
            const auto& cops = cqops(ckey);
            const auto& rops = rqops(rkey);
            if(debug){
               std::cout << " i=" << i 
                  << " pattern=" << pattern.to_string() 
                  << " opkey=" << lkey << ldagger 
                  << ":" << ckey << cdagger 
                  << ":" << rkey << rdagger
                  << " sizes=" << lops.size()
                  << ":" << cops.size()  
                  << ":" << rops.size();
               if(alg_rdm == 0) std::cout << std::endl;
            }

            // for parallel computation
            std::vector<int> reval;
            std::vector<int> leval;
            const bool ifkr = Qm::ifkr;
            int sorb = icomb.get_nphysical()*2;
            auto num_string = pattern.num_string();
            setup_evalmap(num_string, ifkr, sorb, lops, rops, leval, reval, size, rank);

            auto ti = tools::get_time();
            if(alg_rdm == 0){

               if(num_string != "212" or (num_string == "212" and size==1)){
                  rdm_assemble_simple(is_same, pattern, lkey, ckey, rkey, ldagger, cdagger, rdagger, lops, cops, rops, 
                        wf3bra, wf3ket, leval, reval, rdm, aux, rank);
               }else{
#ifndef SERIAL
                  rdm_assemble_simple212(is_same, icomb, pattern, lkey, ckey, rkey, ldagger, cdagger, rdagger, lqops, cqops, rqops, 
                        wf3bra, wf3ket, leval, reval, rdm, aux, size, rank);
#else
                  tools::exit("error: distributed version for pattern=212 is not supported!");
#endif
               }

            }else if(alg_rdm == 1){

               if(num_string != "212" or (num_string == "212" and size==1)){
                  rdm_assemble_omp(is_same, lkey, ckey, rkey, ldagger, cdagger, rdagger, lops, cops, rops, 
                        wf3bra, wf3ket, leval, reval, rdm, aux);
               }else{
#ifndef SERIAL
                  rdm_assemble_omp212(is_same, icomb, lkey, ckey, rkey, ldagger, cdagger, rdagger, lqops, cqops, rqops,
                        wf3bra, wf3ket, leval, reval, rdm, aux, size, rank);
#else
                  tools::exit("error: distributed version for pattern=212 is not supported!");
#endif
               }

            }else{
               tools::exit("error: no such option for alg_rdm");
            } // alg_rdm
            if(debug){
               auto tf = tools::get_time();
               double dt = tools::get_duration(tf-ti);
               std::cout << " TIMING=" << dt << " S" << std::endl;
            }
         } // pattern

         if(debug){
            auto t1 = tools::get_time();
            tools::timing("ctns::rdm_assemble", t0, t1);
         }
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
            rdmaux<Tm>& aux,
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
                  double diff = std::abs(sgn*val - aux.rdm(idx,jdx));
                  std::cout << "rank=" << rank
                     << " pattern=" << pattern.to_string() 
                     << " ldx,cdx,rdx=" << ldx << "," << cdx << "," << rdx 
                     << " rdmstr=" << rdmstr.to_string_spinorb()
                     << " rdmstr2=" << rdmstr2.to_string_spinorb()
                     << " rdmstr=" << rdmstr.to_string()
                     << " rdmstr2=" << rdmstr2.to_string()
                     << " sgn=" << sgn 
                     << " val=" << std::setprecision(12) << sgn*val
                     << " idx,jdx=" << idx << "," << jdx
                     << " aux.rdm=" << aux.rdm(idx,jdx)
                     << " diff=" << diff
                     << std::endl;
                  assert(diff < 1.e-8);
                  if(is_same) rdm(jdx,idx) = tools::conjugate(rdm(idx,jdx));
               }
            }
         }
      }

   template <bool ifab, typename Tm, std::enable_if_t<ifab,int> = 0>
      void rdm_assemble_omp(const bool is_same,
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
            rdmaux<Tm>& aux){
         // assemble rdms
         int lparity = op2parity.at(lkey);
         int cparity = op2parity.at(ckey);
         int rparity = op2parity.at(rkey);
         for(const auto& rdx : reval){
            const auto& rop = rops.at(rdx);
            //std::cout << std::endl;
            //std::cout << "rop: key=" << rkey << " rdx=" << rdx << " normF()=" << rop.normF() << std::endl;
            //rop.to_matrix().print("rop");
            auto rstr = get_calst(rkey, rdx, rdagger);
            auto opxwf1 = oper_kernel_IOwf("cr", wf3ket, rop, rparity, rdagger);
            //wf3ket.print("wf3ket");
            for(const auto& cpr : cops){
               const auto& cdx = cpr.first;
               const auto& cop = cpr.second;
               //std::cout << "cop: key=" << ckey << " cdx=" << cdx << " normF()=" << cop.normF() << std::endl;
               //cop.to_matrix().print("cop");
               auto cstr = get_calst(ckey, cdx, cdagger);
               auto opxwf2 = oper_kernel_OIwf("cr", opxwf1, cop, cdagger);
               if((cparity+rparity)%2 == 1) opxwf2.row_signed();
               auto op2 = contract_qt3_qt3("cr", wf3bra, opxwf2);
               //wf3bra.print("wf3bra");
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
               for(int l=0; l<leval.size(); l++){
                  const auto& ldx = leval[l];
                  auto lop = ldagger? lops.at(ldx).H() : lops.at(ldx);
                  //std::cout << "lop: key=" << lkey << " ldx=" << ldx << " normF()=" << lop.normF() << std::endl;
                  //lop.to_matrix().print("lop");
                  auto lstr = get_calst(lkey, ldx, ldagger);
                  if(tools::is_complex<Tm>()) lop = lop.conj();
                  //op2.to_matrix().print("op2");
                  Tm val = contract_qt2_qt2_full(lop, op2);

                  //auto lop_mat = lop.to_matrix();
                  //auto op2_mat = op2.to_matrix();
                  //auto tmpval = linalg::xdot(lop_mat.size(),lop_mat.data(),op2_mat.data());
                  //std::cout << "tmpval=" << tmpval << std::endl;

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
                     if(aux.alg_mrpt2 != 2){
                    
                        rdm(idx,jdx) = sgn*val;
                        if(is_same) rdm(jdx,idx) = tools::conjugate(rdm(idx,jdx));
                     
                     }else{

                        // on the fly contraction
                        assert(!is_same);
                        aux.dsrg_contract(idx, jdx, sgn*val);
 
                     }
                  } // critical region
               } // l
            } // c
         } // r
      }

#ifndef SERIAL

   template <typename Qm, typename Tm, std::enable_if_t<Qm::ifabelian,int> = 0>
      void rdm_assemble_simple212(const bool is_same,
            const comb<Qm,Tm>& icomb,
            const type_pattern& pattern,
            const char lkey, 
            const char ckey, 
            const char rkey,
            const bool ldagger, 
            const bool cdagger, 
            const bool rdagger, 
            const qoper_dict<Qm::ifabelian,Tm>& lqops, 
            const qoper_dict<Qm::ifabelian,Tm>& cqops, 
            const qoper_dict<Qm::ifabelian,Tm>& rqops, 
            const qtensor3<Qm::ifabelian,Tm>& wf3bra,
            const qtensor3<Qm::ifabelian,Tm>& wf3ket,
            const std::vector<int>& leval,
            const std::vector<int>& reval,
            linalg::matrix<Tm>& rdm,
            rdmaux<Tm>& aux,
            const int size,
            const int rank){
         // assemble rdms
         const auto& lops = lqops(lkey);
         const auto& cops = cqops(ckey);
         const auto& rops = rqops(rkey);
         int lparity = op2parity.at(lkey);
         int cparity = op2parity.at(ckey);
         int rparity = op2parity.at(rkey);
         // generate full set of indices         
         auto rindex2 = (rkey=='A' or rkey=='M')? oper_index_opA(rqops.cindex, rqops.ifkr, rqops.isym) :
            oper_index_opB(rqops.cindex, rqops.ifkr, rqops.isym, rqops.ifhermi);
         for(const auto& rdx : rindex2){
            //---------------------------------------------------------
            // broadcast lop to all processors to perform contractions
            auto iproc = distribute2(rkey,rqops.ifkr,size,rdx,rqops.sorb);
            stensor2<Tm> rop;
            if(iproc == rank){
               rop.init(rops.at(rdx).info);
               linalg::xcopy(rop.size(), rops.at(rdx).data(), rop.data());
            }
            if(size > 1) mpi_wrapper::broadcast(icomb.world, rop, iproc);
            //---------------------------------------------------------
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
                  double diff = std::abs(sgn*val - aux.rdm(idx,jdx));
                  std::cout << "rank=" << rank
                     << " pattern=" << pattern.to_string() 
                     << " ldx,cdx,rdx=" << ldx << "," << cdx << "," << rdx 
                     << " rdmstr=" << rdmstr.to_string_spinorb()
                     << " rdmstr2=" << rdmstr2.to_string_spinorb()
                     << " rdmstr=" << rdmstr.to_string()
                     << " rdmstr2=" << rdmstr2.to_string()
                     << " sgn=" << sgn 
                     << " val=" << std::setprecision(12) << sgn*val
                     << " idx,jdx=" << idx << "," << jdx
                     << " aux.rdm=" << aux.rdm(idx,jdx)
                     << " diff=" << diff
                     << std::endl;
                  assert(diff < 1.e-8);
                  if(is_same) rdm(jdx,idx) = tools::conjugate(rdm(idx,jdx));
               }
            }
         }
      }

   template <typename Qm, typename Tm, std::enable_if_t<Qm::ifabelian,int> = 0>
      void rdm_assemble_omp212(const bool is_same,
            const comb<Qm,Tm>& icomb,
            const char lkey, 
            const char ckey, 
            const char rkey,
            const bool ldagger, 
            const bool cdagger, 
            const bool rdagger, 
            const qoper_dict<Qm::ifabelian,Tm>& lqops, 
            const qoper_dict<Qm::ifabelian,Tm>& cqops, 
            const qoper_dict<Qm::ifabelian,Tm>& rqops, 
            const qtensor3<Qm::ifabelian,Tm>& wf3bra,
            const qtensor3<Qm::ifabelian,Tm>& wf3ket,
            const std::vector<int>& leval,
            const std::vector<int>& reval,
            linalg::matrix<Tm>& rdm,
            rdmaux<Tm>& aux,
            const int size,
            const int rank){
         // assemble rdms
         const auto& lops = lqops(lkey);
         const auto& cops = cqops(ckey);
         const auto& rops = rqops(rkey);
         int lparity = op2parity.at(lkey);
         int cparity = op2parity.at(ckey);
         int rparity = op2parity.at(rkey);
         // generate full set of indices         
         auto rindex2 = (rkey=='A' or rkey=='M')? oper_index_opA(rqops.cindex, rqops.ifkr, rqops.isym) :
            oper_index_opB(rqops.cindex, rqops.ifkr, rqops.isym, rqops.ifhermi);
         for(const auto& rdx : rindex2){
            //---------------------------------------------------------
            // broadcast lop to all processors to perform contractions
            auto iproc = distribute2(rkey,rqops.ifkr,size,rdx,rqops.sorb);
            stensor2<Tm> rop;
            if(iproc == rank){
               rop.init(rops.at(rdx).info);
               linalg::xcopy(rop.size(), rops.at(rdx).data(), rop.data());
            }
            if(size > 1) mpi_wrapper::broadcast(icomb.world, rop, iproc);
            //---------------------------------------------------------
            //std::cout << std::endl;
            //std::cout << "rop: key=" << rkey << " rdx=" << rdx << " normF()=" << rop.normF() << std::endl;
            //rop.to_matrix().print("rop");
            auto rstr = get_calst(rkey, rdx, rdagger);
            auto opxwf1 = oper_kernel_IOwf("cr", wf3ket, rop, rparity, rdagger);
            //wf3ket.print("wf3ket");
            for(const auto& cpr : cops){
               const auto& cdx = cpr.first;
               const auto& cop = cpr.second;
               //std::cout << "cop: key=" << ckey << " cdx=" << cdx << " normF()=" << cop.normF() << std::endl;
               //cop.to_matrix().print("cop");
               auto cstr = get_calst(ckey, cdx, cdagger);
               auto opxwf2 = oper_kernel_OIwf("cr", opxwf1, cop, cdagger);
               if((cparity+rparity)%2 == 1) opxwf2.row_signed();
               auto op2 = contract_qt3_qt3("cr", wf3bra, opxwf2);
               //wf3bra.print("wf3bra");
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
               for(int l=0; l<leval.size(); l++){
                  const auto& ldx = leval[l];
                  auto lop = ldagger? lops.at(ldx).H() : lops.at(ldx);
                  //std::cout << "lop: key=" << lkey << " ldx=" << ldx << " normF()=" << lop.normF() << std::endl;
                  //lop.to_matrix().print("lop");
                  auto lstr = get_calst(lkey, ldx, ldagger);
                  if(tools::is_complex<Tm>()) lop = lop.conj();
                  //op2.to_matrix().print("op2");
                  Tm val = contract_qt2_qt2_full(lop, op2);

                  //auto lop_mat = lop.to_matrix();
                  //auto op2_mat = op2.to_matrix();
                  //auto tmpval = linalg::xdot(lop_mat.size(),lop_mat.data(),op2_mat.data());
                  //std::cout << "tmpval=" << tmpval << std::endl;

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
                     if(aux.alg_mrpt2 != 2){
                    
                        rdm(idx,jdx) = sgn*val;
                        if(is_same) rdm(jdx,idx) = tools::conjugate(rdm(idx,jdx));
                     
                     }else{

                        // on the fly contraction
                        assert(!is_same);
                        aux.dsrg_contract(idx, jdx, sgn*val);
 
                     }
                  } // critical region
               } // l
            } // c
         } // r
      }

#endif

} // ctns

#endif
