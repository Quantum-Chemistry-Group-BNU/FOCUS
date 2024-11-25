#ifndef SWEEP_ONEDOT_SIGMA_H
#define SWEEP_ONEDOT_SIGMA_H

#ifdef _OPENMP
#include <omp.h>
#endif

#include "oper_dict.h"
#include "oper_timer.h"

namespace ctns{

   const bool debug_onedot_sigma = false;
   extern const bool debug_onedot_sigma;

   // construct H*wf: if ifkr=True, construct skeleton sigma vector 

   template <typename Tm>
      stensor3<Tm> onedot_Hx_local(const oper_dict<Tm>& lqops,
            const oper_dict<Tm>& rqops,
            const oper_dict<Tm>& cqops,
            const double& ecore,
            const stensor3<Tm>& wf,
            const int& size,
            const int& rank,
            const bool& ifdist1){
         if(debug_onedot_sigma) std::cout << "onedot_Hx_local" << std::endl;
         const size_t csize1 = lqops.cindex.size();
         const size_t csize2 = rqops.cindex.size();
         const bool ifNC = determine_NCorCN_Ham(lqops.oplist, rqops.oplist, csize1, csize2);
         const Tm scale = lqops.ifkr? 0.5 : 1.0;
         stensor3<Tm> Hwf;
         if(ifNC){
            // 1. H^l + 2. H^cr
            Hwf = oper_compxwf_opH("cr",wf,cqops,rqops,size,rank,ifdist1);
            if(!ifdist1 or rank==0){
               Hwf += contract_qt3_qt2("l",wf,lqops('H').at(0));
            }
         }else{
            // 1. H^lc + 2. H^r
            Hwf = oper_compxwf_opH("lc",wf,lqops,cqops,size,rank,ifdist1);
            if(!ifdist1 or rank==0){
               Hwf += contract_qt3_qt2("r",wf,rqops('H').at(0));
            }
         }
         Hwf *= scale;
         // add const term
         if(rank == 0){
            const Tm fac = scale*ecore;
            linalg::xaxpy(wf.size(), fac, wf.data(), Hwf.data());
         }
         return Hwf;
      }

   // --- Normal-Complementary partition ---
   // Generic formula: L=l, R=cr: A[l]*P[cr]+B[l]*Q[cr]
   // O^l*O^cr|lcr>psi[lcr] = O^l|l>O^cr|cr>(-1)^{p(l)*p(O^cr)}psi[lcr]
   //		         = O^l|l>( (-1)^{p(l)*p(O^cr)} (O^cr|cr>psi[lcr]) )
   template <typename Tm>
      stensor3<Tm> onedot_Hx_CSnc(const int index,
            const oper_dict<Tm>& lqops,
            const oper_dict<Tm>& rqops,
            const oper_dict<Tm>& cqops,
            const integral::two_body<Tm>& int2e,
            const stensor3<Tm>& wf,
            const int& size,
            const int& rank,
            const bool& ifdist1){
         if(debug_onedot_sigma) std::cout << "onedot_Hx_CSnc index=" << index << std::endl;
         const bool dagger = true;
         const auto& op1 = lqops('C').at(index);
         // p^L+*S^CR
         auto qt3n = oper_compxwf_opS("cr",wf,cqops,rqops,int2e,index,size,rank,ifdist1);
         qt3n.row_signed();
         auto Hwf = oper_kernel_OIwf("lc",qt3n,op1); // both lc/lr can work 
                                                     // h.c.
         auto qt3h = oper_compxwf_opS("cr",wf,cqops,rqops,int2e,index,size,rank,ifdist1,dagger);
         qt3h.row_signed();
         Hwf -= oper_kernel_OIwf("lc",qt3h,op1,dagger);
         return Hwf;
      }

   template <typename Tm>
      stensor3<Tm> onedot_Hx_SCnc(const int index,
            const int iformula,
            const oper_dict<Tm>& lqops,
            const oper_dict<Tm>& rqops,
            const oper_dict<Tm>& cqops,
            const integral::two_body<Tm>& int2e,
            const stensor3<Tm>& wf,
            const int& size,
            const int& rank){
         if(debug_onedot_sigma) std::cout << "onedot_Hx_SCnc index=" << index << std::endl;
         const bool dagger = true;
         const auto& op1 = lqops('S').at(index);
         auto qt3n = oper_normxwf_opC("cr",wf,cqops,rqops,index,iformula);
         qt3n.row_signed();
         auto qt3h = oper_normxwf_opC("cr",wf,cqops,rqops,index,iformula,dagger);
         qt3h.row_signed();
         auto Hwf = oper_kernel_OIwf("lc",qt3h,op1,dagger); 
         Hwf -= oper_kernel_OIwf("lc",qt3n,op1); 
         return Hwf;
      }

   template <typename Tm>
      stensor3<Tm> onedot_Hx_APnc(const int index,
            const oper_dict<Tm>& lqops,
            const oper_dict<Tm>& rqops,
            const oper_dict<Tm>& cqops,
            const integral::two_body<Tm>& int2e,
            const stensor3<Tm>& wf,
            const int& size,
            const int& rank){
         if(debug_onedot_sigma) std::cout << "onedot_Hx_APnc index=" << index << std::endl;
         const bool dagger = true;
         auto qt3n = oper_compxwf_opP("cr",wf,cqops,rqops,int2e,index);
         auto qt3h = oper_compxwf_opP("cr",wf,cqops,rqops,int2e,index,dagger);
         const auto& op1 = lqops('A').at(index);
         auto Hwf = oper_kernel_OIwf("lc",qt3n,op1);
         Hwf += oper_kernel_OIwf("lc",qt3h,op1,dagger);
         const Tm wt = lqops.ifkr? wfacAP(index) : 1.0;
         Hwf *= wt;
         return Hwf;
      }

   template <typename Tm>
      stensor3<Tm> onedot_Hx_BQnc(const int index,
            const oper_dict<Tm>& lqops,
            const oper_dict<Tm>& rqops,
            const oper_dict<Tm>& cqops,
            const integral::two_body<Tm>& int2e,
            const stensor3<Tm>& wf,
            const int& size,
            const int& rank){
         if(debug_onedot_sigma) std::cout << "onedot_Hx_BQnc index=" << index << std::endl;
         const bool dagger = true;
         auto qt3n = oper_compxwf_opQ("cr",wf,cqops,rqops,int2e,index);
         auto qt3h = oper_compxwf_opQ("cr",wf,cqops,rqops,int2e,index,dagger);
         const auto& op1 = lqops('B').at(index);
         auto Hwf = oper_kernel_OIwf("lc",qt3n,op1);
         Hwf += oper_kernel_OIwf("lc",qt3h,op1,dagger);
         const Tm wt = lqops.ifkr? wfacBQ(index) : wfac(index);
         Hwf *= wt;
         return Hwf;
      }

   // --- Complementary-Normal partition ---
   // Generic formula: L=lc, R=r: A[lc]*P[r]+B[lc]*Q[r]
   // O^lc*O^r|lcr>psi[lcr] = O^lc|lc>O^r|r>(-1)^{p(lc)*p(O^r)}psi[lcr]
   //		         = O^lc|lc>( (-1)^{p(l)*p(O^cr)} ((-1)^{p(c)*p(O^cr)} O^r|c>psi[lcr]) )
   template <typename Tm>
      stensor3<Tm> onedot_Hx_CScn(const int index,
            const int iformula,
            const oper_dict<Tm>& lqops,
            const oper_dict<Tm>& rqops,
            const oper_dict<Tm>& cqops,
            const integral::two_body<Tm>& int2e,
            const stensor3<Tm>& wf,
            const int& size,
            const int& rank){
         if(debug_onedot_sigma) std::cout << "onedot_Hx_CScn index=" << index << std::endl;
         const bool dagger = true;
         const auto& op2 = rqops('S').at(index);
         auto qt3n = oper_kernel_IOwf("cr",wf,op2,1); // p(c) is taken into account in IOwf
         qt3n.row_signed();
         auto qt3h = oper_kernel_IOwf("cr",wf,op2,1,dagger);
         qt3h.row_signed();
         auto Hwf = oper_normxwf_opC("lc",qt3n,lqops,cqops,index,iformula); // p(l) is taken into account
         Hwf -= oper_normxwf_opC("lc",qt3h,lqops,cqops,index,iformula,dagger);
         return Hwf;
      }

   template <typename Tm>
      stensor3<Tm> onedot_Hx_SCcn(const int index,
            const oper_dict<Tm>& lqops,
            const oper_dict<Tm>& rqops,
            const oper_dict<Tm>& cqops,
            const integral::two_body<Tm>& int2e,
            const stensor3<Tm>& wf,
            const int& size,
            const int& rank,
            const bool& ifdist1){
         if(debug_onedot_sigma) std::cout << "onedot_Hx_SCcn index=" << index << std::endl;
         const bool dagger = true;
         const int& q2 = index;
         const auto& op2 = rqops('C').at(q2);
         auto qt3n = oper_kernel_IOwf("cr",wf,op2,1);
         qt3n.row_signed(); 
         auto qt3h = oper_kernel_IOwf("cr",wf,op2,1,dagger);
         qt3h.row_signed();
         auto Hwf = oper_compxwf_opS("lc",qt3h,lqops,cqops,int2e,q2,size,rank,ifdist1,dagger);
         Hwf -= oper_compxwf_opS("lc",qt3n,lqops,cqops,int2e,q2,size,rank,ifdist1);
         return Hwf;
      }

   template <typename Tm>
      stensor3<Tm> onedot_Hx_PAcn(const int index,
            const oper_dict<Tm>& lqops,
            const oper_dict<Tm>& rqops,
            const oper_dict<Tm>& cqops,
            const integral::two_body<Tm>& int2e,
            const stensor3<Tm>& wf,
            const int& size,
            const int& rank){
         if(debug_onedot_sigma) std::cout << "onedot_Hx_PAcn index=" << index << std::endl;
         const bool dagger = true;
         const auto& op2 = rqops('A').at(index); 
         auto qt3n = oper_kernel_IOwf("cr",wf,op2,0);
         auto qt3h = oper_kernel_IOwf("cr",wf,op2,0,dagger);
         auto Hwf = oper_compxwf_opP("lc",qt3n,lqops,cqops,int2e,index);
         Hwf += oper_compxwf_opP("lc",qt3h,lqops,cqops,int2e,index,dagger);
         const Tm wt = lqops.ifkr? wfacAP(index) : 1.0;
         Hwf *= wt;
         return Hwf;
      }

   template <typename Tm>
      stensor3<Tm> onedot_Hx_QBcn(const int index,
            const oper_dict<Tm>& lqops,
            const oper_dict<Tm>& rqops,
            const oper_dict<Tm>& cqops,
            const integral::two_body<Tm>& int2e,
            const stensor3<Tm>& wf,
            const int& size,
            const int& rank){
         if(debug_onedot_sigma) std::cout << "onedot_Hx_QBcn index=" << index << std::endl;
         const bool dagger = true;
         const auto& op2 = rqops('B').at(index);
         auto qt3n = oper_kernel_IOwf("cr",wf,op2,0);
         auto qt3h = oper_kernel_IOwf("cr",wf,op2,0,dagger);
         auto Hwf = oper_compxwf_opQ("lc",qt3n,lqops,cqops,int2e,index);
         Hwf += oper_compxwf_opQ("lc",qt3h,lqops,cqops,int2e,index,dagger);
         const Tm wt = lqops.ifkr? wfacBQ(index) : wfac(index);
         Hwf *= wt;
         return Hwf;
      }

   // --- driver ---

   // Collect all Hx_funs 
   template <typename Tm>
      Hx_functors<Tm> onedot_Hx_functors(const opersu2_dictmap<Tm>& qops_dict,
            const integral::two_body<Tm>& int2e,
            const double& ecore,
            const stensor3su2<Tm>& wf,
            const int& size,
            const int& rank,
            const bool& ifdist1,
            const bool debug=false){
         std::cout << "error: no implementation of onedot_Hx_functors for su2!" << std::endl;
         exit(1);
      }
   template <typename Tm>
      Hx_functors<Tm> onedot_Hx_functors(const oper_dictmap<Tm>& qops_dict,
            const integral::two_body<Tm>& int2e,
            const double& ecore,
            const stensor3<Tm>& wf,
            const int& size,
            const int& rank,
            const bool& ifdist1,
            const bool debug=false){
         const auto& lqops = qops_dict.at("l");
         const auto& rqops = qops_dict.at("r");
         const auto& cqops = qops_dict.at("c");
         const int isym = lqops.isym;
         const bool ifkr = lqops.ifkr;
         const size_t csize1 = lqops.cindex.size();
         const size_t csize2 = rqops.cindex.size();
         const bool ifNC = determine_NCorCN_Ham(lqops.oplist, rqops.oplist, csize1, csize2);

         Hx_functors<Tm> Hx_funs;

         // Local terms:
         Hx_functor<Tm> Hx("Hloc", 0, 0);
         Hx.opxwf = bind(&onedot_Hx_local<Tm>,
               std::cref(lqops), std::cref(rqops), std::cref(cqops), 
               std::cref(ecore), std::cref(wf), std::cref(size), std::cref(rank),
               std::cref(ifdist1));
         Hx_funs.push_back(Hx);
         
         // One-index terms:
         // 3. p1^l+*Sp1^cr + h.c. or 4. q2^r+*Sq2^lc + h.c. = -Sq2^lc*q2^r + h.c.
         const auto& cnindex = ifNC? lqops.cindex : rqops.cindex;
         auto cnlabel = ifNC? "CSnc" : "SCcn";
         auto cnfun = ifNC? &onedot_Hx_CSnc<Tm> : &onedot_Hx_SCcn<Tm>;
         for(const auto& index : cnindex){
            Hx_functor<Tm> Hx(cnlabel, index, 0);
            Hx.opxwf = bind(cnfun, index,
                  std::cref(lqops), std::cref(rqops), std::cref(cqops), 
                  std::cref(int2e), std::cref(wf), std::cref(size), std::cref(rank),
                  std::cref(ifdist1));
            Hx_funs.push_back(Hx); 
         }
         // 4. q2^cr+*Sq2^l + h.c. or 3. p1^lc+*Sp1^r + h.c.
         auto ccinfo = ifNC? oper_combine_opC(cqops.cindex, rqops.cindex) :
            oper_combine_opC(lqops.cindex, cqops.cindex); 
         auto cclabel = ifNC? "SCnc" : "CScn";
         auto ccfun = ifNC? &onedot_Hx_SCnc<Tm> : &onedot_Hx_CScn<Tm>;
         for(const auto& pr : ccinfo){
            int index = pr.first;
            int iproc = distribute1(ifkr,size,index);
            if(!ifdist1 or iproc==rank){ 
               int iformula = pr.second;
               Hx_functor<Tm> Hx(cclabel, index, iformula);
               Hx.opxwf = bind(ccfun, index, iformula, 
                     std::cref(lqops), std::cref(rqops), std::cref(cqops), 
                     std::cref(int2e), std::cref(wf), std::cref(size), std::cref(rank));
               Hx_funs.push_back(Hx);
            }
         }
         
         // Two-index terms:
         auto aindex_dist = ifNC? oper_index_opA_dist(lqops.cindex, ifkr, isym, size, rank, int2e.sorb) : 
            oper_index_opA_dist(rqops.cindex, ifkr, isym, size, rank, int2e.sorb);
         auto bindex_dist = ifNC? oper_index_opB_dist(lqops.cindex, ifkr, isym, size, rank, int2e.sorb) : 
            oper_index_opB_dist(rqops.cindex, ifkr, isym, size, rank, int2e.sorb);
         auto afun = ifNC? &onedot_Hx_APnc<Tm> : &onedot_Hx_PAcn<Tm>;
         auto bfun = ifNC? &onedot_Hx_BQnc<Tm> : &onedot_Hx_QBcn<Tm>;
         auto alabel = ifNC? "APnc" : "PAcn";
         auto blabel = ifNC? "BQnc" : "QBcn";
         // 5. Apq^l*Ppq^cr + h.c. or Ars^r*Prs^lc + h.c.
         for(const auto& index : aindex_dist){
            Hx_functor<Tm> Hx(alabel, index, 0);
            Hx.opxwf = bind(afun, index, std::cref(lqops), std::cref(rqops), std::cref(cqops), 
                  std::cref(int2e), std::cref(wf), std::cref(size), std::cref(rank));
            Hx_funs.push_back(Hx);
         }
         // 6. Bps^l*Qps^cr (using Hermicity) or Qqr^lc*Bqr^r (using Hermicity)
         for(const auto& index : bindex_dist){
            Hx_functor<Tm> Hx(blabel, index, 0);
            Hx.opxwf = bind(bfun, index, std::cref(lqops), std::cref(rqops), std::cref(cqops), 
                  std::cref(int2e), std::cref(wf), std::cref(size), std::cref(rank));
            Hx_funs.push_back(Hx);
         }

         // debug
         if(rank == 0 and debug){
            std::cout << "onedot_Hx_functors: ifNC=" << ifNC
               << " size=" << Hx_funs.size() 
               << " " << cnlabel << ":" << cnindex.size()
               << " " << cclabel << ":" << ccinfo.size()
               << " " << alabel << ":" << aindex_dist.size()
               << " " << blabel << ":" << bindex_dist.size()
               << std::endl; 
         }
         return Hx_funs;
      }

   template <typename Tm> 
      void onedot_Hx(Tm* y,
            const Tm* x,
            Hx_functors<Tm>& Hx_funs,
            stensor3su2<Tm>& wf,
            const int size,
            const int rank){
         std::cout << "error: no implementation of onedot_Hx for su2!" << std::endl;
         exit(1);
      }
   template <typename Tm> 
      void onedot_Hx(Tm* y,
            const Tm* x,
            Hx_functors<Tm>& Hx_funs,
            stensor3<Tm>& wf,
            const int size,
            const int rank){
         auto t0 = tools::get_time();
#ifdef _OPENMP
         int maxthreads = omp_get_max_threads();
#else
         int maxthreads = 1;
#endif
         if(rank == 0 && debug_onedot_sigma){ 
            std::cout << "ctns::onedot_Hx size=" << size 
               << " maxthreads=" << maxthreads
               << std::endl;
         }
         //=======================
         // Parallel evaluation
         //=======================
         wf.from_array(x);
         // initialization
         std::vector<stensor3<Tm>> Hwfs(maxthreads);
         for(int i=0; i<maxthreads; i++){
            Hwfs[i].init(wf.info);
         }
         auto t1 = tools::get_time();
         // compute
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
         for(int i=0; i<Hx_funs.size(); i++){
#ifdef _OPENMP
            int omprank = omp_get_thread_num();
#else
            int omprank = 0;
#endif
            Hwfs[omprank] += Hx_funs[i]();
         }
         auto t2 = tools::get_time();
         // reduction & save
         for(int i=1; i<maxthreads; i++){
            Hwfs[0] += Hwfs[i];
         }
         Hwfs[0].to_array(y);
         auto t3 = tools::get_time();
         oper_timer.tHxInit += tools::get_duration(t1-t0);
         oper_timer.tHxCalc += tools::get_duration(t2-t1);
         oper_timer.tHxFinl += tools::get_duration(t3-t2);
      }

} // ctns

#endif
