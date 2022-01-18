#ifndef OPER_OP1OP2XWF_H
#define OPER_OP1OP2XWF_H

#include "oper_kernel.h"

namespace ctns{

//
// opwf = (sum_{ij} oij*a1^(d1)[i]*a2^(d2)[i])^d*wf
//

template <typename Tm>
void oper_op1op2xwf_nkr(stensor3<Tm>& opwf,
		        const std::string superblock,
		        const stensor3<Tm>& site,
		        const oper_map<Tm>& qops1C,
		        const oper_map<Tm>& qops2C,
			const qsym& sym_op,
		        const std::map<std::pair<int,int>,Tm>& oij,
			const bool ifdagger1,
			const bool ifdagger2,
			const bool ifdagger){
   const Tm sgn = ifdagger? -1.0 : 1.0;
   const auto& qrow1 = (qops1C.begin()->second).info.qrow;
   const auto& qcol1 = (qops1C.begin()->second).info.qcol;
   const auto& qrow2 = (qops2C.begin()->second).info.qrow;
   const auto& qcol2 = (qops2C.begin()->second).info.qcol;
   if(qops1C.size() <= qops2C.size()){
      // sum_i a1[i] * (sum_j oij a2[j])
      for(const auto& op1C : qops1C){
         int i = op1C.first;
         const auto& op1 = ifdagger1? op1C.second : op1C.second.H();
	 // tmp_op2 = sum_j oij a2[j]
	 stensor2<Tm> tmp_op2(sym_op-op1.info.sym, qrow2, qcol2);
	 int N = tmp_op2.size();
	 for(const auto& op2C : qops2C){
	    bool symAllowed = tmp_op2.info.sym == (ifdagger2? op2C.second.info.sym :
			   				     -op2C.second.info.sym); 
	    if(not symAllowed) continue;
	    int j = op2C.first;
	    const auto& op2 = ifdagger2? op2C.second : op2C.second.H();	
	    //tmp_op2 += oij[std::make_pair(i,j)]*op2;
	    linalg::xaxpy(N, oij.at(std::make_pair(i,j)), op2.data(), tmp_op2.data());
	 }
	 //opwf += sgn*oper_kernel_OOwf(superblock,site,op1,tmp_op2,1,ifdagger);
	 auto tmp = oper_kernel_OOwf(superblock,site,op1,tmp_op2,1,ifdagger); 
	 linalg::xaxpy(opwf.size(), sgn, tmp.data(), opwf.data());
      }
   }else{
      // this part appears when the branch is larger 
      // sum_j (sum_i oij a1[i]) * a2[j]
      for(const auto& op2C : qops2C){
	 int j = op2C.first;
	 const auto& op2 = ifdagger2? op2C.second : op2C.second.H();
	 // tmp_op1 = sum_i oij a1[i]
	 stensor2<Tm> tmp_op1(sym_op-op2.info.sym, qrow1, qcol1);
	 int N = tmp_op1.size();
         for(const auto& op1C : qops1C){
	    bool symAllowed = tmp_op1.info.sym == (ifdagger1? op1C.second.info.sym : 
			    				     -op1C.second.info.sym);
	    if(not symAllowed) continue;
	    int i = op1C.first;
	    const auto& op1 = ifdagger1? op1C.second : op1C.second.H();
	    //tmp_op1 += oij[std::make_pair(i,j)]*op1;
	    linalg::xaxpy(N, oij.at(std::make_pair(i,j)), op1.data(), tmp_op1.data());
	 }
	 //opwf += sgn*oper_kernel_OOwf(superblock,site,tmp_op1,op2,1,ifdagger);
	 auto tmp = oper_kernel_OOwf(superblock,site,tmp_op1,op2,1,ifdagger);
	 linalg::xaxpy(opwf.size(), sgn, tmp.data(), opwf.data());
      }
   }
}

// TRS version: only unbar part of creation ops is stored
template <typename Tm>
void oper_op1op2xwf_kr(stensor3<Tm>& opwf,
		       const std::string superblock,
		       const stensor3<Tm>& site,
		       const oper_map<Tm>& qops1C,
		       const oper_map<Tm>& qops2C,
		       const qsym& sym_op,
		       const std::map<std::pair<int,int>,Tm>& oij,
		       const bool ifdagger1,
		       const bool ifdagger2,
		       const bool ifdagger){
   const Tm sgn = ifdagger? -1.0 : 1.0;
   const auto& qrow1 = (qops1C.begin()->second).info.qrow;
   const auto& qcol1 = (qops1C.begin()->second).info.qcol;
   const auto& qrow2 = (qops2C.begin()->second).info.qrow;
   const auto& qcol2 = (qops2C.begin()->second).info.qcol;
   if(qops1C.size() <= qops2C.size()){
      // sum_i a1[i] * (sum_j oij a2[j])
      for(const auto& op1C : qops1C){
         int ia = op1C.first, ib = ia+1;
	 const auto& op1a = ifdagger1? op1C.second : op1C.second.H();
	 const auto& op1b = op1a.K(1);
	 // top2 = sum_j oij a2[j]
	 stensor2<Tm> top2a(sym_op-op1a.info.sym, qrow2, qcol2);
	 stensor2<Tm> top2b(sym_op-op1b.info.sym, qrow2, qcol2);
	 int Na = top2a.size(), Nb = top2b.size();
	 for(const auto& op2C : qops2C){
	    int ja = op2C.first, jb = ja+1;
	    const auto& op2a = ifdagger2? op2C.second : op2C.second.H();
            const auto& op2b = op2a.K(1);	    
	    //top2a += oij[std::make_pair(ia,ja)]*op2a + oij[std::make_pair(ia,jb)]*op2b;
	    //top2b += oij[std::make_pair(ib,ja)]*op2a + oij[std::make_pair(ib,jb)]*op2b;
	    linalg::xaxpy(Na, oij.at(std::make_pair(ia,ja)), op2a.data(), top2a.data());
	    linalg::xaxpy(Na, oij.at(std::make_pair(ia,jb)), op2b.data(), top2a.data());
	    linalg::xaxpy(Nb, oij.at(std::make_pair(ib,ja)), op2a.data(), top2b.data());
	    linalg::xaxpy(Nb, oij.at(std::make_pair(ib,jb)), op2b.data(), top2b.data());
	 }
	 //opwf += sgn*(oper_kernel_OOwf(superblock,site,op1a,top2a,1,ifdagger)
	 //            +oper_kernel_OOwf(superblock,site,op1b,top2b,1,ifdagger));
	 auto tmpa = oper_kernel_OOwf(superblock,site,op1a,top2a,1,ifdagger);
	 auto tmpb = oper_kernel_OOwf(superblock,site,op1b,top2b,1,ifdagger);
	 linalg::xaxpy(opwf.size(), sgn, tmpa.data(), opwf.data());
	 linalg::xaxpy(opwf.size(), sgn, tmpb.data(), opwf.data());
      }
   // sum_j (sum_i oij a1[i]) * a2[j]
   }else{
      // this part appears when the branch is larger 
      for(const auto& op2C : qops2C){
	 int ja = op2C.first, jb = ja+1;
	 const auto& op2a = ifdagger2? op2C.second : op2C.second.H();
	 const auto& op2b = op2a.K(1);
	 // top1 = sum_i oij a1[i]
	 stensor2<Tm> top1a(sym_op-op2a.info.sym, qrow1, qcol1);
	 stensor2<Tm> top1b(sym_op-op2b.info.sym, qrow1, qcol1);
	 int Na = top1a.size(), Nb = top1b.size();
         for(const auto& op1C : qops1C){
	    int ia = op1C.first, ib = ia+1;
	    const auto& op1a = ifdagger1? op1C.second : op1C.second.H();
	    const auto& op1b = op1a.K(1);
	    //top1a += oij[std::make_pair(ia,ja)]*op1a + oij[std::make_pair(ib,ja)]*op1b;
	    //top1b += oij[std::make_pair(ia,jb)]*op1a + oij[std::make_pair(ib,jb)]*op1b;
	    linalg::xaxpy(Na, oij.at(std::make_pair(ia,ja)), op1a.data(), top1a.data());
	    linalg::xaxpy(Na, oij.at(std::make_pair(ib,ja)), op1b.data(), top1a.data());
	    linalg::xaxpy(Nb, oij.at(std::make_pair(ia,jb)), op1a.data(), top1b.data());
	    linalg::xaxpy(Nb, oij.at(std::make_pair(ib,jb)), op1b.data(), top1b.data());
	 }
	 //opwf += sgn*(oper_kernel_OOwf(superblock,site,top1a,op2a,1,ifdagger)
	 //            +oper_kernel_OOwf(superblock,site,top1b,op2b,1,ifdagger));
	 auto tmpa = oper_kernel_OOwf(superblock,site,top1a,op2a,1,ifdagger);
	 auto tmpb = oper_kernel_OOwf(superblock,site,top1b,op2b,1,ifdagger);
	 linalg::xaxpy(opwf.size(), sgn, tmpa.data(), opwf.data());
	 linalg::xaxpy(opwf.size(), sgn, tmpb.data(), opwf.data());
      }
   }
}

// opwf = (sum_{ij}oij*a1^(d)[i]*a2^(d)[i])*wf
template <typename Tm>
void oper_op1op2xwf(const bool& ifkr,
		    stensor3<Tm>& opwf,
		    const std::string superblock,
		    const stensor3<Tm>& site,
		    const oper_map<Tm>& qops1C,
		    const oper_map<Tm>& qops2C,
		    const qsym& sym_op,
		    const std::map<std::pair<int,int>,Tm>& oij,
		    const bool ifdagger1,
		    const bool ifdagger2,
		    const bool ifdagger){
   if(!ifkr){
      oper_op1op2xwf_nkr(opwf, superblock, site, qops1C, qops2C, 
		         sym_op, oij, ifdagger1, ifdagger2, ifdagger);
   }else{
      oper_op1op2xwf_kr(opwf, superblock, site, qops1C, qops2C, 
		        sym_op, oij, ifdagger1, ifdagger2, ifdagger);
   }
}

} // ctns

#endif
