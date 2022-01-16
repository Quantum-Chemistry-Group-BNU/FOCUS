#ifndef SYMBOLIC_OP1OP2XWF_H
#define SYMBOLIC_OP1OP2XWF_H

#include "symbolic_oper.h"

namespace ctns{

//
// opwf = (sum_{ij} oij*a1^(d1)[i]*a2^(d2)[i])^d*wf
//

template <typename Tm>
void symbolic_op1op2xwf_nkr(symbolic_task<Tm>& formulae,
			    const std::string block1,
	 	            const std::string block2,
		            const std::vector<int>& cindex1,
		            const std::vector<int>& cindex2,
			    const int isym,
			    const qsym& sym_op,
		            const std::map<std::pair<int,int>,Tm>& oij,
		            const bool ifdagger1,
		            const bool ifdagger2,
		            const bool ifdagger){
   symbolic_term<Tm> term;
   if(cindex1.size() <= cindex2.size()){
      // sum_i a1[i] * (sum_j oij a2[j])
      for(const auto& i : cindex1){
	 auto op1C = symbolic_oper(block1,'C',i);      
         auto op1 = ifdagger1? op1C : op1C.H();
	 auto sym_op1 = op1.get_qsym(isym); 
	 // tmp_op2 = sum_j oij a2[j]
	 symbolic_sum<Tm> tmp_op2;
	 for(const auto& j : cindex2){
            auto op2C = symbolic_oper(block2,'C',j);		 
	    auto op2 = ifdagger2? op2C : op2C.H();
	    auto sym_op2 = op2.get_qsym(isym);
	    if(sym_op != sym_op1 + sym_op2) continue;
	    tmp_op2.sum(oij.at(std::make_pair(i,j)),op2);
	 } // j
	 // skip if its size is zero, which is possible for some i
         if(tmp_op2.size() == 0) continue;
	 term = symbolic_term<Tm>(op1,tmp_op2);
	 if(ifdagger) term = term.H();
	 formulae.append(term);
      } // i
   }else{
      // this part appears when the branch is larger 
      // sum_j (sum_i oij a1[i]) * a2[j]
      for(const auto& j : cindex2){
	 auto op2C = symbolic_oper(block2,'C',j);
	 auto op2 = ifdagger2? op2C : op2C.H();
	 auto sym_op2 = ifdagger2? get_qsym_opC(isym,j) : -get_qsym_opC(isym,j);
	 // tmp_op1 = sum_i oij a1[i]
	 symbolic_sum<Tm> tmp_op1;
         for(const auto& i : cindex1){
	    auto op1C = symbolic_oper(block1,'C',i);
	    auto op1 = ifdagger1? op1C : op1C.H();
	    auto sym_op1 = ifdagger1? get_qsym_opC(isym,i) : -get_qsym_opC(isym,i); 
	    if(sym_op != sym_op1 + sym_op2) continue;
	    tmp_op1.sum(oij.at(std::make_pair(i,j)),op1);
	 } // i
         if(tmp_op1.size() == 0) continue;
	 term = symbolic_term<Tm>(tmp_op1,op2);
	 if(ifdagger) term = term.H();
	 formulae.append(term);
      } // j
   } // cindex1.size() <= cindex2.size() 
}

/*
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
*/

// opwf = (sum_{ij}oij*a1^(d)[i]*a2^(d)[i])*wf
template <typename Tm>
void symbolic_op1op2xwf(const bool ifkr,
			symbolic_task<Tm>& formulae,
		        const std::string block1,
	 	        const std::string block2,
		        const std::vector<int>& cindex1,
		        const std::vector<int>& cindex2,
			const int isym,
			const qsym& sym_op,
			const std::map<std::pair<int,int>,Tm>& oij,
		        const bool ifdagger1,
		        const bool ifdagger2,
		        const bool ifdagger){
   if(not ifkr){
      symbolic_op1op2xwf_nkr(formulae,block1,block2,cindex1,cindex2,
		      	     isym,sym_op,oij,ifdagger1,ifdagger2,ifdagger);
   }else{
/*
      symbolic_op1op2xwf_kr(formulae,block1,block2,cindex1,cindex2,
		      	    sym_op,oij,ifdagger1,ifdagger2,ifdagger);
*/
      exit(1);
   }
}

} // ctns

#endif
