#ifndef OPER_RBASIS_H
#define OPER_RBASIS_H

/*
 Construct renormalized operator <r|O|r'> using determinant expansion 
*/

#include "ctns_comb.h" 
#include "oper_dict.h"

namespace ctns{

const bool debug_rops = true;
extern const bool debug_rops;
   
const double thresh_rops = 1.e-5;
extern const double thresh_rops;

inline void setup_orb2pos_map(const std::vector<int>& rsupp,
		              std::vector<int>& rspinorbs,
		              std::map<int,int>& orb2pos){
   int idx = 0;
   for(auto it=rsupp.begin(); it!=rsupp.end(); it++){
      int ks = *it;
      orb2pos[2*ks] = idx; // abab counting and rsupp[0] comes first!
      orb2pos[2*ks+1] = idx+1;
      idx += 2;
      rspinorbs.push_back(2*ks); // spin-orbitals
      rspinorbs.push_back(2*ks+1);
   }
   if(debug_rops){
      tools::print_vector(rsupp,"rsupport");
      tools::print_vector(rspinorbs,"rspinorbs");
   }
}

// check normal operators: C, A, B
template <typename Km>
void oper_check_rbasis(const comb<Km>& bra,
		       const comb<Km>& ket, 
		       const comb_coord& p,
		       oper_dict<typename Km::dtype>& qops,
		       const char opname,
		       const int size,
		       const int rank){
   using Tm = typename Km::dtype;
   if(p == std::make_pair(0,0)) return; // no rbases at the start 
   std::cout << "ctns::oper_check_rbasis coord=" << p 
	     << " opname=" << opname 
	     << " size=" << size
	     << " rank=" << rank
	     << std::endl; 
   int nop = 0, nfail = 0; 
   double maxdiff = 0.0;
   const auto& node = bra.topo.get_node(p);
   const auto& rindex = bra.topo.rindex;
   const auto& bsite = bra.rsites[rindex.at(p)];
   const auto& ksite = ket.rsites[rindex.at(p)];
   const auto& rbasis0 = bra.rbases[rindex.at(p)];
   const auto& rbasis1 = ket.rbases[rindex.at(p)];
   // setup mapping for orbitals to local support
   const auto& rsupp = node.rsupport;
   std::vector<int> rspinorbs;
   std::map<int,int> orb2pos;
   setup_orb2pos_map(rsupp, rspinorbs, orb2pos);
   //----------------
   // check for ap^+
   //----------------
   if(opname == 'C'){
   for(const auto& opC: qops('C')){
      const auto& op = opC.second;
      int orb_p = opC.first;
      int pos = orb2pos.at(orb_p);
      // build
      int dim0 = bsite.info.qrow.get_dimAll();
      int dim1 = ksite.info.qrow.get_dimAll();
      linalg::matrix<Tm> tmat(dim0,dim1);
      int ioff0 = 0;
      for(const auto& rsec0 : rbasis0){
	 const auto& space0 = rsec0.space;
	 const auto& coeff0 = rsec0.coeff;
	 int m0 = coeff0.rows();
	 int n0 = coeff0.cols();
	 int ioff1 = 0;
         for(const auto& rsec1 : rbasis1){
	    const auto& space1 = rsec1.space;
	    const auto& coeff1 = rsec1.coeff;
	    int m1 = coeff1.rows();
	    int n1 = coeff1.cols();
	    // compute <bdet|op|kdet>
	    linalg::matrix<Tm> mat(m0,m1);
	    for(int i1=0; i1<m1; i1++){
	       // ap^+|ket>
	       auto res = space1[i1].cre(pos);
	       if(res.first == 0) continue;
	       // check <bra|
	       for(int i0=0; i0<m0; i0++){
	          if(res.second == space0[i0]){
		     mat(i0,i1) = res.first;
		  }
	       }
	    }
	    // contraction
	    auto tmp1 = linalg::xgemm("C","N",coeff0,mat);
	    auto tmp2 = linalg::xgemm("N","N",tmp1,coeff1);
	    // save
	    for(int i1=0; i1<n1; i1++){
	       for(int i0=0; i0<n0; i0++){
		  tmat(ioff0+i0,ioff1+i1) = tmp2(i0,i1);
	       }
	    }
	    ioff1 += n1;
	 } // rec1
	 ioff0 += n0;
      } // rec0
      auto opmat = op.to_matrix();
      double diff = (opmat-tmat).normF();
      maxdiff = std::max(diff,maxdiff);
      if(diff > thresh_rops) nfail++;
      if(debug_rops){
         std::cout << std::scientific << std::setprecision(8);
         std::cout << " C: p=" << orb_p 
		   << " rank=" << rank
                   << " |opmat|=" << opmat.normF()
                   << " |tmat|=" << tmat.normF()
                   << " diff=" << diff
                   << " fail=" << (diff > thresh_rops)	
                   << std::endl;
      }
      nop++;
   } // op
   } 
   //--------------------------
   // check for Apq = ap^+aq^+
   //--------------------------
   if(opname == 'A'){ 
   for(const auto& opA : qops('A')){
      const auto& op = opA.second;
      auto pq = oper_unpack(opA.first);
      int orb_p = pq.first;
      int orb_q = pq.second;
      int pos_p = orb2pos.at(orb_p);
      int pos_q = orb2pos.at(orb_q);
      // build
      int dim0 = bsite.info.qrow.get_dimAll();
      int dim1 = ksite.info.qrow.get_dimAll();
      linalg::matrix<Tm> tmat(dim0,dim1);
      int ioff0 = 0;
      for(const auto& rsec0 : rbasis0){
	 const auto& space0 = rsec0.space;
	 const auto& coeff0 = rsec0.coeff;
	 int m0 = coeff0.rows();
	 int n0 = coeff0.cols();
	 int ioff1 = 0;
         for(const auto& rsec1 : rbasis1){
	    const auto& space1 = rsec1.space;
	    const auto& coeff1 = rsec1.coeff;
	    int m1 = coeff1.rows();
	    int n1 = coeff1.cols();
	    // compute <bdet|op|kdet>
	    linalg::matrix<Tm> mat(m0,m1);
	    for(int i1=0; i1<m1; i1++){
	       // ap^+aq^+|ket>
	       auto res1 = space1[i1].cre(pos_q);
	       if(res1.first == 0) continue;
	       auto res2 = res1.second.cre(pos_p);
	       if(res2.first == 0) continue;
	       // check <bra|
	       for(int i0=0; i0<m0; i0++){
	          if(res2.second == space0[i0]){
	             mat(i0,i1) = res1.first*res2.first;
	          }
	       }
	    }
	    // contraction
	    auto tmp1 = linalg::xgemm("C","N",coeff0,mat);
	    auto tmp2 = linalg::xgemm("N","N",tmp1,coeff1);
	    // save
	    for(int i1=0; i1<n1; i1++){
	       for(int i0=0; i0<n0; i0++){
		  tmat(ioff0+i0,ioff1+i1) = tmp2(i0,i1);
	       }
	    }
	    ioff1 += n1;
	 }
	 ioff0 += n0;
      }
      auto opmat = op.to_matrix();
      double diff = (opmat-tmat).normF();
      maxdiff = std::max(diff,maxdiff);
      if(diff > thresh_rops) nfail++;
      if(debug_rops){
         std::cout << std::scientific << std::setprecision(8);
         std::cout << " A: p,q=" << orb_p << "," << orb_q
		   << " rank=" << rank
                   << " |opmat|=" << opmat.normF()
                   << " |tmat|=" << tmat.normF()
                   << " diff=" << diff
                   << " fail=" << (diff > thresh_rops)
           	   << std::endl;
      }
      nop++;
   } // op
   }
   //------------------------
   // check for Bpq = ap^+aq
   //------------------------
   if(opname == 'B'){ 
   for(const auto& opB : qops('B')){
      const auto& op = opB.second;
      auto pq = oper_unpack(opB.first);
      int orb_p = pq.first;
      int orb_q = pq.second;
      int pos_p = orb2pos.at(orb_p);
      int pos_q = orb2pos.at(orb_q);
      // build
      int dim0 = bsite.info.qrow.get_dimAll();
      int dim1 = ksite.info.qrow.get_dimAll();
      linalg::matrix<Tm> tmat(dim0,dim1);
      int ioff0 = 0;
      for(const auto& rsec0 : rbasis0){
	 const auto& space0 = rsec0.space;
	 const auto& coeff0 = rsec0.coeff;
	 int m0 = coeff0.rows();
	 int n0 = coeff0.cols();
	 int ioff1 = 0;
         for(const auto& rsec1 : rbasis1){
	    const auto& space1 = rsec1.space;
	    const auto& coeff1 = rsec1.coeff;
	    int m1 = coeff1.rows();
	    int n1 = coeff1.cols();
	    // compute <bdet|op|kdet>
	    linalg::matrix<Tm> mat(m0,m1);
	    for(int i1=0; i1<m1; i1++){
	       // ap^+aq|ket>
	       auto res1 = space1[i1].ann(pos_q);
	       if(res1.first == 0) continue;
	       auto res2 = res1.second.cre(pos_p);
	       if(res2.first == 0) continue;
	       // check <bra|
	       for(int i0=0; i0<m0; i0++){
	          if(res2.second == space0[i0]){
	             mat(i0,i1) = res1.first*res2.first;
	          }
	       }
	    }
	    // contraction
	    auto tmp1 = linalg::xgemm("C","N",coeff0,mat);
	    auto tmp2 = linalg::xgemm("N","N",tmp1,coeff1);
	    // save
	    for(int i1=0; i1<n1; i1++){
	       for(int i0=0; i0<n0; i0++){
		  tmat(ioff0+i0,ioff1+i1) = tmp2(i0,i1);
	       }
	    }
	    ioff1 += n1;
	 }
	 ioff0 += n0;
      }
      auto opmat = op.to_matrix();
      double diff = (opmat-tmat).normF();
      maxdiff = std::max(diff,maxdiff);
      if(diff > thresh_rops) nfail++;
      if(debug_rops){
         std::cout << std::scientific << std::setprecision(8);
         std::cout << " B: p,q=" << orb_p << "," << orb_q
		   << " rank=" << rank
                   << " |opmat|=" << opmat.normF()
                   << " |tmat|=" << tmat.normF()
                   << " diff=" << diff 
                   << " fail=" << (diff > thresh_rops)	
           	   << std::endl;
      }
      nop++;
   } // op
   }
   std::cout << "no. of ops = " << nop 
	     << " failed = " << nfail 
	     << " maxdiff = " << maxdiff << std::endl;
   if(nfail>0 || std::abs(maxdiff)>thresh_rops) exit(1);
}

// check complementary operators: P, Q, S, H
template <typename Km>
void oper_check_rbasis(const comb<Km>& bra,
		       const comb<Km>& ket, 
		       const comb_coord& p,
		       oper_dict<typename Km::dtype>& qops,
		       const char opname,
	               const integral::two_body<typename Km::dtype>& int2e,
	               const integral::one_body<typename Km::dtype>& int1e,
		       const int size,
		       const int rank,
		       const bool ifdist1=false){
   using Tm = typename Km::dtype;
   if(p == std::make_pair(0,0)) return; // no rbases at the start 
   std::cout << "ctns::oper_check_rbasis coord=" << p 
	     << " opname=" << opname
	     << " size=" << size
	     << " rank=" << rank
	     << std::endl; 
   int nop = 0, nfail = 0;
   double maxdiff = 0.0;
   const auto& node = bra.topo.get_node(p);
   const auto& rindex = bra.topo.rindex;
   const auto& bsite = bra.rsites[rindex.at(p)];
   const auto& ksite = ket.rsites[rindex.at(p)];
   const auto& rbasis0 = bra.rbases[rindex.at(p)];
   const auto& rbasis1 = ket.rbases[rindex.at(p)];
   // setup mapping for orbitals to local support
   const auto& rsupp = node.rsupport;
   std::vector<int> rspinorbs;
   std::map<int,int> orb2pos;
   setup_orb2pos_map(rsupp, rspinorbs, orb2pos);
   //-------------------------------------
   // check for Ppq = <pq||sr> aras [r>s]
   //-------------------------------------
   if(opname == 'P'){
   for(const auto& opP : qops('P')){
      const auto& op = opP.second;
      auto pq = oper_unpack(opP.first);
      int orb_p = pq.first;
      int orb_q = pq.second;
      // build
      int dim0 = bsite.info.qrow.get_dimAll();
      int dim1 = ksite.info.qrow.get_dimAll();
      linalg::matrix<Tm> tmat(dim0,dim1);
      int ioff0 = 0;
      for(const auto& rsec0 : rbasis0){
         const auto& space0 = rsec0.space;
         const auto& coeff0 = rsec0.coeff;
         int m0 = coeff0.rows();
         int n0 = coeff0.cols();
         int ioff1 = 0;
         for(const auto& rsec1 : rbasis1){
            const auto& space1 = rsec1.space;
            const auto& coeff1 = rsec1.coeff;
            int m1 = coeff1.rows();
            int n1 = coeff1.cols();
            // compute <bdet|op|kdet>
	    linalg::matrix<Tm> mat(m0,m1);
            for(int i1=0; i1<m1; i1++){
               // <bra|PpLqL^{R}|ket> = <pq||sr> aras [r>s]
               for(int orb_s : rspinorbs){
                  int pos_s = orb2pos[orb_s];
                  auto res1 = space1[i1].ann(pos_s);
                  if(res1.first == 0) continue;
                  for(int orb_r : rspinorbs){
      		     if(orb_r <= orb_s) continue; // r>s
      		     int pos_r = orb2pos[orb_r];
                     auto res2 = res1.second.ann(pos_r);
                     if(res2.first == 0) continue;
                     for(int i0=0; i0<m0; i0++){
     		        if(res2.second != space0[i0]) continue;
                        mat(i0,i1) += int2e.get(orb_p,orb_q,orb_s,orb_r)
                                    * static_cast<double>(res1.first*res2.first);
                     } // i0
                  } // r
               } // s
            } // i1
            // contraction
            auto tmp1 = linalg::xgemm("C","N",coeff0,mat);
            auto tmp2 = linalg::xgemm("N","N",tmp1,coeff1);
            // save
            for(int i1=0; i1<n1; i1++){
               for(int i0=0; i0<n0; i0++){
                  tmat(ioff0+i0,ioff1+i1) = tmp2(i0,i1);
               }
            }
            ioff1 += n1;
         }
         ioff0 += n0;
      }
      auto opmat = op.to_matrix();
      double diff = (opmat-tmat).normF();
      maxdiff = std::max(diff,maxdiff);
      if(diff > thresh_rops) nfail++;
      if(debug_rops){
         std::cout << std::scientific << std::setprecision(8);
         std::cout << " P: p,q=" << orb_p << "," << orb_q
		   << " rank=" << rank
                   << " |opmat|=" << opmat.normF()
                   << " |tmat|=" << tmat.normF()
                   << " diff=" << diff 
                   << " fail=" << (diff > thresh_rops)	
           	   << std::endl;
	 if(diff > thresh_rops){
	    opmat.print("opmat");
	    tmat.print("tmat");
	    exit(1);
	 }
      }
      nop++;
   } // op
   }
   //---------------------------------
   // check for Qps = <pq||sr> aq^+ar
   //---------------------------------
   if(opname == 'Q'){ 
   for(const auto& opQ : qops('Q')){
      const auto& op = opQ.second;
      auto ps = oper_unpack(opQ.first);
      int orb_p = ps.first;
      int orb_s = ps.second;
      // build
      int dim0 = bsite.info.qrow.get_dimAll();
      int dim1 = ksite.info.qrow.get_dimAll();
      linalg::matrix<Tm> tmat(dim0,dim1);
      int ioff0 = 0;
      for(const auto& rsec0 : rbasis0){
         const auto& space0 = rsec0.space;
         const auto& coeff0 = rsec0.coeff;
         int m0 = coeff0.rows();
         int n0 = coeff0.cols();
         int ioff1 = 0;
         for(const auto& rsec1 : rbasis1){
            const auto& space1 = rsec1.space;
            const auto& coeff1 = rsec1.coeff;
            int m1 = coeff1.rows();
            int n1 = coeff1.cols();
            // compute <bdet|op|kdet>
	    linalg::matrix<Tm> mat(m0,m1);
            for(int i1=0; i1<m1; i1++){
   	       // check for <bra|Qps|ket> = <pq||sr> aq^+ar
               for(int orb_r : rspinorbs){
                  int pos_r = orb2pos[orb_r];
                  auto res1 = space1[i1].ann(pos_r);
                  if(res1.first == 0) continue;
                  for(int orb_q : rspinorbs){
                     int pos_q = orb2pos[orb_q];
                     auto res2 = res1.second.cre(pos_q);
                     if(res2.first == 0) continue;
                     for(int i0=0; i0<m0; i0++){
     		        if(res2.second != space0[i0]) continue;
                        mat(i0,i1) += int2e.get(orb_p,orb_q,orb_s,orb_r)
                                    * static_cast<double>(res1.first*res2.first);
                     } // i0
     	          } // q
               } // r
            } // i1
            // contraction
            auto tmp1 = linalg::xgemm("C","N",coeff0,mat);
            auto tmp2 = linalg::xgemm("N","N",tmp1,coeff1);
            // save
            for(int i1=0; i1<n1; i1++){
               for(int i0=0; i0<n0; i0++){
                  tmat(ioff0+i0,ioff1+i1) = tmp2(i0,i1);
               }
            }
            ioff1 += n1;
         }
         ioff0 += n0;
      }
      auto opmat = op.to_matrix();
      double diff = (opmat-tmat).normF();
      maxdiff = std::max(diff,maxdiff);
      if(diff > thresh_rops) nfail++;
      if(debug_rops){
         std::cout << std::scientific << std::setprecision(8);
         std::cout << " Q: p,s=" << orb_p << "," << orb_s
		   << " rank=" << rank
                   << " |opmat|=" << opmat.normF()
                   << " |tmat|=" << tmat.normF()
                   << " diff=" << diff 
                   << " fail=" << (diff > thresh_rops)	
           	   << std::endl;
	 if(diff > thresh_rops){
	    opmat.print("opmat");
	    tmat.print("tmat");
	    exit(1);
	 }
      }
      nop++;
   } // op
   }
   //-----------------------------------------------------
   // check for Sp = 1/2 hpq aq + <pq||sr> aq^+aras [r>s]
   //-----------------------------------------------------
   if(opname == 'S'){
   for(const auto& opS : qops('S')){
      const auto& op = opS.second;
      int orb_p = opS.first;
      int iproc = distribute1(orb_p,size);
      if(ifdist1 and iproc!=rank) continue;
      // build
      int dim0 = bsite.info.qrow.get_dimAll();
      int dim1 = ksite.info.qrow.get_dimAll();
      linalg::matrix<Tm> tmat(dim0,dim1);
      int ioff0 = 0;
      for(const auto& rsec0 : rbasis0){
         const auto& space0 = rsec0.space;
         const auto& coeff0 = rsec0.coeff;
         int m0 = coeff0.rows();
         int n0 = coeff0.cols();
         int ioff1 = 0;
         for(const auto& rsec1 : rbasis1){
            const auto& space1 = rsec1.space;
            const auto& coeff1 = rsec1.coeff;
            int m1 = coeff1.rows();
            int n1 = coeff1.cols();
            // compute <bdet|op|kdet>
	    linalg::matrix<Tm> mat(m0,m1);
            for(int i1=0; i1<m1; i1++){
	       // check for <bra|Sp|ket> = 1/2 hpq aq + <pq||sr> aq^+aras [r>s]
	       for(int orb_s : rspinorbs){
	          // 1/2 hps as
	          int pos_s = orb2pos[orb_s];
	          auto res1 = space1[i1].ann(pos_s);
	          if(res1.first == 0) continue;
		  for(int i0=0; i0<m0; i0++){
		     if(res1.second != space0[i0]) continue;
		     mat(i0,i1) += 0.5*int1e.get(orb_p,orb_s)
			         * static_cast<double>(res1.first);
		  }
	          // <pq||sr> aq^+aras [r>s]
                  for(int orb_r : rspinorbs){
	             if(orb_r <= orb_s) continue; // r>s
                     int pos_r = orb2pos[orb_r];
                     auto res2 = res1.second.ann(pos_r);
     	             if(res2.first == 0) continue;
                     for(int orb_q : rspinorbs){
                        int pos_q = orb2pos[orb_q];
     	                auto res3 = res2.second.cre(pos_q);
     	                if(res3.first == 0) continue;
                        for(int i0=0; i0<m0; i0++){
			   if(res3.second != space0[i0]) continue;
		           mat(i0,i1) += int2e.get(orb_p,orb_q,orb_s,orb_r)
     	                               * static_cast<double>(res1.first*res2.first*res3.first);
			} // i0
	             } // q
                  } // r
               } // s
            } // i1
            // contraction
            auto tmp1 = linalg::xgemm("C","N",coeff0,mat);
            auto tmp2 = linalg::xgemm("N","N",tmp1,coeff1);
            // save
            for(int i1=0; i1<n1; i1++){
               for(int i0=0; i0<n0; i0++){
     	          tmat(ioff0+i0,ioff1+i1) = tmp2(i0,i1);
               }
            }
            ioff1 += n1;
         }
         ioff0 += n0;
      }
      auto opmat = op.to_matrix();
      double diff = (opmat-tmat).normF();
      maxdiff = std::max(diff,maxdiff);
      if(diff > thresh_rops) nfail++;
      if(debug_rops){
         std::cout << std::scientific << std::setprecision(8);
         std::cout << " S: p=" << orb_p
		   << " rank=" << rank
                   << " |opmat|=" << opmat.normF()
                   << " |tmat|=" << tmat.normF()
                   << " diff=" << diff 
                   << " fail=" << (diff > thresh_rops)	
           	   << std::endl;
	 if(diff > thresh_rops){
	    opmat.print("opmat");
	    tmat.print("tmat");
	    exit(1);
	 }
      }
      nop++;
   } // op
   }
   //------------------------------------------------------------
   // check for H = hpq ap^+aq + <pq||sr> ap^+aq^+aras [p<q,r>s]
   //------------------------------------------------------------
   if(opname == 'H'){
   for(const auto& opH : qops('H')){
      const auto& op = opH.second;
      if(ifdist1 and rank!=0) continue;
      // build
      int dim0 = bsite.info.qrow.get_dimAll();
      int dim1 = ksite.info.qrow.get_dimAll();
      linalg::matrix<Tm> tmat(dim0,dim1);
      int ioff0 = 0;
      for(const auto& rsec0 : rbasis0){
         const auto& space0 = rsec0.space;
         const auto& coeff0 = rsec0.coeff;
         int m0 = coeff0.rows();
         int n0 = coeff0.cols();
         int ioff1 = 0;
         for(const auto& rsec1 : rbasis1){
            const auto& space1 = rsec1.space;
            const auto& coeff1 = rsec1.coeff;
            int m1 = coeff1.rows();
            int n1 = coeff1.cols();
            // compute <bdet|op|kdet>
	    linalg::matrix<Tm> mat(m0,m1);
            for(int i1=0; i1<m1; i1++){
	       // check for <bra|H|ket> = hps ap^+as + <pq||sr> ap^+aq^+aras [p<q,r>s]
	       for(int orb_s : rspinorbs){
	          int pos_s = orb2pos[orb_s];
	          auto res1 = space1[i1].ann(pos_s);
	          if(res1.first == 0) continue;
	          // hps ap^+as
	          for(int orb_p : rspinorbs){
	             int pos_p = orb2pos[orb_p];
	     	     auto res2 = res1.second.cre(pos_p);
	             if(res2.first == 0) continue;
		     for(int i0=0; i0<m0; i0++){
		        if(res2.second != space0[i0]) continue;
	     	        mat(i0,i1) += int1e.get(orb_p,orb_s)
				    * static_cast<double>(res1.first*res2.first);
		     }
	          } // p
	          // <pq||sr> ap^+aq^+aras
                  for(int orb_r : rspinorbs){
		     if(orb_r <= orb_s) continue; // r>s
                     int pos_r = orb2pos[orb_r];
                     auto res2 = res1.second.ann(pos_r);
     	             if(res2.first == 0) continue;
                     for(int orb_q : rspinorbs){
                        int pos_q = orb2pos[orb_q];
     	                auto res3 = res2.second.cre(pos_q);
     	                if(res3.first == 0) continue;
	     	        for(int orb_p : rspinorbs){
		           if(orb_p >= orb_q) continue; // p<q
	                   int pos_p = orb2pos[orb_p];
	     	           auto res4 = res3.second.cre(pos_p);
	                   if(res4.first == 0) continue;
                           for(int i0=0; i0<m0; i0++){
			      if(res4.second != space0[i0]) continue;
	     	              mat(i0,i1) += int2e.get(orb_p,orb_q,orb_s,orb_r)
     	                                  * static_cast<double>(res1.first*res2.first*res3.first*res4.first);
			   } // i0
			} // p 
	             } // q
                  } // r
               } // s
            } // i1
            // contraction
            auto tmp1 = linalg::xgemm("C","N",coeff0,mat);
            auto tmp2 = linalg::xgemm("N","N",tmp1,coeff1);
            // save
            for(int i1=0; i1<n1; i1++){
               for(int i0=0; i0<n0; i0++){
     	          tmat(ioff0+i0,ioff1+i1) = tmp2(i0,i1);
               }
            }
            ioff1 += n1;
         }
         ioff0 += n0;
      }
      auto opmat = op.to_matrix();
      double diff = (opmat-tmat).normF();
      maxdiff = std::max(diff,maxdiff);
      if(diff > thresh_rops) nfail++;
      if(debug_rops){
         std::cout << std::scientific << std::setprecision(8);
         std::cout << " H:"
		   << " rank=" << rank
                   << " |opmat|=" << opmat.normF()
                   << " |tmat|=" << tmat.normF()
                   << " diff=" << diff 
                   << " fail=" << (diff > thresh_rops)	
           	   << std::endl;
	 if(diff > thresh_rops){
	    opmat.print("opmat");
	    tmat.print("tmat");
	    exit(1);
	 }
      }
      nop++;
   } // op
   }
   std::cout << "no. of ops = " << nop
	     << " failed = " << nfail 
	     << " maxdiff = " << maxdiff << std::endl;
   if(nfail>0 || std::abs(maxdiff)>thresh_rops) exit(1);
}

} // ctns

#endif
