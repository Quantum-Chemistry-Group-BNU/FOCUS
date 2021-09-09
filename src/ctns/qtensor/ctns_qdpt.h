#ifndef CTNS_QDPT_H
#define CTNS_QDPT_H

#include "ctns_qsym.h"
#include "../../core/tools.h"

namespace ctns{

// direct product table of qbond : V1*V2->V12
// qsym -> list of {(idx1,idx2,ioff)}
using qdpt = std::map<qsym,std::vector<std::tuple<int,int,int>>>;
using qproduct = std::pair<qbond,qdpt>;

// qs12 = qs1*qs2
inline qproduct qmerge(const qbond& qs1, const qbond& qs2){
   // init dpt
   qdpt dpt;
   for(int i1=0; i1<qs1.size(); i1++){
      auto q1 = qs1.get_sym(i1);
      for(int i2=0; i2<qs2.size(); i2++){
	 auto q2 = qs2.get_sym(i2);
	 dpt[q1+q2].push_back(std::make_tuple(i1,i2,0));
      }
   }
   // form qs12 & compute offset
   qbond qs12;
   for(auto& p : dpt){
      const auto& q12 = p.first;
      auto& p12 = p.second;
      int ioff = 0;
      for(int i12=0; i12<p12.size(); i12++){
         int i1 = std::get<0>(p12[i12]);
         int i2 = std::get<1>(p12[i12]);
         int d1 = qs1.get_dim(i1);
	 int d2 = qs2.get_dim(i2);
         p12[i12] = std::make_tuple(i1,i2,ioff);
	 ioff += d1*d2; 
      }
      qs12.dims.push_back(std::make_pair(q12,ioff));
   }
   return std::make_pair(qs12,dpt);
}

// mapping from original PRODUCT basis to kramers paired basis:
// V[odd] = {|le,ro>,|lo,re>}
inline void mapping2krbasis_odd(const qsym& qr,
				const qbond& qs1,
		                const qbond& qs2,
		                const qdpt& dpt,
		                std::vector<int>& pos_new,
				std::vector<double>& phases){
   std::vector<int> pos_up, pos_dw;
   int ioff = 0;
   const auto& comb = dpt.at(qr);
   for(int i=0; i<comb.size(); i++){
      int b1 = std::get<0>(comb[i]);
      int b2 = std::get<1>(comb[i]);
      int ioff = std::get<2>(comb[i]);
      auto q1 = qs1.get_sym(b1);
      auto q2 = qs2.get_sym(b2);
      int  d1 = qs1.get_dim(b1);
      int  d2 = qs2.get_dim(b2);
      // |le,ro> 
      if(q1.parity() == 0 && q2.parity() == 1){
         assert(d2%2 == 0);
         for(int i2=0; i2<d2/2; i2++){
            for(int i1=0; i1<d1; i1++){
               int idxA = ioff + i2*d1 + i1; // |le,ro>
               pos_up.push_back(idxA);
               int idxB = ioff + (i2+d2/2)*d1 + i1; // |le,ro_bar>
               pos_dw.push_back(idxB);
            }
         }
      // |lo,re>   
      }else if(q1.parity() == 1 && q2.parity() == 0){
         assert(d1%2 == 0);
         for(int i2=0; i2<d2; i2++){
            for(int i1=0; i1<d1/2; i1++){
   	       int idxA = ioff + i2*d1 + i1; 
               pos_up.push_back(idxA);
     	       int idxB = ioff + i2*d1 + (i1+d1/2);
               pos_dw.push_back(idxB);
            }
         }
      }else{
         std::cout << "q1p,q2p=" << q1.parity() << "," << q2.parity() << std::endl;
	 tools::exit("error: no such combination of parities!");
      }
      ioff += d1*d2;
   }
   assert(pos_up.size() == pos_dw.size());
   pos_new.clear();
   pos_new.insert(pos_new.end(), pos_up.begin(), pos_up.end());
   pos_new.insert(pos_new.end(), pos_dw.begin(), pos_dw.end());
   phases.resize(pos_dw.size(),1.0);
}

// V[even] = {|le,re>,|lo,ro>}
inline void mapping2krbasis_even(const qsym& qr,
			         const qbond& qs1,
			         const qbond& qs2,
			         const qdpt& dpt,
		                 std::vector<int>& pos_new,
				 std::vector<double>& phases){
   std::vector<int> pos_up, pos_dw, pos_ee;
   int ioff = 0;
   const auto& comb = dpt.at(qr);
   for(int i=0; i<comb.size(); i++){
      int b1 = std::get<0>(comb[i]);
      int b2 = std::get<1>(comb[i]);
      int ioff = std::get<2>(comb[i]);
      auto q1 = qs1.get_sym(b1);
      auto q2 = qs2.get_sym(b2);
      int  d1 = qs1.get_dim(b1);
      int  d2 = qs2.get_dim(b2);
      // |le,re> 
      if(q1.parity() == 0 && q2.parity() == 0){
         for(int i2=0; i2<d2; i2++){
            for(int i1=0; i1<d1; i1++){
               int idx = ioff + i2*d1 + i1;
               pos_ee.push_back(idx);
            }
         }
      // |lo,ro> = {|lo,ro>,|lo_bar,ro>} + {|lo_bar,ro_bar>,|lo,ro_bar>}
      }else if(q1.parity() == 1 && q2.parity() == 1){
         assert(d1%2 == 0 & d2%2 == 0);
         for(int i2=0; i2<d2/2; i2++){
            for(int i1=0; i1<d1/2; i1++){
               int idxA = ioff + i2*d1 + i1; // |lo,ro> 
               pos_up.push_back(idxA);
               int idxB = ioff + (i2+d2/2)*d1 + (i1+d1/2); // |lo_bar,ro_bar>
     	       pos_dw.push_back(idxB);
               phases.push_back(1.0);
	    }
            for(int i1=0; i1<d1/2; i1++){
               int idxA = ioff + i2*d1 + (i1+d1/2); // |lo_bar,ro> 
               pos_up.push_back(idxA);
               int idxB = ioff + (i2+d2/2)*d1 + i1; // |lo,ro_bar>
     	       pos_dw.push_back(idxB);
               phases.push_back(-1.0);
            }
         }
      }else{
         std::cout << "q1p,q2p=" << q1.parity() << "," << q2.parity() << std::endl;
	 tools::exit("error: no such combination of parities!");
      }
      ioff += d1*d2;
   }
   assert(pos_up.size() == pos_dw.size());
   pos_new.clear();
   pos_new.insert(pos_new.end(), pos_up.begin(), pos_up.end());
   pos_new.insert(pos_new.end(), pos_dw.begin(), pos_dw.end());
   pos_new.insert(pos_new.end(), pos_ee.begin(), pos_ee.end());
}

// mapping product basis to kramers paired basis 
inline void mapping2krbasis(const qsym& qr,
		            const qbond& qs1,
		            const qbond& qs2,
		            const qdpt& dpt,
		            std::vector<int>& pos_new,
		            std::vector<double>& phases){
   if(qr.parity() == 1){
      mapping2krbasis_odd(qr,qs1,qs2,dpt,pos_new,phases);
   }else{
      mapping2krbasis_even(qr,qs1,qs2,dpt,pos_new,phases);
   }
}

} // ctns

#endif
