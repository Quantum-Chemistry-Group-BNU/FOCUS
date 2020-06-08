#include "../settings/global.h"
#include "../core/matrix.h"
#include "../core/linalg.h"
#include "tns_comb.h" 
#include "tns_qtensor.h"
#include "tns_oper.h"
#include <string>
#include <iomanip>

using namespace std;
using namespace linalg;
using namespace tns;

// normal operators
void tns::oper_rbases(const comb& bra,
		      const comb& ket, 
		      const comb_coord& p,
		      oper_dict& qops,
		      const char opname){
   if(p == make_pair(0,0)) return; // no rbases at the start 
   const double thresh = 1.e-6;
   int nfail = 0;
   double mdiff = -1.0;
   cout << "tns::oper_rbases opname=" << opname << endl;
   int i = p.first, j = p.second;
   const auto& bsite = bra.rsites.at(p);
   const auto& ksite = ket.rsites.at(p);
   const auto& rbasis0 = bra.rbases.at(p);
   const auto& rbasis1 = ket.rbases.at(p);

   // setup mapping for orbitals to local support
   auto rsupp = bra.rsupport.at(p);
   vector<int> rspinorbs;
   map<int,int> orb2pos;
   int idx = 0;
   for(auto it=rsupp.begin(); it!=rsupp.end(); it++){
      int ks = *it;
      orb2pos[2*ks] = idx; // abab counting and rsupp[0] comes first!
      orb2pos[2*ks+1] = idx+1;
      idx += 2;
      rspinorbs.push_back(2*ks); // spin-orbitals
      rspinorbs.push_back(2*ks+1);
   }
   cout << "p=(" << i << "," << j << ")[" << bra.topo[i][j] << "] " << endl;
   cout << "rsupp=";
   for(int i : rsupp) cout << i << " ";
   cout << endl;
   cout << "rspinorbs=";
   for(int i : rspinorbs) cout << i << " ";
   cout << endl;

   // check for ap^+
   if(opname == 'C'){ 
   for(const auto& opC: qops['C']){
      const auto& op = opC.second;
      int orb_p = opC.first;
      int pos = orb2pos.at(orb_p);
      // build
      int dim0 = bsite.get_dim_row();
      int dim1 = ksite.get_dim_row();
      matrix tmat(dim0,dim1);
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
	    matrix mat(m0,m1);
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
	    auto tmp1 = dgemm("T","N",coeff0,mat);
	    auto tmp2 = dgemm("N","N",tmp1,coeff1);
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
      double diff = normF(opmat-tmat);
      mdiff = max(diff,mdiff);
      if(diff > thresh){
         cout << scientific << setprecision(8);
         cout << "C: p=" << orb_p 
              << " |op|=" << normF(opmat)
              << " |dm|=" << normF(tmat)
              << " diff=" << diff
	      << endl;
	 nfail++;
      }
   } // op
   } 

   // check for Apq = ap^+aq^+
   if(opname == 'A'){ 
   for(const auto& opA : qops['A']){
      const auto& op = opA.second;
      auto pq = oper_unpack(opA.first);
      int orb_p = pq.first;
      int orb_q = pq.second;
      int pos_p = orb2pos.at(orb_p);
      int pos_q = orb2pos.at(orb_q);
      // build
      int dim0 = bsite.get_dim_row();
      int dim1 = ksite.get_dim_row();
      matrix tmat(dim0,dim1);
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
	    matrix mat(m0,m1);
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
	    auto tmp1 = dgemm("T","N",coeff0,mat);
	    auto tmp2 = dgemm("N","N",tmp1,coeff1);
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
      double diff = normF(opmat-tmat);
      mdiff = max(diff,mdiff);
      if(diff > thresh){
         cout << scientific << setprecision(8);
         cout << "A: p,q=" << orb_p << "," << orb_q
              << " |op|=" << normF(opmat)
              << " |dm|=" << normF(tmat)
              << " diff=" << diff << endl;
	 nfail++;
      }
   } // op
   }

   // check for Bpq = ap^+aq
   if(opname == 'B'){ 
   for(const auto& opB : qops['B']){
      const auto& op = opB.second;
      auto pq = oper_unpack(opB.first);
      int orb_p = pq.first;
      int orb_q = pq.second;
      int pos_p = orb2pos.at(orb_p);
      int pos_q = orb2pos.at(orb_q);
      // build
      int dim0 = bsite.get_dim_row();
      int dim1 = ksite.get_dim_row();
      matrix tmat(dim0,dim1);
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
	    matrix mat(m0,m1);
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
	    auto tmp1 = dgemm("T","N",coeff0,mat);
	    auto tmp2 = dgemm("N","N",tmp1,coeff1);
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
      double diff = normF(opmat-tmat);
      mdiff = max(diff,mdiff);
      if(diff > thresh){
         cout << scientific << setprecision(8);
         cout << "B: p,q=" << orb_p << "," << orb_q
              << " |op|=" << normF(opmat)
              << " |dm|=" << normF(tmat)
              << " diff=" << diff << endl;
	 nfail++;
      }
   } // op
   }
   
   cout << "no. of failed cases = " << nfail 
	<< " maxdiff = " << mdiff
        << endl;
   cout << endl;
   //if(nfail > 0) exit(1);
}

// complementary operators
void tns::oper_rbases(const comb& bra,
		      const comb& ket, 
		      const comb_coord& p,
		      oper_dict& qops,
		      const char opname,
	              const integral::two_body& int2e,
	              const integral::one_body& int1e){
   if(p == make_pair(0,0)) return; // no rbases at the start 
   const double thresh = 1.e-5;
   int nfail = 0;
   double mdiff = -1.0;
   cout << "tns::oper_rbases opname=" << opname << endl;
   int i = p.first, j = p.second;
   const auto& bsite = bra.rsites.at(p);
   const auto& ksite = ket.rsites.at(p);
   const auto& rbasis0 = bra.rbases.at(p);
   const auto& rbasis1 = ket.rbases.at(p);

   // setup mapping for orbitals to local support
   auto rsupp = bra.rsupport.at(p);
   vector<int> rspinorbs;
   map<int,int> orb2pos;
   int idx = 0;
   for(auto it=rsupp.begin(); it!=rsupp.end(); it++){
      int ks = *it;
      orb2pos[2*ks] = idx; // abab counting and rsupp[0] comes first!
      orb2pos[2*ks+1] = idx+1;
      idx += 2;
      rspinorbs.push_back(2*ks); // spin-orbitals
      rspinorbs.push_back(2*ks+1);
   }
   cout << "p=(" << i << "," << j << ")[" << bra.topo[i][j] << "] " << endl;
   cout << "rsupp=";
   for(int i : rsupp) cout << i << " ";
   cout << endl;
   cout << "rspinorbs=";
   for(int i : rspinorbs) cout << i << " ";
   cout << endl;

   // check for Ppq = <pq||sr> aras [r>s]
   if(opname == 'P'){ 
   for(const auto& opP : qops['P']){
      const auto& op = opP.second;
      auto pq = oper_unpack(opP.first);
      int orb_p = pq.first;
      int orb_q = pq.second;
      // build
      int dim0 = bsite.get_dim_row();
      int dim1 = ksite.get_dim_row();
      matrix tmat(dim0,dim1);
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
            matrix mat(m0,m1);
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
                        mat(i0,i1) += int2e.getAnti(orb_p,orb_q,orb_s,orb_r)
                                    * res1.first*res2.first;
                     } // i0
                  } // r
               } // s
            } // i1
            // contraction
            auto tmp1 = dgemm("T","N",coeff0,mat);
            auto tmp2 = dgemm("N","N",tmp1,coeff1);
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
      double diff = normF(opmat-tmat);
      mdiff = max(diff,mdiff);
      if(diff > thresh){
         cout << scientific << setprecision(8);
         cout << "P: p,q=" << orb_p << "," << orb_q
              << " |op|=" << normF(opmat)
              << " |dm|=" << normF(tmat)
              << " diff=" << diff << endl;
         nfail++;
      }
   } // op
   }

   // check for Qps = <pq||sr> aq^+ar
   if(opname == 'Q'){ 
   for(const auto& opQ : qops['Q']){
      const auto& op = opQ.second;
      auto ps = oper_unpack(opQ.first);
      int orb_p = ps.first;
      int orb_s = ps.second;
      // build
      int dim0 = bsite.get_dim_row();
      int dim1 = ksite.get_dim_row();
      matrix tmat(dim0,dim1);
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
            matrix mat(m0,m1);
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
                        mat(i0,i1) += int2e.getAnti(orb_p,orb_q,orb_s,orb_r)
                                    * res1.first*res2.first;
                     } // i0
     	          } // q
               } // r
            } // i1
            // contraction
            auto tmp1 = dgemm("T","N",coeff0,mat);
            auto tmp2 = dgemm("N","N",tmp1,coeff1);
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
      double diff = normF(opmat-tmat);
      mdiff = max(diff,mdiff);
      if(diff > thresh){
         cout << scientific << setprecision(8);
         cout << "Q: p,s=" << orb_p << "," << orb_s
              << " |op|=" << normF(opmat)
              << " |dm|=" << normF(tmat)
              << " diff=" << diff << endl;
         nfail++;
      }
   } // op
   }
  
   // check for Sp = 1/2 hpq aq + <pq||sr> aq^+aras [r>s]
   if(opname == 'S'){
   for(const auto& opS : qops['S']){
      const auto& op = opS.second;
      int orb_p = opS.first;
      // build
      int dim0 = bsite.get_dim_row();
      int dim1 = ksite.get_dim_row();
      matrix tmat(dim0,dim1);
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
            matrix mat(m0,m1);
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
			         * res1.first;
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
		           mat(i0,i1) += int2e.getAnti(orb_p,orb_q,orb_s,orb_r)
     	                               * res1.first*res2.first*res3.first;
			} // i0
	             } // q
                  } // r
               } // s
            } // i1
            // contraction
            auto tmp1 = dgemm("T","N",coeff0,mat);
            auto tmp2 = dgemm("N","N",tmp1,coeff1);
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
      double diff = normF(opmat-tmat);
      mdiff = max(diff,mdiff);
      if(diff > thresh){
         cout << scientific << setprecision(8);
         cout << "S: p=" << orb_p
              << " |op|=" << normF(opmat)
              << " |dm|=" << normF(tmat)
              << " diff=" << diff << endl;
	 nfail++;
      }
   } // op
   }

   // check for H = hpq ap^+aq + <pq||sr> ap^+aq^+aras [p<q,r>s]
   if(opname == 'H'){
   for(const auto& opH : qops['H']){
      const auto& op = opH.second;
      // build
      int dim0 = bsite.get_dim_row();
      int dim1 = ksite.get_dim_row();
      matrix tmat(dim0,dim1);
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
            matrix mat(m0,m1);
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
				    * res1.first*res2.first;
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
	     	              mat(i0,i1) += int2e.getAnti(orb_p,orb_q,orb_s,orb_r)
     	                                  * res1.first*res2.first*res3.first*res4.first;
			   } // i0
			} // p 
	             } // q
                  } // r
               } // s
            } // i1
            // contraction
            auto tmp1 = dgemm("T","N",coeff0,mat);
            auto tmp2 = dgemm("N","N",tmp1,coeff1);
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
      double diff = normF(opmat-tmat);
      mdiff = max(diff,mdiff);
      if(diff > thresh){
         cout << scientific << setprecision(8);
         cout << "H:"
              << " |op|=" << normF(opmat)
              << " |dm|=" << normF(tmat)
              << " diff=" << diff << endl;
	 nfail++;
      }
   } // op
   }
   
   cout << "no. of failed cases = " << nfail 
	<< " maxdiff = " << mdiff
        << endl;
   cout << endl;
   //if(nfail > 0) exit(1);
}
