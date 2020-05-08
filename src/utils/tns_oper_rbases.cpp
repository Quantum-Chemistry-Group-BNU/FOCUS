#include "../settings/global.h"
#include "tns_comb.h" 
#include "tns_qtensor.h"
#include "tns_oper.h"
#include "../core/matrix.h"
#include "../core/linalg.h"
#include <string>
#include <iomanip>

using namespace std;
using namespace linalg;
using namespace tns;

// normal operators
void tns::oper_rbases(const comb& bra,
		      const comb& ket, 
		      const comb_coord& p,
		      const string scratch,
		      const string optype){
   if(p == make_pair(0,0)) return; // no rbases at the start 
   const double thresh = 1.e-5;
   int nfail = 0;
   double mdiff = -1.0;
   cout << "tns::oper_rbases optype=" << optype << endl;
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

   // load renormalized operators 
   qopers qops_C, qops_A, qops_B;
   string fname;
    
   // check for ap^+
   if(optype == "C"){ 
   fname  = oper_fname(scratch, p, "rightC");
   oper_load(fname, qops_C);
   for(const auto& op: qops_C){
      int orb_p = op.index[0];
      int pos = orb2pos.at(orb_p);
      // build
      int dim0 = bsite.get_dim_col();
      int dim1 = ksite.get_dim_col();
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
              << " diff=" << diff << endl;
	 nfail++;
      }
   } // op
   } 

   // check for Apq = ap^+aq^+
   if(optype == "A"){ 
   fname = oper_fname(scratch, p, "rightA");
   oper_load(fname, qops_A);
   for(const auto& op : qops_A){
      int orb_p = op.index[0];
      int orb_q = op.index[1];
      int pos_p = orb2pos.at(orb_p);
      int pos_q = orb2pos.at(orb_q);
      // build
      int dim0 = bsite.get_dim_col();
      int dim1 = ksite.get_dim_col();
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
   if(optype == "B"){ 
   fname = oper_fname(scratch, p, "rightB");
   oper_load(fname, qops_B);
   for(const auto& op : qops_B){
      int orb_p = op.index[0];
      int orb_q = op.index[1];
      int pos_p = orb2pos.at(orb_p);
      int pos_q = orb2pos.at(orb_q);
      // build
      int dim0 = bsite.get_dim_col();
      int dim1 = ksite.get_dim_col();
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
   if(nfail > 0) exit(1);
}

// complementary operators
void tns::oper_rbases(const comb& bra,
		      const comb& ket, 
		      const comb_coord& p,
	              const integral::two_body& int2e,
	              const integral::one_body& int1e,
		      const string scratch,
		      const string optype){
   if(p == make_pair(0,0)) return; // no rbases at the start 
   const double thresh = 1.e-5;
   int nfail = 0;
   double mdiff = -1.0;
   cout << "tns::oper_rbases optype=" << optype << endl;
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

   // load renormalized operators 
   qopers qops_P, qops_Q, qops_S, qops_H;
   string fname;
 
   // check for Ppq = <pq||sr> aras [r>s]
   if(optype == "P"){ 
   fname = oper_fname(scratch, p, "rightP");
   oper_load(fname, qops_P);
   for(const auto& op : qops_P){
      int orb_p = op.index[0];
      int orb_q = op.index[1];
      // build
      int dim0 = bsite.get_dim_col();
      int dim1 = ksite.get_dim_col();
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
   if(optype == "Q"){ 
   fname = oper_fname(scratch, p, "rightQ");
   oper_load(fname, qops_Q);
   for(const auto& op : qops_Q){
      int orb_p = op.index[0];
      int orb_s = op.index[1];
      // build
      int dim0 = bsite.get_dim_col();
      int dim1 = ksite.get_dim_col();
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
   if(optype == "S"){
   fname = oper_fname(scratch, p, "rightS");
   oper_load(fname, qops_S);
   for(const auto& op : qops_S){
      int orb_p = op.index[0];
      // build
      int dim0 = bsite.get_dim_col();
      int dim1 = ksite.get_dim_col();
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
   if(optype == "H"){
   fname = oper_fname(scratch, p, "rightH");
   oper_load(fname, qops_H);
   for(const auto& op : qops_H){
      // build
      int dim0 = bsite.get_dim_col();
      int dim1 = ksite.get_dim_col();
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
   if(nfail > 0) exit(1);
}
