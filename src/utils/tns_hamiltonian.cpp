#include "tns_hamiltonian.h"
#include "tns_oper.h"

using namespace std;
using namespace tns;

vector<double> tns::get_Hdiag(const comb& icomb,
			      const comb_coord& p,
	              	      const integral::two_body& int2e,
	              	      const integral::one_body& int1e,
			      const double ecore,
			      const string scratch,
			      qtensor3& wf){
   cout << "\ntns::get_Hdiag" << endl;
   // load Hl, Hc, Hr
   auto pl = icomb.get_l(p);
   auto pc = icomb.get_c(p);
   auto pr = icomb.get_r(p);
   string fname0l, fname0c, fname0r;
   qopers lqops_H, cqops_H, rqops_H;
   if(icomb.ifbuild_l(p)){
      int kp = icomb.get_kp(pl);
      auto lqops_ca = oper_dot_ca(kp);
      oper_dot_H(kp, int2e, int1e, lqops_ca, lqops_H);
   }else{
      fname0l = oper_fname(scratch, pl, "lopH");
      oper_load(fname0l, lqops_H);
   }
   if(icomb.ifbuild_c(p)){
      int kp = icomb.get_kp(p);
      auto cqops_ca = oper_dot_ca(kp);
      oper_dot_H(kp, int2e, int1e, cqops_ca, cqops_H);
   }else{
      // load only if C is branch, which is in right canonical form 
      fname0c = oper_fname(scratch, pc, "ropH"); 
      oper_load(fname0c, cqops_H);
   }
   if(icomb.ifbuild_r(p)){
      int kp = icomb.get_kp(pr);
      auto rqops_ca = oper_dot_ca(kp);
      oper_dot_H(kp, int2e, int1e, rqops_ca, rqops_H);
   }else{
      fname0r = oper_fname(scratch, pr, "ropH");
      oper_load(fname0r, rqops_H);
   }
   auto Hl = lqops_H[0];
   auto Hc = cqops_H[0];
   auto Hr = rqops_H[0];
   // <lcr|H|lcr> = <lcr|Hl*Ic*Ir+...|lcr> = Hll + Hcc + Hrr
   for(const auto& pm : wf.qmid){
      const auto& qm = pm.first;
      int mdim = pm.second; 
      for(const auto& pr : wf.qrow){
         const auto& qr = pr.first;
         int rdim = pr.second;
         for(const auto& pc : wf.qcol){
            const auto& qc = pc.first;
            int cdim = pc.second;
	    auto key = make_tuple(qm,qr,qc);
	    auto& blk = wf.qblocks.at(key);
	    if(blk.size() > 0){
	       int size = rdim*cdim;
	       auto lkey = make_pair(qr,qr); // left->row
	       auto ckey = make_pair(qm,qm); // central->mid
	       auto rkey = make_pair(qc,qc); // row->col
	       auto& lblk = Hl.qblocks.at(lkey);
	       auto& cblk = Hc.qblocks.at(ckey);
	       auto& rblk = Hr.qblocks.at(rkey);
	       for(int m=0; m<mdim; m++){
		  for(int c=0; c<cdim; c++){
		     for(int r=0; r<rdim; r++){
			blk[m](r,c) = ecore 
				    + lblk(r,r) 
				    + cblk(m,m)
				    + rblk(c,c);
		     } // r
		  } // c
	       } // m
	    }
	 } // qc
      } // qr
   } // qm
   auto diag = wf.to_vector();
   return diag;
}

void tns::get_Hx(double* y,
		 const double* x,
		 const comb& icomb,
		 const comb_coord& p,
	         const integral::two_body& int2e,
	         const integral::one_body& int1e,
		 const double ecore,
		 const string scratch,
		 qtensor3& wf){
   bool debug = true;
   if(debug) cout << "\ntns::get_Hx" << endl;
   int dim = wf.get_dim();
/*
   // const term
   vector<double> yi(dim);
   transform(x, x+dim, yi.begin(),
             [&ecore](const double& xi){ return ecore*xi; });
   wf.from_vector(yi);
*/
   wf.from_array(x);
   // load Hl, Hc, Hr
   auto pl = icomb.get_l(p);
   auto pc = icomb.get_c(p);
   auto pr = icomb.get_r(p);
   string fname0l, fname0c, fname0r;
   qopers lqops_H, cqops_H, rqops_H;
   if(icomb.ifbuild_l(p)){
      int kp = icomb.get_kp(pl);
      auto lqops_ca = oper_dot_ca(kp);
      oper_dot_H(kp, int2e, int1e, lqops_ca, lqops_H);
   }else{
      fname0l = oper_fname(scratch, pl, "lopH");
      oper_load(fname0l, lqops_H);
   }
   if(icomb.ifbuild_c(p)){
      int kp = icomb.get_kp(p);
      auto cqops_ca = oper_dot_ca(kp);
      oper_dot_H(kp, int2e, int1e, cqops_ca, cqops_H);
   }else{
      // load only if C is branch, which is in right canonical form 
      fname0c = oper_fname(scratch, pc, "ropH"); 
      oper_load(fname0c, cqops_H);
   }
   if(icomb.ifbuild_r(p)){
      int kp = icomb.get_kp(pr);
      auto rqops_ca = oper_dot_ca(kp);
      oper_dot_H(kp, int2e, int1e, rqops_ca, rqops_H);
   }else{
      fname0r = oper_fname(scratch, pr, "ropH");
      oper_load(fname0r, rqops_H);
   }
   auto Hl = lqops_H[0];
   auto Hc = cqops_H[0];
   auto Hr = rqops_H[0];
   // Hl*wf
   auto Hwf = contract_qt3_qt2_l(wf,Hl);
   //wf += contract_qt3_qt2_c(wf,Hc);
   
   auto Hlm = Hl.to_matrix();
   Hlm.print("Hlm");

   // copy back
   Hwf.to_array(y);
}
