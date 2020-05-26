#include "../settings/global.h"
#include "tns_comb.h" 
#include "tns_qtensor.h"
#include "tns_oper.h"

using namespace std;
using namespace tns;

// renormalize ops
oper_dict tns::oper_renorm_ops(const string& superblock,
			       const comb& bra,
			       const comb& ket,
		               const comb_coord& p,
			       oper_dict& qops1,
			       oper_dict& qops2,
		               const integral::two_body& int2e,
		               const integral::one_body& int1e,
			       const bool debug){
   bool ifrops = false;
   auto t0 = global::get_time();
   int ip  =  p.first, jp  =  p.second;
   // support for index 
   qtensor3 bsite, ksite;
   vector<int> supp;
   bool ifAB; 
   if(superblock == "cr"){
      bsite = bra.rsites.at(p);
      ksite = ket.rsites.at(p);
      supp = bra.lsupport.at(p);
      ifAB = !(jp == 0 && ip <= bra.iswitch);
   }else if(superblock == "lc"){
      bsite = bra.lsites.at(p);
      ksite = ket.lsites.at(p);
      supp = bra.rsupport.at(bra.get_r(p));
      ifAB = (jp == 0 && ip < bra.iswitch-1);
   }else if(superblock == "lr"){
      bsite = bra.lsites.at(p);
      ksite = ket.lsites.at(p);
      supp = bra.rsupport.at(bra.get_c(p));
      ifAB = false;
   }
   cout << "tns::oper_renorm_ops"
        << " coord=(" << ip << "," << jp << ")"
	<< "[" << bra.topo[ip][jp] << "]" 
	<< " iswitch=" << (p == make_pair(bra.iswitch,0)) 
        << " (" << superblock[0] << ":" << oper_dict_opnames(qops1) << "," 
	<< superblock[1] << ":" << oper_dict_opnames(qops2) << ")"
	<< "->" << (ifAB? "AB" : "PQ")
        << endl;	
   // three kinds of sites 
   oper_dict qops;
   // C,S,H
   oper_renorm_opC(superblock,bsite,ksite,qops1,qops2,qops,debug);
   if(debug && ifrops) oper_rbases(bra,ket,p,qops,'C');
   oper_renorm_opS(superblock,bsite,ksite,qops1,qops2,qops,
   		   supp,int2e,int1e,debug);
   if(debug && ifrops) oper_rbases(bra,ket,p,qops,'S',int2e,int1e);
   oper_renorm_opH(superblock,bsite,ksite,qops1,qops2,qops,
   		   int2e,int1e,debug);
   if(debug && ifrops) oper_rbases(bra,ket,p,qops,'H',int2e,int1e);
   // consistency check
   auto H = qops['H'].at(0);
   auto diffH = (H-H.T()).normF();
   if(diffH > 1.e-10){
      H.print("H",2);
      cout << "error: H-H.T is too large! diffH=" << diffH << endl;
      exit(1);
   }
   // AB/PQ
   if(ifAB){
      oper_renorm_opA(superblock,bsite,ksite,qops1,qops2,qops,debug);
      if(debug && ifrops) oper_rbases(bra,ket,p,qops,'A');
      oper_renorm_opB(superblock,bsite,ksite,qops1,qops2,qops,debug);
      if(debug && ifrops) oper_rbases(bra,ket,p,qops,'B');
   }else{
      oper_renorm_opP(superblock,bsite,ksite,qops1,qops2,qops,
        	      supp,int2e,int1e,debug);
      if(debug && ifrops) oper_rbases(bra,ket,p,qops,'P',int2e,int1e);
      oper_renorm_opQ(superblock,bsite,ksite,qops1,qops2,qops,
      	              supp,int2e,int1e,debug);
      if(debug && ifrops) oper_rbases(bra,ket,p,qops,'Q',int2e,int1e);
   }
   auto t1 = global::get_time();
   if(debug){
      cout << "timing for tns::oper_renorm_ops : " << setprecision(2) 
           << global::get_duration(t1-t0) << " s" << endl;
   }
   return qops;
}

void tns::oper_env_right(const comb& bra, 
  		         const comb& ket,
		         const integral::two_body& int2e,
		         const integral::one_body& int1e,
			 const string scratch){
   auto t0 = global::get_time();
   cout << "\ntns::oper_env_right" << endl;
   oper_build_boundary(bra, int2e, int1e, scratch);
   for(int idx=0; idx<bra.rcoord.size(); idx++){
      auto p = bra.rcoord[idx];
      if(bra.type.at(p) != 0 || p.first == 0){
         auto qops1 = oper_get_cqops(bra, p, scratch);
         auto qops2 = oper_get_rqops(bra, p, scratch);
	 const string superblock = "cr";
         auto qops = oper_renorm_ops(superblock, bra, ket, p, 
			 	     qops1, qops2, int2e, int1e);
         string fname = oper_fname(scratch, p, "rop");
         oper_save(fname, qops);
      }
   } // i
   auto t1 = global::get_time();
   cout << "\ntiming for tns::oper_env_right : " << setprecision(2) 
        << global::get_duration(t1-t0) << " s" << endl;
}

void tns::oper_renorm_onedot(const comb& icomb, 
		             const comb_coord& p, 
		             const bool forward, 
		             const bool cturn, 
			     oper_dict& cqops,
		             oper_dict& lqops,
			     oper_dict& rqops,	
		             const integral::two_body& int2e, 
		             const integral::one_body& int1e,
			     const string scratch){ 
   oper_dict qops;
   if(forward){
      if(!cturn){
	 qops = oper_renorm_ops("lc", icomb, icomb, p, lqops, cqops, int2e, int1e);
      }else{
	 qops = oper_renorm_ops("lr", icomb, icomb, p, lqops, rqops, int2e, int1e);
      }
      string fname = oper_fname(scratch, p, "lop");
      oper_save(fname, qops);
   }else{
      qops = oper_renorm_ops("cr", icomb, icomb, p, cqops, rqops, int2e, int1e);
      string fname = oper_fname(scratch, p, "rop");
      oper_save(fname, qops);
   }
}

void tns::oper_renorm_twodot(const comb& icomb, 
		             const comb_coord& p, 
		             const bool forward, 
		             const bool cturn, 
			     oper_dict& c1qops,
			     oper_dict& c2qops,
		             oper_dict& lqops,
			     oper_dict& rqops,	
		             const integral::two_body& int2e, 
		             const integral::one_body& int1e, 
			     const string scratch){ 
   oper_dict qops;
   if(forward){
      if(!cturn){
	 qops = oper_renorm_ops("lc", icomb, icomb, p, lqops, c1qops, int2e, int1e);
      }else{
	 qops = oper_renorm_ops("lr", icomb, icomb, p, lqops, rqops, int2e, int1e);
      }
      string fname = oper_fname(scratch, p, "lop");
      oper_save(fname, qops);
   }else{
      if(!cturn){
         qops = oper_renorm_ops("cr", icomb, icomb, p, c2qops, rqops, int2e, int1e);
      }else{
         qops = oper_renorm_ops("cr", icomb, icomb, p, c1qops, c2qops, int2e, int1e);
      }
      string fname = oper_fname(scratch, p, "rop");
      oper_save(fname, qops);
   }
}
