#include "../settings/global.h"
#include "tns_comb.h" 
#include "tns_qtensor.h"
#include "tns_oper.h"

using namespace std;
using namespace tns;

// build local operators
void tns::oper_build_cops(const int kp,
		          const integral::two_body& int2e,
		          const integral::one_body& int1e,
		          oper_dict& qops){
   vector<int> rsupp;
   for(int k=0; k<int1e.sorb/2; k++){
      if(k == kp) continue;
      rsupp.push_back(k);
   }
   oper_dot_C(kp, qops);
   oper_dot_A(kp, qops);
   oper_dot_B(kp, qops);
   oper_dot_S(kp, int2e, int1e, rsupp, qops);
   oper_dot_H(kp, int2e, int1e, qops);
}
 
void tns::oper_renorm_rops(const comb& bra,
			   const comb& ket,
		           const comb_coord& p,
		           const integral::two_body& int2e,
		           const integral::one_body& int1e,
			   const string scratch){
   bool debug = false;
   auto t0 = global::get_time();
   int ip  =  p.first, jp  =  p.second;
   cout << "\ntns::oper_renorm_rops iswitch=" 
	<< (p == make_pair(bra.iswitch,0)) 
        << " coord=(" << ip << "," << jp << ")"
	<< "[" << bra.topo[ip][jp] << "]" 
	<< " type=" << bra.type.at(p) << endl;
   oper_dict qops;
   // construct directly for boundary case {C,A,B,S,H}
   if(bra.type.at(p) == 0){
      int kp = bra.get_kp(p);
      oper_build_cops(kp, int2e, int1e, qops);
      string fname = oper_fname(scratch, p, "rop");
      oper_save(fname, qops);
   // construct by renormalization
   }else{
      oper_dict cqops, rqops;
      if(bra.ifbuild_c(p)){
         int kp = bra.get_kp(p);
         oper_build_cops(kp, int2e, int1e, cqops);
      }else{
         auto pc = bra.get_c(p);
         string fname0c = oper_fname(scratch, pc, "rop");
         oper_load(fname0c, cqops);
      }
      auto pr = bra.get_r(p);
      string fname0r = oper_fname(scratch, pr, "rop");
      oper_load(fname0r, rqops);
      // three kinds of sites 
      const auto& bsite = bra.rsites.at(p);
      const auto& ksite = ket.rsites.at(p);
      const auto& orbord = bra.orbord;
      const auto& lsupp = bra.lsupport.at(p);
      bool left = (jp == 0 && ip < bra.iswitch);
      bool swpt = (jp == 0 && ip == bra.iswitch);
      bool rest = !(left || swpt);
      oper_renorm_ropC(bsite,ksite,cqops,rqops,qops,debug);
      if(rest){
         oper_renorm_ropA(bsite,ksite,cqops,rqops,qops,debug);
         oper_renorm_ropB(bsite,ksite,cqops,rqops,qops,debug);
         /*
         // debug
         if(ip==0){
           int nb = bra.nphysical*2;
           matrix rdmA(nb,nb),rdmB(nb,nb),rdmC(nb,nb);
           for(const auto& op : qops['B']){
              int r = op.index[0];
              int s = op.index[1];
              rdmA(r,s) = op.to_matrix()(0,0);
              rdmB(r,s) = op.to_matrix()(2,0);
              rdmC(r,s) = op.to_matrix()(1,2);
           }
           cout << setprecision(10) << endl;
           cout << "trA=" << rdmA.trace() << " |A|=" << normF(rdmA)
             << " A-A.t=" << normF(rdmA-rdmA.T()) << endl;
           cout << "trA=" << rdmB.trace() << " |A|=" << normF(rdmB)
             << " A-A.t=" << normF(rdmB-rdmB.T()) << endl;
           cout << "trA=" << rdmC.trace() << " |A|=" << normF(rdmC)
             << " A-A.t=" << normF(rdmC-rdmC.T()) << endl;
           cout << qops.size() << endl;
   
           matrix rdm1a,rdm1b,rdm1c;
           rdm1a.load("fci_rdm1a");
           rdm1b.load("fci_rdm1b");
           rdm1c.load("fci_rdm1c");
           cout << "fci_drm1:" << endl;
           cout << "trA=" << rdm1a.trace() << " |A|=" << normF(rdm1a)
             << " A-A.t=" << normF(rdm1a-rdm1a.T()) << endl;
           cout << "trA=" << rdm1b.trace() << " |A|=" << normF(rdm1b)
             << " A-A.t=" << normF(rdm1b-rdm1b.T()) << endl;
           cout << "trA=" << rdm1c.trace() << " |A|=" << normF(rdm1c)
             << " A-A.t=" << normF(rdm1c-rdm1c.T()) << endl;
   
           cout << "diff=" << normF(rdmA-rdm1a) << endl;
           cout << "diff=" << normF(rdmB-rdm1b) << endl;
           cout << "diff=" << normF(rdmC-rdm1c) << endl;
         }
         */
      }
      if(left || swpt){
         auto ifAB = swpt;
         oper_renorm_ropP(bsite,ksite,orbord,lsupp,ifAB,int2e,int1e,cqops,rqops,qops,debug);
         oper_renorm_ropQ(bsite,ksite,lsupp,ifAB,int2e,int1e,cqops,rqops,qops,debug);
      }
      auto ifAB = swpt || rest;
      oper_renorm_ropS(bsite,ksite,lsupp,ifAB,int2e,int1e,cqops,rqops,qops,debug);
      oper_renorm_ropH(bsite,ksite,lsupp,ifAB,int2e,int1e,cqops,rqops,qops,debug);
      /*
      // debug against fci::get_Hij
      if(ip==0){
         H.print("H",2);
         auto Hmat = H.to_matrix();
         matrix Hm;
         Hm.load("fci_Hmat");
         auto diffH = Hmat-Hm;
         diffH.print("diffH");
         cout << normF(diffH) << endl;
      }
      */
      // save
      string fname = oper_fname(scratch, p, "rop");
      oper_save(fname, qops);
   }
   auto t1 = global::get_time();
   cout << "timing for tns::oper_renorm_rops : " << setprecision(2) 
        << global::get_duration(t1-t0) << " s" << endl;
}

void tns::oper_env_right(const comb& bra, 
  		         const comb& ket,
		         const integral::two_body& int2e,
		         const integral::one_body& int1e,
			 const string scratch){
   auto t0 = global::get_time();
   cout << "\ntns::oper_env_right" << endl;
   for(int idx=0; idx<bra.rcoord.size(); idx++){
      auto p = bra.rcoord[idx];
      oper_renorm_rops(bra,ket,p,int2e,int1e,scratch);
   } // i
   auto t1 = global::get_time();
   cout << "\ntiming for tns::oper_env_right : " << setprecision(2) 
        << global::get_duration(t1-t0) << " s" << endl;
}
