#ifndef CTNS_OPER_HELPER_H
#define CTNS_OPER_HELPER_H

#include "ctns_io.h"
#include "ctns_comb.h"
#include "ctns_oper_util.h"
#include "ctns_oper_dot.h"

namespace ctns{

/*
#include "../settings/global.h"
#include "tns_comb.h" 
#include "tns_qtensor.h"
#include "tns_oper.h"

using namespace std;
using namespace tns;

oper_dict tns::oper_get_cqops(const comb& icomb,
		              const comb_coord& p,
			      const string scratch){
   oper_dict cqops;
   if(icomb.ifbuild_c(p)){
      string fname0c = oper_fname(scratch, p, "cop");
      oper_load(fname0c, cqops);
   }else{
      auto pc = icomb.get_c(p);
      string fname0c = oper_fname(scratch, pc, "rop");
      oper_load(fname0c, cqops);
   }
   return cqops;
}

oper_dict tns::oper_get_rqops(const comb& icomb,
		              const comb_coord& p,
			      const string scratch){
   oper_dict rqops;
   auto pr = icomb.get_r(p);
   string fname0r = oper_fname(scratch, pr, "rop");
   oper_load(fname0r, rqops);
   return rqops;
}

oper_dict tns::oper_get_lqops(const comb& icomb,
		              const comb_coord& p,
			      const string scratch){
   oper_dict lqops;
   auto pl = icomb.get_l(p);
   string fname0l = oper_fname(scratch, pl, "lop");
   oper_load(fname0l, lqops);
   return lqops;
}
*/

/*
// renormalize ops
oper_dict oper_renorm_ops(const string& superblock,
			       const comb& bra,
			       const comb& ket,
		               const comb_coord& p,
			       oper_dict& qops1,
			       oper_dict& qops2,
		               const integral::two_body& int2e,
		               const integral::one_body& int1e,
			       const bool debug){
   bool ifrops = false;
   auto t0 = tools::get_time();
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
   cout << "ctns::oper_renorm_ops"
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
   auto t1 = tools::get_time();
   if(debug){
      cout << "timing for ctns::oper_renorm_ops : " << setprecision(2) 
           << tools::get_duration(t1-t0) << " s" << endl;
   }
   return qops;
}

void oper_renorm_onedot(const comb& icomb, 
		             const directed_bond& dbond,
			     oper_dict& cqops,
		             oper_dict& lqops,
			     oper_dict& rqops,	
		             const integral::two_body& int2e, 
		             const integral::one_body& int1e,
			     const string scratch){
   auto p0 = get<0>(dbond);
   auto p1 = get<1>(dbond);
   auto forward = get<2>(dbond);
   auto p = forward? p0 : p1;
   bool cturn = (icomb.type.at(p0) == 3 && p1.second == 1);
   cout << "ctns::oper_renorm_onedot" << endl;
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

void oper_renorm_twodot(const comb& icomb, 
		             const directed_bond& dbond,
			     oper_dict& c1qops,
			     oper_dict& c2qops,
		             oper_dict& lqops,
			     oper_dict& rqops,	
		             const integral::two_body& int2e, 
		             const integral::one_body& int1e, 
			     const string scratch){ 
   auto p0 = get<0>(dbond);
   auto p1 = get<1>(dbond);
   auto forward = get<2>(dbond);
   auto p = forward? p0 : p1;
   bool cturn = (icomb.type.at(p0) == 3 && p1.second == 1);
   cout << "ctns::oper_renorm_twodot" << endl;
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
*/

// construct directly for boundary case {C,A,B,S,H}
template <typename Tm>
oper_dict<Tm> oper_init_local(const int kp,
		              const integral::two_body<Tm>& int2e,
		              const integral::one_body<Tm>& int1e){
   std::vector<int> krest;
   for(int k=0; k<int1e.sorb/2; k++){
      if(k == kp) continue;
      krest.push_back(k);
   }
   oper_dict<Tm> qops;
   oper_dot_C(kp, qops);
   oper_dot_A(kp, qops);
   oper_dot_B(kp, qops);
   oper_dot_S(kp, int2e, int1e, krest, qops);
   oper_dot_H(kp, int2e, int1e, qops);
   return qops;
}

// construct directly for boundary case {C,A,B,S,H} [sites with type=0]
template <typename Tm>
void oper_init(const comb<Tm>& icomb,
	       const integral::two_body<Tm>& int2e,
	       const integral::one_body<Tm>& int1e,
	       const std::string scratch){
   for(int idx=0; idx<icomb.topo.rcoord.size(); idx++){
      auto p = icomb.topo.rcoord[idx];
      auto& node = icomb.topo.nodes[p.first][p.second];
      // local operators on physical sites
      if(node.type != 3){
         int kp = node.pindex;
         auto qops = oper_init_local(kp, int2e, int1e);
	 std::string fname = oper_fname(scratch, p, "cop");
         oper_save(fname, qops);
      }
      // right boundary (exclude the start point)
      if(node.type == 0 && p.first != 0){
         int kp = node.pindex;
         auto qops = oper_init_local(kp, int2e, int1e);
	 std::string fname = oper_fname(scratch, p, "rop");
         oper_save(fname, qops);
      }
   }
   // left boundary at the start
   auto p = std::make_pair(0,0);
   int kp = icomb.topo.nodes[0][0].pindex;
   auto qops = oper_init_local(kp, int2e, int1e);
   std::string fname = oper_fname(scratch, p, "lop");
   oper_save(fname, qops);
}

template <typename Tm>
void oper_env_right(const comb<Tm>& icomb, 
		    const integral::two_body<Tm>& int2e,
		    const integral::one_body<Tm>& int1e,
		    const std::string scratch){
   auto t0 = tools::get_time();
   std::cout << "ctns::oper_env_right" << std::endl;
   oper_init(icomb, int2e, int1e, scratch);
/*
   for(int idx=0; idx<icomb.rcoord.size(); idx++){
      auto p = icomb.rcoord[idx];
      if(icomb.type.at(p) != 0 || p.first == 0){
         auto qops1 = oper_get_cqops(icomb, p, scratch);
         auto qops2 = oper_get_rqops(icomb, p, scratch);
	 const std::string superblock = "cr";
         auto qops = oper_renorm_ops(superblock, icomb, p, 
			 	     qops1, qops2, int2e, int1e);
	 std::string fname = oper_fname(scratch, p, "rop");
         oper_save(fname, qops);
      }
   } // i
*/
   auto t1 = tools::get_time();
   std::cout << "timing for ctns::oper_env_right : " << std::setprecision(2) 
             << tools::get_duration(t1-t0) << " s" << std::endl;
}

template <typename Tm>
linalg::matrix<Tm> get_Hmat(const comb<Tm>& icomb, 
		            const integral::two_body<Tm>& int2e,
		            const integral::one_body<Tm>& int1e,
		            const double ecore,
		            const std::string scratch){
   std::cout << "\nctns::get_Hmat" << std::endl;
   // build operators for environement
   oper_env_right(icomb, int2e, int1e, scratch);
   exit(1);
/*
   // load
   oper_dict qops;
   auto p = make_pair(0,0); 
   std::string fname = oper_fname(scratch, p, "rop");
   oper_load(fname, qops);
   auto Hmat = qops['H'][0].to_matrix();
   Hmat += ecore*linalg::identity_matrix(Hmat.rows());
   return Hmat;
*/
   linalg::matrix<Tm> mat;
   return mat;
}

} // ctns

#endif
