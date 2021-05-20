#ifndef OPER_RENORM_H
#define OPER_RENORM_H

#include "oper_rbasis.h"
#include "oper_combine.h"
#include "oper_normxwf.h"
#include "oper_compxwf.h"

namespace ctns{

// kernel for computing renormalized ap^+
template <typename Tm>
void oper_renorm_opC(const std::string& superblock,
		     const qtensor3<Tm>& site,
		     oper_dict<Tm>& qops1,
		     oper_dict<Tm>& qops2,
		     oper_dict<Tm>& qops,
		     const bool debug=false){
   if(debug) std::cout << "\nctns::oper_renorm_opC" << std::endl;
   auto t0 = tools::get_time();
   // preprocess
   std::vector<std::pair<int,int>> info;
   oper_combine_opC(qops1.cindex, qops2.cindex, info);
   // compute
   for(const auto p : info){
      int iformula = p.first;
      int index = p.second;
      auto opwf = oper_normxwf_opC(superblock,site,qops1,qops2,iformula,index); 
      qops('C')[index] = oper_kernel_renorm(superblock,site,opwf); 
   }
   auto t1 = tools::get_time();
   if(debug){ 
      std::cout << "timing for ctns::oper_renorm_opC : " << std::setprecision(2) 
                << tools::get_duration(t1-t0) << " s" << std::endl;
   }
}

// kernel for computing renormalized Apq=ap^+aq^+
template <typename Tm>
void oper_renorm_opA(const std::string& superblock,
		     const qtensor3<Tm>& site,
		     oper_dict<Tm>& qops1,
		     oper_dict<Tm>& qops2,
		     oper_dict<Tm>& qops,
		     const bool& ifkr,
		     const bool debug=false){
   if(debug) std::cout << "\nctns::oper_renorm_opA" << std::endl;
   auto t0 = tools::get_time();
   // preprocess
   std::vector<std::pair<int,int>> info;
   oper_combine_opA(qops1.cindex, qops2.cindex, ifkr, info);
   // compute
   for(const auto p : info){
      int iformula = p.first;
      int index = p.second;
      auto opwf = oper_normxwf_opA(superblock,site,qops1,qops2,ifkr,iformula,index);
      qops('A')[index] = oper_kernel_renorm(superblock,site,opwf);
   }
   auto t1 = tools::get_time();
   if(debug){
      std::cout << "timing for ctns::oper_renorm_opA : " << std::setprecision(2) 
                << tools::get_duration(t1-t0) << " s" << std::endl;
   }
}

// kernel for computing renormalized ap^+aq
template <typename Tm>
void oper_renorm_opB(const std::string& superblock,
		     const qtensor3<Tm>& site,
		     oper_dict<Tm>& qops1,
		     oper_dict<Tm>& qops2,
		     oper_dict<Tm>& qops,
		     const bool& ifkr,
		     const bool debug=false){
   if(debug) std::cout << "\nctns::oper_renorm_opB" << std::endl;
   auto t0 = tools::get_time();
   // preprocess
   std::vector<std::pair<int,int>> info;
   oper_combine_opB(qops1.cindex, qops2.cindex, ifkr, info);
   // compute
   for(const auto p : info){
      int iformula = p.first;
      int index = p.second;
      auto opwf = oper_normxwf_opB(superblock,site,qops1,qops2,ifkr,iformula,index);
      qops('B')[index] = oper_kernel_renorm(superblock,site,opwf);
   }
   auto t1 = tools::get_time();
   if(debug){ 
      std::cout << "timing for tns::oper_renorm_opB : " << std::setprecision(2) 
                << tools::get_duration(t1-t0) << " s" << std::endl;
   }
}

// Ppq = <pq||sr> aras [r>s] (p<q)
template <typename Tm>
void oper_renorm_opP(const std::string& superblock,	
		     const qtensor3<Tm>& site,
		     oper_dict<Tm>& qops1,
		     oper_dict<Tm>& qops2,
		     oper_dict<Tm>& qops,
		     const int& isym,
		     const bool& ifkr,
		     const std::vector<int>& krest,
	             const integral::two_body<Tm>& int2e,
	             const integral::one_body<Tm>& int1e,
		     const bool debug=false){
   if(debug) std::cout << "\nctns::oper_renorm_opP" << std::endl;
   auto t0 = tools::get_time();
   // preprocess
   std::vector<int> info;
   oper_combine_opP(krest, ifkr, info);
   // compute
   for(const auto index : info){
      auto opwf = oper_compxwf_opP(superblock,site,qops1,qops2,isym,ifkr,int2e,int1e,index);
      qops('P')[index] = oper_kernel_renorm(superblock,site,opwf);
   }
   auto t1 = tools::get_time();
   if(debug){
      std::cout << "timing for ctns::oper_renorm_opP : " << std::setprecision(2) 
	        << tools::get_duration(t1-t0) << " s" << std::endl;
   }
}

// Qps = <pq||sr> aq^+ar
template <typename Tm>
void oper_renorm_opQ(const std::string& superblock,
		     const qtensor3<Tm>& site,
		     oper_dict<Tm>& qops1,
		     oper_dict<Tm>& qops2,
		     oper_dict<Tm>& qops,
		     const int& isym,
		     const bool& ifkr,
		     const std::vector<int>& krest,
	             const integral::two_body<Tm>& int2e,
	             const integral::one_body<Tm>& int1e,
		     const bool debug=false){
   if(debug) std::cout << "\nctns::oper_renorm_opQ" << std::endl;
   auto t0 = tools::get_time();
   // preprocess
   std::vector<int> info;
   oper_combine_opQ(krest, ifkr, info);
   // compute
   for(const int index : info){
      auto opwf = oper_compxwf_opQ(superblock,site,qops1,qops2,isym,ifkr,int2e,int1e,index);
      qops('Q')[index] = oper_kernel_renorm(superblock,site,opwf);
   }
   auto t1 = tools::get_time();
   if(debug){ 
      std::cout << "timing for ctns::oper_renorm_opQ : " << std::setprecision(2) 
	        << tools::get_duration(t1-t0) << " s" << std::endl;
   }
}

// Sp = 1/2 hpq aq + <pq||sr> aq^+aras [r>s]
template <typename Tm>
void oper_renorm_opS(const std::string& superblock,
		     const qtensor3<Tm>& site,
		     oper_dict<Tm>& qops1,
		     oper_dict<Tm>& qops2,
		     oper_dict<Tm>& qops,
		     const int& isym,
		     const bool& ifkr,
		     const std::vector<int>& krest,
	             const integral::two_body<Tm>& int2e,
	             const integral::one_body<Tm>& int1e,
		     const bool debug=false){
   if(debug) std::cout << "\nctns::oper_renorm_opS" << std::endl;
   auto t0 = tools::get_time();
   // preprocess
   std::vector<int> info;
   oper_combine_opS(krest, ifkr, info);
   // compute
   for(const int index : info){
      auto opwf = oper_compxwf_opS(superblock,site,qops1,qops2,isym,ifkr,int2e,int1e,index);
      qops('S')[index] = oper_kernel_renorm(superblock,site,opwf);
   }
   auto t1 = tools::get_time();
   if(debug){
      std::cout << "timing for ctns::oper_renorm_opS : " << std::setprecision(2) 
	        << tools::get_duration(t1-t0) << " s" << std::endl;
   }
}

template <typename Tm>
void oper_renorm_opH(const std::string& superblock,
		     const qtensor3<Tm>& site,
		     oper_dict<Tm>& qops1,
		     oper_dict<Tm>& qops2,
		     oper_dict<Tm>& qops,
		     const int& isym,
		     const bool& ifkr,
	             const integral::two_body<Tm>& int2e,
	             const integral::one_body<Tm>& int1e,
		     const bool debug=false){
   if(debug) std::cout << "\nctns::oper_renorm_opH" << std::endl;
   auto t0 = tools::get_time();
   // compute 
   auto opwf = oper_compxwf_opH(superblock,site,qops1,qops2,isym,ifkr,int2e,int1e);
   qops('H')[0] = oper_kernel_renorm(superblock,site,opwf);
   auto t1 = tools::get_time();
   if(debug){
      std::cout << "timing for tns::oper_renorm_opH : " << std::setprecision(2) 
	        << tools::get_duration(t1-t0) << " s" << std::endl;
   }
}

// renormalize ops
template <typename Km>
void oper_renorm_opAll(const std::string& superblock,
		       const comb<Km>& icomb,
		       const comb_coord& p,
		       const integral::two_body<typename Km::dtype>& int2e,
		       const integral::one_body<typename Km::dtype>& int1e,
		       oper_dict<typename Km::dtype>& qops1,
		       oper_dict<typename Km::dtype>& qops2,
		       oper_dict<typename Km::dtype>& qops){
   const bool debug = debug_oper_dict; 
   const bool ifcheck = false; // check operators against explicit construction
   auto t0 = tools::get_time();
   const bool ifkr = kind::is_kramers<Km>();
   const int isym = Km::isym;
   std::cout << "ctns::oper_renorm_opAll coord=" << p 
             << " superblock=" << superblock 
	     << " ifkr=" << ifkr << std::endl;
  
   // settings for current site & krest 
   auto& node = icomb.topo.get_node(p);
   qtensor3<typename Km::dtype> site;
   std::vector<int> krest; // support for index of complementary ops 
   if(superblock == "cr"){
      site = icomb.rsites.at(p);
      krest = node.lsupport;
   }else if(superblock == "lc"){
      site = icomb.lsites.at(p);
      auto pr = node.right;
      krest = icomb.topo.get_node(pr).rsupport;
   }else if(superblock == "lr"){
      site = icomb.lsites.at(p);
      auto pc = node.center;
      krest = icomb.topo.get_node(pc).rsupport;
   }

   // combine cindex first 
   qops.cindex = oper_combine_cindex(qops1.cindex, qops2.cindex);
 
   // C
   oper_renorm_opC(superblock, site, qops1, qops2, qops, debug);
   if(ifcheck) oper_check_rbasis(icomb, icomb, p, qops, 'C');
   // A
   oper_renorm_opA(superblock, site, qops1, qops2, qops, ifkr, debug);
   if(ifcheck) oper_check_rbasis(icomb, icomb, p, qops, 'A');
   // B
   oper_renorm_opB(superblock, site, qops1, qops2, qops, ifkr, debug);
   if(ifcheck) oper_check_rbasis(icomb, icomb, p, qops, 'B');
   // P
   oper_renorm_opP(superblock, site, qops1, qops2, qops, isym, ifkr, krest, int2e, int1e, debug);
   if(ifcheck) oper_check_rbasis(icomb, icomb, p, qops, 'P', int2e, int1e);
   // Q
   oper_renorm_opQ(superblock, site, qops1, qops2, qops, isym, ifkr, krest, int2e, int1e, debug);
   if(ifcheck) oper_check_rbasis(icomb, icomb, p, qops, 'Q', int2e, int1e);
   // S
   oper_renorm_opS(superblock, site, qops1, qops2, qops, isym, ifkr, krest, int2e, int1e, debug);
   if(ifcheck) oper_check_rbasis(icomb, icomb, p, qops, 'S', int2e, int1e);
   
   // H
   oper_renorm_opH(superblock, site, qops1, qops2, qops, isym, ifkr, int2e, int1e, debug);
   if(ifcheck) oper_check_rbasis(icomb, icomb, p, qops, 'H', int2e, int1e);
   // consistency check for Hamiltonian
   const auto& H = qops('H').at(0);
   auto diffH = (H-H.H()).normF();
   if(diffH > 1.e-10){
      H.print("H",2);
      std::string msg = "error: H-H.H() is too large! diffH=";
      tools::exit(msg+std::to_string(diffH));
   }

   auto t1 = tools::get_time();
   if(debug){
      std::cout << "timing for ctns::oper_renorm_opAll : " << std::setprecision(2) 
                << tools::get_duration(t1-t0) << " s" << std::endl;
      std::cout << std::endl;
   }
}

} // ctns

#endif
