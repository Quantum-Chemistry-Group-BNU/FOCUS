#ifndef OPER_RENORM_H
#define OPER_RENORM_H

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ctns{

// renormalize ops
template <typename Km, typename Tm>
void oper_renorm_opAll(const std::string& superblock,
		       const comb<Km>& icomb,
		       const comb_coord& p,
		       const integral::two_body<Tm>& int2e,
		       const integral::one_body<Tm>& int1e,
		       oper_dict<Tm>& qops1,
		       oper_dict<Tm>& qops2,
		       oper_dict<Tm>& qops){
   int size = 1, rank = 0;
#ifndef SERIAL
   size = icomb.world.size();
   rank = icomb.world.rank();
#endif   
   const int isym = Km::isym;
   const bool ifkr = qkind::is_kramers<Km>();
   if(rank == 0){ 
      std::cout << "ctns::oper_renorm_opAll coord=" << p 
	        << " superblock=" << superblock 
	     	<< " isym=" << isym << " ifkr=" << ifkr
		<< " mpisize=" << size << std::endl;
   }
   auto t0 = tools::get_time();
   exit(1);
/*  
   // support for index of complementary ops 
   auto& node = icomb.topo.get_node(p);
   std::vector<int> krest; 
   if(superblock == "cr"){
      krest = node.lsupport;
   }else if(superblock == "lc"){
      auto pr = node.right;
      krest = icomb.topo.get_node(pr).rsupport;
   }else if(superblock == "lr"){
      auto pc = node.center;
      krest = icomb.topo.get_node(pc).rsupport;
   }

   oper_timer.clear();
   
   // combine cindex first 
   qops.cindex = oper_combine_cindex(qops1.cindex, qops2.cindex);
   const auto& site = (superblock=="cr")? icomb.rsites.at(p) : icomb.lsites.at(p);
   const bool ifcheck = false; // check operators against explicit construction
   // C
   oper_renorm_opC(superblock, icomb, p, site, qops1, qops2, qops);
   if(ifcheck) oper_check_rbasis(icomb, icomb, p, qops, 'C');
   // A
   oper_renorm_opA(superblock, icomb, p, site, qops1, qops2, qops, ifkr);
   if(ifcheck) oper_check_rbasis(icomb, icomb, p, qops, 'A');
   // B
   oper_renorm_opB(superblock, icomb, p, site, qops1, qops2, qops, ifkr);
   if(ifcheck) oper_check_rbasis(icomb, icomb, p, qops, 'B');
   // P
   oper_renorm_opP(superblock, icomb, p, site, qops1, qops2, qops, isym, ifkr, krest, int2e, int1e);
   if(ifcheck) oper_check_rbasis(icomb, icomb, p, qops, 'P', int2e, int1e);
   // Q
   oper_renorm_opQ(superblock, icomb, p, site, qops1, qops2, qops, isym, ifkr, krest, int2e, int1e);
   if(ifcheck) oper_check_rbasis(icomb, icomb, p, qops, 'Q', int2e, int1e);
   // S
   oper_renorm_opS(superblock, icomb, p, site, qops1, qops2, qops, isym, ifkr, krest, int2e, int1e);
   if(ifcheck) oper_check_rbasis(icomb, icomb, p, qops, 'S', int2e, int1e);
   // H
   oper_renorm_opH(superblock, icomb, p, site, qops1, qops2, qops, isym, ifkr, int2e, int1e);
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
   if(rank == 0){ 
      qops.print("qops");
      oper_timer.analysis();
      tools::timing("ctns::oper_renorm_opAll", t0, t1);
   }
*/
   exit(1);
}

} // ctns

#endif
