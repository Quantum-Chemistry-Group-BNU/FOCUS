#ifndef SWEEP_TWODOT_LOCAL_H
#define SWEEP_TWODOT_LOCAL_H

#include "oper_functors.h"

namespace ctns{

template <typename Km>
void twodot_guess(comb<Km>& icomb, 
	          const directed_bond& dbond,
	          const int nsub,
	          const int neig,
	          stensor4<typename Km::dtype>& wf,
	          std::vector<typename Km::dtype>& v0){
   const bool debug_twodot_guess = true;
   if(debug_twodot_guess) std::cout << "ctns::twodot_guess ";
   auto pdx0 = icomb.topo.rindex.at(dbond.p0);
   auto pdx1 = icomb.topo.rindex.at(dbond.p1);
   assert(icomb.psi.size() == neig);
   if(dbond.forward){
      if(!dbond.cturn){

         if(debug_twodot_guess) std::cout << "|lc1>" << std::endl;
	 for(int i=0; i<neig; i++){
	    // psi[l,a,c1] => cwf[lc1,a]
	    auto cwf = icomb.psi[i].merge_lc(); 
	    // cwf[lc1,a]*r[a,r,c2] => wf3[lc1,r,c2]
            auto wf3 = contract_qt3_qt2_l(icomb.rsites[pdx1],cwf); 
	    // wf3[lc1,r,c2] => wf4[l,r,c1,c2]
	    auto wf4 = wf3.split_lc1(wf.info.qrow, wf.info.qmid);
	    assert(wf4.size() == nsub);
            wf4.to_array(&v0[nsub*i]);
         }

      }else{
	 
	 //
	 //     c2
	 //      |
	 // c1---p1 
	 //      |
	 //  l---p0---r
	 //     [psi]
	 //
	 if(debug_twodot_guess) std::cout << "|lr>(comb)" << std::endl;
	 for(int i=0; i<neig; i++){
            // psi[l,r,a] => cwf[lr,a]		 
	    auto cwf = icomb.psi[i].merge_lr(); // on backone
	    // r[a,c2,c1] => r[a,c1c2], cwf[lr,a]*r[a,c1c2] => wf2[lr,c1c2]
	    auto wf2 = cwf.dot(icomb.rsites[pdx1].merge_cr());
	    // wf2[lr,c1c2] => wf4[l,r,c1,c2] 
	    auto wf4 = wf2.split_lr_c1c2(wf.info.qrow, wf.info.qcol, wf.info.qmid, wf.info.qver);
	    assert(wf4.size() == nsub);
            wf4.to_array(&v0[nsub*i]);
	 }

      } // cturn
   }else{
      if(!dbond.cturn){

	 if(debug_twodot_guess) std::cout << "|c2r>" << std::endl;
	 for(int i=0; i<neig; i++){
            // psi[a,r,c2] => cwf[a,c2r]
	    auto cwf = icomb.psi[i].merge_cr();
	    // l[l,a,c1]*cwf[a,c2r] => wf3[l,c2r,c1]
	    auto wf3 = contract_qt3_qt2_r(icomb.lsites[pdx0],cwf.T());
	    // wf3[l,c2r,c1] => wf4[l,r,c1,c2] 
	    auto wf4 = wf3.split_c2r(wf.info.qver, wf.info.qcol);
	    assert(wf4.size() == nsub);
            wf4.to_array(&v0[nsub*i]);
	 }

      }else{

	 //
	 //     c2
	 //      |
	 // c1---p0 [psi]
	 //      |
	 //  l---p1---r
	 //
	 if(debug_twodot_guess) std::cout << "|c1c2>(comb)" << std::endl;
	 for(int i=0; i<neig; i++){
	    // psi[a,c2,c1] => cwf[a,c1c2]
	    auto cwf = icomb.psi[i].merge_cr(); // on branch
	    // l[l,r,a] => l[lr,a], l[lr,a]*cwf[a,c1c2] => wf2[lr,c1c2]
	    auto wf2 = icomb.lsites[pdx0].merge_lr().dot(cwf);
            // wf2[lr,c1c2] => wf4[l,r,c1,c2]
            auto wf4 = wf2.split_lr_c1c2(wf.info.qrow, wf.info.qcol, wf.info.qmid, wf.info.qver);
	    wf4.permCR_signed(); // back to backbone
	    assert(wf4.size() == nsub);
            wf4.to_array(&v0[nsub*i]);
	 }

      } // cturn
   } // forward
}

// local CI solver	
template <typename Km>
void twodot_localCI(comb<Km>& icomb,
		    const int nsub,
		    const int neig,
   		    std::vector<double>& diag,
		    HVec_type<typename Km:: dtype> HVec,
   		    std::vector<double>& eopt,
   		    linalg::matrix<typename Km::dtype>& vsol,
		    int& nmvp,
		    const int cisolver,
		    const bool guess,
		    const double eps,
		    const int maxcycle,
		    const int parity,
		    const directed_bond& dbond,
		    stensor4<typename Km::dtype>& wf){
   using Tm = typename Km::dtype;
   int size = 1, rank = 0;
#ifndef SERIAL
   size = icomb.world.size();
   rank = icomb.world.rank();
#endif

   // without kramers restriction
   assert(!qkind::is_kramers<Km>());
   pdvdsonSolver_nkr<Tm> solver(nsub, neig, eps, maxcycle);
   solver.Diag = diag.data();
   solver.HVec = HVec;
#ifndef SERIAL
   solver.world = icomb.world;
#endif
   if(cisolver == 0){

      // full diagonalization for debug
      solver.solve_diag(eopt.data(), vsol.data(), true);

   }else if(cisolver == 1){ 
	   
      // davidson
      if(!guess){
	 // davidson without initial guess
         solver.solve_iter(eopt.data(), vsol.data()); 
      }else{     
	 //------------------------------------
	 // prepare initial guess     
	 //------------------------------------
         std::vector<Tm> v0(nsub*neig);
	 if(rank == 0){
	    // starting guess 
            if(icomb.psi.size() == 0) onedot_guess_psi0(icomb, neig);
	    // specific to twodot 
            twodot_guess(icomb, dbond, nsub, neig, wf, v0);
	    // reorthogonalization
            int nindp = linalg::get_ortho_basis(nsub, neig, v0); 
            assert(nindp == neig);
	 }
	 //------------------------------------
         solver.solve_iter(eopt.data(), vsol.data(), v0.data());
      }

   }
   nmvp = solver.nmvp;
}

/*
template <>
inline void twodot_localCI(comb<qkind::cNK>& icomb,
		    const int nsub,
		    const int neig,
   		    std::vector<double>& diag,
		    HVec_type<std::complex<double>> HVec,
   		    std::vector<double>& eopt,
   		    linalg::matrix<std::complex<double>>& vsol,
		    int& nmvp,
		    const int cisolver,
		    const bool guess,
		    const double eps,
		    const int maxcycle,
		    const int parity,
		    const directed_bond& dbond,
		    stensor4<std::complex<double>>& wf){
   using Tm = std::complex<double>;
   int size = 1, rank = 0;
#ifndef SERIAL
   size = icomb.world.size();
   rank = icomb.world.rank();
#endif

   // kramers restricted (currently works only for iterative with guess!) 
   assert(qkind::is_kramers<Km>());
   assert(cisolver == 1 && guess);
   pdvdsonSolver_kr<Tm,stensor4<Tm>> solver(nsub, neig, eps, maxcycle, parity, wf); 
   solver.Diag = diag.data();
   solver.HVec = HVec;
#ifndef SERIAL
   solver.world = icomb.world;
#endif
   //------------------------------------
   // prepare initial guess     
   //------------------------------------
   std::vector<Tm> v0;
   if(rank == 0){
      if(icomb.psi.size() == 0) onedot_guess_psi0(icomb,neig); // starting guess 
      auto psi4 = twodot_guess(icomb, dbond, nsub, neig, wf);
      // load initial guess from previous opt
      solver.init_guess(psi4, v0);
   }
   //------------------------------------
   solver.solve_iter(eopt.data(), vsol.data(), v0.data());
   nmvp = solver.nmvp;
}
*/

} // ctns

#endif
