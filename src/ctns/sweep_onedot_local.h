#ifndef SWEEP_ONEDOT_LOCAL_H
#define SWEEP_ONEDOT_LOCAL_H

#include "oper_functors.h"

namespace ctns{

// generate initial guess for initial sweep optimization at p=(1,0)
template <typename Km>
void onedot_guess_psi0(comb<Km>& icomb, const int nroots){
   const auto& rindex = icomb.topo.rindex;
   const auto& rsite0 = icomb.rsites[rindex.at(std::make_pair(0,0))];
   const auto& rsite1 = icomb.rsites[rindex.at(std::make_pair(1,0))];
   const auto& qrow = icomb.rwfuns[0].info.qrow;
   assert(qrow.size() == 1); // only same symmetry of wfs
   if(icomb.get_nroots() < nroots){
      std::cout << "dim(psi0)=" << icomb.get_nroots() << " nroots=" << nroots << std::endl;
      tools::exit("error in onedot_guess_psi0: requested nroots exceed!");
   }
   auto sym_state = qrow.get_sym(0);
   icomb.psi.resize(nroots);
   for(int iroot=0; iroot<nroots; iroot++){
      // qt2(1,r)
      auto qt2 = icomb.rwfuns[iroot];
      // qt2(1,r)*rsite0(r,r0,n0) = qt3(1,r0,n0)
      auto qt3 = contract_qt3_qt2("l",rsite0,qt2);
      // qt3(1,r0,n0) -> cwf(n0,r0)
      stensor2<typename Km::dtype> cwf(sym_state, rsite0.info.qmid, rsite0.info.qcol, {1,1});
      for(int br=0; br<cwf.rows(); br++){
	 for(int bc=0; bc<cwf.cols(); bc++){
	    auto blk = cwf(br,bc);
	    if(blk.empty()) continue;
	    const auto blk0 = qt3(0,bc,br);
	    int rdim = cwf.info.qrow.get_dim(br); 
	    int cdim = cwf.info.qcol.get_dim(bc);
	    for(int ir=0; ir<rdim; ir++){
	       for(int ic=0; ic<cdim; ic++){
	     	  blk(ir,ic) = blk0(0,ic,ir); 
	       } // ic
	    } // ir
	 } // bc
      } // br
      // cwf(n0,r0)*rsite1(r0,r1,n1) = psi(n0,r1,n1)
      icomb.psi[iroot] = contract_qt3_qt2("l",rsite1,cwf);
   } // iroot
}

// local CI solver	
template <typename Km>
void onedot_localCI(comb<Km>& icomb,
		    const input::schedule& schd,
		    const double eps,
		    const int parity,
		    const int nsub,
		    const int neig,
   		    std::vector<double>& diag,
		    HVec_type<typename Km:: dtype> HVec,
   		    std::vector<double>& eopt,
   		    linalg::matrix<typename Km::dtype>& vsol,
		    int& nmvp,
		    stensor3<typename Km::dtype>& wf){
   using Tm = typename Km::dtype;
   int size = 1, rank = 0;
#ifndef SERIAL
   size = icomb.world.size();
   rank = icomb.world.rank();
#endif

   // without kramers restriction
   assert(Km::ifkr == false);
   pdvdsonSolver_nkr<Tm> solver(nsub, neig, eps, schd.ctns.maxcycle);
   solver.iprt = schd.ctns.verbose;
   solver.Diag = diag.data();
   solver.HVec = HVec;
#ifndef SERIAL
   solver.world = icomb.world;
#endif
   if(schd.ctns.cisolver == 0){

      // full diagonalization for debug
      solver.solve_diag(eopt.data(), vsol.data(), true);

   }else if(schd.ctns.cisolver == 1){ 
      
      // davidson
      if(!schd.ctns.guess){
	 // davidson without initial guess
         solver.solve_iter(eopt.data(), vsol.data()); 
      }else{    
	 //------------------------------------
	 // prepare initial guess     
	 //------------------------------------
         std::vector<Tm> v0;
	 if(rank == 0){ 
	    // starting guess 
            if(icomb.psi.size() == 0) onedot_guess_psi0(icomb, neig); 
            assert(icomb.psi.size() == neig && icomb.psi[0].size() == nsub);
            // load initial guess from previous opt
	    v0.resize(nsub*neig);
            for(int i=0; i<neig; i++){
               icomb.psi[i].to_array(&v0[nsub*i]);
            }
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

template <>
inline void onedot_localCI(comb<qkind::cNK>& icomb,
			   const input::schedule& schd,
		           const double eps,
		           const int parity,
		           const int nsub,
		           const int neig,
   		           std::vector<double>& diag,
		           HVec_type<std::complex<double>> HVec,
   		           std::vector<double>& eopt,
   		           linalg::matrix<std::complex<double>>& vsol,
		           int& nmvp,
		           stensor3<std::complex<double>>& wf){
   using Tm = std::complex<double>;
   int size = 1, rank = 0;
#ifndef SERIAL
   size = icomb.world.size();
   rank = icomb.world.rank();
#endif

   // kramers restricted (currently works only for iterative with guess!)
   assert(schd.ctns.cisolver == 1 && schd.ctns.guess);
   pdvdsonSolver_kr<Tm,stensor3<Tm>> solver(nsub, neig, eps, schd.ctns.maxcycle, parity, wf); 
   solver.iprt = schd.ctns.verbose;
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
      if(icomb.psi.size() == 0) onedot_guess_psi0(icomb, neig); // starting guess 
      // load initial guess from previous opt
      solver.init_guess(icomb.psi, v0);
   }
   //------------------------------------
   solver.solve_iter(eopt.data(), vsol.data(), v0.data());
   nmvp = solver.nmvp;
}

} // ctns

#endif
