#ifndef SWEEP_ONEDOT_H
#define SWEEP_ONEDOT_H

#include <functional> // for std::function
#include "../core/tools.h"
#include "../core/linalg.h"
#include "qtensor/qtensor.h"
#include "sweep_guess.h"
#include "sweep_onedot_ham.h"
#include "sweep_onedot_renorm.h"
#include "ctns_sys.h"

namespace ctns{

template <typename Tm>
using HVec_type = std::function<void(Tm*, const Tm*)>;

template <typename Km>
void onedot_localCI(comb<Km>& icomb,
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
		    qtensor3<typename Km::dtype>& wf){
   int size = 1, rank = 0;
#ifndef SERIAL
   size = icomb.world.size();
   rank = icomb.world.rank();
#endif
   using Tm = typename Km::dtype;
   // without kramers restriction
   dvdsonSolver_nkr<Tm> solver(nsub, neig, eps, maxcycle);
   solver.Diag = diag.data();
   solver.HVec = HVec;
#ifndef SERIAL
   solver.world = icomb.world;
#endif
   if(cisolver == 0){
      solver.solve_diag(eopt.data(), vsol.data(), true); // full diagonalization for debug
   }else if(cisolver == 1){ // davidson
      if(!guess){
         solver.solve_iter(eopt.data(), vsol.data()); // davidson without initial guess
      }else{    
         std::vector<Tm> v0(nsub*neig);
	 if(rank == 0){ 
            if(icomb.psi.size() == 0) onedot_guess_psi0(icomb, neig); // starting guess 
            assert(icomb.psi.size() == neig);
            assert(icomb.psi[0].get_dim() == nsub);
            // load initial guess from previous opt
            for(int i=0; i<neig; i++){
               icomb.psi[i].to_array(&v0[nsub*i]);
            }
            int nindp = linalg::get_ortho_basis(nsub, neig, v0); // reorthogonalization
            assert(nindp == neig);
	 }
         solver.solve_iter(eopt.data(), vsol.data(), v0.data());
      }
   }
   nmvp = solver.nmvp;
}
template <>
inline void onedot_localCI(comb<kind::cNK>& icomb,
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
		    qtensor3<std::complex<double>>& wf){
   int size = 1, rank = 0;
#ifndef SERIAL
   size = icomb.world.size();
   rank = icomb.world.rank();
#endif
   using Tm = std::complex<double>;
   // kramers restricted (currently works only for iterative with guess!)
   assert(cisolver == 1 && guess);
   dvdsonSolver_kr<Tm,qtensor3<Tm>> solver(nsub, neig, eps, maxcycle, parity, wf); 
   solver.Diag = diag.data();
   solver.HVec = HVec;
#ifndef SERIAL
   solver.world = icomb.world;
#endif
   std::vector<Tm> v0;
   if(rank == 0){
      if(icomb.psi.size() == 0) onedot_guess_psi0(icomb, neig); // starting guess 
      // load initial guess from previous opt
      solver.init_guess(icomb.psi, v0);
   }
   solver.solve_iter(eopt.data(), vsol.data(), v0.data());
   nmvp = solver.nmvp;
}

template <typename Km>
void sweep_onedot(const input::schedule& schd,
		  sweep_data& sweeps,
		  const int isweep,
		  const int ibond,
                  comb<Km>& icomb,
                  const integral::two_body<typename Km::dtype>& int2e,
                  const integral::one_body<typename Km::dtype>& int1e,
                  const double ecore){
   const bool debug_sweep = schd.ctns.verbose > 0; 
   int size = 1, rank = 0;
#ifndef SERIAL
   size = icomb.world.size();
   rank = icomb.world.rank();
#endif   
   if(rank == 0) std::cout << "ctns::sweep_onedot" << std::endl;
   const int isym = Km::isym;
   const bool ifkr = kind::is_kramers<Km>();
   auto& timing = sweeps.opt_timing[isweep][ibond];
   timing.t0 = tools::get_time();

   // 0. processing partition & symmetry
   auto dbond = sweeps.seq[ibond];
   auto p = dbond.p;
   std::vector<int> suppc, suppl, suppr;
   qbond qc, ql, qr;
   if(rank == 0 && debug_sweep) std::cout << "support info:" << std::endl;
   suppc = icomb.get_suppc(p, rank == 0 && debug_sweep); 
   suppl = icomb.get_suppl(p, rank == 0 && debug_sweep);
   suppr = icomb.get_suppr(p, rank == 0 && debug_sweep);
   qc = icomb.get_qc(p); 
   ql = icomb.get_ql(p);
   qr = icomb.get_qr(p);
   int sc = suppc.size();
   int sl = suppl.size();
   int sr = suppr.size();
   assert(sc+sl+sr == icomb.topo.nphysical);
   if(rank == 0){
      if(debug_sweep) std::cout << "qbond info:" << std::endl;
      qc.print("qc", debug_sweep);
      ql.print("ql", debug_sweep);
      qr.print("qr", debug_sweep);
   }

   // 1. load operators 
   using Tm = typename Km::dtype;
   oper_dict<Tm> cqops, lqops, rqops;
   oper_load_qops(icomb, p, schd.scratch, "c", cqops);
   oper_load_qops(icomb, p, schd.scratch, "l", lqops);
   oper_load_qops(icomb, p, schd.scratch, "r", rqops);
   if(schd.ctns.verbose > 1){
      for(int iproc=0; iproc<size; iproc++){
          if(rank == iproc){
             std::cout << "qops info: rank=" << rank << std::endl;
             const int level = 0;
             cqops.print("cqops", level);
             lqops.print("lqops", level);
             rqops.print("rqops", level);
	  }
#ifndef SERIAL
	  icomb.world.barrier();
#endif
      } // iproc
   }
   timing.ta = tools::get_time();

   // 2. Davidson solver for wf
   qsym sym_state = (isym == 1)? qsym(schd.nelec) : qsym(schd.nelec, schd.twoms);
   qtensor3<Tm> wf(sym_state, qc, ql, qr, {1,1,1});
   if(rank == 0 && debug_sweep) std::cout << "dim(localCI)=" << wf.get_dim() << std::endl;   
   int nsub = wf.get_dim();
   int neig = sweeps.nstates;
   auto& nmvp = sweeps.opt_result[isweep][ibond].nmvp;
   auto& eopt = sweeps.opt_result[isweep][ibond].eopt;
   linalg::matrix<Tm> vsol(nsub,neig);

   // 2.1 Hdiag 
   std::vector<double> diag(nsub,1.0);
   diag = onedot_Hdiag(ifkr, cqops, lqops, rqops, ecore, wf, size, rank);
#ifndef SERIAL
   // reduction of partial Hdiag: no need to broadcast, if only rank=0 
   // executes the preconditioning in Davidson's algorithm
   if(size > 1){
      std::vector<double> diag2(nsub);
      boost::mpi::reduce(icomb.world, diag, diag2, std::plus<double>(), 0);
      diag = diag2;
   }
#endif 
   timing.tb = tools::get_time();

   // 2.2 Solve local problem: Hc=cE
   using std::placeholders::_1;
   using std::placeholders::_2;
   auto HVec = bind(&ctns::onedot_Hx<Tm>, _1, _2, 
           	    std::cref(isym), std::cref(ifkr), 
           	    std::ref(cqops), std::ref(lqops), std::ref(rqops), 
           	    std::cref(int2e), std::cref(int1e), std::cref(ecore), 
           	    std::ref(wf), std::cref(size), std::cref(rank));
   onedot_localCI(icomb, nsub, neig, diag, HVec, eopt, vsol, nmvp,
		  schd.ctns.cisolver, sweeps.guess, sweeps.ctrls[isweep].eps, 
		  schd.ctns.maxcycle, (schd.nelec)%2, wf);
   timing.tc = tools::get_time();
   if(rank == 0) sweeps.print_eopt(isweep, ibond);

   // 3. decimation & renormalize operators
   onedot_renorm(sweeps, isweep, ibond, icomb, vsol, wf, 
		 cqops, lqops, rqops, int2e, int1e, schd.scratch);

   timing.t1 = tools::get_time();
   if(rank == 0){
      tools::timing("ctns::sweep_onedot", timing.t0, timing.t1);
      timing.analysis();
      get_sys_status();
   }
}

// use one dot algorithm to produce a final wavefunction
// in right canonical form for later usage 
template <typename Km>
void sweep_rwfuns(const input::schedule& schd,
		  comb<Km>& icomb,
		  const integral::two_body<typename Km::dtype>& int2e,
	          const integral::one_body<typename Km::dtype>& int1e,
		  const double ecore){
   using Tm = typename Km::dtype;
   int size = 1, rank = 0;
#ifndef SERIAL
   size = icomb.world.size();
   rank = icomb.world.rank();
#endif   
   if(rank == 0) std::cout << "ctns::sweep_rwfuns" << std::endl;

   // perform an additional onedot opt  
   auto p0 = std::make_pair(0,0);
   auto p1 = std::make_pair(1,0);
   auto cturn = icomb.topo.is_cturn(p0,p1);
   auto dbond = directed_bond(p0,p1,0,p1,cturn); // fake dbond
   const int dcut1 = -1;
   const double eps = schd.ctns.ctrls[schd.ctns.maxsweep-1].eps; // take the last eps 
   input::params_sweep ctrl = {0, 1, dcut1, eps, 0.0};
   sweep_data sweeps({dbond}, schd.ctns.nroots, schd.ctns.guess, 
		     0, 1, {ctrl}, 0, schd.ctns.rdm_vs_svd);
   sweep_onedot(schd, sweeps, 0, 0, icomb, int2e, int1e, ecore);

   if(rank == 0){
      std::cout << "deal with site0 by decimation for rsite0 & rwfuns" << std::endl;
      const auto& wf = icomb.psi[0]; // only rank-0 has psi from renorm
      // decimation
      qtensor2<Tm> rot;
      std::vector<qtensor2<Tm>> wfs2;
      for(int i=0; i<schd.ctns.nroots; i++){
         auto wf2 = icomb.psi[i].merge_cr().T();
	 wfs2.push_back(wf2);
      }
      const int dcut = schd.ctns.nroots;
      double dwt; 
      int deff;
      const bool ifkr = tools::is_complex<Km>();
      decimation_row(ifkr, wf.qmid, wf.qcol, dcut, schd.ctns.rdm_vs_svd, wfs2,
		     rot, dwt, deff);
      rot = rot.T(); 
      icomb.rsites[p0] = rot.split_cr(wf.qmid, wf.qcol);
      // form rwfuns(istate,irbas)
      auto& sym_state = icomb.psi[0].sym;
      qbond qrow({{sym_state, schd.ctns.nroots}});
      auto& qcol = rot.qrow; 
      qtensor2<typename Km::dtype> rwfuns(qsym(), qrow, qcol, {0, 1});
      assert(qcol.size() == 1);
      int rdim = qrow.get_dim(0);
      int cdim = qcol.get_dim(0);
      for(int i=0; i<schd.ctns.nroots; i++){
         auto cwf = icomb.psi[i].merge_cr().dot(rot.H()); // <-W[1,alpha]->
         for(int c=0; c<cdim; c++){
            rwfuns(0,0)(i,c) = cwf(0,0)(0,c);	      
         }
      } // iroot
      icomb.rwfuns = std::move(rwfuns);
   } // rank0
}
   
} // ctns

#endif
