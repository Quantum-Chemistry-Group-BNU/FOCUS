#ifndef PDVDSON_H
#define PDVDSON_H

#ifndef SERIAL
#include <boost/mpi.hpp>
#endif

#include "../../core/ortho.h"

namespace ctns{

// solver
template <typename Tm>	
struct pdvdsonSolver_nkr{
   public:

      // simple constructor
      pdvdsonSolver_nkr(const int _ndim, const int _neig, const double _crit_v, const int _maxcycle){
         ndim = _ndim;
	 neig = _neig;
	 crit_v = _crit_v;
	 maxcycle = _maxcycle;
      }

      // iteration info
      void print_iter(const int iter,
		      const int nsub,
		      const linalg::matrix<double>& eigs,
		      const linalg::matrix<double>& rnorm,
		      const double t){
	 const std::string ifconverge = "-+";
         if(iter == 1){
            if(iprt > 0){		 
               std::cout << std::defaultfloat; 
               std::cout << "settings: ndim=" << ndim 
                         << " neig=" << neig
                         << " nbuff=" << nbuff  
                         << " maxcycle=" << maxcycle << std::endl; 
	       std::cout << "          damping=" << damping << std::scientific 
                         << " crit_v=" << crit_v 
                         << " crit_e=" << crit_e 
                         << " crit_indp=" << crit_indp << std::endl;
	    }
	    std::cout << "iter   ieig        eigenvalue        ediff      rnorm   nsub  nmvp   time/s    tav/s" << std::endl;
         }
         for(int i=0; i<neig; i++){
            std::cout << std::setw(5) << iter << " " 
                 << std::setw(3) << i << " "
                 << std::setw(1) << ifconverge[(rnorm(i,iter)<crit_v)] << " "
                 << std::setw(20) << std::setprecision(12) << std::fixed << eigs(i,iter) << " "
                 << std::setw(10) << std::setprecision(2) << std::scientific << eigs(i,iter)-eigs(i,iter-1) << " "
                 << std::setw(10) << std::setprecision(2) << std::scientific << rnorm(i,iter) << " "
          	 << std::setw(4) << nsub << " " 
                 << std::setw(5) << nmvp << " "
                 << std::setw(10) << std::setprecision(2) << std::scientific << t 
                 << std::setw(10) << std::setprecision(2) << std::scientific << t/nmvp 
		 << std::endl;
         } // i
      }

      // perform H*x for a set of input vectors: x(nstate,ndim)
      void HVecs(const int nstate, Tm* y, const Tm* x){
         int size = 1, rank = 0;
#ifndef SERIAL
         size = world.size();
	 rank = world.rank();
	 world.barrier();
#endif
	 double tcal = 0.0, tcomm = 0.0;
         auto ti = tools::get_time();
         for(int istate=0; istate<nstate; istate++){
	    auto t0 = tools::get_time();
            HVec(y+istate*ndim, x+istate*ndim); // y=H*x
	    auto t1 = tools::get_time();
	    tcal += tools::get_duration(t1-t0);
#ifndef SERIAL
	    if(size > 1){
	       std::vector<Tm> y_sum(ndim);
	       boost::mpi::reduce(world, y+istate*ndim, ndim, y_sum.data(), std::plus<Tm>(), 0);
	       linalg::xcopy(ndim, y_sum.data(), y+istate*ndim);
	    }
	    auto t2 = tools::get_time();
	    tcomm += tools::get_duration(t2-t1); 
#endif
         }
         nmvp += nstate;
         auto tf = tools::get_time();
         if(rank == 0){
/*
            auto dt = tools::get_duration(tf-ti);
	    std::cout << "T(tot/cal/comm)=" << dt << "," 
		      << tcal << "," << tcomm
                      << " for nstate=" << nstate 
		      << " tav=" << dt/nstate 
		      << std::endl;
*/
	    t_cal += tcal;
	    t_comm += tcomm;
         }
      }

      // check by full diag
      void solve_diag(double* es, Tm* vs, const bool ifCheckDiag=false){
         int size = 1, rank = 0;
#ifndef SERIAL
         size = world.size();
	 rank = world.rank();
#endif
	 if(rank == 0) std::cout << "ctns::pdvdsonSolver_nkr:solve_diag" << std::endl;
         auto t0 = tools::get_time();
	 linalg::matrix<Tm> id = linalg::identity_matrix<Tm>(ndim);
         linalg::matrix<Tm> H(ndim,ndim);
         HVecs(ndim,H.data(),id.data());
	 if(rank == 0){
            // check consistency with diag
            if(ifCheckDiag){
               std::cout << "ndim=" << ndim << std::endl;
   	       std::cout << std::setprecision(12);
               double diff = 0.0;
               for(int i=0; i<ndim; i++){
	          std::cout << "i=" << i 
                            << " H(i,i)=" << H(i,i) 
                            << " Diag[i]=" << Diag[i] 
                            << " diff=" << Diag[i]-H(i,i)
                            << std::endl;
                  diff += std::abs(Diag[i]-H(i,i));
                  if(diff>1.e-10) tools::exit("error: |Diag[i]-H(i,i)| is too large!");
               } // i
	       std::cout << "CheckDiag passed successfully!" << std::endl;
            }
            // check symmetry
            auto sdiff = H.diff_hermitian();
	    std::cout << "|H-H.h|=" << sdiff << std::endl;
            if(sdiff > 1.e-5){
               (H-H.H()).print("H-H.h");
	       tools::exit("error: H is not symmetric in ctns::pdvdsonSolver_nkr::solve_diag!");
            }
            // solve eigenvalue problem by diagonalization
	    std::vector<double> e(ndim);
	    linalg::matrix<Tm> V;
	    linalg::eig_solver(H, e, V);
	    std::cout << "eigenvalues:\n" << std::setprecision(12);
            for(int i=0; i<ndim; i++){
	       std::cout << "i=" << i << " e=" << e[i] << std::endl;
            }
            // copy results
	    linalg::xcopy(neig, e.data(), es);
            linalg::xcopy(ndim*neig, V.data(), vs);
	 } // rank-0
         auto t1 = tools::get_time();
         if(rank == 0) tools::timing("solve_diag", t0, t1);
      }

      // subspace problem
      void subspace_solver(const int ndim, 
		      	   const int nsub,
		      	   const int nt,
      	       		   std::vector<Tm>& vbas,
      	       		   std::vector<Tm>& wbas,
      	       		   std::vector<double>& tmpE,
      	       		   std::vector<Tm>& tmpV,
			   std::vector<Tm>& rbas){
         // 1. form H in the subspace: H = V^+W, V(ndim,nsub), W(ndim,nsub)
         const Tm alpha = 1.0, beta=0.0;
	 linalg::matrix<Tm> tmpH(nsub,nsub);
	 linalg::xgemm("C","N",&nsub,&nsub,&ndim,
                       &alpha,vbas.data(),&ndim,wbas.data(),&ndim,
                       &beta,tmpH.data(),&nsub);
         // 2. check symmetry property
         double diff = tmpH.diff_hermitian();
         if(diff > crit_skewH){
            tmpH.print("tmpH");
            std::string msg = "error: ctns::pdvdsonSolver_nkr::subspace_solver: diff_skewH=";
	    tools::exit(msg+std::to_string(diff)); 
         }
         // 3. solve eigenvalue problem
	 linalg::matrix<Tm> tmpU;
	 linalg::eig_solver(tmpH, tmpE, tmpU);
	 linalg::xcopy(nsub*nt, tmpU.data(), tmpV.data());
         // 4. form full residuals: Res[i]=HX[i]-e[i]*X[i]
         // vbas = X[i]
	 linalg::xcopy(ndim*nsub, vbas.data(), rbas.data()); 
	 linalg::xgemm("N","N",&ndim,&nt,&nsub,
                       &alpha,rbas.data(),&ndim,tmpV.data(),&nsub,
                       &beta,vbas.data(),&ndim);
         // wbas = HX[i]
	 linalg::xcopy(ndim*nsub, wbas.data(), rbas.data()); 
	 linalg::xgemm("N","N",&ndim,&nt,&nsub,
                       &alpha,rbas.data(),&ndim,tmpV.data(),&nsub,
                       &beta,wbas.data(),&ndim);
         // rbas = HX[i]-e[i]*X[i]
         for(int i=0; i<nt; i++){
            std::transform(&wbas[i*ndim],&wbas[i*ndim]+ndim,&vbas[i*ndim],&rbas[i*ndim],
                           [i,&tmpE](const Tm& w, const Tm& x){ return w-x*tmpE[i]; }); 
         }
      }

      // Davidson iterative algorithm for Hv=ve 
      void solve_iter(double* es, Tm* vs, Tm* vguess=nullptr){
         auto ti = tools::get_time();
         int size = 1, rank = 0;
#ifndef SERIAL
         size = world.size();
	 rank = world.rank();
#endif
	 if(rank == 0 && iprt > 0){
	    std::cout << "ctns::pdvdsonSolver_nkr::solve_iter"
	              << " is_complex=" << tools::is_complex<Tm>() 
		      << " mpisize=" << size
		      << std::endl;
	 }
         if(neig > ndim){
	    std::string msg = "error: neig>ndim in pdvdson! neig/ndim=";	
	    tools::exit(msg+std::to_string(neig)+","+std::to_string(ndim));
	 }
         // clear counter
         nmvp = 0;

         // 1. generate initial subspace - vbas
         int nl = std::min(ndim,neig+nbuff); // maximal subspace size
	 std::vector<Tm> vbas(ndim*nl), wbas(ndim*nl);
	 if(rank == 0){
            if(vguess != nullptr){
	       linalg::xcopy(ndim*neig, vguess, vbas.data());
            }else{
               auto index = tools::sort_index(ndim, Diag);
               for(int i=0; i<neig; i++){
                  vbas[i*ndim+index[i]] = 1.0;
               }
            }
	    linalg::check_orthogonality(ndim, neig, vbas);
	 }
#ifndef SERIAL
	 if(size > 1) boost::mpi::broadcast(world, vbas, 0);
#endif
         HVecs(neig, wbas.data(), vbas.data());

         // 2. begin to solve
	 std::vector<Tm> rbas(ndim*nl), tbas(ndim*nl), tmpV(nl*nl);
	 std::vector<double> tmpE(nl), tnorm(neig);
         std::vector<bool> rconv(neig);
	 // record history
	 linalg::matrix<double> eigs(neig,maxcycle+1,1.e3), rnorm(neig,maxcycle+1); 
         double damp = damping;
         bool ifconv = false;
         int nsub = neig;
         for(int iter=1; iter<maxcycle+1; iter++){
	    // ONLY rank-0 solve the subspace problem
	    if(rank == 0){
	       //------------------------------------------------------------------------
               // solve subspace problem and form full residuals: Res[i]=HX[i]-w[i]*X[i]
	       //------------------------------------------------------------------------
               subspace_solver(ndim,nsub,neig,vbas,wbas,tmpE,tmpV,rbas);
	       //------------------------------------------------------------------------
	       // compute norm of residual
               for(int i=0; i<neig; i++){
                  auto norm = linalg::xnrm2(ndim, &rbas[i*ndim]);
                  eigs(i,iter) = tmpE[i];
                  rnorm(i,iter) = norm;
                  rconv[i] = (norm < crit_v)? true : false;
               }
               auto t1 = tools::get_time();
               if(iprt >= 0) print_iter(iter,nsub,eigs,rnorm,tools::get_duration(t1-ti));
               // check convergence and return (e,v) if applied 
               ifconv = (count(rconv.begin(), rconv.end(), true) == neig);
	    }
#ifndef SERIAL
            if(size > 1) boost::mpi::broadcast(world, ifconv, 0);
#endif	  
            if(ifconv || iter == maxcycle){
#ifndef SERIAL
	       if(size > 1){
                  // broadcast results to all processors
		  boost::mpi::broadcast(world, tmpE.data(), neig, 0);
                  boost::mpi::broadcast(world, vbas.data(), ndim*neig, 0);
	       }
#endif
	       linalg::xcopy(neig, tmpE.data(), es);
               linalg::xcopy(ndim*neig, vbas.data(), vs);
               break;
            }
            // if not converged, improve the subspace by ri/(abs(D-ei)+damp) 
	    int nindp = 0;
	    if(rank == 0){
               int nres = 0;
               for(int i=0; i<neig; i++){
                  if(rconv[i]) continue;
	          std::transform(&rbas[i*ndim], &rbas[i*ndim]+ndim, Diag, &tbas[nres*ndim],
                                 [i,&tmpE,&damp](const Tm& r, const double& d){ return r/(std::abs(d-tmpE[i])+damp); });
                  tnorm[nres] = linalg::xnrm2(ndim,&tbas[nres*ndim]);
                  nres += 1;		
               }
               // *** this part is critical for better performance ***
               // ordering the residual to be added from large to small
               auto index = tools::sort_index(nres, tnorm.data(), 1);
               for(int i=0; i<nres; i++){
	          linalg::xcopy(ndim, &tbas[index[i]*ndim], &rbas[i*ndim]); 
               }
               // re-orthogonalization and get nindp
               nindp = linalg::get_ortho_basis(ndim,neig,nres,vbas,rbas,crit_indp);
	    }
#ifndef SERIAL
            if(size > 1) boost::mpi::broadcast(world, nindp, 0);	    
#endif
            if(nindp == 0){
	       std::cout << "Convergence failure: unable to generate new direction: nindp=0!" << std::endl;
               exit(1);
            }else{
               // expand V and W
               nindp = std::min(nindp,nbuff);
#ifndef SERIAL
	       if(size > 1) boost::mpi::broadcast(world, &rbas[0], ndim*nindp, 0);
#endif	       
	       linalg::xcopy(ndim*nindp, &rbas[0], &vbas[ndim*neig]);
	       HVecs(nindp, &wbas[ndim*neig], &vbas[ndim*neig]);
               nsub = neig+nindp;
	       if(rank == 0) linalg::check_orthogonality(ndim,nsub,vbas);
            }
         } // iter
         if(rank == 0 && !ifconv){
            std::cout << "convergence failure: out of maxcycle=" << maxcycle << std::endl;
         }
	 auto tf = tools::get_time();    
	 if(rank == 0){
	    t_tot = tools::get_duration(tf-ti);
	    t_rest = t_tot - t_cal - t_comm;
            std::cout << "TIMING for Davidson : " << t_tot
		      << "  T(cal/comm/rest)=" << t_cal << ","
		      << t_comm << "," << t_rest
		      << std::endl;
	 }
      }
   public:
      // basics
      int ndim = 0;
      int neig = 0;
      double* Diag;
      std::function<void(Tm*, const Tm*)> HVec;
      double crit_v = 1.e-5;  // used control parameter
      int maxcycle = 200;
      // settings
      int iprt = 0;
      double crit_e = 1.e-12; // not used actually
      double crit_indp = 1.e-12;
      double crit_skewH = 1.e-8;
      double damping = 1.e-1;
      int nbuff = 4; // maximal additional vectors
      int nmvp = 0;
#ifndef SERIAL
      boost::mpi::communicator world;
#endif
      double t_tot = 0.0;
      double t_cal = 0.0; // Hx
      double t_comm = 0.0; // reduce
      double t_rest = 0.0; // solver
};

} // ctns

#endif
