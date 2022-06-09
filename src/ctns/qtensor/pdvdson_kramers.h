#ifndef PDVDSON_KRAMERS_H
#define PDVDSON_KRAMERS_H

#include "../../core/ortho.h"

#ifndef SERIAL
#include <boost/mpi.hpp>
#endif

namespace ctns{

// solver
template <typename Tm, typename QTm>
struct pdvdsonSolver_kr{
   public:

      // simple constructor
      pdvdsonSolver_kr(const int _ndim, const int _neig, const double _crit_v, const int _maxcycle, 
		      const int _parity, QTm& _wf){
         ndim = _ndim;
	 neig = _neig;
	 crit_v = _crit_v;
	 maxcycle = _maxcycle;
	 parity = _parity;
	 pwf = &_wf;
	 // consistency check
	 if(parity == 1 && neig%2 == 1){
	    tools::exit(std::string("error: odd-electron case requires even neig=")+std::to_string(neig));
	 }
      }

      // iteration info
      void print_iter(const int iter,
		      const int nsub,
		      const linalg::matrix<double>& eigs,
		      const linalg::matrix<double>& rnorm,
		      const double t){
	 //const std::string line(87,'-');
	 const std::string ifconverge = "-+";
         if(iter == 1){
            std::cout << std::defaultfloat; 
            std::cout << "settings: ndim=" << ndim 
                      << " neig=" << neig
                      << " nbuff=" << nbuff  
                      << " maxcycle=" << maxcycle << std::endl; 
	    std::cout << "          damping=" << damping << std::scientific 
                      << " crit_v=" << crit_v 
                      << " crit_e=" << crit_e 
                      << " crit_indp=" << crit_indp << std::endl;
	    std::cout << "iter   ieig        eigenvalue        ediff      rnorm   nsub  nmvp   time/s    tav/s" << std::endl;
            //std::cout << line << std::endl;
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
	 //std::cout << line << std::endl;
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
            HVec(y+istate*ndim, x+istate*ndim); // y = tilde{H}*x
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
            //-------------------------------------------------------------
            // Even-electron case: get full sigma vector from skeleton one
            //-------------------------------------------------------------
	    if(parity == 0){
	       pwf->from_array( y+istate*ndim );
	       *pwf += pwf->K();
	       pwf->to_array( y+istate*ndim );
	    }
            //-------------------------------------------------------------
         } // istate
         nmvp += nstate;
         auto tf = tools::get_time();
         if(rank == 0){
            auto dt = tools::get_duration(tf-ti);
/*
	    std::cout << "T(tot/cal/comm)=" << dt << "," 
		      << tcal << "," << tcomm
                      << " for nstate=" << nstate 
		      << " tav=" << dt/nstate << " s" 
		      << std::endl;
*/
	    t_cal += tcal;
	    t_comm += tcomm;
         }
      }

      // initialization
      void init_guess(const std::vector<QTm>& psi, 
		      std::vector<Tm>& v0){
	 std::cout << "ctns::pdvdsonSolver_kr::init_guess parity=" << parity << std::endl;
	 assert(psi.size() == neig && psi[0].size() == ndim);
         v0.resize(ndim*neig*2);
	 int nindp = 0;
         if(parity == 0){
            // even-electron case
            const std::complex<double> iunit(0.0,1.0);
/*
            for(int i=0; i<neig; i++){
	       std::cout << "I=" << i << " - " << psi[i].normF() << std::endl;
	       auto psiK = psi[i].K();
	       std::cout << "I=" << i << " - " << psi[i].normF() << " - " << psiK.normF() << std::endl;
	       exit(1);
	    }
    	    exit(1);	   
*/ 
            for(int i=0; i<neig; i++){
/*
     		    std::cout << "I=" << i << " - " << psi[i].normF() << std::endl;
*/
	       auto psiK = psi[i].K();

               auto tmp1 = (psi[i] + psiK);
               auto tmp2 = (psi[i] - psiK)*iunit; 
               tmp1.to_array(&v0[ndim*(i)]); // put all plus combination before
               tmp2.to_array(&v0[ndim*(i+neig)]);
               std::cout << " iguess=" << i 
		         << " |psi|=" << psi[i].normF()
		         << " |psiK|=" << psiK.normF()
           	         << " |psi+psiK|=" << tmp1.normF() 
           	         << " |i(psi-psiK)|=" << tmp2.normF() 
           	         << std::endl;
            } // i
            nindp = linalg::get_ortho_basis(ndim, neig*2, v0); // reorthogonalization
         }else{
            // odd-electron case: needs to first generate Kramers paired basis
            for(int i=0; i<neig; i++){
               psi[i].to_array(&v0[ndim*(2*i)]);
               psi[i].K().to_array(&v0[ndim*(2*i+1)]);
	       std::cout << " iguess=" << i 
		         << " |psi(K)|=" << psi[i].normF()
			 << std::endl;
            } // i
            nindp = kramers::get_ortho_basis_qt(ndim, neig*2, v0, *pwf); // reorthogonalization
         }
         std::cout << " neig,nindp=" << neig << "," << nindp << std::endl;
         assert(nindp >= neig);
	 v0.resize(ndim*nindp);
      }

      //-------------------------------------
      // Case 0: even-electron Hilbert space 
      //-------------------------------------
      void subspace_solver_even(const int ndim, 
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
	    std::string msg = "error in ctns::pdvdsonSolver_kr::subspace_solver_even!";
	    tools::exit(msg+" diff_skewH="+std::to_string(diff));
         }
	 //-------------------------------------------------------- 
         // 3. solve eigenvalue problem [in real alrithemics]
	 //-------------------------------------------------------- 
	 linalg::matrix<double> tmpX;
         linalg::eig_solver(tmpH.real(), tmpE, tmpX);
	 auto tmpU = tmpX.as_complex();
	 linalg::xcopy(nsub*nt, tmpU.data(), tmpV.data());
	 //-------------------------------------------------------- 
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

      //-------------------------------------
      // Case 1: odd-electron Hilbert space 
      //-------------------------------------
      void subspace_solver_odd(const int ndim, 
		      	       const int nsub,
		      	       const int nt,
      	       		       std::vector<Tm>& vbas,
      	       		       std::vector<Tm>& wbas,
      	       		       std::vector<double>& tmpE,
      	       		       std::vector<Tm>& tmpV,
			       std::vector<Tm>& rbas){
         // 1. form H in the subspace: H = V^+W, V(ndim,nsub), W(ndim,nsub)
         const Tm alpha = 1.0, beta=0.0;
	 linalg::matrix<Tm> tmpH2(nsub,nsub);
	 linalg::xgemm("C","N",&nsub,&nsub,&ndim,
                       &alpha,vbas.data(),&ndim,wbas.data(),&ndim,
                       &beta,tmpH2.data(),&nsub);
	 //-----------------------------------------------------------
	 // 2. construct full Hamiltonian from skeleton sigma vector
	 //-----------------------------------------------------------
	 assert(nsub%2 == 0);
	 int nsub2 = nsub/2;
	 std::vector<int> pos_new(nsub);
	 linalg::matrix<Tm> tmpH(nsub,nsub);
	 for(int i=0; i<nsub2; i++){
	    for(int j=0; j<nsub2; j++){
	       auto a = tmpH2(2*i,2*j);
	       auto b = tmpH2(2*i,2*j+1);
	       auto c = tmpH2(2*i+1,2*j);
	       auto d = tmpH2(2*i+1,2*j+1);
	       auto h11 = a + tools::conjugate(d);
	       auto h12 = b - tools::conjugate(c);
	       tmpH(i,j) = h11; 
	       tmpH(i+nsub2,j+nsub2) = tools::conjugate(h11);
	       tmpH(i,j+nsub2) = h12;
	       tmpH(i+nsub2,j) = -tools::conjugate(h12);
	    }
	    pos_new[i] = 2*i;
	    pos_new[i+nsub2] = 2*i+1;
	 }
       	 // check symmetry property
         double diff = tmpH.diff_hermitian();
         if(diff > crit_skewH){
            tmpH.print("tmpH");
	    std::string msg = "error in ctns::pdvdsonSolver_kr::subspace_solver_odd!";
	    tools::exit(msg+" diff_skewH="+std::to_string(diff)); 
         }
	 //-----------------------------------------------------------
         // 3. solve eigenvalue problem  
	 //-----------------------------------------------------------
         // TRS-preserving diagonalization (only half eigs are output) 
	 std::vector<double> tmpE2(nsub);
	 linalg::matrix<Tm> tmpU;
	 kramers::zquatev(tmpH, tmpE, tmpU);
	 for(int i=0; i<nsub2; i++){
	    tmpE2[2*i] = tmpE[i];
	    tmpE2[2*i+1] = tmpE[i];
	 }
	 tmpE = tmpE2;
	 tmpU = tmpU.reorder_rowcol(pos_new, pos_new, 1);
	 linalg::xcopy(nsub*nt, tmpU.data(), tmpV.data());
	 //-----------------------------------------------------------
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
	 std::vector<Tm> krvec(ndim);
         for(int i=0; i<nt/2; i++){
	    int i0 = 2*i;
	    int i1 = 2*i+1;
            //-------------------------------------------------------------------------
            // We need to first contruct full sigma vector from skeleton one using TRS
            //-------------------------------------------------------------------------
	    // sigma[o]
	    kramers::get_krvec_qt(&wbas[i1*ndim], krvec.data(), *pwf, 0);
            std::transform(&wbas[i0*ndim], &wbas[i0*ndim]+ndim, krvec.begin(), &rbas[i0*ndim],
			   [](const Tm& x, const Tm& y){ return x + y; });
            std::transform(&rbas[i0*ndim], &rbas[i0*ndim]+ndim, &vbas[i0*ndim], &rbas[i0*ndim],
                           [i0,&tmpE](const Tm& w, const Tm& x){ return w-x*tmpE[i0]; }); 
	    // sigma[o_bar] 
	    kramers::get_krvec_qt(&wbas[i0*ndim], krvec.data(), *pwf);
            std::transform(&wbas[i1*ndim], &wbas[i1*ndim]+ndim, krvec.begin(), &rbas[i1*ndim], 
			   [](const Tm& x, const Tm& y){ return x + y; });
            std::transform(&rbas[i1*ndim], &rbas[i1*ndim]+ndim, &vbas[i1*ndim], &rbas[i1*ndim],
                           [i1,&tmpE](const Tm& w, const Tm& x){ return w-x*tmpE[i1]; }); 
            //-------------------------------------------------------------------------
         } // i
      }

      // Davidson iterative algorithm for Hv=ve
      void solve_iter(double* es, Tm* vs, Tm* vguess){
         auto ti = tools::get_time();
         int size = 1, rank = 0;
#ifndef SERIAL
         size = world.size();
	 rank = world.rank();
#endif
	 if(rank == 0){
	    std::cout << "ctns::pdvdsonSolver_kr::solve_iter"
	              << " parity=" << parity 
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
	    linalg::xcopy(ndim*neig, vguess, vbas.data()); // copying neig states from vguess
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
	       if(parity == 0){
                  subspace_solver_even(ndim,nsub,neig,vbas,wbas,tmpE,tmpV,rbas);
	       }else{
                  subspace_solver_odd(ndim,nsub,neig,vbas,wbas,tmpE,tmpV,rbas);
	       }
	       //------------------------------------------------------------------------
               // compute norm of residual
               for(int i=0; i<neig; i++){
                  auto norm = linalg::xnrm2(ndim, &rbas[i*ndim]);
                  eigs(i,iter) = tmpE[i];
                  rnorm(i,iter) = norm;
                  rconv[i] = (norm < crit_v)? true : false;
               }
               auto t1 = tools::get_time();
               if(iprt > 0) print_iter(iter,nsub,eigs,rnorm,tools::get_duration(t1-ti));
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
	          //-------------------------------------------
	          // Kramers projection to ensure K|psi>=|psi>
	          //-------------------------------------------
	          if(parity == 0){
   	             pwf->from_array( &tbas[nres*ndim] );
   	             *pwf += pwf->K();
   	             pwf->to_array( &tbas[nres*ndim] );
	          }
	          //-------------------------------------------
                  tnorm[nres] = linalg::xnrm2(ndim,&tbas[nres*ndim]);
                  nres += 1;
               }
               // *** this part is critical for better performance ***
               // ordering the residual to be added from large to small
               auto index = tools::sort_index(nres, tnorm.data(), 1);
               for(int i=0; i<nres; i++){
	          linalg::xcopy(ndim, &tbas[index[i]*ndim], &rbas[i*ndim]); 
               }
	       //------------------------------------------------------------------
               // re-orthogonalization and get nindp for different cases of parity
	       //------------------------------------------------------------------
	       if(parity == 0){
	          nindp = linalg::get_ortho_basis(ndim,neig,nres,vbas,rbas,crit_indp);
	       }else{
                  nindp = kramers::get_ortho_basis_qt(ndim,neig,nres,vbas,rbas,*pwf,crit_indp);
	       }
	       //------------------------------------------------------------------
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
            std::cout << "TIMING for Davdison : " << t_tot
		      << " T(cal/comm/rest)=" << t_cal << ","
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
      int maxcycle = 1000;
      //--------------------
      // Kramers projection
      //--------------------
      int parity = 0;
      QTm* pwf; 
      //--------------------
      // settings
      int iprt = 1;
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
