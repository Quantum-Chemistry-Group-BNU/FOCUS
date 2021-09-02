#ifndef DVDSON_H
#define DVDSON_H

#include <iostream>
#include <iomanip>
#include <vector>
#include <functional> // for std::function
#include "tools.h"
#include "matrix.h"
#include "linalg.h"

namespace linalg{

// orthogonality of vbas(ndim,mstate) as in Fortran
template <typename Tm>
double check_orthogonality(const linalg::matrix<Tm>& V,
  		           const double thresh=1.e-10){
   int n = V.rows();
   int m = V.cols();
   linalg::matrix<Tm> dev = xgemm("C","N",V,V) - identity_matrix<Tm>(m);
   double diff = normF(dev)/static_cast<double>(m);
   if(diff > thresh){
      //dev.print("dev");
      std::cout << "error in check_orthogonality: dim=" << m 
	        << " diff=" << std::scientific << diff 
		<< " thresh=" << thresh
	        << std::endl;
      std::exit(1);
   }
   return diff;
}
template <typename Tm>
double check_orthogonality(const int n, const int m, 
   	      	           const std::vector<Tm>& vbas,
  		           const double thresh=1.e-10){
   linalg::matrix<Tm> V(n,m,vbas.data());
   return check_orthogonality(V, thresh);
}

// modified Gram-Schmidt orthogonalization of 
// rbas(ndim,nres) against vbas(ndim,neig)
template <typename Tm>
int get_ortho_basis(const int ndim,
  	      	    const int neig,
  		    const int nres,
  		    const std::vector<Tm>& vbas,
  		    std::vector<Tm>& rbas,
  		    const double crit_indp=1.e-12){
   const Tm one = 1.0, mone = -1.0, zero = 0.0;
   const int maxtimes = 2;
   // 1. projection (1-V*V^+)*R = R-V*(V^+R)
   std::vector<Tm> vtr(neig*nres);
   for(int repeat=0; repeat<maxtimes; repeat++){
      linalg::xgemm("C","N",&neig,&nres,&ndim,
	            &one,vbas.data(),&ndim,rbas.data(),&ndim,
	            &zero,vtr.data(),&neig);
      linalg::xgemm("N","N",&ndim,&nres,&neig,
	            &mone,vbas.data(),&ndim,vtr.data(),&neig,
	            &one,rbas.data(),&ndim);
   }
   // 2. form new basis from rbas by modified Gram-Schmidt procedure
   std::vector<Tm> rtr(nres*nres/4);
   int nindp = 0;
   for(int i=0; i<nres; i++){
      double rii = linalg::xnrm2(ndim, &rbas[i*ndim]); // normalization constant
      if(rii < crit_indp) continue;
      // normalized |r[i]> 
      for(int repeat=0; repeat<maxtimes; repeat++){
	 std::transform(&rbas[i*ndim], &rbas[i*ndim]+ndim, &rbas[i*ndim],
		        [rii](const Tm& x){ return x/rii; });
         rii = linalg::xnrm2(ndim, &rbas[i*ndim]);
      }
      // copy
      std::copy(&rbas[i*ndim], &rbas[i*ndim]+ndim, &rbas[nindp*ndim]);
      nindp +=1;
      // project out |r[i]>-component from other basis
      // essentially equivalent to https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_procesis
      // since [r[i+1:]> is changing when a new |r[i]> is find.  
      int N = nres-1-i;
      if(N == 0) break;
      for(int repeat=0; repeat<maxtimes; repeat++){
         // R_rest = (1-V*V^+)*R_rest
	 linalg::xgemm("C","N",&neig,&N,&ndim,
                       &one,vbas.data(),&ndim,&rbas[(i+1)*ndim],&ndim,
                       &zero,vtr.data(),&neig);
         linalg::xgemm("N","N",&ndim,&N,&neig,
                       &mone,vbas.data(),&ndim,vtr.data(),&neig,
                       &one,&rbas[(i+1)*ndim],&ndim);
         // R_rest = (1-Rnew*Rnew^+)*R_rest
         linalg::xgemm("C","N",&nindp,&N,&ndim,
                       &one,&rbas[0],&ndim,&rbas[(i+1)*ndim],&ndim,
                       &zero,rtr.data(),&nindp);
         linalg::xgemm("N","N",&ndim,&N,&nindp,
                       &mone,&rbas[0],&ndim,rtr.data(),&nindp,
                       &one,&rbas[(i+1)*ndim],&ndim);
      } // repeat
   } // i
   return nindp;
}

// MGS for rbas of size rbas(ndim,nres)
template <typename Tm>
int get_ortho_basis(const int ndim,
		    const int nres,
		    std::vector<Tm>& rbas,
		    const double crit_indp=1.e-12){
   const Tm one = 1.0, mone = -1.0, zero = 0.0;
   const int maxtimes = 2;
   // 2. form new basis from rbas by modified Gram-Schmidt procedure
   std::vector<Tm> rtr(nres*nres/4);
   int nindp = 0;
   for(int i=0; i<nres; i++){
      double rii = linalg::xnrm2(ndim, &rbas[i*ndim]); // normalization constant
      if(rii < crit_indp) continue;
      // normalized |r[i]> 
      for(int repeat=0; repeat<maxtimes; repeat++){
	 std::transform(&rbas[i*ndim], &rbas[i*ndim]+ndim, &rbas[i*ndim],
		        [rii](const Tm& x){ return x/rii; });
         rii = linalg::xnrm2(ndim, &rbas[i*ndim]);
      }
      // copy
      std::copy(&rbas[i*ndim], &rbas[i*ndim]+ndim, &rbas[nindp*ndim]);
      nindp +=1;
      // project out |r[i]>-component from other basis
      // essentially equivalent to https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_procesis
      // since [r[i+1:]> is changing when a new |r[i]> is find.  
      int N = nres-1-i;
      if(N == 0) break;
      for(int repeat=0; repeat<maxtimes; repeat++){
         // R_rest = (1-Rnew*Rnew^+)*R_rest
	 linalg::xgemm("C","N",&nindp,&N,&ndim,
               	       &one,&rbas[0],&ndim,&rbas[(i+1)*ndim],&ndim,
               	       &zero,rtr.data(),&nindp);
	 linalg::xgemm("N","N",&ndim,&N,&nindp,
                       &mone,&rbas[0],&ndim,rtr.data(),&nindp,
                       &one,&rbas[(i+1)*ndim],&ndim);
      } // repeat
   } // i
   return nindp;
}

// solver
template <typename Tm>	
struct dvdsonSolver{
   public:
      // simple constructor
      dvdsonSolver(){};
      dvdsonSolver(const int _ndim, const int _neig, const double _crit_v, const int _maxcycle){
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
	 std::string ifconverge = "-+";
         if(iter == 1){
            std::cout << std::defaultfloat; 
            std::cout << "settings: ndim=" << ndim 
                      << " neig=" << neig
                      << " nbuff=" << nbuff  
                      << " maxcycle= " << maxcycle << std::endl; 
	    std::cout << "          damping=" << damping 
                      << " crit_v=" << crit_v 
                      << " crit_e=" << crit_e 
                      << " crit_indp=" << crit_indp << std::endl;
	    std::cout << "iter   ieig        eigenvalue        ediff      rnorm   nsub  nmvp  time/s" << std::endl;
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
                 << std::setw(10) << std::setprecision(2) << std::scientific << t << std::endl;
         } // i
      } 
      // perform H*x for a set of input vectors: x(nstate,ndim)
      void HVecs(const int nstate, Tm* y, const Tm* x){
         auto t0 = tools::get_time();
         for(int istate=0; istate<nstate; istate++){
            HVec(y+istate*ndim, x+istate*ndim); // y=H*x
         }
         nmvp += nstate;
         auto t1 = tools::get_time();
	 /*
         bool debug = true;
         if(debug){
	    std::cout << "timing for HVecs : " << std::setprecision(2)  
                      << tools::get_duration(t1-t0) << " s" 
                      << " for nstate = " << nstate << std::endl;
         }
	 */
      }
      // check by full diag
      void solve_diag(double* es, Tm* vs, const bool ifCheckDiag=false){
	 std::cout << "linalg::dvdsonSolver:solve_diag" << std::endl;
	 linalg::matrix<Tm> id = identity_matrix<Tm>(ndim);
         linalg::matrix<Tm> H(ndim,ndim);
         HVecs(ndim,H.data(),id.data());
         // final check consistency with diag
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
         auto sdiff = linalg::symmetric_diff(H);
	 std::cout << "|H-H.h|=" << sdiff << std::endl;
         if(sdiff > 1.e-5){
            (H-H.H()).print("H-H.h");
	    tools::exit("error: H is not symmetric in linalg::dvdsonSolver::solve_diag!");
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
	 std::copy(e.begin(), e.begin()+neig, es);
         std::copy(V.data(), V.data()+ndim*neig, vs);
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
         double diff = linalg::symmetric_diff(tmpH);
         if(diff > crit_skewH){
            tmpH.print("tmpH");
            std::string msg = "error: linalg::dvdsonSolver::subspace_solver: diff_skewH=";
	    tools::exit(msg+std::to_string(diff)); 
         }
         // 3. solve eigenvalue problem
	 linalg::matrix<Tm> tmpU;
	 linalg::eig_solver(tmpH, tmpE, tmpU);
	 std::copy(tmpU.data(), tmpU.data()+nsub*nt, tmpV.data());
         // 4. form full residuals: Res[i]=HX[i]-e[i]*X[i]
         // vbas = X[i]
	 std::copy(vbas.data(), vbas.data()+ndim*nsub, rbas.data()); 
	 linalg::xgemm("N","N",&ndim,&nt,&nsub,
                       &alpha,rbas.data(),&ndim,tmpV.data(),&nsub,
                       &beta,vbas.data(),&ndim);
         // wbas = HX[i]
	 std::copy(wbas.data(), wbas.data()+ndim*nsub, rbas.data()); 
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
	 std::cout << "linalg::dvdsonSolver::solve_iter"
		   << " is_complex=" << tools::is_complex<Tm>() << std::endl;
         if(neig > ndim){
	    std::string msg = "error: neig>ndim in dvdson! neig/ndim=";	
	    tools::exit(msg+std::to_string(neig)+","+std::to_string(ndim));
	 }
         // clear counter
         nmvp = 0;
         auto t0 = tools::get_time();

         // generate initial subspace - vbas
         int nl = std::min(ndim,neig+nbuff); // maximal subspace size
	 std::vector<Tm> vbas(ndim*nl), wbas(ndim*nl);
         if(vguess != nullptr){
	    std::copy(vguess, vguess+ndim*neig, vbas.data());
         }else{
            auto index = tools::sort_index(ndim, Diag);
            for(int i=0; i<neig; i++){
               vbas[i*ndim+index[i]] = 1.0;
            }
         }
         check_orthogonality(ndim, neig, vbas);
         HVecs(neig, wbas.data(), vbas.data());

         // Begin to solve
	 std::vector<Tm> rbas(ndim*nl), tbas(ndim*nl), tmpV(nl*nl);
	 std::vector<double> tmpE(nl), tnorm(neig);
         std::vector<bool> rconv(neig);
	 // record history
	 linalg::matrix<double> eigs(neig,maxcycle+1,1.e3), rnorm(neig,maxcycle+1); 
         double damp = damping;
         bool ifconv = false;
         int nsub = neig;
         for(int iter=1; iter<maxcycle+1; iter++){
           
            // solve subspace problem and form full residuals: Res[i]=HX[i]-w[i]*X[i]
            subspace_solver(ndim,nsub,neig,vbas,wbas,tmpE,tmpV,rbas);
            // check convergence
            for(int i=0; i<neig; i++){
               auto norm = linalg::xnrm2(ndim, &rbas[i*ndim]);
               eigs(i,iter) = tmpE[i];
               rnorm(i,iter) = norm;
               rconv[i] = (norm < crit_v)? true : false;
            }
            auto t1 = tools::get_time();
            if(iprt > 0) print_iter(iter,nsub,eigs,rnorm,tools::get_duration(t1-t0));
            t0 = tools::get_time();
            
            // check convergence and return (e,v) if applied 
            ifconv = (count(rconv.begin(), rconv.end(), true) == neig);
            if(ifconv || iter == maxcycle){
	       std::copy(tmpE.data(), tmpE.data()+neig, es);
               std::copy(vbas.data(), vbas.data()+ndim*neig, vs);
               break;
            }
            // if not converged, improve the subspace by ri/(abs(D-ei)+damp) 
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
	       std::copy(&tbas[index[i]*ndim], &tbas[index[i]*ndim]+ndim, &rbas[i*ndim]); 
            }
            // re-orthogonalization and get nindp
            int nindp = get_ortho_basis(ndim,neig,nres,vbas,rbas,crit_indp);
            if(nindp == 0){
	       std::cout << "Convergence failure: unable to generate new direction: nindp=0!" << std::endl;
               exit(1);
            }else{
               // expand V and W
               nindp = std::min(nindp,nbuff);
	       std::copy(&rbas[0],&rbas[0]+ndim*nindp,&vbas[ndim*neig]);
               HVecs(nindp, &wbas[ndim*neig], &vbas[ndim*neig]);
               nsub = neig+nindp;
               check_orthogonality(ndim,nsub,vbas);
            }
         } // iter
         if(!ifconv){
            std::cout << "convergence failure: out of maxcycle =" << maxcycle << std::endl;
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
      // settings
      int iprt = 1;
      double crit_e = 1.e-12; // not used actually
      double crit_indp = 1.e-12;
      double crit_skewH = 1.e-8;
      double damping = 1.e-1;
      int nbuff = 4; // maximal additional vectors
      int nmvp = 0;
};

} // linalg

#endif
