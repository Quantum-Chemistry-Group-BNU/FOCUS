#ifndef SWEEP_DVDSON_H
#define SWEEP_DVDSON_H

#include "../core/dvdson.h"

namespace ctns{

const bool debug_krsym = true;
extern const bool debug_krsym;

template <typename Tm> 
void onedot_Proj(Tm* y, qtensor3<Tm>& wf){
   wf.from_array(y);
   if(debug_krsym) std::cout << "deviation before=" << (wf-wf.K()).normF() << std::endl;
   wf = 0.5*(wf + wf.K());
   if(debug_krsym) std::cout << "deviation after=" << (wf-wf.K()).normF() << std::endl;
   wf.to_array(y);
}

template <typename Tm> 
void get_krvec(Tm* y, Tm* ykr, qtensor3<Tm>& wf, const int parity=1){
   wf.from_array(y);
//   wf = (parity==0)? wf.K() : wf.K(1);
   wf = wf.K();
   wf.to_array(ykr);
}

// modified Gram-Schmidt orthogonalization of 
// rbas(ndim,nres) against vbas(ndim,neig)
template <typename Tm>
int get_ortho_basis(const int ndim,
  	      	    const int neig,
  		    const int nres,
  		    const std::vector<Tm>& vbas,
  		    std::vector<Tm>& rbas,
		    qtensor3<Tm>& wf,
  		    const double crit_indp=1.e-12){
   const bool debug = false;
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
   };
   // 2. form new basis from rbas by modified Gram-Schmidt procedure
   std::vector<Tm> krvec(ndim);
   std::vector<Tm> rbas_new;
   int nindp = 0;
   for(int i=0; i<nres; i++){
      double rii = linalg::xnrm2(ndim, &rbas[i*ndim]); // normalization constant
      if(debug) std::cout << "\ni=" << i << " rii=" << rii << std::endl;
      if(rii < crit_indp) continue;
      // normalized |r[i]> 
      for(int repeat=0; repeat<maxtimes; repeat++){
	 std::transform(&rbas[i*ndim], &rbas[i*ndim]+ndim, &rbas[i*ndim],
		        [rii](const Tm& x){ return x/rii; });
         rii = linalg::xnrm2(ndim, &rbas[i*ndim]);
      }
      //-------------------------------------------------------------
      rbas_new.resize(ndim*(nindp+2));
      // copy
      std::copy(&rbas[i*ndim], &rbas[i*ndim]+ndim, &rbas_new[nindp*ndim]);
      nindp += 1;
      // add its time-reversal partner
      get_krvec(&rbas[i*ndim], krvec.data(), wf);
      std::copy(krvec.cbegin(), krvec.cend(), &rbas_new[nindp*ndim]);
      nindp += 1;
      if(debug){
         linalg::matrix<Tm> V(ndim,neig+nindp);
	 std::copy(vbas.begin(), vbas.begin()+ndim*neig, V.col(0));
	 std::copy(rbas_new.begin(), rbas_new.begin()+ndim*nindp, V.col(neig));
/*
	 // test
	 get_krvec(V.col(0), krvec.data(), wf);
	 Tm tmp0, tmp1;
	 tmp0 = linalg::xdot(ndim, krvec.data(), &vbas[0]);
	 tmp1 = linalg::xdot(ndim, krvec.data(), &vbas[ndim]);
         std::cout << "tmp0,tmp1=" << tmp0 << "," << tmp1 << std::endl;
	 
	 tmp0 = linalg::xdot(ndim, &vbas[0], &vbas[0]);
	 tmp1 = linalg::xdot(ndim, &vbas[0], &vbas[ndim]);
         std::cout << "tmp0,tmp1=" << tmp0 << "," << tmp1 << std::endl;
	 
	 tmp0 = linalg::xdot(ndim, krvec.data(), krvec.data());
	 tmp1 = linalg::xdot(ndim, krvec.data(), &vbas[0]);
         std::cout << "tmp0,tmp1=" << tmp0 << "," << tmp1 << std::endl;
	 exit(1); 
*/
         auto ova = xgemm("C","N",V,V);
         ova.print("ova");
	 auto dev = ova - linalg::identity_matrix<Tm>(neig+nindp);
	 double diff = normF(dev);
	 std::cout << "diff=" << diff << std::endl;
	 //exit(1);
      }
      //-------------------------------------------------------------
      // project out |r[i]>-component from other basis
      int N = nres-1-i;
      if(N == 0) continue;
      std::vector<Tm> vtr(neig*N), rtr(nindp*N);
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
                       &one,&rbas_new[0],&ndim,&rbas[(i+1)*ndim],&ndim,
                       &zero,rtr.data(),&nindp);
         linalg::xgemm("N","N",&ndim,&N,&nindp,
                       &mone,&rbas_new[0],&ndim,rtr.data(),&nindp,
                       &one,&rbas[(i+1)*ndim],&ndim);
      } // repeat
   } // i
   rbas = rbas_new;
   return nindp;
}

// MGS for rbas of size rbas(ndim,nres)
template <typename Tm>
int get_ortho_basis(const int ndim,
		    const int nres,
		    std::vector<Tm>& rbas,
		    qtensor3<Tm>& wf,
		    const double crit_indp=1.e-12){
   const bool debug = false;
   const Tm one = 1.0, mone = -1.0, zero = 0.0;
   const int maxtimes = 2;
   // 2. form new basis from rbas by modified Gram-Schmidt procedure
   std::vector<Tm> krvec(ndim);
   std::vector<Tm> rbas_new;
   int nindp = 0;
   for(int i=0; i<nres; i++){
      double rii = linalg::xnrm2(ndim, &rbas[i*ndim]); // normalization constant
      if(debug) std::cout << "\ni=" << i << " rii=" << rii << std::endl;
      if(rii < crit_indp) continue;
      // normalized |r[i]> 
      for(int repeat=0; repeat<maxtimes; repeat++){
	 std::transform(&rbas[i*ndim], &rbas[i*ndim]+ndim, &rbas[i*ndim],
		        [rii](const Tm& x){ return x/rii; });
         rii = linalg::xnrm2(ndim, &rbas[i*ndim]);
      }
      //-------------------------------------------------------------
      rbas_new.resize(ndim*(nindp+2));
      // copy
      std::copy(&rbas[i*ndim], &rbas[i*ndim]+ndim, &rbas_new[nindp*ndim]);
      nindp += 1;
      // add its time-reversal partner
      get_krvec(&rbas[i*ndim], krvec.data(), wf);
      std::copy(krvec.cbegin(), krvec.cend(), &rbas_new[nindp*ndim]);
      nindp += 1;
      if(debug){
         // check psi[lr] = psi_bar[l_bar,r_bar]*
         std::vector<Tm> tmp(ndim);
         get_krvec(&rbas_new[(nindp-1)*ndim], tmp.data(), wf, 0);
         std::transform(tmp.begin(), tmp.end(), &rbas[i*ndim], krvec.begin(),
                        [](const Tm& x, const Tm& y){ return x-y; }); 
	 auto diff = linalg::xnrm2(ndim, krvec.data());
         std::cout << "diff[psi]=" << diff << std::endl;
	 if(diff > 1.e-10){
	    std::cout << "error: in consistent psi and psi_bar!" << std::endl;
	 }
	 // check overlap matrix for basis
         linalg::matrix<Tm> V(ndim,nindp,rbas_new.data());
         auto ova = xgemm("C","N",V,V);
         ova.print("ova");
      }
      //-------------------------------------------------------------
      // project out |r[i]>-component from other basis
      int N = nres-1-i;
      if(N == 0) break;
      std::vector<Tm> rtr(nindp*N);
      // R_rest = (1-Rnew*Rnew^+)*R_rest
      for(int repeat=0; repeat<maxtimes; repeat++){
	 // rtr = Rnew^+*R_rest
	 linalg::xgemm("C","N",&nindp,&N,&ndim,
               	       &one,&rbas_new[0],&ndim,&rbas[(i+1)*ndim],&ndim,
               	       &zero,rtr.data(),&nindp);
	 // R_rest -= Rnew*rtr
	 linalg::xgemm("N","N",&ndim,&N,&nindp,
                       &mone,&rbas_new[0],&ndim,rtr.data(),&nindp,
                       &one,&rbas[(i+1)*ndim],&ndim);
      } // repeat
   } // i
   rbas = rbas_new;
   return nindp;
}

// solver
template <typename Tm>	
struct dvdsonSolver{
   public:
      // simple constructor
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
         bool debug = true;
         if(debug){
	    std::cout << "timing for HVecs : " << std::setprecision(2)  
                      << tools::get_duration(t1-t0) << " s" 
                      << " for nstate = " << nstate << std::endl;
         }
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
            std::cout << "error in ctns::dvdsonSolver::subspace_solver: diff_skewH=" 
                      << diff << std::endl;
            tmpH.print("tmpH");
            exit(1);
         }
	 //-------------------------------------------------------- 
         // 3. solve eigenvalue problem [in real alrithemics]
	 //-------------------------------------------------------- 
	 linalg::matrix<double> tmpX;
         linalg::eig_solver(tmpH.real(), tmpE, tmpX);
	 auto tmpU = tmpX.as_complex();
	 std::copy(tmpU.data(), tmpU.data()+nsub*nt, tmpV.data());
	 //-------------------------------------------------------- 
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
      void solve_iter(double* es, Tm* vs, Tm* vguess){
	 std::cout << "ctns::dvdsonSolver::solve_iter is_complex=" << tools::is_complex<Tm>() << std::endl;
         if(neig > ndim){
            std::cout << "error in dvdson: neig>ndim, neig/ndim=" << neig << "," << ndim << std::endl; 
            exit(1);
         }
         // clear counter
         nmvp = 0;
         auto t0 = tools::get_time();

         // generate initial subspace - vbas
         int nl = std::min(ndim,neig+nbuff); // maximal subspace size
	 std::vector<Tm> vbas(ndim*nl), wbas(ndim*nl);
	  
	 std::copy(vguess, vguess+ndim*neig, vbas.data());
	 linalg::check_orthogonality(ndim, neig, vbas);
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
                              [i,&tmpE,&damp](const Tm& r, const double& d){ return r/(abs(d-tmpE[i])+damp); });
	       //-----------------------
	       // Kramers projection
	       //-----------------------
	       Proj(&tbas[nres*ndim]);
	       //-----------------------
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
            int nindp = linalg::get_ortho_basis(ndim,neig,nres,vbas,rbas,crit_indp);
            if(nindp == 0){
	       std::cout << "Convergence failure: unable to generate new direction: nindp=0!" << std::endl;
               exit(1);
            }else{
               // expand V and W
               nindp = std::min(nindp,nbuff);
	       std::copy(&rbas[0],&rbas[0]+ndim*nindp,&vbas[ndim*neig]);
               HVecs(nindp, &wbas[ndim*neig], &vbas[ndim*neig]);
               nsub = neig+nindp;
               linalg::check_orthogonality(ndim,nsub,vbas);
            }
         } // iter
         if(!ifconv){
            std::cout << "convergence failure: out of maxcycle =" << maxcycle << std::endl;
         }
      }

      // subspace problem
      void subspace_solver2(const int ndim, 
		      	   const int nsub,
		      	   const int nt,
      	       		   std::vector<Tm>& vbas,
      	       		   std::vector<Tm>& wbas,
      	       		   std::vector<double>& tmpE,
      	       		   std::vector<Tm>& tmpV,
			   std::vector<Tm>& rbas,
			   qtensor3<Tm>& wf){
         const bool debug = true;
	 const double thresh = 1.e-10;
	 //--------------------------------------------------------
   	 std::vector<double> etmp(nsub);
  	 linalg::matrix<Tm> vtmp;
   	 linalg::matrix<Tm> htmp(nsub,nsub), stmp(nsub,nsub);
         if(debug){
            // check time-reversal relation among basis vectors
	    std::cout << "debug basis of subspace:" << std::endl;
	    std::vector<Tm> krtmp(ndim);
            for(int i=0; i<nsub/2; i++){
	       int i0 = 2*i;
	       int i1 = 2*i+1;
	       get_krvec(&vbas[i0*ndim], krtmp.data(), wf);
	       std::transform(&vbas[i1*ndim],&vbas[i1*ndim]+ndim,krtmp.data(),krtmp.data(),
	   		   [](const Tm& x, const Tm& y){ return x-y; });
	       std::cout << "i=" << i << std::endl;
	       std::cout << std::setprecision(8);
	       auto diff0 = linalg::xnrm2(ndim,krtmp.data());
	       std::cout << "diff(V0)=" << diff0 << std::endl; 
	       get_krvec(&vbas[i1*ndim], krtmp.data(), wf, 0);
	       std::transform(&vbas[i0*ndim],&vbas[i0*ndim]+ndim,krtmp.data(),krtmp.data(),
	   		   [](const Tm& x, const Tm& y){ return x-y; });
	       auto diff1 = linalg::xnrm2(ndim,krtmp.data());
	       std::cout << "diff(V1)=" << diff1 << std::endl;
	       if(diff0 > thresh || diff1 > thresh){
		  std::cout << "error: too large diff!" << std::endl;
		  exit(1);
	       }
	    } // i 
            // check Hamiltonian
	    std::cout << "debug matrix elements:" << std::endl;
	    int nsub2 = nsub/2;
            for(int i=0; i<nsub2; i++){
	       int i0 = 2*i;
	       int i1 = 2*i+1;
	       for(int j=0; j<nsub2; j++){
	          int j0 = 2*j;
		  int j1 = 2*j+1;
		  
		  // construct Sij
		  stmp(i,j) = linalg::xdot(ndim, &vbas[i0*ndim], &vbas[j0*ndim]);
		  stmp(i+nsub2,j) = linalg::xdot(ndim, &vbas[i1*ndim], &vbas[j0*ndim]);
		  stmp(i,j+nsub2) = linalg::xdot(ndim, &vbas[i0*ndim], &vbas[j1*ndim]);
		  stmp(i+nsub2,j+nsub2) = linalg::xdot(ndim, &vbas[i1*ndim], &vbas[j1*ndim]);
		  
		  // construct sigma vector
	          std::vector<Tm> sigmaj0(ndim), sigmaj1(ndim);
	          HVecs(1, sigmaj0.data(), &vbas[j0*ndim]);
	          HVecs(1, sigmaj1.data(), &vbas[j1*ndim]);

		  std::vector<Tm> vdiff(ndim,0.0);
	          std::transform(sigmaj0.begin(), sigmaj0.end(), &wbas[j0*ndim], vdiff.begin(),
			         [](const Tm& x, const Tm& y){ return x - y; });
	          auto diffW0 = linalg::xnrm2(ndim, vdiff.data());
		  std::cout << ">>> j0=" << j0 
			    << " |sigmaj0|=" << linalg::xnrm2(ndim, sigmaj0.data())
			    << " |wj0|=" << linalg::xnrm2(ndim, &wbas[j0*ndim]) 
			    << std::endl;
                  std::cout << "diff_W0=" << diffW0 << std::endl;
            	  std::transform(sigmaj1.begin(), sigmaj1.end(), &wbas[j1*ndim], vdiff.begin(),
			         [](const Tm& x, const Tm& y){ return x - y; });
	          auto diffW1 = linalg::xnrm2(ndim, vdiff.data());
		  std::cout << ">>> j1=" << j1 
			    << " |sigmaj1|=" << linalg::xnrm2(ndim, sigmaj1.data())
			    << " |wj1|=" << linalg::xnrm2(ndim, &wbas[j1*ndim]) 
			    << std::endl;
                  std::cout << "diff_W1=" << diffW1 << std::endl;
	          if(diffW0 > thresh || diffW1 > thresh){
	             std::cout << "error: too large diffW!" << std::endl;
	             exit(1);
	          }

		  // complete sigma vector by time-reversal operation
		  std::vector<Tm> sigmak0(ndim), sigmak1(ndim);
		  get_krvec(sigmaj0.data(), sigmak1.data(), wf);
	  	  get_krvec(sigmaj1.data(), sigmak0.data(), wf, 0);
                  std::transform(sigmaj0.begin(), sigmaj0.end(), sigmak0.data(), sigmaj0.data(),
			         [](const Tm& x, const Tm& y){ return x + y; });
                  std::transform(sigmaj1.begin(), sigmaj1.end(), sigmak1.data(), sigmaj1.data(),
			         [](const Tm& x, const Tm& y){ return x + y; });
		  // construct Hij
		  htmp(i,j) = linalg::xdot(ndim, &vbas[i0*ndim], sigmaj0.data());
		  htmp(i+nsub2,j) = linalg::xdot(ndim, &vbas[i1*ndim], sigmaj0.data());
		  htmp(i,j+nsub2) = linalg::xdot(ndim, &vbas[i0*ndim], sigmaj1.data());
		  htmp(i+nsub2,j+nsub2) = linalg::xdot(ndim, &vbas[i1*ndim], sigmaj1.data());

	       } // j
	    } // i
	    stmp.print("stmp");
	    htmp.print("htmp");
	    zquatev(htmp, etmp, vtmp);
	    for(int i=0; i<nsub; i++){
	       std::cout << "i=" << i << " eigenvalue=" << std::setprecision(8) << etmp[i] << std::endl;
	    } // i 
	 } // debug
	 //--------------------------------------------------------

         // 1. form H in the subspace: H = V^+W, V(ndim,nsub), W(ndim,nsub)
         const Tm alpha = 1.0, beta=0.0;
	 linalg::matrix<Tm> tmpH2(nsub,nsub);
	 linalg::xgemm("C","N",&nsub,&nsub,&ndim,
                       &alpha,vbas.data(),&ndim,wbas.data(),&ndim,
                       &beta,tmpH2.data(),&nsub);
	 //--------------------------------------------------------
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

	 if(debug){
	    tmpH.print("tmpH");
	    auto diffH = normF(htmp - tmpH);
	    std::cout << "diffH =" << diffH << std::endl;
	    if(diffH > thresh){
	       std::cout << "error: diffH is too large!" << std::endl;
	       exit(1);
	    }
	 }

	 //-------------------------------------------------------- 
       	 // 2. check symmetry property
         double diff = linalg::symmetric_diff(tmpH);
         if(diff > crit_skewH){
            std::cout << "error in ctns::dvdsonSolver::subspace_solver: diff_skewH=" 
                      << diff << std::endl;
            tmpH.print("tmpH");
            exit(1);
         }
	 //-------------------------------------------------------- 
         // 3. solve eigenvalue problem  
	 //--------------------------------------------------------
         // TRS-preserving diagonalization (only half eigs are output) 
	 std::vector<double> tmpE2(nsub);
	 linalg::matrix<Tm> tmpU;
	 zquatev(tmpH, tmpE, tmpU);
	 
	 //tmpE = etmp;
	 //tmpU = vtmp; 
	 
	 for(int i=0; i<nsub2; i++){
	    tmpE2[2*i] = tmpE[i];
	    tmpE2[2*i+1] = tmpE[i];
	 }
	 tmpE = tmpE2;
	 tmpU = tmpU.reorder_rowcol(pos_new, pos_new, 1);
	 std::copy(tmpU.data(), tmpU.data()+nsub*nt, tmpV.data());
	 
	 //-------------------------------------------------------- 
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

	 if(debug){
	    for(int i=0; i<nt; i++){
	       std::vector<Tm> tmpHx(ndim), diffv(ndim);
	       HVecs(1, tmpHx.data(), &vbas[i*ndim]);
	       std::transform(tmpHx.begin(), tmpHx.end(), &wbas[i*ndim], diffv.data(),
	           	   [](const Tm& x, const Tm& y){ return x - y; });
	       auto diff = linalg::xnrm2(ndim,diffv.data());
	       std::cout << "### i=" << i 
	                 << " |vi|=" << linalg::xnrm2(ndim,&vbas[i*ndim])
	                 << " |wi|=" << linalg::xnrm2(ndim,&wbas[i*ndim])
	                 << " |hvi|=" << linalg::xnrm2(ndim,tmpHx.data())
	                 << " diff(hvi-wi)=" << diff
	                 << std::endl;
	    } // ieig
	 }
         
	 // rbas = HX[i]-e[i]*X[i]
	 std::vector<Tm> krvec(ndim);
         for(int i=0; i<nt/2; i++){
	    int i0 = 2*i;
	    int i1 = 2*i+1;
            //-------------------------------------------	    
	    // sigma[o] 
	    get_krvec(&wbas[i1*ndim], krvec.data(), wf, 0);
            std::transform(&wbas[i0*ndim], &wbas[i0*ndim]+ndim, krvec.begin(), &rbas[i0*ndim],
			   [](const Tm& x, const Tm& y){ return x + y; });
            std::transform(&rbas[i0*ndim], &rbas[i0*ndim]+ndim, &vbas[i0*ndim], &rbas[i0*ndim],
                           [i0,&tmpE](const Tm& w, const Tm& x){ return w-x*tmpE[i0]; }); 
	    // sigma[o_bar] 
            get_krvec(&wbas[i0*ndim], krvec.data(), wf);
            std::transform(&wbas[i1*ndim], &wbas[i1*ndim]+ndim, krvec.begin(), &rbas[i1*ndim], 
			   [](const Tm& x, const Tm& y){ return x + y; });
            std::transform(&rbas[i1*ndim], &rbas[i1*ndim]+ndim, &vbas[i1*ndim], &rbas[i1*ndim],
                           [i1,&tmpE](const Tm& w, const Tm& x){ return w-x*tmpE[i1]; }); 
            //-------------------------------------------	    
	    if(debug){
	       get_krvec(&rbas[i0*ndim], krvec.data(), wf);
	       std::transform(&rbas[i1*ndim],&rbas[i1*ndim]+ndim,krvec.data(),krvec.data(),
	           	      [](const Tm& x, const Tm& y){ return x-y; });
	       auto diffr = linalg::xnrm2(ndim,krvec.data());
	       std::cout << "i=" << i << " : e=" << tmpE[i0] 
	                 << " |r[o]|=" << linalg::xnrm2(ndim, &rbas[i0*ndim]) 
	                 << " |r[o_bar]|=" << linalg::xnrm2(ndim, &rbas[i1*ndim])
			 << " diff(R)=" << diffr 
	                 << std::endl;
	       if(diffr > thresh){
	          std::cout << "error: too large diffr!" << std::endl;
	          exit(1);
	       }
	    }
         } // i
      }
 
      // Davidson iterative algorithm for Hv=ve 
      void solve_iter2(double* es, Tm* vs, Tm* vguess, qtensor3<Tm>& wf){
	 std::cout << "ctns::dvdsonSolver::solve_iter is_complex=" << tools::is_complex<Tm>() << std::endl;
         if(neig > ndim){
            std::cout << "error in dvdson: neig>ndim, neig/ndim=" << neig << "," << ndim << std::endl; 
            exit(1);
         }
         // clear counter
         nmvp = 0;
         auto t0 = tools::get_time();

         // generate initial subspace - vbas
         int nl = std::min(ndim,neig+nbuff); // maximal subspace size
	 std::vector<Tm> vbas(ndim*nl), wbas(ndim*nl);
	  
	 std::copy(vguess, vguess+ndim*neig, vbas.data());
	 linalg::check_orthogonality(ndim, neig, vbas);
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
            subspace_solver2(ndim,nsub,neig,vbas,wbas,tmpE,tmpV,rbas,wf);
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
                              [i,&tmpE,&damp](const Tm& r, const double& d){ return r/(abs(d-tmpE[i])+damp); });
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
            int nindp = get_ortho_basis(ndim,neig,nres,vbas,rbas,wf,crit_indp);

	    std::cout << "nindp=" << nindp << std::endl;

            if(nindp == 0){
	       std::cout << "Convergence failure: unable to generate new direction: nindp=0!" << std::endl;
               exit(1);
            }else{
               // expand V and W
               nindp = std::min(nindp,nbuff);
	       std::copy(&rbas[0],&rbas[0]+ndim*nindp,&vbas[ndim*neig]);
               HVecs(nindp, &wbas[ndim*neig], &vbas[ndim*neig]);
               nsub = neig+nindp;
               linalg::check_orthogonality(ndim,nsub,vbas);
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
      // Kramers projection
      std::function<void(Tm*)> Proj;
      // settings
      int iprt = 1;
      double crit_e = 1.e-12; // not used actually
      double crit_indp = 1.e-12;
      double crit_skewH = 1.e-8;
      double damping = 1.e-1;
      int nbuff = 4; // maximal additional vectors
      int nmvp = 0;
};

} // ctns

#endif
