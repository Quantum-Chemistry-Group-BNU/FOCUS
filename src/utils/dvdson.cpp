#include "dvdson.h"
#include "matrix.h"
#include "linalg.h"
#include "tools.h"
#include <memory>

using namespace std;
using namespace linalg;

// just for debug 
void dvdsonSolver::full_diag(double* es, double* vs){
   cout << "\ndvdsonSolver:full_diag" << endl;
   matrix id = identity_matrix(ndim);
   matrix H(ndim,ndim);
   HVecs(ndim,H.data(),id.data());
   cout << "symmetric_diff=" << symmetric_diff(H) << endl;
   // eigenvalue problem
   vector<double> e(ndim);
   matrix V(H);
   eigen_solver(V,e);
   cout << "eigenvalues:\n" << setprecision(10);
   for(int i=0; i<ndim; i++){
      cout << "i=" << i << " e=" << e[i] << endl;
   }
   // copy results
   copy(e.begin(), e.begin()+neig, es);
   copy(V.data(), V.data()+ndim*neig, vs);
}

// Davidson iterative algorithm for Hv=ve
void dvdsonSolver::solve_iter(double* es, double* vs, double* vguess){
   cout << "\ndvdsonSolver::solve_iter" << endl;
   if(neig > ndim){
      cout << "error in dvdson: neig>ndim, neig/ndim=" << neig << "," << ndim << endl; 
      exit(1);
   }
   nmvp = 0;
   // generate initial subspace - vbas
   vector<double> vbas(ndim*neig);
   if(vguess != nullptr){
      copy(vguess, vguess+ndim*neig, vbas.begin());
   }else{
      auto index = tools::sort_index(ndim, Diag, 1);
      for(int i=0; i<neig; i++){
	 vbas[i*ndim+index[i]] = 1.0;
      }
   }
   check_orthogonality(ndim,neig,vbas.data());
   exit(1);  
/*
   vector<double> wbas(neig*ndim);
   HVecs(neig, wbas.data(), vbas.data());
   // Begin to solve
   int nsub = neig;
   int nl = min(ndim,neig+nbuff);
   vector<double> tmpH(nl*nl),tmpV(nl*nl),tmpE(nl*nl);
   vector<double> eigs(neig,1.e3);
   for(int niter=1; niter<maxcycle; niter++){
      subspace_solver(nsub,ndim,vbas,wbas,tmpH,tmpV,tmpE);
   }
*/
}

/*
   // vbas 
   bool ifconv = false;
   for(int niter=1; niter<maxcycle; niter++){
      // check eigenvalue convergence
      int nconv1 = 0;
      vector<bool> econv(neig,false);
      for(int i=0; i<neig; i++){
         if(abs(teigs[i]-eigs[i]) < crit_e){
	    econv[i] = true;
	    nconv1 += 1;
	 }
	 eigs[i] = teigs[i]; 
      }
      // full residuals: Res[i]=Res'[i]-w[i]*X[i]
      matrix rbas(neig,ndim);
      dgemm("N","N",1.0,vr,vbas,0.0,rbas);
      for(int i=0; i<neig; i++){
	 rbas.row_scale(i, eigs[i]);
      }
      dgemm("N","N",1.0,vr,wbas,-1.0,rbas);
      // check eigenvector convergence
      int nconv2 = 0;
      vector<bool> rconv(neig,false);
      for(int i=0; i<neig; i++){
	 auto tmp = dnrm2(ndim, rbas.row(i));
	 if(tmp < crit_vec){
	    nconv2 += 1;
	    rconv[i] = true;
         }
      }
      int nconv = count(rconv.cbegin(), rconv.cend(), true);
      if(nconv == neig || niter == maxcycle-1){
	 for_each(eigs.begin(), eigs.end(), [cnst](int& x) {x+cnst;});
         dgemm("N","N",1.0,vr,vbas,0.0,rbas);
         break;
      }
      // subspace size too large, then reduce the basis to span{x[k],x[k]-x[k-1]}
      if(nsub > nfac*neig){
       
         dgemm("N","N",1.0,vr,vbas,0.0,rbas);
      }
      // if not converged
      for(int i=0; i<neig; i++){
	 if(rconv[i]) continue;
         // ri = ri/(abs(D-ei)+lshift)
	 transform(rbas.row(i), rbas.row(i)+ndim, Diag.begin(), rbas.row(i),
		   [eigs[i],lshift](const double& x, const double& y){ 
		   return x/(abs(y-eigs[i])+lshift); });
      }
      // orthogonalization and add to the previous ones
      ndinp,vbas2 = dvdson_ortho(vbas,rbas[rindx,:],crit_indp);
      if(iprt > 0) cout << "final nindp=" << nindp << endl;
      if(nindp != 0){
	 matrix wbas2(nindp,ndim);
   	 HVecs(wbas2, vbas2);
	 // stack together
	 copy(vbas2.data(), vbas2.data()+vbas2.size(), back_inserter(vbas));
         copy(wbas2.data(), wbas2.data()+wbas2.size(), back_inserter(wbas));
      }
   } // niter 
   exit(1);
   return 0;
}
*/ 
/*
   int nbas = vbas.rows();
   int ndim = vbas.cols();
   double norm = dnrm2(ndim, vbas.row(0));
   assert(norm > cindp);
   vbas.row_scale(0,1.0/norm);
   if(nbas > 1){
*/
/*
// raw implementation of modified Gram-Schmidt orthogonalization
void dvdsonSolver::gen_ortho_basis(matrix& rbas, 
				   const matrix& vbas,
			  	   const double cindp){
   const int maxtimes = 2;
   int nv = vbas.rows();
   int nr = rbas.rows();
   int ndim = vbas.cols();
   // Clean projection (1-V*V^+)*R => ((V*Vh)*R)^t = Rt*Vht*Vt 
   vector<double> r_vh(nr*nv);
   for(int repeat=0; repeat<maxtimes; repeat++){
      ::dgemm_("N","T",&nr,&nv,&ndim,
	       1.0,rbas.data(),&nr,vbas.data(),&nv,
	       0.0,r_vh.get(),nr);
      ::dgemm_("N","N",&nr,&ndim,&nv,
	       -1.0,r_vh.get(),&nr,vbas.data(),&nv,
	       1.0,rbas.data(),&nr);
   };
   // Form new basis from rbas
   vector<double> r_vh2;
   vector<double> vbas2;
   int nindp = 0;
   for(int i=0; i<nr; i++){
      // normalize
      double rii = dnrm2(ndim, rbas.row(i));
      if(rii < cindp) continue;
      rbas.scale_row(i, 1.0/rii);
      rii = dnrm2(ndim, rbas.row(i)); // repeat
      rbas.scale_row(i, 1.0/rii);
      copy(rbas.row(i), rbas.row(i)+ndim, back_inserter(vbas2));
      nindp +=1;
      // orthogonalization againt vbas
      int M = nr-i;
      r_vh2.resize(M*nindp);
      for(int repeat=0; repeat<maxtimes; repeat++){
	 // rbas[i:,:] -= reduce(numpy.dot,(rbas[i:,:],vbas.T.conj(),vbas))
         ::dgemm_("N","T",&M,&nv,&ndim,1.0,
		  rbas.row(i+1),&M,vbas.data(),&nv,
		  0.0,r_vh.get(),&M);
 	 ::dgemm_("N","N",&M,&ndim,&nv,-1.0,
		  r_vh1.data(),&M,vbas.data(),&nv,
		  1.0,rbas.row(i+1),&M);
	 // rbas[i:,:] -= reduce(numpy.dot,(rbas[i:,:],vbas2[:nindp,:].T.conj(),vbas2[:nindp,:]))
	 ::dgemm_("N","T",&M,&nindp,&ndim,1.0,
	          rbas.row(i+1),&M,vbas2.get(),&nindp,
		  0.0,r_vh2.get(),&M);
	 ::dgemm_("N","N",&M,&ndim,&nindp,-1.0,
	          r_vh2.get(),&M,vbas2.get(),&nindp,
		  1.0,rbas.row(i+1),&M);
      } // repeat
   } // i
}
*/
