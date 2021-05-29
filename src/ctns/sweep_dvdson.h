#ifndef SWEEP_DVDSON_KR_H
#define SWEEP_DVDSON_KR_H

#include "../core/dvdson.h"
#ifndef SERIAL
#include <boost/mpi.hpp>
#endif

namespace ctns{

const bool debug_ortho = false;
extern const bool debug_ortho;

template <typename Tm, typename QTm> 
void get_krvec(Tm* y, Tm* ykr, QTm& wf, const int parity=1){
   wf.from_array(y);
   wf.K(parity).to_array(ykr);
}

// MGS orthogonalization of rbas(ndim,nres) against vbas(ndim,neig)
template <typename Tm, typename QTm>
int kr_get_ortho_basis(const int ndim,
  	      	       const int neig,
  		       const int nres,
  		       const std::vector<Tm>& vbas,
  		       std::vector<Tm>& rbas,
		       QTm& wf,
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
   std::vector<Tm> krvec(ndim);
   std::vector<Tm> rbas_new;
   int nindp = 0;
   for(int i=0; i<nres; i++){
      double rii = linalg::xnrm2(ndim, &rbas[i*ndim]); // normalization constant
      if(debug_ortho) std::cout << "\ni=" << i << " rii=" << rii << std::endl;
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
      if(debug_ortho){
         linalg::matrix<Tm> V(ndim,neig+nindp);
	 std::copy(vbas.begin(), vbas.begin()+ndim*neig, V.col(0));
	 std::copy(rbas_new.begin(), rbas_new.begin()+ndim*nindp, V.col(neig));
         auto ova = xgemm("C","N",V,V);
         ova.print("ova");
	 auto dev = ova - linalg::identity_matrix<Tm>(neig+nindp);
	 double diff = normF(dev);
	 std::cout << "diff=" << diff << std::endl;
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
template <typename Tm, typename QTm>
int kr_get_ortho_basis(const int ndim,
		       const int nres,
		       std::vector<Tm>& rbas,
		       QTm& wf,
		       const double crit_indp=1.e-12){
   const Tm one = 1.0, mone = -1.0, zero = 0.0;
   const int maxtimes = 2;
   // 2. form new basis from rbas by modified Gram-Schmidt procedure
   std::vector<Tm> krvec(ndim);
   std::vector<Tm> rbas_new;
   int nindp = 0;
   for(int i=0; i<nres; i++){
      double rii = linalg::xnrm2(ndim, &rbas[i*ndim]); // normalization constant
      if(debug_ortho) std::cout << "\ni=" << i << " rii=" << rii << std::endl;
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
      if(debug_ortho){
         // check psi[lr] = psi_bar[l_bar,r_bar]*
         std::vector<Tm> tmp(ndim);
         get_krvec(&rbas_new[(nindp-1)*ndim], tmp.data(), wf, 0);
         std::transform(tmp.begin(), tmp.end(), &rbas[i*ndim], krvec.begin(),
                        [](const Tm& x, const Tm& y){ return x-y; }); 
	 auto diff = linalg::xnrm2(ndim, krvec.data());
         std::cout << "diff[psi]=" << diff << std::endl;
	 if(diff > 1.e-10) tools::exit("error: in consistent psi and psi_bar!");
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
template <typename Tm, typename QTm>
struct dvdsonSolver_kr{
   public:
      // simple constructor
      dvdsonSolver_kr(const int _ndim, const int _neig, const double _crit_v, const int _maxcycle, 
		      const int _parity, QTm& _wf){
         ndim = _ndim;
	 neig = _neig;
	 crit_v = _crit_v;
	 maxcycle = _maxcycle;
	 parity = _parity;
	 wf = _wf;
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
	 const std::string line(87,'-');
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
            std::cout << line << std::endl;
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
	 std::cout << line << std::endl;
      } 
      // perform H*x for a set of input vectors: x(nstate,ndim)
      void HVecs(const int nstate, Tm* y, const Tm* x){
         int size = 1, rank = 0;
#ifndef SERIAL
         size = world.size();
	 rank = world.rank();
	 world.barrier();
#endif
         auto t0 = tools::get_time();
         for(int istate=0; istate<nstate; istate++){
            HVec(y+istate*ndim, x+istate*ndim); // y = tilde{H}*x
#ifndef SERIAL
	    if(size > 1){
	       std::vector<Tm> y_sum(ndim);
	       boost::mpi::reduce(world, y+istate*ndim, ndim, y_sum.data(), std::plus<Tm>(), 0);
	       std::copy(y_sum.begin(), y_sum.end(), y+istate*ndim);
	    }
#endif
            //-------------------------------------------------------------
            // Even-electron case: get full sigma vector from skeleton one
            //-------------------------------------------------------------
	    if(parity == 0){
	       wf.from_array( y+istate*ndim );
	       wf += wf.K();
	       wf.to_array( y+istate*ndim );
	    }
            //-------------------------------------------------------------
         } // istate
         nmvp += nstate;
         auto t1 = tools::get_time();
	 /*
         if(rank == 0){
            auto dt = tools::get_duration(t1-t0);
	    std::cout << "timing for HVecs : " << std::setprecision(2)  
                      << dt << " s" 
                      << " for nstate = " << nstate
		      << " tav = " << dt/nstate << " s" 
		      << " size = " << size << std::endl;
         }
	 */
      }
      // initialization
      void init_guess(std::vector<QTm>& psi, std::vector<Tm>& v0){
	 std::cout << "ctns::dvdsonSolver_kr::init_guess parity=" << parity << std::endl;
	 assert(psi.size() == neig);
         assert(psi[0].get_dim() == ndim);
         v0.resize(ndim*neig*2);
	 int nindp = 0;
         if(parity == 0){
            // even-electron case
            const std::complex<double> iunit(0.0,1.0);
            for(int i=0; i<neig; i++){
               auto tmp1 = (psi[i] + psi[i].K());
               auto tmp2 = (psi[i] - psi[i].K())*iunit; 
               tmp1.to_array(&v0[ndim*(i)]); // put all plus combination before
               tmp2.to_array(&v0[ndim*(i+neig)]);
               std::cout << " iguess=" << i 
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
            nindp = kr_get_ortho_basis(ndim, neig*2, v0, wf); // reorthogonalization
         }
         std::cout << " neig,nindp=" << neig << "," << nindp << std::endl;
         assert(nindp >= neig);
	 v0.resize(ndim*nindp);
      }
      //-------------------------------------
      // Case 0: even-electron Hilbert space 
      //-------------------------------------
      void subspace_solver0(const int ndim, 
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
	    std::string msg = "error in ctns::dvdsonSolver_kr::subspace_solver0!";
	    tools::exit(msg+" diff_skewH="+std::to_string(diff));
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
      //-------------------------------------
      // Case 1: odd-electron Hilbert space 
      //-------------------------------------
      void subspace_solver1(const int ndim, 
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
         double diff = linalg::symmetric_diff(tmpH);
         if(diff > crit_skewH){
            tmpH.print("tmpH");
	    std::string msg = "error in ctns::dvdsonSolver_kr::subspace_solver1!";
	    tools::exit(msg+" diff_skewH="+std::to_string(diff)); 
         }
	 //-----------------------------------------------------------
         // 3. solve eigenvalue problem  
	 //-----------------------------------------------------------
         // TRS-preserving diagonalization (only half eigs are output) 
	 std::vector<double> tmpE2(nsub);
	 linalg::matrix<Tm> tmpU;
	 zquatev(tmpH, tmpE, tmpU);
	 for(int i=0; i<nsub2; i++){
	    tmpE2[2*i] = tmpE[i];
	    tmpE2[2*i+1] = tmpE[i];
	 }
	 tmpE = tmpE2;
	 tmpU = tmpU.reorder_rowcol(pos_new, pos_new, 1);
	 std::copy(tmpU.data(), tmpU.data()+nsub*nt, tmpV.data());
	 //-----------------------------------------------------------
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
	 std::vector<Tm> krvec(ndim);
         for(int i=0; i<nt/2; i++){
	    int i0 = 2*i;
	    int i1 = 2*i+1;
            //-------------------------------------------------------------------------
            // We need to first contruct full sigma vector from skeleton one using TRS
            //-------------------------------------------------------------------------
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
            //-------------------------------------------------------------------------
         } // i
      }
      // Davidson iterative algorithm for Hv=ve
      void solve_iter(double* es, Tm* vs, Tm* vguess){
         int size = 1, rank = 0;
#ifndef SERIAL
         size = world.size();
	 rank = world.rank();
#endif
	 if(rank == 0){
	    std::cout << "ctns::dvdsonSolver_kr::solve_iter"
	              << " is_complex=" << tools::is_complex<Tm>()
		      << " size=" << size 
	              << " parity=" << parity << std::endl;
	 }
         if(neig > ndim){
	    std::string msg = "error: neig>ndim in dvdson! neig/ndim=";
            tools::exit(msg+std::to_string(neig)+","+std::to_string(ndim));
	 }
         // clear counter
         nmvp = 0;
         auto t0 = tools::get_time();
         
	 // 1. generate initial subspace - vbas 
         int nl = std::min(ndim,neig+nbuff); // maximal subspace size
	 std::vector<Tm> vbas(ndim*nl), wbas(ndim*nl);
	 if(rank == 0){
	    std::copy(vguess, vguess+ndim*neig, vbas.data()); // copying neig states from vguess
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
                  subspace_solver0(ndim,nsub,neig,vbas,wbas,tmpE,tmpV,rbas);
	       }else{
                  subspace_solver1(ndim,nsub,neig,vbas,wbas,tmpE,tmpV,rbas);
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
               if(iprt > 0) print_iter(iter,nsub,eigs,rnorm,tools::get_duration(t1-t0));
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
	       std::copy(tmpE.data(), tmpE.data()+neig, es);
               std::copy(vbas.data(), vbas.data()+ndim*neig, vs);
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
	          //-----------------------
	          // Kramers projection
	          //-----------------------
	          if(parity == 0){
   	             wf.from_array( &tbas[nres*ndim] );
   	             wf += wf.K();
   	             wf.to_array( &tbas[nres*ndim] );
	          }
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
	       //------------------------------------------------------------------
               // re-orthogonalization and get nindp for different cases of parity
	       //------------------------------------------------------------------
	       if(parity == 0){
	          nindp = linalg::get_ortho_basis(ndim,neig,nres,vbas,rbas,crit_indp);
	       }else{
                  nindp = kr_get_ortho_basis(ndim,neig,nres,vbas,rbas,wf,crit_indp);
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
	       std::copy(&rbas[0],&rbas[0]+ndim*nindp,&vbas[ndim*neig]);
               HVecs(nindp, &wbas[ndim*neig], &vbas[ndim*neig]);
               nsub = neig+nindp;
               if(rank == 0) linalg::check_orthogonality(ndim,nsub,vbas);
            }
         } // iter
         if(rank == 0 && !ifconv){
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
      //--------------------
      // Kramers projection
      //--------------------
      int parity = 0;
      QTm wf; 
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
};

// solver
template <typename Tm>	
struct dvdsonSolver_nkr{
   public:
      // simple constructor
      dvdsonSolver_nkr(const int _ndim, const int _neig, const double _crit_v, const int _maxcycle){
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
	 const std::string line(87,'-');
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
            std::cout << line << std::endl;
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
	 std::cout << line << std::endl;
      } 
      // perform H*x for a set of input vectors: x(nstate,ndim)
      void HVecs(const int nstate, Tm* y, const Tm* x){
         int size = 1, rank = 0;
#ifndef SERIAL
         size = world.size();
	 rank = world.rank();
	 world.barrier();
#endif
         auto t0 = tools::get_time();
         for(int istate=0; istate<nstate; istate++){
            HVec(y+istate*ndim, x+istate*ndim); // y=H*x
#ifndef SERIAL
	    if(size > 1){
	       std::vector<Tm> y_sum(ndim);
	       boost::mpi::reduce(world, y+istate*ndim, ndim, y_sum.data(), std::plus<Tm>(), 0);
	       std::copy(y_sum.begin(), y_sum.end(), y+istate*ndim);
	    }
#endif
         }
         nmvp += nstate;
         auto t1 = tools::get_time();
	 /*
         if(rank == 0){
            auto dt = tools::get_duration(t1-t0);
	    std::cout << "timing for HVecs : " << std::setprecision(2)  
                      << dt << " s" 
                      << " for nstate = " << nstate 
		      << " tav = " << dt/nstate << " s" 
		      << " size = " << size << std::endl;
         }
	 */
      }
      // check by full diag
      void solve_diag(double* es, Tm* vs, const bool ifCheckDiag=false){
         int size = 1, rank = 0;
#ifndef SERIAL
         size = world.size();
	 rank = world.rank();
#endif
	 if(rank == 0) std::cout << "ctns::dvdsonSolver_nkr:solve_diag" << std::endl;
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
            auto sdiff = linalg::symmetric_diff(H);
	    std::cout << "|H-H.h|=" << sdiff << std::endl;
            if(sdiff > 1.e-5){
               (H-H.H()).print("H-H.h");
	       tools::exit("error: H is not symmetric in ctns::dvdsonSolver_nkr::solve_diag!");
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
	 } // rank-0
         auto t1 = tools::get_time();
         if(rank == 0){
	    std::cout << "timing for solve_diag : " << std::setprecision(2)  
                      << tools::get_duration(t1-t0) << " s" << std::endl;
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
            tmpH.print("tmpH");
            std::string msg = "error: ctns::dvdsonSolver_nkr::subspace_solver: diff_skewH=";
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
         int size = 1, rank = 0;
#ifndef SERIAL
         size = world.size();
	 rank = world.rank();
#endif
	 if(rank == 0){
	    std::cout << "ctns::dvdsonSolver_nkr::solve_iter"
	              << " is_complex=" << tools::is_complex<Tm>() 
		      << " size=" << size
		      << std::endl;
	 }
         if(neig > ndim){
	    std::string msg = "error: neig>ndim in dvdson! neig/ndim=";	
	    tools::exit(msg+std::to_string(neig)+","+std::to_string(ndim));
	 }
         // clear counter
         nmvp = 0;
         auto t0 = tools::get_time();

         // 1. generate initial subspace - vbas
         int nl = std::min(ndim,neig+nbuff); // maximal subspace size
	 std::vector<Tm> vbas(ndim*nl), wbas(ndim*nl);
	 if(rank == 0){
            if(vguess != nullptr){
	       std::copy(vguess, vguess+ndim*neig, vbas.data());
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
               if(iprt > 0) print_iter(iter,nsub,eigs,rnorm,tools::get_duration(t1-t0));
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
	       std::copy(tmpE.data(), tmpE.data()+neig, es);
               std::copy(vbas.data(), vbas.data()+ndim*neig, vs);
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
	          std::copy(&tbas[index[i]*ndim], &tbas[index[i]*ndim]+ndim, &rbas[i*ndim]); 
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
	       std::copy(&rbas[0],&rbas[0]+ndim*nindp,&vbas[ndim*neig]);
	       HVecs(nindp, &wbas[ndim*neig], &vbas[ndim*neig]);
               nsub = neig+nindp;
	       if(rank == 0) linalg::check_orthogonality(ndim,nsub,vbas);
            }
         } // iter
         if(rank == 0 && !ifconv){
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
#ifndef SERIAL
      boost::mpi::communicator world;
#endif
};

} // ctns

#endif
