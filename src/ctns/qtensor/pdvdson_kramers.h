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
               const std::string ifconverge = "-+";
               if(iter == 1){
                  if(iprt > 0){
                     int nmax = std::min(ndim,neig+nbuff); // maximal subspace size
                     std::cout << std::defaultfloat; 
                     std::cout << "settings: ndim=" << ndim 
                        << " neig=" << neig
                        << " maxcycle=" << maxcycle 
                        << " nbuff=" << nbuff  
                        << " memory=" << tools::sizeGB<Tm>(ndim*(3*nmax)) << "GB"
                        << std::endl; 
                     std::cout << "          damping=" << damping << std::scientific 
                        << " crit_v=" << crit_v 
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
               if(rank == 0){
                  t_cal += tcal;
                  t_comm += tcomm;
               }
            }

            // initialization
            void init_guess(const std::vector<QTm>& psi, 
                  std::vector<Tm>& v0){
               if(iprt > 0) std::cout << "ctns::pdvdsonSolver_kr::init_guess parity=" << parity << std::endl;
               assert(psi.size() == neig && psi[0].size() == ndim);
               v0.resize(ndim*neig*2);
               int nindp = 0;
               if(parity == 0){
                  // even-electron case
                  const std::complex<double> iunit(0.0,1.0);
                  for(int i=0; i<neig; i++){
                     auto psiK = psi[i].K();
                     auto tmp1 = (psi[i] + psiK);
                     auto tmp2 = (psi[i] - psiK)*iunit; 
                     tmp1.to_array(&v0[ndim*(i)]); // put all plus combination before
                     tmp2.to_array(&v0[ndim*(i+neig)]);
                     if(iprt > 0){
                        std::cout << " iguess=" << i 
                           << " |psi|=" << psi[i].normF()
                           << " |psiK|=" << psiK.normF()
                           << " |psi+psiK|=" << tmp1.normF() 
                           << " |i(psi-psiK)|=" << tmp2.normF() 
                           << std::endl;
                     }
                  } // i
                  nindp = linalg::get_ortho_basis(ndim, neig*2, v0.data()); 
               }else{
                  // odd-electron case: needs to first generate Kramers paired basis
                  for(int i=0; i<neig; i++){
                     psi[i].to_array(&v0[ndim*(2*i)]);
                     psi[i].K().to_array(&v0[ndim*(2*i+1)]);
                     if(iprt > 0){
                        std::cout << " iguess=" << i 
                           << " |psi(K)|=" << psi[i].normF()
                           << std::endl;
                     }
                  } // i
                  nindp = kramers::get_ortho_basis_qt(ndim, neig*2, v0, *pwf);
               }
               if(iprt > 0) std::cout << " neig,nindp=" << neig << "," << nindp << std::endl;
               assert(nindp >= neig);
               v0.resize(ndim*nindp);
            }

            //-------------------------------------
            // Case 0: even-electron Hilbert space 
            //-------------------------------------
            int subspace_solver_even(const int ndim, 
                  const int nsub,
                  const int neig,
                  const int naux,
                  const std::vector<bool>& rconv,
                  std::vector<Tm>& vbas,
                  std::vector<Tm>& wbas,
                  std::vector<double>& tmpE,
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
               //---------------------------------------------------------------------------------
               // 4. Rotated basis to minimal subspace that can give the exact [neig] eigenvalues
               //    Also, the difference vector = xold - xnew as corrections (or simply xold) 
               //---------------------------------------------------------------------------------
               auto delU = linalg::identity_matrix<Tm>(nsub);
               int nres = 0;
               for(int i=0; i<neig; i++){
                  if(rconv[i]) continue;
                  linalg::xcopy(nsub,delU.col(i),delU.col(nres));
                  nres++;
               }
               int nindp = linalg::get_ortho_basis(nsub,neig,nres,tmpU.data(),delU.data(),crit_indp);
               if(nindp >= naux) nindp = 2;
               if(nindp > 0) linalg::xcopy(nsub*nindp, delU.data(), tmpU.col(neig));
               int nsub1 = neig + nindp;
               assert(nsub1 <= nsub);
               //---------------------------------------------------------------------------------
               // 4. form full residuals: Res[i]=HX[i]-e[i]*X[i]
               // vbas = X[i]
               linalg::xcopy(ndim*nsub, vbas.data(), rbas.data()); 
               linalg::xgemm("N","N",&ndim,&nsub1,&nsub,
                     &alpha,rbas.data(),&ndim,tmpU.data(),&nsub,
                     &beta,vbas.data(),&ndim);
               // wbas = HX[i]
               linalg::xcopy(ndim*nsub, wbas.data(), rbas.data()); 
               linalg::xgemm("N","N",&ndim,&nsub1,&nsub,
                     &alpha,rbas.data(),&ndim,tmpU.data(),&nsub,
                     &beta,wbas.data(),&ndim);
               // rbas = HX[i]-e[i]*X[i]
               linalg::xcopy(ndim*neig, wbas.data(), rbas.data()); 
               for(int i=0; i<neig; i++){
                  linalg::xaxpy(ndim, -tmpE[i], &vbas[i*ndim], &rbas[i*ndim]); 
               }
               return nindp;
            }

            //-------------------------------------
            // Case 1: odd-electron Hilbert space 
            //-------------------------------------
            int subspace_solver_odd(const int ndim, 
                  const int nsub,
                  const int neig,
                  const int naux,
                  const std::vector<bool>& rconv,
                  std::vector<Tm>& vbas,
                  std::vector<Tm>& wbas,
                  std::vector<double>& tmpE,
                  std::vector<Tm>& rbas){
               // 1. form H in the subspace: H = V^+W, V(ndim,nsub), W(ndim,nsub)
               const Tm alpha = 1.0, beta=0.0;
               linalg::matrix<Tm> tmpH2(nsub,nsub);
               linalg::xgemm("C","N",&nsub,&nsub,&ndim,
                     &alpha,vbas.data(),&ndim,wbas.data(),&ndim,
                     &beta,tmpH2.data(),&nsub);
               //-----------------------------------------------------------
               // 2. construct full Hamiltonian from skeleton sigma vector
               //    convert ordering of basis from abab to aabb (pow_new).
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
               std::vector<double> tmpe(nsub);
               linalg::matrix<Tm> tmpU; // U(aabb,aabb) for zquatev
               kramers::zquatev(tmpH, tmpe, tmpU);
               for(int i=0; i<nsub2; i++){
                  tmpE[2*i] = tmpe[i];
                  tmpE[2*i+1] = tmpe[i];
               }
               // convert ordering of basis from aabb back to abab 
               tmpU = tmpU.reorder_rowcol(pos_new, pos_new, 1); // U(abab,abab)
               //---------------------------------------------------------------------------------
               // 4. Rotated basis to minimal subspace that can give the exact [neig] eigenvalues
               //    Also, the difference vector = xold - xnew as corrections (or simply xold) 
               //---------------------------------------------------------------------------------
               auto delU = linalg::identity_matrix<Tm>(nsub);
               int nres = 0;
               for(int i=0; i<neig; i++){
                  if(rconv[i]) continue;
                  linalg::xcopy(nsub,delU.col(i),delU.col(nres));
                  nres++;
               }
               tmpU = tmpU.reorder_row(pos_new, 0); // U(aabb,abab)
               delU = delU.reorder_row(pos_new, 0);
               std::vector<double> phases(nsub2,1.0);
               int nindp = kramers::get_ortho_basis_odd(nsub,neig,nres,tmpU,delU,phases,crit_indp);
               tmpU = tmpU.reorder_row(pos_new, 1); // U(abab,abab)
               delU = delU.reorder_row(pos_new, 1);
               if(nindp >= naux) nindp = 2;
               if(nindp > 0) linalg::xcopy(nsub*nindp, delU.data(), tmpU.col(neig));
               int nsub1 = neig + nindp;
               assert(nsub1 <= nsub);
               // 4. form full residuals: Res[i]=HX[i]-e[i]*X[i]
               // vbas = X[i]
               linalg::xcopy(ndim*nsub, vbas.data(), rbas.data()); 
               linalg::xgemm("N","N",&ndim,&nsub1,&nsub,
                     &alpha,rbas.data(),&ndim,tmpU.data(),&nsub,
                     &beta,vbas.data(),&ndim);
               // wbas = HX[i]
               linalg::xcopy(ndim*nsub, wbas.data(), rbas.data()); 
               linalg::xgemm("N","N",&ndim,&nsub1,&nsub,
                     &alpha,rbas.data(),&ndim,tmpU.data(),&nsub,
                     &beta,wbas.data(),&ndim);
               // rbas = HX[i]-e[i]*X[i]
               linalg::xcopy(ndim*neig, wbas.data(), rbas.data());
               for(int i=0; i<neig/2; i++){
                  int i0 = 2*i;
                  int i1 = 2*i+1;
                  //-------------------------------------------------------------------------
                  // We need to first contruct full sigma vector from skeleton one using TRS
                  //-------------------------------------------------------------------------
                  // sigma[o] = (H+KHKi)xo - e*xo = Hxo + K(Hxob)Ki - e*xo
                  kramers::get_krvec_qt(&wbas[i1*ndim], &rbas[i0*ndim], *pwf, 0); // K(Hxob)Ki
                  linalg::xaxpy(ndim, 1.0, &wbas[i0*ndim], &rbas[i0*ndim]); // +Hxo 
                  linalg::xaxpy(ndim, -tmpE[i0], &vbas[i0*ndim], &rbas[i0*ndim]); // -e*xo 
                  // sigma[o_bar] = (H+KHKi)xob - e*xob = Hxob + K(-Hxo)Ki - e*xob
                  kramers::get_krvec_qt(&wbas[i0*ndim], &rbas[i1*ndim], *pwf); // K(-Hxo)Ki
                  linalg::xaxpy(ndim, 1.0, &wbas[i1*ndim], &rbas[i1*ndim]); // +Hxob
                  linalg::xaxpy(ndim, -tmpE[i1], &vbas[i1*ndim], &rbas[i1*ndim]); // -e*xob
                  //-------------------------------------------------------------------------
               } // i
               return nindp;
            }

            // Precondition of a residual
            void precondition(const Tm* rvec, Tm* tvec, const double& ei){
               for(int j=0; j<ndim; j++){
                  tvec[j] = rvec[j]/(std::abs(Diag[j]-ei)+damping);
               }
            }

            // Davidson iterative algorithm for Hv=ve
            void solve_iter(double* es, Tm* vs, Tm* vguess){
               auto ti = tools::get_time();
               int size = 1, rank = 0;
#ifndef SERIAL
               size = world.size();
               rank = world.rank();
#endif
               if(rank == 0 && iprt > 0){
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

               // 0. initialization
               int nmax = std::min(ndim,neig+nbuff); // maximal subspace size
               int naux = nmax-neig;
               std::vector<Tm> vbas(ndim*nmax), wbas(ndim*nmax), rbas(ndim*nmax);
               std::vector<double> tmpE(nmax);
               std::vector<bool> rconv(neig);
               linalg::matrix<double> eigs(neig,maxcycle+1,1.e3), rnorm(neig,maxcycle+1); // history
               nmvp = 0;

               // 1. generate initial subspace - vbas 
               if(rank == 0){
                  linalg::xcopy(ndim*neig, vguess, vbas.data()); // copying neig states from vguess
                  linalg::check_orthogonality(ndim, neig, vbas);
               }
#ifndef SERIAL
               if(size > 1) boost::mpi::broadcast(world, vbas, 0);
#endif
               HVecs(neig, wbas.data(), vbas.data());

               // 2. begin to solve
               bool ifconv = false;
               int nsub = (rank==0)? neig : 0;
               int nindp = 0;
               for(int iter=1; iter<maxcycle+1; iter++){

                  // rank-0: solve the subspace problem
                  if(rank == 0){
                     //------------------------------------------------------------------------
                     // solve subspace problem and form full residuals: Res[i]=HX[i]-w[i]*X[i]
                     //------------------------------------------------------------------------
                     if(parity == 0){
                        nindp = subspace_solver_even(ndim,nsub,neig,naux,rconv,vbas,wbas,tmpE,rbas);
                     }else{
                        nindp = subspace_solver_odd(ndim,nsub,neig,naux,rconv,vbas,wbas,tmpE,rbas);
                     }
                     //------------------------------------------------------------------------
                     // compute norm of residual
                     for(int i=0; i<neig; i++){
                        double norm = linalg::xnrm2(ndim, &rbas[i*ndim]);
                        eigs(i,iter) = tmpE[i];
                        rnorm(i,iter) = norm;
                        rconv[i] = (norm < crit_v)? true : false;
                     }
                     auto t1 = tools::get_time();
                     if(iprt >= 0) print_iter(iter,nsub,eigs,rnorm,tools::get_duration(t1-ti));
                     nsub = neig+nindp;
                     ifconv = (count(rconv.begin(), rconv.end(), true) == neig);
                  }

#ifndef SERIAL
                  if(size > 1) boost::mpi::broadcast(world, ifconv, 0);
#endif	  
                  // if converged, return (es,vs) 
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

                  // if not converged, improve the subspace by adding preconditioned residues
                  if(rank == 0){
                     int nres = 0;
                     for(int i=0; i<neig; i++){
                        if(rconv[i]) continue;
                        precondition(&rbas[i*ndim],&rbas[nres*ndim],tmpE[i]);
                        //------------------------------------------------------------------
                        // Kramers projection to ensure K|psi>=|psi> for even electron case
                        //------------------------------------------------------------------
                        if(parity == 0){
                           pwf->from_array( &rbas[nres*ndim] );
                           *pwf += pwf->K();
                           pwf->to_array( &rbas[nres*ndim] );
                        }
                        //------------------------------------------------------------------
                        nres += 1;
                     }
                     //------------------------------------------------------------------
                     // re-orthogonalization and get nindp for different cases of parity
                     //------------------------------------------------------------------
                     if(parity == 0){
                        nindp = linalg::get_ortho_basis(ndim,nsub,nres,vbas.data(),rbas.data(),crit_indp);
                     }else{
                        nindp = kramers::get_ortho_basis_qt(ndim,nsub,nres,vbas,rbas,*pwf,crit_indp);
                     }
                     //------------------------------------------------------------------
                     nindp = std::min(nindp, nmax-nsub);
                  }

#ifndef SERIAL
                  if(size > 1) boost::mpi::broadcast(world, nindp, 0);	    
#endif
                  if(nindp == 0){
                     std::cout << "Convergence failure: unable to generate new direction: nindp=0!" << std::endl;
                     exit(1);
                  }else{
#ifndef SERIAL
                     if(size > 1) boost::mpi::broadcast(world, &rbas[0], ndim*nindp, 0);
#endif	      
                     linalg::xcopy(ndim*nindp, &rbas[0], &vbas[ndim*nsub]);
                     HVecs(nindp, &wbas[ndim*nsub], &vbas[ndim*nsub]);
                     if(rank == 0){
                        nsub += nindp; // expand the subspace
                        linalg::check_orthogonality(ndim,nsub,vbas);
                     }
                  }

               } // iter
               if(rank == 0){
                  if(!ifconv) std::cout << "convergence failure: out of maxcycle=" << maxcycle << std::endl;
                  auto tf = tools::get_time();    
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
            int maxcycle = 30;
            int nbuff = 4; // maximal additional vectors
            //--------------------
            // Kramers projection
            //--------------------
            int parity = 0;
            QTm* pwf; 
            //--------------------
            // settings
            int iprt = 0;
            double crit_indp = 1.e-12;
            double crit_skewH = 1.e-8;
            double damping = 1.e-12;
#ifndef SERIAL
            boost::mpi::communicator world;
#endif
            // statistics
            int nmvp = 0;
            double t_tot = 0.0;
            double t_cal = 0.0; // Hx
            double t_comm = 0.0; // reduce
            double t_rest = 0.0; // solver
      };

} // ctns

#endif
