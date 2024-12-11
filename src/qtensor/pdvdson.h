#ifndef PDVDSON_H
#define PDVDSON_H

#include "../core/ortho.h"
#ifndef SERIAL
#include "../core/mpi_wrapper.h"
#endif
#ifdef GPU
#include "../gpu/gpu_env.h"
#endif

namespace ctns{

   // solver
   template <typename Tm>	
      struct pdvdsonSolver_nkr{
         public:

            // simple constructor
            pdvdsonSolver_nkr(const size_t _ndim, const int _neig, const double _crit_v, const int _maxcycle){
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
                     size_t nmax = std::min(ndim,size_t(neig+nbuff)); // maximal subspace size
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
                  std::cout << "iter   ieig        eigenvalue        ediff      rnorm   nsub  nmvp   time/s    tav/s   tav[Hx]/s" << std::endl;
               }
               for(int i=0; i<neig; i++){
                  std::cout << std::setw(5) << iter << " " 
                     << std::setw(3) << i << " "
                     << std::setw(1) << ifconverge[(rnorm(i,iter)<crit_v)] << " "
                     << std::setw(20) << std::setprecision(12) << std::fixed << eigs(i,iter) << " "
                     << std::setw(10) << std::setprecision(3) << std::scientific << eigs(i,iter)-eigs(i,iter-1) << " "
                     << std::setw(10) << std::setprecision(3) << std::scientific << rnorm(i,iter) << " "
                     << std::setw(4) << nsub << " " 
                     << std::setw(5) << nmvp << " "
                     << std::setw(10) << std::setprecision(3) << std::scientific << t 
                     << std::setw(10) << std::setprecision(3) << std::scientific << t/nmvp 
                     << std::setw(10) << std::setprecision(3) << std::scientific << t_cal/nmvp 
                     << std::endl;
               } // i
            }

            // perform H*x for a set of input vectors: x(nstate,ndim)
            void HVecs(const int nstate, Tm* y, const Tm* x){
               int size = 1, rank = 0;
#ifndef SERIAL
               size = world.size();
               rank = world.rank();
               world.barrier(); // barrier
#endif
               double tcal = 0.0, tcomm = 0.0;
               for(int istate=0; istate<nstate; istate++){
                  auto t0 = tools::get_time();
                  HVec(y+istate*ndim, x+istate*ndim); // y=H*x
                  auto t1 = tools::get_time();
                  tcal += tools::get_duration(t1-t0);
#ifndef SERIAL
                  if(!ifnccl && size > 1){
                     mpi_wrapper::reduce(world, y+istate*ndim, ndim, 0);
                  }
                  auto t2 = tools::get_time();
                  tcomm += tools::get_duration(t2-t1); 
#endif
               }
               nmvp += nstate;
               if(rank == 0){
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
               HVecs(ndim, H.data(), id.data());

               std::vector<double> e(ndim);
               linalg::matrix<Tm> V(ndim,ndim);
               
               if(rank == 0){
              
                  //H.print("Hmat");
                  // check symmetry
                  auto sdiff = H.diff_hermitian();
                  std::cout << "|H-H.h|=" << sdiff << std::endl;
                  if(sdiff > 1.e-5){
                     tools::exit("error: H is not Hermitian in ctns::pdvdsonSolver_nkr::solve_diag!");
                  }
              
                  // check consistency with diag
                  double diff = 0.0;
                  if(ifCheckDiag){
                     std::cout << "CheckDiag[master]: ndim=" << ndim << std::endl;
                     std::cout << std::setprecision(12);
                     for(int i=0; i<ndim; i++){
                        std::cout << "i=" << i 
                           << " H(i,i)=" << H(i,i) 
                           << " Diag[i]=" << Diag[i] 
                           << " diff=" << Diag[i]-H(i,i)
                           << std::endl;
                        diff += std::abs(Diag[i]-H(i,i));
                     } // i
                  }

                  // solve eigenvalue problem by diagonalization
                  linalg::eig_solver(H, e, V);
                  std::cout << "eigenvalues:\n" << std::setprecision(12);
                  for(size_t i=0; i<ndim; i++){
                     std::cout << "i=" << i << " e=" << e[i] << std::endl;
                  }
                     
                  if(ifCheckDiag){
                     if(diff>1.e-10){
                        std::cout << "error: |Diag[i]-H(i,i)| is too large! diff=" << diff << std::endl;
                        exit(1);
                     }
                     std::cout << "CheckDiag passed successfully!" << std::endl;
                  }
               } // rank-0
#ifndef SERIAL
               // broadcast results
               if(size > 1){
                  boost::mpi::broadcast(world, e.data(), neig, 0);
                  mpi_wrapper::broadcast(world, V.data(), ndim*neig, 0);
               }
#endif
               
               // copy results
               linalg::xcopy(neig, e.data(), es);
               linalg::xcopy(ndim*neig, V.data(), vs);
               auto t1 = tools::get_time();
               if(rank == 0) tools::timing("solve_diag", t0, t1);
            }

            // subspace problem
            int subspace_solver(const size_t ndim, 
                  const int nsub,
                  const int neig,
                  const int naux,
                  const std::vector<bool>& rconv,
                  Tm* vbas,
                  Tm* wbas,
                  std::vector<double>& tmpE,
                  Tm* rbas){
               auto t0 = tools::get_time();
               // 1. form H in the subspace: H = V^+W, V(ndim,nsub), W(ndim,nsub)
               const Tm alpha = 1.0, beta=0.0;
               linalg::matrix<Tm> tmpH(nsub,nsub);
               linalg::xgemm("C", "N", nsub, nsub, ndim,
                     alpha, vbas, ndim, wbas, ndim,
                     beta, tmpH.data(), nsub);
               auto t0a = tools::get_time();
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
               auto t0b = tools::get_time();
               int nindp = linalg::get_ortho_basis(nsub,neig,nres,tmpU.data(),delU.data(),crit_indp);
               if(nindp >= naux) nindp = std::min(2,naux); // ZL@20221210 for naux=0
               if(nindp > 0) linalg::xcopy(nsub*nindp, delU.data(), tmpU.col(neig));
               int nsub1 = neig + nindp;
               assert(nsub1 <= nsub);
               auto t0c = tools::get_time();
               //---------------------------------------------------------------------------------
               // 5. form full residuals: Res[i]=HX[i]-e[i]*X[i]
               // vbas = {X[i]}
               linalg::xcopy(ndim*nsub, vbas, rbas); 
               linalg::xgemm("N", "N", ndim, nsub1, nsub,
                     alpha, rbas, ndim, tmpU.data(), nsub,
                     beta, vbas, ndim);
               // wbas = {HX[i]}
               linalg::xcopy(ndim*nsub, wbas, rbas); 
               linalg::xgemm("N", "N", ndim, nsub1, nsub,
                     alpha, rbas, ndim, tmpU.data(), nsub,
                     beta, wbas, ndim);
               auto t0d = tools::get_time();
               // rbas = HX[i]-e[i]*X[i]
               linalg::xcopy(ndim*neig, wbas, rbas); 
               for(int i=0; i<neig; i++){
                  linalg::xaxpy(ndim, -tmpE[i], &vbas[i*ndim], &rbas[i*ndim]); 
               }
               auto t1 = tools::get_time();
               t_sub += tools::get_duration(t1-t0);
               std::cout << "T(sub)=" << tools::get_duration(t1-t0)
                  << " T(gemm/eig/ortho/gemm/axpy)="
                  << tools::get_duration(t0a-t0) << ","
                  << tools::get_duration(t0b-t0a) << ","
                  << tools::get_duration(t0c-t0b) << ","
                  << tools::get_duration(t0d-t0c) << ","
                  << tools::get_duration(t1-t0d)
                  << std::endl;
               return nindp;
            }

            // Precondition of a residual
            void precondition(const Tm* rvec, Tm* tvec, const double& ei){
               auto t0 = tools::get_time();
               if(precond){
#ifdef _OPENMP
                  #pragma omp parallel for schedule(static,1048576)
#endif
                  for(size_t j=0; j<ndim; j++){
                     tvec[j] = rvec[j]/(std::abs(Diag[j]-ei)+damping);
                  }
               }else{
                  linalg::xcopy(ndim, rvec, tvec);
               }
               auto t1 = tools::get_time();
               t_precond += tools::get_duration(t1-t0);
            }
	
	    bool ifconverged(const int iter, const double edel, const double norm){
	       return (norm < crit_v) and (iter==1 or (iter>1 and edel<std::max(1.e-14,1.e-2*crit_v*crit_v)));
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
                     << " ndim=" << ndim 
                     << " is_complex=" << tools::is_complex<Tm>() 
                     << " mpisize=" << size
                     << std::endl;
               }
               if(neig > ndim){
                  std::string msg = "error: neig>ndim in pdvdson! neig/ndim=";	
                  tools::exit(msg+std::to_string(neig)+","+std::to_string(ndim));
               }

               // 0. initialization
               size_t nmax = std::min(ndim,size_t(neig+nbuff)); // maximal subspace size
               int naux = nmax-neig; // additional dimension for subspace
               Tm* workspace = new Tm[ndim*nmax*3];
               Tm* vbas = workspace;
               Tm* wbas = workspace + ndim*nmax;
               Tm* rbas = workspace + ndim*nmax*2;
               std::vector<double> tmpE(nmax);
               std::vector<bool> rconv(neig);
               linalg::matrix<double> eigs(neig,maxcycle+1,1.e3), rnorm(neig,maxcycle+1); // history
               nmvp = 0;

               // 1. generate initial subspace - vbas
               if(rank == 0){
                  if(vguess != nullptr){
                     auto t0x = tools::get_time();
                     linalg::xcopy(ndim*neig, vguess, vbas);
                     auto t1x = tools::get_time();
                     t_xcopy += tools::get_duration(t1x-t0x);
                  }else{
                     auto index = tools::sort_index(ndim, Diag);
                     memset(vbas, 0, ndim*neig*sizeof(Tm)); 
                     for(int i=0; i<neig; i++){
                        vbas[i*ndim+index[i]] = 1.0;
                     }
                  }
                  if(debug) linalg::check_orthogonality(ndim, neig, vbas);
               }
#ifndef SERIAL
               if(!ifnccl && size > 1) mpi_wrapper::broadcast(world, vbas, ndim*neig, 0);
#endif
               HVecs(neig, wbas, vbas);

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
                     nindp = subspace_solver(ndim,nsub,neig,naux,rconv,vbas,wbas,tmpE,rbas);
                     //------------------------------------------------------------------------
                     // compute norm of residual
                     auto t0r = tools::get_time();
                     for(int i=0; i<neig; i++){
                        double edel = std::abs(tmpE[i]-eigs(i,iter-1));
			double norm = linalg::xnrm2(ndim, &rbas[i*ndim]);
                        eigs(i,iter) = tmpE[i];
                        rnorm(i,iter) = norm;
                        rconv[i] = this->ifconverged(iter, edel, norm);
                     }
                     auto t1 = tools::get_time();
                     t_xnrm2 += tools::get_duration(t1-t0r);
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
                     // broadcast converged results to all processors
                     if(size > 1){
                        auto t0b = tools::get_time();
                        boost::mpi::broadcast(world, tmpE.data(), neig, 0);
                        mpi_wrapper::broadcast(world, vbas, ndim*neig, 0);
                        auto t1b = tools::get_time();
                        t_comm += tools::get_duration(t1b-t0b);
                     }
#endif
                     auto t0x = tools::get_time();
                     linalg::xcopy(neig, tmpE.data(), es);
                     linalg::xcopy(ndim*neig, vbas, vs);
                     auto t1x = tools::get_time();
                     t_xcopy += tools::get_duration(t1x-t0x);
                     break;
                  }

                  // if not converged, improve the subspace by adding preconditioned residues
                  if(rank == 0){
                     int nres = 0; // no. of residuals to be added
                     for(int i=0; i<neig; i++){
                        if(rconv[i]) continue;
                        precondition(&rbas[i*ndim],&rbas[nres*ndim],tmpE[i]);
                        nres += 1;		
                     }
                     auto t0o = tools::get_time();
                     nindp = linalg::get_ortho_basis(ndim,nsub,nres,vbas,rbas,crit_indp);
                     auto t1o = tools::get_time();
                     t_ortho += tools::get_duration(t1o-t0o);
                     nindp = std::min(nindp, int(nmax-nsub));
                  }
#ifndef SERIAL
                  if(size > 1) boost::mpi::broadcast(world, nindp, 0);	    
#endif
                  if(nindp == 0){
                     std::cout << "Convergence failure: unable to generate new direction: nindp=0!" << std::endl;
                     exit(1);
                  }else{
#ifndef SERIAL
                     if(!ifnccl && size > 1) mpi_wrapper::broadcast(world, &rbas[0], ndim*nindp, 0);
#endif	     
                     auto t0x = tools::get_time(); 
                     linalg::xcopy(ndim*nindp, &rbas[0], &vbas[ndim*nsub]);
                     auto t1x = tools::get_time(); 
                     t_xcopy += tools::get_duration(t1x-t0x);
                     HVecs(nindp, &wbas[ndim*nsub], &vbas[ndim*nsub]);
                     if(rank == 0){ 
                        nsub += nindp; // expand the subspace 
                        if(debug) linalg::check_orthogonality(ndim,nsub,vbas);
                     }
                  }

               } // iter
               delete [] workspace;
               if(rank == 0){
                  if(!ifconv) std::cout << "convergence failure: out of maxcycle=" << maxcycle << std::endl;
                  auto tf = tools::get_time();    
                  t_tot = tools::get_duration(tf-ti);
                  t_rest = t_tot - t_cal - t_comm;
                  std::cout << "TIMING FOR Davidson : " << t_tot 
                     << "  T(cal/comm/rest)=" 
                     << t_cal << "," << t_comm << "," << t_rest
                     << std::endl;
                  double t_other =  t_rest - t_sub - t_precond - t_ortho - t_xcopy - t_xnrm2;
                  std::cout << "decomposed t_rest=" << t_rest << std::endl;
                  std::cout << " T(sub)    =" << t_sub     << " per=" << t_sub/t_rest*100 << std::endl;
                  std::cout << " T(precond)=" << t_precond << " per=" << t_precond/t_rest*100 << std::endl;
                  std::cout << " T(ortho)  =" << t_ortho   << " per=" << t_ortho/t_rest*100 << std::endl;
                  std::cout << " T(xcopy)  =" << t_xcopy   << " per=" << t_xcopy/t_rest*100 << std::endl;
                  std::cout << " T(xnrm2)  =" << t_xnrm2   << " per=" << t_xnrm2/t_rest*100 << std::endl;
                  std::cout << " T(other)  =" << t_other   << " per=" << t_other/t_rest*100 << std::endl;
               }
            }
         public:
            // basics
            size_t ndim = 0;
            int neig = 0;
            double* Diag;
            std::function<void(Tm*, const Tm*)> HVec;
            double crit_v = 1.e-5;  // used control parameter
            int maxcycle = 50;
            int nbuff = 4; // maximal additional vectors
            // settings
            int iprt = 0;
            double crit_indp = 1.e-12;
            double crit_skewH = 1.e-8;
            double damping = 1.e-12;
            bool precond = true;
            bool ifnccl = false;
#ifndef SERIAL
            boost::mpi::communicator world;
#endif
            // statistics
            int nmvp = 0;
            double t_tot = 0.0;
            double t_cal = 0.0; // Hx
            double t_comm = 0.0; // reduce
            double t_rest = 0.0; 
            bool debug = false;
            // decomposition of t_rest
            double t_sub = 0.0; // subspace
            double t_precond = 0.0; // subspace
            double t_ortho = 0.0; // ortho
            double t_xcopy = 0.0; // copy
            double t_xnrm2 = 0.0; // nrm2 
      };

} // ctns

#endif
