#ifndef PLINEAR_H
#define PLINEAR_H

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
      struct plinearSolver_nkr{
         public:

            // simple constructor
            plinearSolver_nkr(const size_t _ndim, const int _neig, const double _crit_v, const int _maxcycle,
                  const int _icase){
               ndim = _ndim;
               neig = _neig;
               crit_v = _crit_v;
               maxcycle = _maxcycle;
               icase = _icase;
            }

            // iteration info
            void print_iter(const int iter,
                  double eigs,
                  double rnorm,
                  const double t){
               int nsub = iter;
               const std::string ifconverge = "-+";
               if(iter == 1){
                  if(iprt > 0){		 
                     size_t nmax = 7; // no. of vectors needed
                     std::cout << std::defaultfloat; 
                     std::cout << "settings: ndim=" << ndim 
                        << " neig=" << neig
                        << " maxcycle=" << maxcycle 
                        << " memory=" << tools::sizeGB<Tm>(ndim*nmax) << "GB"
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
                     << std::setw(1) << ifconverge[rnorm<crit_v] << " "
                     << std::setw(20) << std::setprecision(12) << std::fixed << eigs << " "
                     << std::setw(10) << std::setprecision(3) << std::scientific << eigs - eigs_old << " "
                     << std::setw(10) << std::setprecision(3) << std::scientific << rnorm << " "
                     << std::setw(4) << nsub << " " 
                     << std::setw(5) << nmvp << " "
                     << std::setw(10) << std::setprecision(3) << std::scientific << t 
                     << std::setw(10) << std::setprecision(3) << std::scientific << t/nmvp 
                     << std::setw(10) << std::setprecision(3) << std::scientific << t_cal/nmvp 
                     << std::endl;
               } // i
               eigs_old = eigs;
            }

            // perform H*x for a set of input vectors: x(nstate,ndim)
            void HVecs(const int nstate, Tm* y, const Tm* x, Tm* work, const int cases){
               int size = 1, rank = 0;
#ifndef SERIAL
               size = world.size();
               rank = world.rank();
               world.barrier(); // barrier
#endif
               double tcal = 0.0, tcomm = 0.0;
               for(int istate=0; istate<nstate; istate++){
                  auto t0 = tools::get_time();
                  if(cases == 0){
                     HVec(work, x+istate*ndim); // y=[(H-omegaR)^2+omegaI^2]*x
                     linalg::xaxpy(ndim, -omegaR, x+istate*ndim, work);
                     HVec(y+istate*ndim, work);
                     linalg::xaxpy(ndim, -omegaR, work, y+istate*ndim);
                     linalg::xaxpy(ndim, omegaI*omegaI, x+istate*ndim, y+istate*ndim);
                  }else if(cases == 1){
                     HVec(y+istate*ndim, x+istate*ndim); // y=H*x
                  }else{
                     tools::exit("error: no such cases for HVecs!");
                  }
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
            void solve_diag(double* es, Tm* vs, const bool debug_hmat=false){
               int size = 1, rank = 0;
#ifndef SERIAL
               size = world.size();
               rank = world.rank();
#endif
               if(rank == 0) std::cout << "ctns::plinearSolver_nkr:solve_diag" << std::endl;
               auto t0 = tools::get_time();
               
               linalg::matrix<Tm> id = linalg::identity_matrix<Tm>(ndim);
               linalg::matrix<Tm> H(ndim,ndim);
               std::vector<Tm> work(ndim);
               HVecs(ndim, H.data(), id.data(), work.data(), icase);
               work.clear();

               std::vector<Tm> rhs(ndim);
               linalg::xcopy(ndim, RHS, rhs.data());
              
               std::vector<double> e(1);
               std::vector<Tm> V(ndim);
               
               if(rank == 0){
              
                  // check symmetry
                  auto sdiff = H.diff_hermitian();
                  std::cout << "|H-H.h|=" << sdiff << std::endl;
                  if(sdiff > 1.e-5){
                     tools::exit("error: H is not Hermitian in ctns::plinearSolver_nkr::solve_diag!");
                  }
              
                  // solve eigenvalue problem by diagonalization
                  linalg::linear_solver(H, rhs, e, V);
                  std::cout << "eigenvalues:\n" << std::setprecision(12);
                  for(size_t i=0; i<1; i++){
                     std::cout << "i=" << i << " e=" << e[i] << std::endl;
                  }
                     
                  // ZL@2025/09/14: save for debug
                  if(debug_hmat){
                     HVecs(ndim, H.data(), id.data(), work.data(), 1); // save H
                     H.save_txt("Hmat", 12);
                     linalg::matrix<Tm> RHSmat(ndim, 1, RHS);
                     RHSmat.save_txt("RHSmat", 12);
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

            // Precondition of a residual
            void precondition(const Tm* rvec, Tm* tvec){
               auto t0 = tools::get_time();
               if(precond){
                  if(icase == 0){
#ifdef _OPENMP
                     #pragma omp parallel for schedule(static,1048576)
#endif
                     for(size_t j=0; j<ndim; j++){
                        tvec[j] = rvec[j]/(std::norm(Diag[j]-omegaR)+std::norm(omegaI));
                     }
                  }else{
                     tools::exit("error: precondition not supported yet!");
                  }
               }else{
                  linalg::xcopy(ndim, rvec, tvec);
               }
               auto t1 = tools::get_time();
               t_precond += tools::get_duration(t1-t0);
            }

            // Conjugate gradient algorithm for Ax=b 
            void solve_iter(double* es, Tm* vs, Tm* vguess=nullptr){
               auto ti = tools::get_time();
               int size = 1, rank = 0;
#ifndef SERIAL
               size = world.size();
               rank = world.rank();
#endif
               if(rank == 0 && iprt > 0){
                  std::cout << "ctns::plinearSolver_nkr::solve_iter"
                     << " ndim=" << ndim 
                     << " is_complex=" << tools::is_complex<Tm>() 
                     << " mpisize=" << size
                     << std::endl;
               }
               if(neig != 1){
                  std::string msg = "error: neig must be 1 in plinear! neig=";
                  tools::exit(msg+std::to_string(neig));
               }

               std::vector<Tm> x0(ndim), r0(ndim), z0(ndim), p0(ndim), Ap(ndim), Ax(ndim), work(ndim);
               std::vector<double> tmpE(1);

               // 1. generate initial subspace - vbas
               if(rank == 0){
                  if(vguess != nullptr){
                     auto t0x = tools::get_time();
                     linalg::xcopy(ndim, vguess, x0.data());
                     auto t1x = tools::get_time();
                     t_xcopy += tools::get_duration(t1x-t0x);
                  }else{
                     memset(x0.data(), 0, ndim*sizeof(Tm));
                  }
               }
#ifndef SERIAL
               if(!ifnccl && size > 1) mpi_wrapper::broadcast(world, x0.data(), ndim, 0);
#endif
   
               // r = b - A @ x
               HVecs(neig, Ax.data(), x0.data(), work.data(), icase); 
               std::memset(r0.data(), 0, ndim*sizeof(Tm));
               linalg::xaxpy(ndim, -1.0, Ax.data(), r0.data());
               linalg::xaxpy(ndim,  1.0, RHS, r0.data()); 
               
               precondition(r0.data(), z0.data()); // Mz0=r0
               linalg::xcopy(ndim, z0.data(), p0.data());
               Tm rtz = linalg::xdot(ndim, r0.data(), z0.data());

               // 2. begin to solve
               bool ifconv = false;
               for(int iter=1; iter<maxcycle+1; iter++){
                  // perform A*p in parallel 
                  HVecs(neig, Ap.data(), p0.data(), work.data(), icase);
                  // rank-0: solve the subspace problem
                  if(rank == 0){
                     Tm ptAp = linalg::xdot(ndim, p0.data(), Ap.data()); 
                     Tm alpha = rtz / ptAp;
                     linalg::xaxpy(ndim,  alpha, p0.data(), x0.data());
                     linalg::xaxpy(ndim,  alpha, Ap.data(), Ax.data());
                     linalg::xaxpy(ndim, -alpha, Ap.data(), r0.data());
                     double rnorm = linalg::xnrm2(ndim, r0.data());
	                  ifconv = rnorm<crit_v;
                     // result = x^t*A*x - 2*b^t*x [variational]
                     tmpE[0] = std::real(linalg::xdot(ndim, x0.data(), Ax.data()) \
                                    - 2.0*linalg::xdot(ndim, RHS, x0.data())); 
                     auto t1 = tools::get_time();
                     if(iprt >= 0) print_iter(iter,tmpE[0],rnorm,tools::get_duration(t1-ti));
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
                        mpi_wrapper::broadcast(world, x0.data(), ndim*neig, 0);
                        auto t1b = tools::get_time();
                        t_comm += tools::get_duration(t1b-t0b);
                     }
#endif
                     auto t0x = tools::get_time();
                     linalg::xcopy(neig, tmpE.data(), es);
                     linalg::xcopy(ndim*neig, x0.data(), vs);
                     auto t1x = tools::get_time();
                     t_xcopy += tools::get_duration(t1x-t0x);
                     break;
                  }

                  // if not converged yet
                  if(rank == 0){
                     precondition(r0.data(), z0.data());
                     Tm rtz_new = linalg::xdot(ndim, r0.data(), z0.data());
                     Tm beta = rtz_new / rtz;
                     // p = z + beta * p
                     linalg::xscal(ndim, beta, p0.data());
                     linalg::xaxpy(ndim, 1.0, z0.data(), p0.data());
                     rtz = rtz_new; 
                  }
#ifndef SERIAL
                  if(!ifnccl && size > 1) mpi_wrapper::broadcast(world, p0.data(), ndim, 0);
#endif	     
               } // iter
               if(rank == 0){
                  if(!ifconv) std::cout << "convergence failure: out of maxcycle=" << maxcycle << std::endl;
                  auto tf = tools::get_time();    
                  t_tot = tools::get_duration(tf-ti);
                  t_rest = t_tot - t_cal - t_comm;
                  std::cout << "TIMING FOR Davidson : " << t_tot 
                     << "  T(cal/comm/rest)=" 
                     << t_cal << "," << t_comm << "," << t_rest
                     << std::endl;
                  double t_other =  t_rest - t_precond - t_ortho - t_xcopy - t_xnrm2;
                  std::cout << "decomposed t_rest=" << t_rest << std::endl;
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
            Tm* RHS;
            double omegaR = 0.0;
            double omegaI = 0.0;
            int icase = 0;
            double eigs_old = 0.0;
            double crit_v = 1.e-5;  // used control parameter
            int maxcycle = 30;
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
            double t_precond = 0.0; // subspace
            double t_ortho = 0.0; // ortho
            double t_xcopy = 0.0; // copy
            double t_xnrm2 = 0.0; // nrm2 
      };

} // ctns

#endif
