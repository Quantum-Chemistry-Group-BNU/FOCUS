#ifndef KRAMERS_ORTHO_H
#define KRAMERS_ORTHO_H

/*
  Generation of Kramers-Symmetry-Adapted basis:
  0. even-electron subspace: {|b>} with K|b>=|b>
  1. odd-electron subspace: {|b>,K|b>}
*/

#include "../../core/ortho.h"

namespace kramers{

   const bool debug_ortho = false;
   extern const bool debug_ortho;

   const double crit_indp_kr = 1.e-6;
   extern const double crit_indp_kr;

   //================================
   // Functions for kramers_linalg.h 
   //================================

   //----------------------------------------------------------
   // 1. Generate |ykr> = K|y> for |y> expanded in KSA basis
   //----------------------------------------------------------

   // Even-electron case: suppose |v> = |D>A + |Df>B + |D0>C
   // then K|v> = |Df>sA* + |D>sB* + |D0>C* 
   //           = |D>sB* + |Df>sA* + |D0>C*
   // using K|D>=|Df>s and K|Df>=|D>s
   template <typename Tm> 
      void get_krvec_even(Tm* y, Tm* ykr, const std::vector<double>& phases, const size_t ndim0){
         size_t ndim1 = phases.size();
         for(size_t i=0; i<ndim1; i++){
            ykr[i] = phases[i]*tools::conjugate(y[ndim1+i]);
         } 
         for(size_t i=0; i<ndim1; i++){
            ykr[ndim1+i] = phases[i]*tools::conjugate(y[i]);
         }
         for(size_t i=0; i<ndim0; i++){
            ykr[2*ndim1+i] = tools::conjugate(y[2*ndim1+i]);
         }
      }

   // Odd-electron case: suppose |v> = |D>A + |Df>B 
   // then K|v> = |Df>sA* + |D>(-sB*)
   //           = |D>(-sB*) + |Df>sA*
   // using K|D>=|Df>s and K|Df>=|D>(-s)
   template <typename Tm> 
      void get_krvec_odd(Tm* y, Tm* ykr, const std::vector<double>& phases){
         size_t ndim2 = phases.size();
         for(size_t i=0; i<ndim2; i++){
            ykr[i] = -phases[i]*tools::conjugate(y[ndim2+i]);
         } 
         for(size_t i=0; i<ndim2; i++){
            ykr[ndim2+i] = phases[i]*tools::conjugate(y[i]);
         }
      }

   //----------------------------------------------------------
   // 2. Generate KSA basis from a given set of vectors
   //----------------------------------------------------------

   // Even-electron case: MGS for rbas of size rbas(ndim,nres)
   template <typename Tm>
      int get_ortho_basis_even(linalg::matrix<Tm>& rbas, 
            const int nres,
            const std::vector<double>& phases,
            const double crit_indp=crit_indp_kr){
         const double isqrt2 = 1.0/std::sqrt(2.0);
         const std::complex<double> iunit(0.0,1.0);
         if(debug_ortho) std::cout << "kramers::get_ortho_basis_even crit_indp=" << crit_indp << std::endl;
         // form new basis from rbas by modified Gram-Schmidt procedure
         size_t ndim = rbas.rows();
         size_t ndim1 = phases.size();
         size_t ndim0 = ndim-ndim1*2;
         // Form {1/sqrt2(|b>+K|b>), i/sqrt2(|b>-K|b>)} from {|b>}
         linalg::matrix<Tm> rbas_new(ndim,2*nres);
         std::vector<Tm> vec(ndim), krvec(ndim);
         for(int i=0; i<nres; i++){
            // |v> = |r[i]>/sqrt2
            std::transform(rbas.col(i), rbas.col(i)+ndim, vec.begin(),
                  [&isqrt2](const Tm& x){ return x*isqrt2; });
            get_krvec_even(vec.data(), krvec.data(), phases, ndim0);
            // |+> = (|v>+K|v>)
            std::transform(vec.begin(), vec.begin()+ndim, krvec.begin(), rbas_new.col(2*i),
                  [](const Tm& x, const Tm& y){ return x+y; }); 
            // |-> = i(|v>-K|v>)
            std::transform(vec.begin(), vec.begin()+ndim, krvec.begin(), rbas_new.col(2*i+1),
                  [&iunit](const Tm& x, const Tm& y){ return (x-y)*iunit; }); 
         }
         // Orthonormalization
         int nindp = linalg::get_ortho_basis(ndim, 2*nres, rbas_new.data(), crit_indp);
         if(debug_ortho) std::cout << "final nindp=" << nindp << std::endl;
         assert(nindp > 0);
         rbas.resize(ndim, nindp);
         linalg::xcopy(ndim*nindp, rbas_new.data(), rbas.data());
         // Orthonormality is essential
         linalg::check_orthogonality(rbas);
         return nindp;
      }

   // Odd-electron case: MGS for rbas of size rbas(ndim,nres)
   template <typename Tm>
      int get_ortho_basis_odd(linalg::matrix<Tm>& rbas, 
            const int nres,
            const std::vector<double>& phases,
            const double crit_indp=crit_indp_kr){
         const int maxtimes = 2;
         if(debug_ortho) std::cout << "kramers::get_ortho_basis_odd crit_indp=" << crit_indp << std::endl;
         // form new basis from rbas by modified Gram-Schmidt procedure
         size_t ndim = rbas.rows();
         std::vector<Tm> krvec(ndim);
         std::vector<Tm> rbas_new;
         int nindp = 0;
         for(int i=0; i<nres; i++){
            double rii = linalg::xnrm2(ndim, rbas.col(i)); // normalization constant
            if(debug_ortho) std::cout << " i=" << i << " rii=" << rii << std::endl;
            if(rii < crit_indp) continue;
            // normalized |r[i]> 
            for(int repeat=0; repeat<maxtimes; repeat++){
               linalg::xscal(ndim, 1.0/rii, rbas.col(i));
               rii = linalg::xnrm2(ndim, rbas.col(i));
            }
            //-------------------------------------------------------------
            rbas_new.resize(ndim*(nindp+2));
            linalg::xcopy(ndim, rbas.col(i), &rbas_new[nindp*ndim]);
            nindp += 1;
            // add its time-reversal partner
            get_krvec_odd(rbas.col(i), krvec.data(), phases);
            linalg::xcopy(ndim, krvec.data(), &rbas_new[nindp*ndim]);
            nindp += 1;
            //-------------------------------------------------------------
            // project out |r[i]>-component from other basis
            int nleft = nres-1-i;
            if(nleft == 0) break;
            for(int repeat=0; repeat<maxtimes; repeat++){
               // R_rest = (1-Rnew*Rnew^+)*R_rest
               linalg::ortho_projection(ndim,nindp,nleft,&rbas_new[0],rbas.col(i+1));
            } // repeat
         } // i
         if(debug_ortho) std::cout << "final nindp=" << nindp << std::endl;
         assert(nindp > 0 && nindp%2 == 0);
         rbas.resize(ndim, nindp);
         linalg::xcopy(ndim*nindp, rbas_new.data(), rbas.data());
         // reorder into {|b>,K|b>} structure
         std::vector<int> pos_new(nindp);
         for(int i=0; i<nindp; i++){
            pos_new[i] = i%2==0? i/2 : i/2+nindp/2;
         }
         rbas = rbas.reorder_col(pos_new, 1);
         // Orthonormality is essential
         linalg::check_orthogonality(rbas);
         return nindp;
      }

   template <typename Tm>
      int get_ortho_basis_kr(const ctns::qsym& qr,
            const std::vector<double>& phases,
            linalg::matrix<Tm>& rbas, 
            const int nres,
            const double crit_indp=crit_indp_kr){
         assert(tools::is_complex<Tm>());
         int nindp = 0;
         if(qr.parity() == 0){
            nindp = get_ortho_basis_even(rbas, nres, phases, crit_indp);
         }else{
            nindp = get_ortho_basis_odd(rbas, nres, phases, crit_indp);
         }
         return nindp;
      }

   //===================================================
   // Functions for pdvdson_kramers.h (QTensor-related) 
   //===================================================

   // Odd-electron case: MGS for rbas of size rbas(ndim,nres)
   template <typename Tm>
      int get_ortho_basis_odd(const size_t ndim,
            const int neig,
            const int nres,
            const linalg::matrix<Tm>& vbas,
            linalg::matrix<Tm>& rbas, 
            const std::vector<double>& phases,
            const double crit_indp=1.e-12){
         const int maxtimes = 2;
         // 1. projection (1-V*V^+)*R = R-V*(V^+R)
         for(int repeat=0; repeat<maxtimes; repeat++){
            linalg::ortho_projection(ndim,neig,nres,vbas.data(),rbas.data());
         }
         // 2. form new basis from rbas by modified Gram-Schmidt procedure
         std::vector<Tm> krvec(ndim);
         std::vector<Tm> rbas_new;
         int nindp = 0;
         for(int i=0; i<nres; i++){
            double rii = linalg::xnrm2(ndim, rbas.col(i)); // normalization constant
            if(rii < crit_indp) continue;
            // normalized |r[i]> 
            for(int repeat=0; repeat<maxtimes; repeat++){
               linalg::xscal(ndim, 1.0/rii, rbas.col(i));
               rii = linalg::xnrm2(ndim, rbas.col(i));
            }
            //-------------------------------------------------------------
            rbas_new.resize(ndim*(nindp+2));
            linalg::xcopy(ndim, rbas.col(i), &rbas_new[nindp*ndim]);
            nindp += 1;
            // add its time-reversal partner
            get_krvec_odd(rbas.col(i), krvec.data(), phases);
            linalg::xcopy(ndim, krvec.data(), &rbas_new[nindp*ndim]);
            nindp += 1;
            //-------------------------------------------------------------
            // project out |r[i]>-component from other basis
            int nleft = nres-1-i;
            if(nleft == 0) break;
            for(int repeat=0; repeat<maxtimes; repeat++){
               // R_rest = (1-V*V^+)*R_rest
               //linalg::ortho_projection(ndim,neig,nleft,vbas.data(),rbas.col(i+1));
               linalg::ortho_projection(ndim,neig,nleft,vbas.data(),&rbas._data[(i+1)*ndim]);
               // R_rest = (1-Rnew*Rnew^+)*R_rest
               //linalg::ortho_projection(ndim,nindp,nleft,&rbas_new[0],rbas.col(i+1));
               linalg::ortho_projection(ndim,nindp,nleft,&rbas_new[0],&rbas._data[(i+1)*ndim]);
            } // repeat
         } // i
         assert(nindp%2 == 0);
         if(nindp > 0){
            rbas.resize(ndim, nindp);
            linalg::xcopy(ndim*nindp, rbas_new.data(), rbas.data());
            // this part is not needed in subspace_solver_odd (pdvdson_kramers.h)
            /*
            // reorder into {|b>,K|b>} structure
            std::vector<int> pos_new(nindp);
            for(int i=0; i<nindp; i++){
            pos_new[i] = i%2==0? i/2 : i/2+nindp/2;
            }
            rbas = rbas.reorder_col(pos_new, 1);
            */
            // Orthonormality is essential
            linalg::check_orthogonality(rbas);
         }
         return nindp;
      }

   template <typename Tm, typename QTm> 
      void get_krvec_qt(Tm* y, Tm* ykr, QTm& wf, const int parity=1){
         wf.from_array(y);
         wf.K(parity).to_array(ykr);
      }

   //-------------------------------------------------
   // Generate KSA basis for Odd electron case, while
   // Even electron case is treated as usual.
   //-------------------------------------------------

   template <typename Tm, typename QTm>
      int get_ortho_basis_qt(const size_t ndim,
            const int neig,
            const int nres,
            const std::vector<Tm>& vbas,
            std::vector<Tm>& rbas,
            QTm& wf,
            const double crit_indp=1.e-12){
         const int maxtimes = 2;
         // projection (1-V*V^+)*R = R-V*(V^+R)
         for(int repeat=0; repeat<maxtimes; repeat++){
            linalg::ortho_projection(ndim,neig,nres,vbas.data(),rbas.data());
         }
         // form new basis from rbas by modified Gram-Schmidt procedure
         std::vector<Tm> krvec(ndim);
         std::vector<Tm> rbas_new;
         int nindp = 0;
         for(int i=0; i<nres; i++){
            double rii = linalg::xnrm2(ndim, &rbas[i*ndim]); // normalization constant
            if(debug_ortho) std::cout << " i=" << i << " rii=" << rii << std::endl;
            if(rii < crit_indp) continue;
            // normalized |r[i]> 
            for(int repeat=0; repeat<maxtimes; repeat++){
               linalg::xscal(ndim, 1.0/rii, &rbas[i*ndim]);
               rii = linalg::xnrm2(ndim, &rbas[i*ndim]);
            }
            //-------------------------------------------------------------
            rbas_new.resize(ndim*(nindp+2));
            linalg::xcopy(ndim, &rbas[i*ndim], &rbas_new[nindp*ndim]);
            nindp += 1;
            // add its time-reversal partner
            get_krvec_qt(&rbas[i*ndim], krvec.data(), wf);
            linalg::xcopy(ndim, krvec.data(), &rbas_new[nindp*ndim]);
            nindp += 1;
            // debug
            if(debug_ortho){
               linalg::matrix<Tm> V(ndim,neig+nindp);
               linalg::xcopy(ndim*neig, vbas.data(), V.col(0));
               linalg::xcopy(ndim*nindp, rbas_new.data(), V.col(neig));
               auto ova = linalg::xgemm("C","N",V,V);
               ova.print("ova");
               double diff = (ova - linalg::identity_matrix<Tm>(neig+nindp)).normF();
               std::cout << "diff=" << diff << std::endl;
            }
            //-------------------------------------------------------------
            // project out |r[i]>-component from other basis
            int nleft = nres-1-i;
            if(nleft == 0) continue;
            for(int repeat=0; repeat<maxtimes; repeat++){
               // R_rest = (1-V*V^+)*R_rest
               linalg::ortho_projection(ndim,neig,nleft,vbas.data(),&rbas[(i+1)*ndim]);
               // R_rest = (1-Rnew*Rnew^+)*R_rest
               linalg::ortho_projection(ndim,nindp,nleft,&rbas_new[0],&rbas[(i+1)*ndim]);
            } // repeat
         } // i
         assert(nindp > 0);
         rbas = rbas_new;
         // Orthonormality is essential
         linalg::check_orthogonality(ndim, nindp, rbas);
         return nindp;
      }

   template <typename Tm, typename QTm>
      int get_ortho_basis_qt(const size_t ndim,
            const int nres,
            std::vector<Tm>& rbas,
            QTm& wf,
            const double crit_indp=1.e-12){
         const int maxtimes = 2;
         // form new basis from rbas by modified Gram-Schmidt procedure
         std::vector<Tm> krvec(ndim);
         std::vector<Tm> rbas_new;
         int nindp = 0;
         for(int i=0; i<nres; i++){
            double rii = linalg::xnrm2(ndim, &rbas[i*ndim]); // normalization constant
            if(debug_ortho) std::cout << " i=" << i << " rii=" << rii << std::endl;
            if(rii < crit_indp) continue;
            // normalized |r[i]> 
            for(int repeat=0; repeat<maxtimes; repeat++){
               linalg::xscal(ndim, 1.0/rii, &rbas[i*ndim]);
               rii = linalg::xnrm2(ndim, &rbas[i*ndim]);
            }
            //-------------------------------------------------------------
            rbas_new.resize(ndim*(nindp+2));
            linalg::xcopy(ndim, &rbas[i*ndim], &rbas_new[nindp*ndim]);
            nindp += 1;
            // add its time-reversal partner
            get_krvec_qt(&rbas[i*ndim], krvec.data(), wf);
            linalg::xcopy(ndim, krvec.data(), &rbas_new[nindp*ndim]);
            nindp += 1;
            // debug
            if(debug_ortho){
               // check psi[lr] = psi_bar[l_bar,r_bar]*
               std::vector<Tm> tmp(ndim);
               get_krvec_qt(&rbas_new[(nindp-1)*ndim], tmp.data(), wf, 0);
               std::transform(tmp.begin(), tmp.end(), &rbas[i*ndim], krvec.begin(),
                     [](const Tm& x, const Tm& y){ return x-y; }); 
               auto diff = linalg::xnrm2(ndim, krvec.data());
               std::cout << "diff[psi]=" << diff << std::endl;
               if(diff > 1.e-10) tools::exit("error: in consistent psi and psi_bar!");
               // check overlap matrix for basis
               linalg::matrix<Tm> V(ndim,nindp,rbas_new.data());
               auto ova = linalg::xgemm("C","N",V,V);
               ova.print("ova");
            }
            //-------------------------------------------------------------
            // project out |r[i]>-component from other basis
            int nleft = nres-1-i;
            if(nleft == 0) break;
            for(int repeat=0; repeat<maxtimes; repeat++){
               // R_rest = (1-Rnew*Rnew^+)*R_rest
               linalg::ortho_projection(ndim,nindp,nleft,&rbas_new[0],&rbas[(i+1)*ndim]);
            } // repeat
         } // i
         assert(nindp > 0);
         rbas = rbas_new;
         // Orthonormality is essential
         linalg::check_orthogonality(ndim, nindp, rbas);
         return nindp;
      }

} // kramers

#endif
