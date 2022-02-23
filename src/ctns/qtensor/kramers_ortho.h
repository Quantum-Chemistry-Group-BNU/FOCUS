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

//
// Even-electron case: suppose |v> = |D>A + |Df>B + |D0>C
// then K|v> = |Df>sA* + |D>sB* + |D0>C* 
//           = |D>sB* + |Df>sA* + |D0>C*
// using K|D>=|Df>s and K|Df>=|D>s
//
template <typename Tm> 
void get_krvec_even(Tm* y, Tm* ykr, const std::vector<double>& phases, const int ndim0){
   int ndim1 = phases.size();
   for(int i=0; i<ndim1; i++){
      ykr[i] = phases[i]*tools::conjugate(y[ndim1+i]);
   } 
   for(int i=0; i<ndim1; i++){
      ykr[ndim1+i] = phases[i]*tools::conjugate(y[i]);
   }
   for(int i=0; i<ndim0; i++){
      ykr[2*ndim1+i] = tools::conjugate(y[2*ndim1+i]);
   }
}

//
// Odd-electron case: suppose |v> = |D>A + |Df>B 
// then K|v> = |Df>sA* + |D>(-sB*)
//           = |D>(-sB*) + |Df>sA*
// using K|D>=|Df>s and K|Df>=|D>(-s)
//
template <typename Tm> 
void get_krvec_odd(Tm* y, Tm* ykr, const std::vector<double>& phases){
   int ndim2 = phases.size();
   for(int i=0; i<ndim2; i++){
      ykr[i] = -phases[i]*tools::conjugate(y[ndim2+i]);
   } 
   for(int i=0; i<ndim2; i++){
      ykr[ndim2+i] = phases[i]*tools::conjugate(y[i]);
   }
}

// Even-electron case: MGS for rbas of size rbas(ndim,nres)
template <typename Tm>
int get_ortho_basis_even(linalg::matrix<Tm>& rbas, 
		         const int nres,
		         const std::vector<double>& phases,
		         const double crit_indp=crit_indp_kr){
   const Tm one = 1.0, mone = -1.0, zero = 0.0;
   const double isqrt2 = 1.0/std::sqrt(2.0);
   const std::complex<double> iunit(0.0,1.0);
   const int maxtimes = 2;
   if(debug_ortho) std::cout << "kramers::get_ortho_basis_even crit_indp=" << crit_indp << std::endl;
   // form new basis from rbas by modified Gram-Schmidt procedure
   int ndim = rbas.rows(), ndim1 = phases.size(), ndim0 = ndim-ndim1*2;
   // Form {1/sqrt2(|b>+K|b>), i/sqrt2(|b>-K|b>)} from {|b>}
   linalg::matrix<Tm> rbas_new(ndim,2*nres);
   std::vector<Tm> vec(ndim), krvec(ndim);
   for(int i=0; i<nres; i++){
      std::transform(rbas.col(i), rbas.col(i)+ndim, vec.begin(),
		     [isqrt2](const Tm& x){ return x*isqrt2; });
      get_krvec_even(vec.data(), krvec.data(), phases, ndim0);
      // |+> = (|v>+K|v>)
      std::transform(vec.begin(), vec.begin()+ndim, krvec.begin(), rbas_new.col(2*i),
		     [](const Tm& x, const Tm& y){ return x+y; }); 
      // |-> = i(|v>-K|v>)
      std::transform(vec.begin(), vec.begin()+ndim, krvec.begin(), rbas_new.col(2*i+1),
		     [iunit](const Tm& x, const Tm& y){ return (x-y)*iunit; }); 
   }
   // Orthonormalization
   int nindp = get_ortho_basis(rbas_new, 2*nres, crit_indp);
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
   const Tm one = 1.0, mone = -1.0, zero = 0.0;
   const int maxtimes = 2;
   if(debug_ortho) std::cout << "kramers::get_ortho_basis_odd crit_indp=" << crit_indp << std::endl;
   // form new basis from rbas by modified Gram-Schmidt procedure
   int ndim = rbas.rows();
   std::vector<Tm> krvec(ndim);
   std::vector<Tm> rbas_new;
   int nindp = 0;
   for(int i=0; i<nres; i++){
      double rii = linalg::xnrm2(ndim, rbas.col(i)); // normalization constant
      if(debug_ortho) std::cout << " i=" << i << " rii=" << rii << std::endl;
      if(rii < crit_indp) continue;
      // normalized |r[i]> 
      for(int repeat=0; repeat<maxtimes; repeat++){
	 std::transform(rbas.col(i), rbas.col(i)+ndim, rbas.col(i),
		        [rii](const Tm& x){ return x/rii; });
         rii = linalg::xnrm2(ndim, rbas.col(i));
      }
      //-------------------------------------------------------------
      rbas_new.resize(ndim*(nindp+2));
      // copy
      linalg::xcopy(ndim, rbas.col(i), &rbas_new[nindp*ndim]);
      nindp += 1;
      // add its time-reversal partner
      get_krvec_odd(rbas.col(i), krvec.data(), phases);
      linalg::xcopy(ndim, krvec.data(), &rbas_new[nindp*ndim]);
      nindp += 1;
      //-------------------------------------------------------------
      // project out |r[i]>-component from other basis
      int N = nres-1-i;
      if(N == 0) break;
      std::vector<Tm> rtr(nindp*N);
      // R_rest = (1-Rnew*Rnew^+)*R_rest
      for(int repeat=0; repeat<maxtimes; repeat++){
	 // rtr = Rnew^+*R_rest
	 linalg::xgemm("C","N",&nindp,&N,&ndim,
               	       &one,&rbas_new[0],&ndim,rbas.col(i+1),&ndim,
               	       &zero,rtr.data(),&nindp);
	 // R_rest -= Rnew*rtr
	 linalg::xgemm("N","N",&ndim,&N,&nindp,
                       &mone,&rbas_new[0],&ndim,rtr.data(),&nindp,
                       &one,rbas.col(i+1),&ndim);
      } // repeat
   } // i
   if(debug_ortho) std::cout << "final nindp=" << nindp << std::endl;
   assert(nindp%2 == 0);
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
   assert(nindp > 0);
   return nindp;
}

//
// QTensor-related, used in pdvdson_kramers.h
//
template <typename Tm, typename QTm> 
void get_krvec_qt(Tm* y, Tm* ykr, QTm& wf, const int parity=1){
   wf.from_array(y);
   wf.K(parity).to_array(ykr);
}

// Odd case: 
template <typename Tm, typename QTm>
int get_ortho_basis_qt(const int ndim,
  	      	       const int neig,
  		       const int nres,
  		       const std::vector<Tm>& vbas,
  		       std::vector<Tm>& rbas,
		       QTm& wf,
  		       const double crit_indp=1.e-12){
   const Tm one = 1.0, mone = -1.0, zero = 0.0;
   const int maxtimes = 2;
   // projection (1-V*V^+)*R = R-V*(V^+R)
   std::vector<Tm> vtr(neig*nres);
   for(int repeat=0; repeat<maxtimes; repeat++){
      linalg::xgemm("C","N",&neig,&nres,&ndim,
	            &one,vbas.data(),&ndim,rbas.data(),&ndim,
	            &zero,vtr.data(),&neig);
      linalg::xgemm("N","N",&ndim,&nres,&neig,
	            &mone,vbas.data(),&ndim,vtr.data(),&neig,
	            &one,rbas.data(),&ndim);
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
	 std::transform(&rbas[i*ndim], &rbas[i*ndim]+ndim, &rbas[i*ndim],
		        [rii](const Tm& x){ return x/rii; });
         rii = linalg::xnrm2(ndim, &rbas[i*ndim]);
      }
      //-------------------------------------------------------------
      rbas_new.resize(ndim*(nindp+2));
      // copy
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
   // Orthonormality is essential
   linalg::check_orthogonality(ndim, nindp, rbas);
   return nindp;
}

// MGS for rbas of size rbas(ndim,nres)
template <typename Tm, typename QTm>
int get_ortho_basis_qt(const int ndim,
		       const int nres,
		       std::vector<Tm>& rbas,
		       QTm& wf,
		       const double crit_indp=1.e-12){
   const Tm one = 1.0, mone = -1.0, zero = 0.0;
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
	 std::transform(&rbas[i*ndim], &rbas[i*ndim]+ndim, &rbas[i*ndim],
		        [rii](const Tm& x){ return x/rii; });
         rii = linalg::xnrm2(ndim, &rbas[i*ndim]);
      }
      //-------------------------------------------------------------
      rbas_new.resize(ndim*(nindp+2));
      // copy
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
   // Orthonormality is essential
   linalg::check_orthogonality(ndim, nindp, rbas);
   return nindp;
}

} // kramers

#endif
