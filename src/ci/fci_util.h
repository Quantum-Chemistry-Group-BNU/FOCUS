#ifndef FCI_UTIL_H
#define FCI_UTIL_H

#include <complex>
#include "sparse_ham.h"

namespace fci{

   // compute S & H
   template <typename Tm>
      linalg::matrix<Tm> get_Smat(const fock::onspace& space,
            const linalg::matrix<Tm>& vs){
         int dim = space.size();
         int n = vs.cols();
         linalg::matrix<Tm> Smat(n,n);
         for(int j=0; j<n; j++){
            for(int i=0; i<n; i++){
               // SIJ = <I|S|J>
               Smat(i,j) = linalg::xdot(dim,vs.col(i),vs.col(j));
            }
         }
         return Smat;
      }

   // matrix-vector product using stored H
   template <typename Tm>
      void get_Hx(Tm* y,
            const Tm* x,
            const sparse_hamiltonian<Tm>& sparseH){
         // y[i] = sparseH.diag[i]*x[i]; 
         std::transform(sparseH.diag.begin(), sparseH.diag.end(), x, y,
               [](const double& d, const Tm& c){return d*c;}); 
         // y[i] = sum_j H[i,j]*x[j] 
         for(int i=0; i<sparseH.dim; i++){
            for(int jdx=0; jdx<sparseH.connect[i].size(); jdx++){
               int j = sparseH.connect[i][jdx];
               Tm Hij = sparseH.value[i][jdx];
               y[i] += Hij*x[j]; // j>i
               y[j] += tools::conjugate(Hij)*x[i]; // j<i 
            }
         }
      }

   template <typename Tm>
      linalg::matrix<Tm> get_Hmat(const fock::onspace& space,
            const linalg::matrix<Tm>& vs,
            const integral::two_body<Tm>& int2e,
            const integral::one_body<Tm>& int1e,
            const double ecore){
         const bool Htype = tools::is_complex<Tm>();
         // compute sparse_hamiltonian
         sparse_hamiltonian<Tm> sparseH;
         sparseH.get_hamiltonian(space, int2e, int1e, ecore, Htype);
         int dim = space.size();
         int n = vs.cols();
         linalg::matrix<Tm> Hmat(n,n);
         for(int j=0; j<n; j++){
            std::vector<Tm> Hx(dim,0.0);
            fci::get_Hx(Hx.data(),vs.col(j),sparseH);
            for(int i=0; i<n; i++){
               // HIJ = <I|H|J>
               Hmat(i,j) = linalg::xdot(dim,vs.col(i),Hx.data());
            }
         }
         return Hmat;
      }

   // truncate CI coefficients for later use in initializing TNS
   template <typename Tm>
      void ci_truncate(fock::onspace& space,
            linalg::matrix<Tm>& vs,
            const int maxdets,
            const bool ifortho=true){
         const bool debug = true;
         std::cout << "\nfci::ci_truncate maxdets=" << maxdets 
            << " ifortho=" << ifortho << std::endl;
         int nsub = space.size();
         int neig = vs.cols();
         int nred = std::min(nsub,maxdets);
         std::cout << " reduction from " << nsub << " to " << nred << " dets" << std::endl;
         // select important basis
         std::vector<double> cmax(nsub,0.0);
         for(int j=0; j<neig; j++){
            for(int i=0; i<nsub; i++){
               cmax[i] += std::norm(vs(i,j));
            }
         }
         auto index = tools::sort_index(cmax, 1); 
         // orthogonalization if required
         linalg::matrix<Tm> vs2(nred,neig);
         for(int j=0; j<neig; j++){
            for(int i=0; i<nred; i++){
               vs2(i,j) = vs(index[i],j);
            }
         }
         if(ifortho){
            int nindp = linalg::get_ortho_basis(nred,neig,vs2.data());
            if(nindp != neig){
               std::string msg = "error: thresh is too large for ci_truncate! nindp,neig=";
               tools::exit(msg+std::to_string(nindp)+","+std::to_string(neig));
            }
         }
         // check the quality of truncated states
         if(debug){
            for(int j=0; j<neig; j++){
               std::vector<Tm> vec(nred);
               for(int i=0; i<nred; i++){
                  vec[i] = vs(index[i],j);
               }
               // <vs2|vec> = <vs2|full vec>
               auto ova = linalg::xdot(nred, vs2.col(j), vec.data());
               std::cout << " iroot=" << j << " ova=" 
                  << std::setprecision(12) << ova << std::endl;
            }
         }
         // copy basis and coefficients
         fock::onspace space2(nred);
         for(int i=0; i<nred; i++){
            space2[i] = space[index[i]];	
         }
         // replace (space,vs) by the reduced counterpart 
         space = space2;
         vs = vs2;
      }

} // fci

#endif
