#ifndef KRAMERS_LINALG_H
#define KRAMERS_LINALG_H

#include "../../../extlibs/zquatev/zquatev.h"
#include "ctns_qsym.h"
#include "kramers_basis.h"

namespace kramers{

// used in initial guess part 
template <typename Tm>
class blockMatrix{
   public:
      // pass dims by value allow temporary vector as input
      blockMatrix(const std::vector<int> dims1, const std::vector<int> dims2){
	 // row
	 nblks1 = dims1.size();
	 _dims1 = dims1;
	 _offs1.resize(nblks1+1,0);
	 for(int iblk=0; iblk<nblks1; iblk++){
	    _offs1[iblk+1] = _offs1[iblk] + _dims1[iblk];
	 }
	 // col
	 nblks2 = dims2.size();
	 _dims2 = dims2;
	 _offs2.resize(nblks2+1,0);
	 for(int iblk=0; iblk<nblks2; iblk++){
	    _offs2[iblk+1] = _offs2[iblk] + _dims2[iblk];
	 }
	 // data (initialized with zeros)
	 _blocks.resize(nblks1);
	 for(int iblk=0; iblk<nblks1; iblk++){
	    _blocks[iblk].resize(nblks2);
	    int ni = _dims1[iblk];
	    if(ni == 0) continue; // in case some dimensions are zero!
	    for(int jblk=0; jblk<nblks2; jblk++){
	       int nj = _dims2[jblk];
	       if(nj == 0) continue; // in case some dimensions are zero!
	       linalg::matrix<Tm> blk(ni,nj);
	       _blocks[iblk][jblk] = blk;
	    }
	 }
	 //print(); 
      }
      // for debug
      void print() const{
	 std::cout << "blockMatrix: rblks,cblks=" << nblks1 << "," << nblks2 << std::endl;
	 std::cout << "dims1=";
	 for(const auto d : _dims1) std::cout << " " << d;
	 std::cout << std::endl;
	 std::cout << "dims2=";
	 for(const auto d : _dims2) std::cout << " " << d;
	 std::cout << std::endl;
	 std::cout << "offs1=";
	 for(const auto d : _offs1) std::cout << " " << d;
	 std::cout << std::endl;
	 std::cout << "offs2=";
	 for(const auto d : _offs2) std::cout << " " << d;
	 std::cout << std::endl;
      }
      // copy data
      blockMatrix& operator =(const linalg::matrix<Tm>& mat){
	 for(int iblk=0; iblk<nblks1; iblk++){
	    int ni = _dims1[iblk]; 
	    if(ni == 0) continue; // in case some dimensions are zero!
	    int ioff = _offs1[iblk];
	    for(int jblk=0; jblk<nblks2; jblk++){
	       int nj = _dims2[jblk];
	       if(nj == 0) continue; // in case some dimensions are zero!
	       int joff = _offs2[jblk];
	       linalg::matrix<Tm> blk(ni,nj);
	       for(int j=0; j<nj; j++){
		  auto p = mat.col(joff+j)+ioff;
	 	  std::copy(p, p+ni, blk.col(j));
	       }
	       _blocks[iblk][jblk] = blk;
	    }
	 }
	 return *this;
      }
      // convert back 
      linalg::matrix<Tm> to_matrix() const{
	 linalg::matrix<Tm> mat(_offs1[nblks1], _offs2[nblks2]);
	 for(int iblk=0; iblk<nblks1; iblk++){
	    int ni = _dims1[iblk];
	    if(ni == 0) continue; // in case some dimensions are zero!
	    int ioff = _offs1[iblk];
	    for(int jblk=0; jblk<nblks2; jblk++){
	       int nj = _dims2[jblk];
	       if(nj == 0) continue; // in case some dimensions are zero!
	       int joff = _offs2[jblk];
	       const auto& blk = _blocks[iblk][jblk];
	       for(int j=0; j<nj; j++){
	 	  std::copy(blk.col(j), blk.col(j)+ni, mat.col(joff+j)+ioff);
	       }
	    }
	 }
	 return mat;
      }
      // access block of matrix
      const linalg::matrix<Tm> operator()(const int iblk, const int jblk) const{
	 assert(iblk>=0 && iblk<nblks1 && jblk>=0 && jblk<nblks2);
	 return _blocks[iblk][jblk];
      }
      linalg::matrix<Tm>& operator()(const int iblk, const int jblk){
	 assert(iblk>=0 && iblk<nblks1 && jblk>=0 && jblk<nblks2);
	 return _blocks[iblk][jblk];
      }
   private:
      int nblks1, nblks2; // row, column
      std::vector<int> _dims1, _dims2;
      std::vector<int> _offs1, _offs2;
      std::vector<std::vector<linalg::matrix<Tm>>> _blocks;
};

// interface to the zquatev library for diagonalizing a quaternion matrix
inline void zquatev(const linalg::matrix<std::complex<double>>& A, 
	     	    std::vector<double>& e,
	     	    linalg::matrix<std::complex<double>>& U,
	     	    const int order=0){
   U = (order == 0)? A : -A;
   int n2 = A.rows(); 
   int nld2 = n2;
   int info = ts::zquatev(n2, U.data(), nld2, e.data());
   if(order == 1){ std::transform(e.begin(),e.end(),e.begin(),[](const double& x){ return -x; }); }
   if(info) tools::exit("error: zquatev failed!");
}

// A(l,r) = B(bar{l},bar{r})^* given parity of qr and qc
template <typename Tm>
linalg::matrix<Tm> time_reversal(const linalg::matrix<Tm>& blk1,
		                 const int pr,  
				 const int pc){
   int dr = blk1.rows();
   int dc = blk1.cols();
   linalg::matrix<Tm> blk(dr,dc);
   // even-even block:
   //    <e|\bar{O}|e> = p{O} <e|O|e>^*
   if(pr == 0 && pc == 0){
      blk = blk1.conj();
   // even-odd block:
   //    <e|\bar{O}|o> = p{O} <e|O|\bar{o}>^*
   //    <e|\bar{O}|\bar{o}> = p{O} <e|O|o>^* (-1)
   //    [A,B] -> p{O}[B*,-A*]  
   }else if(pr == 0 && pc == 1){
      assert(dc%2 == 0);
      int dc2 = dc/2;
      // copy blocks <e|O|o>^*
      for(int ic=0; ic<dc2; ic++){
         std::transform(blk1.col(ic),blk1.col(ic)+dr,blk.col(ic+dc2),
          	        [](const Tm& x){ return -tools::conjugate(x); });
      }
      // copy blocks <e|O|\bar{o}>
      for(int ic=0; ic<dc2; ic++){
         std::transform(blk1.col(ic+dc2),blk1.col(ic+dc2)+dr,blk.col(ic),
         	        [](const Tm& x){ return tools::conjugate(x); });
      }
   // odd-even block:
   //    [A]        [ B*]
   //    [ ] -> p{O}[   ]
   //    [B]        [-A*] 
   }else if(pr == 1 && pc == 0){
      assert(dr%2 == 0);
      int dr2 = dr/2;
      for(int ic=0; ic<dc; ic++){
         std::transform(blk1.col(ic),blk1.col(ic)+dr2,blk.col(ic)+dr2,
         	        [](const Tm& x){ return -tools::conjugate(x); });
         std::transform(blk1.col(ic)+dr2,blk1.col(ic)+dr,blk.col(ic),
         	        [](const Tm& x){ return tools::conjugate(x); });
      }
   // odd-odd block:
   //    [A B]        [ D* -C*]
   //    [   ] -> p{O}[       ]
   //    [C D]        [-B*  A*]
   }else if(pr == 1 && pc == 1){
      assert(dr%2 == 0 && dc%2 == 0);
      int dr2 = dr/2, dc2 = dc/2;
      for(int ic=0; ic<dc2; ic++){
         std::transform(blk1.col(ic),blk1.col(ic)+dr2,blk.col(ic+dc2)+dr2,
         	        [](const Tm& x){ return tools::conjugate(x); });
         std::transform(blk1.col(ic)+dr2,blk1.col(ic)+dr,blk.col(ic+dc2),
         	        [](const Tm& x){ return -tools::conjugate(x); });
      }
      for(int ic=0; ic<dc2; ic++){
         std::transform(blk1.col(ic+dc2),blk1.col(ic+dc2)+dr2,blk.col(ic)+dr2,
         	        [](const Tm& x){ return -tools::conjugate(x); });
         std::transform(blk1.col(ic+dc2)+dr2,blk1.col(ic+dc2)+dr,blk.col(ic),
         	        [](const Tm& x){ return tools::conjugate(x); });
      }
   } // (pr,pc)
   return blk;
}

//---------------------------------------------------------
// eig_solver with Kramers-Symmetry Adapation (Projection)
//---------------------------------------------------------

// Even-electron case:
// from original basis {|D>,|Df>,|D0>} to {|D>,|Dbar>,|D0>} to TR basis {|->,|+>,|0>}
//  |-> = i(|D> - |Dbar>)/sqrt2 = i(|D> - s|Df>)/sqrt2
//  |+> =  (|D> + |Dbar>)/sqrt2 =  (|D> + s|Df>)/sqrt2
template <typename Tm>
void eig_solver_even(const linalg::matrix<Tm>& rhor,
		     const std::vector<double>& phases,
		     std::vector<double>& eigs,
		     linalg::matrix<Tm>& U){
   const bool debug = false;
   int dim = rhor.rows();
   int dim1 = phases.size();
   int dim0 = dim-2*dim1;
   assert(dim0 >= 0);
   std::vector<int> partition = {dim1,dim1,dim0};
   blockMatrix<std::complex<double>> rmat(partition,partition);
   rmat = rhor;
   // col-1 & row-1
   rmat(0,1).colscale(phases);
   rmat(1,1).colscale(phases);
   rmat(2,1).colscale(phases);
   rmat(1,0).rowscale(phases);
   rmat(1,1).rowscale(phases);
   rmat(1,2).rowscale(phases);
   //  
   // Check: time-reversal symmetric rhor in a kramers-paired basis 
   //        should have the following structure
   // 
   //  [  A   B   C  ]
   //  [  B*  A*  C* ]
   //  [  Cd  Ct  E  ]
   //
   if(debug){
      rhor.print("rhor");
      rmat(0,0).print("A");
      rmat(1,1).print("A*");
      rmat(0,1).print("B");
      rmat(1,0).print("B*");
      rmat(0,2).print("C");
      rmat(1,2).print("C*");
      rmat(2,0).print("Cd");
      rmat(2,1).print("Ct");
      rmat(2,2).print("E");
   }
   //
   // Kramers symmetrization:
   //
   auto A = 0.5*(rmat(0,0) + rmat(1,1).conj());
   auto B = 0.5*(rmat(0,1) + rmat(1,0).conj());
   auto C = 0.5*(rmat(0,2) + rmat(1,2).conj());
   auto E = 0.5*(rmat(2,2) + rmat(2,2).conj());
   //
   // real matrix representation in {|->,|+>,|0>}
   //
   //  [   (a-b)r   (a+b)i   sqrt2*ci ]
   //  [  -(a-b)i   (a+b)r   sqrt2*cr ] 
   //  [ sqrt2*ciT sqrt2*crT     e    ]
   auto ApB = A+B;
   auto AmB = A-B;
   const double sqrt2 = std::sqrt(2.0), isqrt2 = 1.0/sqrt2;
   auto Cr = sqrt2*C.real();
   auto Ci = sqrt2*C.imag();
   blockMatrix<double> matr(partition,partition);
   matr(0,0) = AmB.real();
   matr(1,0) = -AmB.imag();
   matr(2,0) = Ci.T();
   matr(0,1) = ApB.imag();
   matr(1,1) = ApB.real();
   matr(2,1) = Cr.T();
   matr(0,2) = Ci;
   matr(1,2) = Cr;
   matr(2,2) = E.real();
   // diagonalization
   linalg::matrix<double> rhor_kr = matr.to_matrix();
   linalg::matrix<double> Ur;
   linalg::eig_solver(rhor_kr,eigs,Ur,1);
   //
   // back to determinant basis {|D>,|Df>,|D0>} from {|->,|+>,|0>}
   // [   i     1    0  ]       [ u[-] ]   [    u[+]+i*u[-]  ]
   // [-s*i   s*1    0  ]/sqrt2 [ u[+] ] = [ s*(u[+]-i*u[-]) ]/sqrt2
   // [   0     0  sqrt2]       [ u[0] ]   [   sqrt2*u[0]    ]
   // where the sign comes from |Dbar>=|Df>*s
   //
   blockMatrix<double> matu(partition,{dim});
   matu = Ur;
   // back transformation to original basis
   blockMatrix<std::complex<double>> umat(partition,{dim});
   const std::complex<double> iunit(0.0,1.0);
   umat(0,0) = (matu(1,0) + iunit*matu(0,0))*isqrt2;
   umat(1,0) = umat(0,0).conj();
   umat(1,0).rowscale(phases);
   umat(2,0) = matu(2,0).as_complex();
   U = umat.to_matrix();
}

// Odd-electron subspace V[odd]=span{|D>,|Df>}
// phase: from original bare basis {|D>,|Df>} to TR basis {|D>,|Dbar>}, |Dbar>=|Df>*phase
template <typename Tm>
void eig_solver_odd(const linalg::matrix<Tm>& rhor,
		    const std::vector<double>& phases,
		    std::vector<double>& eigs,
		    linalg::matrix<Tm>& U){
   const bool debug = false;
   int dim = rhor.rows();
   int dim1 = phases.size();
   assert(dim = 2*dim1);
   std::vector<int> partition = {dim1,dim1};
   blockMatrix<Tm> rmat(partition,partition);
   rmat = rhor;
   rmat(0,1).colscale(phases);
   rmat(1,1).colscale(phases);
   rmat(1,0).rowscale(phases);
   rmat(1,1).rowscale(phases);
   //  
   // Check: time-reversal symmetric rhor in a kramers-paired basis 
   //        should have the following structure
   // 
   //  [  A   B  ]
   //  [ -B*  A* ]
   //
   if(debug){
      rhor.print("rhor");
      rmat(0,0).print("A");
      rmat(1,1).print("A*");
      rmat(0,1).print("B");
      rmat(1,0).print("B*");
   }
   //
   // Kramers symmetrization:
   //
   //  [ rho_AA rho_BA ]        [ rho_AA+rho_BB^*  rho_BA-rho_AB^* ]   [  A   B  ]
   //  [	       ] -> 1/2 [				   ] = [         ]
   //  [ rho_AB rho_BB ]        [ rho_AB-rho_BA^*  rho_BB+rho_AA^* ]   [ -B*  A* ]
   //
   auto A = 0.5*(rmat(0,0) + rmat(1,1).conj());
   auto B = 0.5*(rmat(0,1) - rmat(1,0).conj()); 
   rmat(0,0) = A;
   rmat(0,1) = B;
   rmat(1,0) = -B.conj();
   rmat(1,1) = A.conj();
   auto rhor_kr = rmat.to_matrix();
   // TRS-preserving diagonalization (only half eigs are output)
   zquatev(rhor_kr,eigs,U,1);
   std::copy(eigs.begin(), eigs.begin()+dim1, eigs.begin()+dim1); // duplicate eigs!
   // back to original basis {|D>,|Df>}
   blockMatrix<Tm> umat(partition,{dim});
   umat = U;
   umat(1,0).rowscale(phases);
   U = umat.to_matrix();
}

template <typename Tm>
void eig_solver_kr(const ctns::qsym& qr,
	           const std::vector<double>& phases,
	           const linalg::matrix<Tm>& rhor,
	           std::vector<double>& eigs,
	           linalg::matrix<Tm>& U){
   assert(tools::is_complex<Tm>());
   if(qr.parity() == 0){
      eig_solver_even(rhor,phases,eigs,U); 
   }else{
      eig_solver_odd(rhor,phases,eigs,U); 
   }
}

//---------------------------------------------------------------
// Compute right renormalized states from a set of wavefunctions
//---------------------------------------------------------------
const int svd_iop = 13;
extern const int svd_iop;

// Input: a vector of matrices {c[l,r]}
// Output: the reduced basis U[r,alpha]
template <typename Tm> 
void get_renorm_states_nkr(const std::vector<linalg::matrix<Tm>>& clr,
		           std::vector<double>& sigs2,
		           linalg::matrix<Tm>& U,
			   const double rdm_vs_svd,
			   const bool debug_basis=false){
   int nroots = clr.size();
   int diml = clr[0].rows();
   int dimr = clr[0].cols();
   if(dimr <= static_cast<int>(rdm_vs_svd*diml)){ 
      
      if(debug_basis){ 
         std::cout << " RDM-based decimation: dim(l,r)=" << diml << "," << dimr << std::endl;
      }
      linalg::matrix<Tm> rhor(dimr,dimr);
      for(int iroot=0; iroot<nroots; iroot++){
         rhor += linalg::xgemm("T","N",clr[iroot],clr[iroot].conj());
      } // iroot
      rhor *= 1.0/nroots;   
      sigs2.resize(dimr);
      linalg::eig_solver(rhor, sigs2, U, 1);

   }else{

      if(debug_basis){ 
         std::cout << " SVD-based decimation: dim(l,r)=" << diml << "," << dimr << std::endl;
      }
      linalg::matrix<Tm> vrl(dimr,diml*nroots);
      for(int iroot=0; iroot<nroots; iroot++){
         auto crl = clr[iroot].T();
         std::copy(crl.data(), crl.data()+dimr*diml, vrl.col(iroot*diml));
      } // iroot
      vrl *= 1.0/std::sqrt(nroots);
      linalg::matrix<Tm> vt; // size of sig2,U,vt will be determined inside svd_solver!
      linalg::svd_solver(vrl, sigs2, U, vt, svd_iop);
      std::transform(sigs2.begin(), sigs2.end(), sigs2.begin(),
  		     [](const double& x){ return x*x; });

   }
   if(debug_basis){ 
      std::cout << " sigs2[final]: ";
      for(const auto sig2 : sigs2) std::cout << sig2 << " ";
      std::cout << std::endl;                               
      //U.print("U"); 
   }
   linalg::check_orthogonality(U); // orthonormality is essential for variational calculations
}

template <typename Tm> 
void get_renorm_states_kr(const ctns::qsym& qr,
	                  const std::vector<double>& phases,	
      	                  const std::vector<linalg::matrix<Tm>>& clr,
		          std::vector<double>& sigs2,
		          linalg::matrix<Tm>& U,
			  const double rdm_vs_svd,
			  const bool debug_basis=false){
   const double thresh_kept = 1.e-16;
   assert(tools::is_complex<Tm>());
   int nroots = clr.size();
   int diml = clr[0].rows();
   int dimr = clr[0].cols();
   if(dimr <= static_cast<int>(rdm_vs_svd*diml)){ 
      
      if(debug_basis){ 
         std::cout << " RDM-based decimation: dim(l,r)=" << diml << "," << dimr << std::endl;
      }
      linalg::matrix<Tm> rhor(dimr,dimr);
      for(int iroot=0; iroot<nroots; iroot++){
         rhor += linalg::xgemm("T","N",clr[iroot],clr[iroot].conj());
      } // iroot
      rhor *= 1.0/nroots;   
      sigs2.resize(dimr);
      eig_solver_kr<std::complex<double>>(qr, phases, rhor, sigs2, U);

   }else{
   
      if(debug_basis){ 
         std::cout << " SVD-based decimation: dim(l,r)=" << diml << "," << dimr << std::endl;
      }
      //------------------------------------------
      // 0. Perform usual SVD to get renorm_basis
      //------------------------------------------
      linalg::matrix<Tm> vrl(dimr,diml*nroots);
      for(int iroot=0; iroot<nroots; iroot++){
         auto crl = clr[iroot].T();
         std::copy(crl.data(), crl.data()+dimr*diml, vrl.col(iroot*diml));
      } // iroot
      vrl *= 1.0/std::sqrt(nroots);
      linalg::matrix<Tm> vt; // size of sig2,U,vt will be determined inside svd_solver!
      linalg::svd_solver(vrl, sigs2, U, vt, svd_iop);
      std::transform(sigs2.begin(), sigs2.end(), sigs2.begin(),
  		     [](const double& x){ return x*x; });
      if(debug_basis){
         std::cout << " sigs2[SVD]: ";
         for(const auto sig2 : sigs2) std::cout << sig2 << " ";
         std::cout << std::endl;                              
	 //U.print("U[SVD]"); 
      }
      //------------------------------------------
      // 1. Generate KRS-adapted basis
      //------------------------------------------
      int nkept = 0;
      for(int i=0; i<sigs2.size(); i++){
         if(sigs2[i] > thresh_kept){ 
            nkept += 1;
         }else{
            break;
         }
      }
      if(debug_basis) std::cout << " nkept=" << nkept << std::endl;
      // in case no state in this symmetry sector
      if(nkept == 0){
         sigs2.resize(0);
         U.resize(dimr,0);
         return;
      }
      int nindp = get_ortho_basis_kr(qr, phases, U, nkept);
      //if(debug_basis) U.print("U[ortho]");
      //------------------------------------------
      // 2. Re-diagonalize RDM in the KRS-basis
      // rhor_proj = U^+ rho_r U
      //           = U^+ psi^T psi^* U
      //           = X^T X*    with X = psi U^*
      //------------------------------------------
      linalg::matrix<Tm> rhor_proj(nindp,nindp);
      for(int iroot=0; iroot<nroots; iroot++){
         auto clru = linalg::xgemm("N","N",clr[iroot],U.conj());
         rhor_proj += linalg::xgemm("T","N",clru,clru.conj());
      } // iroot
      rhor_proj *= 1.0/nroots;
      // re-diagonalize rhor_proj 
      std::vector<double> phases_fake;
      if(qr.parity() == 0){ 
         phases_fake.resize(0);
      }else{
	 phases_fake.resize(nindp/2, 1.0);
      }
      sigs2.resize(nindp);
      linalg::matrix<Tm> Urot;
      eig_solver_kr<std::complex<double>>(qr, phases_fake, rhor_proj, sigs2, Urot);
      //if(debug_basis) Urot.print("Urot");
      U = linalg::xgemm("N","N",U,Urot);

   }
   if(debug_basis){ 
      std::cout << " sigs2[final]: ";
      for(const auto sig2 : sigs2) std::cout << sig2 << " ";
      std::cout << std::endl;                               
      //U.print("U"); 
   }
   linalg::check_orthogonality(U);
}

} // kramers

#endif
