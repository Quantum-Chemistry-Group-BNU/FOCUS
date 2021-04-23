#ifndef CTNS_KRAMERS_H
#define CTNS_KRAMERS_H

#include "../../extlibs/zquatev/zquatev.h"
#include "../core/matrix.h"

namespace ctns{

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
void zquatev(const linalg::matrix<std::complex<double>>& A, 
	     std::vector<double>& e,
	     linalg::matrix<std::complex<double>>& U,
	     const int order=0){
   U = (order == 0)? A : -A;
   int n2 = A.rows(); 
   int nld2 = n2;
   int info = ts::zquatev(n2, U.data(), nld2, e.data());
   if(order == 1){ std::transform(e.begin(),e.end(),e.begin(),[](const double& x){ return -x; }); }
   if(info){
      std::cout << "zquatev failed" << std::endl;
      exit(1);
   }
}

// A(l,r) = B(bar{l},bar{r})^*
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

// mapping from original PRODUCT basis to kramers paired basis
// V[odd] = {|le,ro>,|lo,re>}
inline void mapping2krbasis_odd(const qsym& qr,
				const qbond& qs1,
		                const qbond& qs2,
		                const qdpt& dpt,
		                std::vector<int>& pos_new,
				std::vector<double>& phases){
   std::vector<int> pos_up, pos_dw;
   int ioff = 0;
   const auto& comb = dpt.at(qr);
   for(int i=0; i<comb.size(); i++){
      int b1 = std::get<0>(comb[i]);
      int b2 = std::get<1>(comb[i]);
      int ioff = std::get<2>(comb[i]);
      auto q1 = qs1.get_sym(b1);
      auto q2 = qs2.get_sym(b2);
      int  d1 = qs1.get_dim(b1);
      int  d2 = qs2.get_dim(b2);
      // |le,ro> 
      if(q1.parity() == 0 && q2.parity() == 1){
         assert(d2%2 == 0);
         for(int i2=0; i2<d2/2; i2++){
            for(int i1=0; i1<d1; i1++){
               int idxA = ioff + i2*d1 + i1; // |le,ro>
               pos_up.push_back(idxA);
               int idxB = ioff + (i2+d2/2)*d1 + i1; // |le,ro_bar>
               pos_dw.push_back(idxB);
            }
         }
      // |lo,re>   
      }else if(q1.parity() == 1 && q2.parity() == 0){
         assert(d1%2 == 0);
         for(int i2=0; i2<d2; i2++){
            for(int i1=0; i1<d1/2; i1++){
   	       int idxA = ioff + i2*d1 + i1; 
               pos_up.push_back(idxA);
     	       int idxB = ioff + i2*d1 + (i1+d1/2);
               pos_dw.push_back(idxB);
            }
         }
      }else{
         std::cout << "error: no such combination of parities!" << std::endl;
         std::cout << "q1p,q2p=" << q1.parity() << "," << q2.parity() << std::endl;
         exit(1);
      }
      ioff += d1*d2;
   }
   assert(pos_up.size() == pos_dw.size());
   pos_new.clear();
   pos_new.insert(pos_new.end(), pos_up.begin(), pos_up.end());
   pos_new.insert(pos_new.end(), pos_dw.begin(), pos_dw.end());
   phases.resize(pos_dw.size(),1.0);
}

// V[even] = {|le,re>,|lo,ro>}
inline void mapping2krbasis_even(const qsym& qr,
			         const qbond& qs1,
			         const qbond& qs2,
			         const qdpt& dpt,
		                 std::vector<int>& pos_new,
				 std::vector<double>& phases){
   std::vector<int> pos_up, pos_dw, pos_ee;
   int ioff = 0;
   const auto& comb = dpt.at(qr);
   for(int i=0; i<comb.size(); i++){
      int b1 = std::get<0>(comb[i]);
      int b2 = std::get<1>(comb[i]);
      int ioff = std::get<2>(comb[i]);
      auto q1 = qs1.get_sym(b1);
      auto q2 = qs2.get_sym(b2);
      int  d1 = qs1.get_dim(b1);
      int  d2 = qs2.get_dim(b2);
      // |le,re> 
      if(q1.parity() == 0 && q2.parity() == 0){
         for(int i2=0; i2<d2; i2++){
            for(int i1=0; i1<d1; i1++){
               int idx = ioff + i2*d1 + i1;
               pos_ee.push_back(idx);
            }
         }
      // |lo,ro> = {|lo,ro>,|lo_bar,ro>} + {|lo_bar,ro_bar>,|lo,ro_bar>}
      }else if(q1.parity() == 1 && q2.parity() == 1){
         assert(d1%2 == 0 & d2%2 == 0);
         for(int i2=0; i2<d2/2; i2++){
            for(int i1=0; i1<d1/2; i1++){
               int idxA = ioff + i2*d1 + i1; // |lo,ro> 
               pos_up.push_back(idxA);
               int idxB = ioff + (i2+d2/2)*d1 + (i1+d1/2); // |lo_bar,ro_bar>
     	       pos_dw.push_back(idxB);
               phases.push_back(1.0);
	    }
            for(int i1=0; i1<d1/2; i1++){
               int idxA = ioff + i2*d1 + (i1+d1/2); // |lo_bar,ro> 
               pos_up.push_back(idxA);
               int idxB = ioff + (i2+d2/2)*d1 + i1; // |lo,ro_bar>
     	       pos_dw.push_back(idxB);
               phases.push_back(-1.0);
            }
         }
      }else{
         std::cout << "error: no such combination of parities!" << std::endl;
         std::cout << "q1p,q2p=" << q1.parity() << "," << q2.parity() << std::endl;
         exit(1);
      }
      ioff += d1*d2;
   }
   assert(pos_up.size() == pos_dw.size());
   pos_new.clear();
   pos_new.insert(pos_new.end(), pos_up.begin(), pos_up.end());
   pos_new.insert(pos_new.end(), pos_dw.begin(), pos_dw.end());
   pos_new.insert(pos_new.end(), pos_ee.begin(), pos_ee.end());
}

// mapping product basis to kramers paired basis 
inline void mapping2krbasis(const qsym& qr,
		            const qbond& qs1,
		            const qbond& qs2,
		            const qdpt& dpt,
		            std::vector<int>& pos_new,
		            std::vector<double>& phases){
   if(qr.parity() == 1){
      mapping2krbasis_odd(qr,qs1,qs2,dpt,pos_new,phases);
   }else{
      mapping2krbasis_even(qr,qs1,qs2,dpt,pos_new,phases);
   }
}

// Odd-electron subspace V[odd]=span{|D>,|Df>}
// phase: from original bare basis {|D>,|Df>} to TR basis {|D>,|Dbar>}, |Dbar>=|Df>*phase
template <typename Tm>
void eig_solver_kr_odd(const linalg::matrix<Tm>& rhor,
		       std::vector<double>& eigs,
		       linalg::matrix<Tm>& U,
		       std::vector<double>& phases){
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

// Even-electron case:
// from original basis {|D>,|Df>,|D0>} to {|D>,|Dbar>,|D0>} to TR basis {|->,|+>,|0>}
//  |-> = i(|D> - |Dbar>)/sqrt2 = i(|D> - s|Df>)/sqrt2
//  |+> =  (|D> + |Dbar>)/sqrt2 =  (|D> + s|Df>)/sqrt2
template <typename Tm>
void eig_solver_kr_even(const linalg::matrix<Tm>& rhor,
		        std::vector<double>& eigs,
		        linalg::matrix<Tm>& U,
		        std::vector<double>& phases){
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
   // Kramers projection
   auto A = 0.5*(rmat(0,0) + rmat(1,1).conj());
   auto B = 0.5*(rmat(0,1) + rmat(1,0).conj());
   auto C = 0.5*(rmat(0,2) + rmat(1,2).conj());
   auto E = 0.5*(rmat(2,2) + rmat(2,2).conj());
   // real matrix representation in {|->,|+>,|0>}
   //  [   (a-b)r   (a+b)i   sqrt2*ci ]
   //  [  -(a-b)i   (a+b)r   sqrt2*cr ] 
   //  [ sqrt2*ciT sqrt2*crT     e    ]
   auto ApB = A+B;
   auto AmB = A-B;
   const double sqrt2 = sqrt(2.0), invsqrt2 = 1.0/sqrt2;
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
   linalg::matrix<double> rho = matr.to_matrix();
   linalg::matrix<double> Ur;
   linalg::eig_solver(rho,eigs,Ur,1);
   // back to determinant basis {|D>,|Df>,|D0>} from {|->,|+>,|0>}
   // [   i     1    0  ]       [ u[-] ]   [    u[+]+i*u[-]  ]
   // [-s*i   s*1    0  ]/sqrt2 [ u[+] ] = [ s*(u[+]-i*u[-]) ]/sqrt2
   // [   0     0  sqrt2]       [ u[0] ]   [   sqrt2*u[0]    ]
   // where the sign comes from |Dbar>=|Df>*s
   blockMatrix<double> matu(partition,{dim});
   matu = Ur;
   // back transformation to original basis
   blockMatrix<std::complex<double>> umat(partition,{dim});
   const std::complex<double> iunit(0.0,1.0);
   umat(0,0) = (matu(1,0) + iunit*matu(0,0))*invsqrt2;
   umat(1,0) = umat(0,0).conj();
   umat(1,0).rowscale(phases);
   umat(2,0) = matu(2,0).as_complex();
   U = umat.to_matrix();
}

template <typename Tm>
void eig_solver_kr(const qsym& qr,
		   const linalg::matrix<Tm>& rhor,
		   std::vector<double>& eigs,
		   linalg::matrix<Tm>& U,
		   std::vector<double>& phases){
   assert(tools::is_complex<Tm>());
   if(qr.parity() == 1){
      eig_solver_kr_odd(rhor,eigs,U,phases); 
   }else{
      eig_solver_kr_even(rhor,eigs,U,phases); 
   }
}

} // ctns

#endif
