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

} // ctns

#endif
