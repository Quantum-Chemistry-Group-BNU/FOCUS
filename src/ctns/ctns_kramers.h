#ifndef CTNS_KRAMERS_H
#define CTNS_KRAMERS_H

#include "../../extlibs/zquatev/zquatev.h"
#include "../core/matrix.h"

namespace ctns{

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

} // ctns

#endif
