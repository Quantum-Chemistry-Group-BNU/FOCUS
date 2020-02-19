#ifndef INTEGRAL_H
#define INTEGRAL_H

#include <vector>
#include <string>
#include "../settings/global.h"

using namespace global;
using namespace std;

struct oneInt{
   public:
      void initSpace(){
         store.clear();
	 store.resize(norbs*(norbs+1)/2,zero);
      };

      // note that using sizeof(oneInt) only return 40,
      // since unlike array, vector only contains 3 pointers (3*8=24).
      inline double memSpace(){
         return memSize(store.size());
      };

      inline DType& operator()(int i, int j){
	 if(i%2 != j%2) return zero;
         int I = i/2, J = j/2;
	 int IJ = max(I,J)*(max(I,J)+1)/2 + min(I,J);
	 return store[IJ];
      };
   public:
      DType zero = 0.0;
      int norbs;
      vector<DType> store;
};

struct twoInt{
   public:
      void initSpace(){
         store.clear();
	 int npair = norbs*(norbs+1)/2;     
	 store.resize(npair*(npair+1)/2,zero);
      };

      inline double memSpace(){
         return memSize(store.size());
      };

      // input spin-orbital indices   
      inline DType& operator()(int i, int j, int k, int l){
	 if(i%2 != j%2) return zero;
	 if(k%2 != l%2) return zero;
         int I = i/2, J = j/2, K = k/2, L = l/2;
	 int IJ = max(I,J)*(max(I,J)+1)/2 + min(I,J);
	 int KL = max(K,L)*(max(K,L)+1)/2 + min(K,L);
	 int A = max(IJ,KL), B = min(IJ,KL);
	 return store[A*(A+1)/2+B];
      }
   public:
      DType zero = 0.0;
      int norbs;
      vector<DType> store;
};

void readIntegral(string fcidump,
		  twoInt& I2,
		  oneInt& I1,
		  double& coreE);

#endif
