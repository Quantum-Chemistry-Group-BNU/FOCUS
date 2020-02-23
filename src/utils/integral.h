#ifndef INTEGRAL_H
#define INTEGRAL_H

#include <vector>
#include <string>
#include "../settings/global.h"

namespace integral{

struct one_body{
   public:
      void init_space(){
         data.clear();
	 data.resize(sorb*(sorb+1)/2,zero);
      };

      // note that using sizeof(oneInt) only return 40,
      // since unlike array, vector only contains 3 pointers (3*8=24).
      inline double get_mem_space(){
         return global::mem_size(data.size());
      };

      // non-relativistic h1e[i,j]=[i|h|j] where i<=j
      inline double& operator()(const int i, const int j){
	 if(i%2 != j%2) return zero; // h[A,B]=0
         int I = i/2, J = j/2;
	 int IJ = std::max(I,J)*(std::max(I,J)+1)/2 + std::min(I,J);
	 return data[IJ];
      };
   public:
      double zero = 0.0;
      int sorb;
      std::vector<double> data;
};

struct two_body{
   public:
      void init_space(){
         data.clear();
	 int npair = sorb*(sorb+1)/2;     
	 data.resize(npair*(npair+1)/2,zero);
      };

      inline double get_mem_space(){
         return global::mem_size(data.size());
      };

      // non-relativistic h2e[i,j,k,l]=[ij|kl] where i<=j,k<=l,(ij)<=(kl)
      inline double& operator()(const int i, const int j, const int k, const int l){
	 if(i%2 != j%2) return zero;
	 if(k%2 != l%2) return zero;
         int I = i/2, J = j/2, K = k/2, L = l/2;
	 int IJ = std::max(I,J)*(std::max(I,J)+1)/2 + std::min(I,J);
	 int KL = std::max(K,L)*(std::max(K,L)+1)/2 + std::min(K,L);
	 int A = std::max(IJ,KL), B = std::min(IJ,KL);
	 return data[A*(A+1)/2+B];
      }
   public:
      double zero = 0.0;
      int sorb;
      std::vector<double> data;
};

void read_integral(std::string fcidump,
  		   two_body& int2e,
  		   one_body& int1e,
  		   double& ecore);

}

#endif
