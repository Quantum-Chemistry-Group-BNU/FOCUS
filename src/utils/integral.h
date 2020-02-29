#ifndef INTEGRAL_H
#define INTEGRAL_H

#include <vector>
#include <string>
#include "../settings/global.h"
#include "tools.h"

namespace integral{

struct one_body{
   public:
      void init_space(){
         data.clear();
	 data.resize(sorb*(sorb+1)/2,zero);
      }
      // note that using sizeof(oneInt) only return 40,
      // since unlike array, vector only contains 3 pointers (3*8=24).
      double get_mem_space(){
         return global::mem_size(data.size());
      }
      // non-relativistic h1e[i,j]=[i|h|j] where i<=j
      double operator ()(const int i, const int j) const{
	 if(i%2 != j%2) return zero; // h[A,B]=0
	 int IJ = tools::canonical_pair(i/2,j/2);
	 return data[IJ];
      }
      double& operator ()(const int i, const int j){
	 if(i%2 != j%2) return zero; // h[A,B]=0
	 int IJ = tools::canonical_pair(i/2,j/2);
	 return data[IJ];
      }
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
      double get_mem_space(){
         return global::mem_size(data.size());
      };
      // non-relativistic h2e[i,j,k,l]=[ij|kl] where i<=j,k<=l,(ij)<=(kl)
      double operator ()(const int i, const int j, const int k, const int l) const{
	 if(i%2 != j%2) return zero;
	 if(k%2 != l%2) return zero;
	 int IJ = tools::canonical_pair(i/2,j/2);
	 int KL = tools::canonical_pair(k/2,l/2);
	 return data[tools::canonical_pair(IJ,KL)];
      }
      double& operator ()(const int i, const int j, const int k, const int l){
	 if(i%2 != j%2) return zero;
	 if(k%2 != l%2) return zero;
	 int IJ = tools::canonical_pair(i/2,j/2);
	 int KL = tools::canonical_pair(k/2,l/2);
	 return data[tools::canonical_pair(IJ,KL)];
      }
   public:
      double zero = 0.0;
      int sorb;
      std::vector<double> data;
};

void read_integral(two_body& int2e, one_body& int1e, double& ecore,
		   std::string fcidump="FCIDUMP");

}

#endif
