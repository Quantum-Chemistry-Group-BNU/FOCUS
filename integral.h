#ifndef INTEGRAL_HEADER_H
#define INTEGRAL_HEADER_H
#include <vector>
#include <string>
#include "global.h"

using namespace std;

struct oneInt{
   public:
      inline DType& operator()(int i, int j){ return store.at(i*norbs+j); }; 
   public:
      int norbs;
      std::vector<DType> store;
};

struct twoInt{
   public:
      int norbs;
      std::vector<DType> store;
};

void readIntegral(string fcidump,
		  twoInt& I2,
		  oneInt& I1,
		  double& coreE);

#endif
