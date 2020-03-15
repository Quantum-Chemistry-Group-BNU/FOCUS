#ifndef HAMILTONIAN_H
#define HAMILTONIAN_H

#include "onstate.h"
#include "integral.h"
#include <tuple>

namespace fock{

inline size_t pack_ph1(const int k, const int* p, const int* q){
   return p[0]+q[0]*k;
}

inline void unpack_ph1(const size_t ph, const int k, int* p, int* q){
   p[0] = ph%k;
   q[0] = ph/k; 
}

inline size_t pack_ph2(const int k, const int* p, const int* q){
   return p[0]+(q[0]+(p[1]+q[1]*k)*k)*k;
}

inline void unpack_ph2(const size_t ph, const int k, int* p, int* q){
   size_t tmp = ph;
   p[0] = tmp%k;
   tmp = tmp/k;
   q[0] = tmp%k;
   tmp = tmp/k;
   p[1] = tmp%k;
   q[1] = tmp/k;
}

double get_Hii(const onstate& state1,
	       const integral::two_body& int2e,
	       const integral::one_body& int1e);

std::pair<double,size_t> get_HijS(const onstate& state1, 
			          const onstate& state2,
	        	          const integral::two_body& int2e,
	        	          const integral::one_body& int1e);

std::pair<double,size_t> get_HijD(const onstate& state1, 
			          const onstate& state2,
	        	          const integral::two_body& int2e,
	        	          const integral::one_body& int1e);

double get_Hij(const onstate& state1, 
	       const onstate& state2,
	       const integral::two_body& int2e,
	       const integral::one_body& int1e);

}

#endif
