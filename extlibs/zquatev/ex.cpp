#include "zquatev.h"
#include "f77.h"
#include <iostream>

using namespace std;

/*
[[ 0.7824-0.j      0.3761-0.5175j  0.    -0.j      0.7761-0.1509j]
 [ 0.3761+0.5175j  0.6949-0.j     -0.7761+0.1509j  0.    -0.j    ]
 [-0.    -0.j     -0.7761-0.1509j  0.7824+0.j      0.3761+0.5175j]
 [ 0.7761+0.1509j -0.    -0.j      0.3761-0.5175j  0.6949+0.j    ]]
*/

int main(){

   const int n = 2;
   const int n2 = 2*n;
   unique_ptr<complex<double>[]> D(new complex<double>[n2*n2]);
   unique_ptr<double[]> E(new double[n2]);

   D[0] = 0.7824;
   D[1] = 0.3761+0.5175j;
   D[2] = 0.0;
   D[3] = 0.7761+0.1509j;

   D[4] =  0.3761-0.5175j;
   D[5] =  0.6949-0.j    ;
   D[6] = -0.7761-0.1509j;
   D[7] = -0.    -0.j    ;

   D[8] =  0.    -0.j     ; 
   D[9] = -0.7761+0.1509j  ;
   D[10] =  0.7824+0.j      ;
   D[11] =  0.3761-0.5175j ;

   D[12] = 0.7761-0.1509j;
   D[13] = 0.    -0.j    ;
   D[14] = 0.3761+0.5175j;
   D[15] = 0.6949+0.j    ;

   for(size_t i = 0; i != n2; ++i){
      for(size_t j = 0; j != n2; ++j){
  	 cout << D[j*n2+i] << " "; 
      }
      cout << endl;
   }

   const int info2 = ts::zquatev(n2, D.get(), n2, E.get());

   for(size_t i = 0; i != n2; ++i){
      cout << "i=" << E[i] << endl;
   }
   cout << endl;

   for(size_t i = 0; i != n2; ++i){
      for(size_t j = 0; j != n2; ++j){
  	 cout << D[j*n2+i] << " "; 
      }
      cout << endl;
   }

}
