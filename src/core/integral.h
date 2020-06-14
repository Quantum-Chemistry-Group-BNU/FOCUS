#ifndef INTEGRAL_H
#define INTEGRAL_H

#include <cassert>
#include <vector>
#include <string>
#include <complex>
// --- load ---
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <algorithm>
#include "../settings/global.h"

namespace integral{

// simple strategy to enable both cases
inline double conjugate(const double x){ return x; }
inline std::complex<double> conjugate(const std::complex<double> x){ return std::conj(x); };
inline double realpart(const double x){ return x; }
inline double realpart(const std::complex<double> x){ return x.real(); }

template <typename Tm>
struct one_body{
   public:
      void init_mem(){
	 assert(sorb > 0);
	 data.resize(sorb*sorb);
      }
      Tm get(const size_t i, const size_t j) const{
         return data[j*sorb+i];
      }
      void set(const size_t i, const size_t j, const Tm val){
         data[j*sorb+i] = val; // column-major storage
      }
   public:
      int sorb;
   private:
      std::vector<Tm> data; // Oij = <i|O|j>	
};

template <typename Tm>
struct two_body{
   public:
      // store <ij||kl> where i>j, k>l, (ij)>(kl)
      void init_mem(){
	 assert(sorb > 0);
	 size_t p = sorb*(sorb-1)/2;
	 data.resize(p*(p+1)/2,0.0);
	 Q.resize(p);
      }
      Tm get(const size_t i, const size_t j, 
	     const size_t k, const size_t l) const{
         if((i == j) || (k==l)) return 0.0;
	 size_t ij = i>j? i*(i-1)/2+j : j*(j-1)/2+i;
	 size_t kl = k>l? k*(k-1)/2+l : l*(l-1)/2+k;
	 double sgn = 1;
	 sgn = i>j? sgn : -sgn;
	 sgn = k>l? sgn : -sgn;
	 Tm val;
	 if(ij >= kl){
            size_t ijkl = ij*(ij+1)/2+kl;
	    val = sgn*data[ijkl];
	 }else{
	    size_t ijkl = kl*(kl+1)/2+ij;
	    val = sgn*conjugate(data[ijkl]);
	 }
	 return val;
      }
      void set(const size_t i, const size_t j, 
	       const size_t k, const size_t l, 
	       const Tm val){
         if((i == j) || (k==l)) return;
	 size_t ij = i>j? i*(i-1)/2+j : j*(j-1)/2+i;
	 size_t kl = k>l? k*(k-1)/2+l : l*(l-1)/2+k;
	 double sgn = 1;
	 sgn = i>j? sgn : -sgn;
	 sgn = k>l? sgn : -sgn;
	 if(ij >= kl){
            size_t ijkl = ij*(ij+1)/2+kl;
	    data[ijkl] = sgn*val;
	 }else{
	    size_t ijkl = kl*(kl+1)/2+ij;
	    data[ijkl] = sgn*conjugate(val);
	 }
      }
      // Qij = <ij||ij> (i>j);
      void initQ(){
	 for(int i=0; i<sorb; i++){
	    for(int j=0; j<i; j++){
	       int ij = i*(i-1)/2+j;
	       Q[ij] = realpart(get(i,i,j,j) - get(i,j,j,i));
	    }
	 }
      }
      double getQ(const size_t i, const size_t j){
	 if(i == j) return 0.0;
	 double val;
	 if(i>j){
	    int ij = i*(i-1)/2+j;
	    val = Q[ij];
	 }else{
            int ji = j*(j-1)/2+i;
    	    val = -Q[ji];	    
	 }
	 return val;
      }
   public:
      int sorb;
   private:   
      std::vector<Tm> data; // <ij||kl>
      std::vector<double> Q;  // <ij||ij> 
};

template <typename Tm>	
void load(two_body<Tm>& int2e,
	  one_body<Tm>& int1e,
	  double& ecore,
	  std::string fname){
   auto t0 = global::get_time();
   std::cout << "\nintegral::load_integral fname = " << fname << std::endl; 
   std::ifstream istrm(fname);
   if(!istrm){
      std::cout << "failed to open " << fname << std::endl;
      exit(1);
   }
   // parse MOLEINFO file
   std::string line;
   std::getline(istrm,line);
   int sorb = std::stoi(line);
   std::cout << "sorb = " << sorb << std::endl;
   // load integrals
   int1e.sorb = sorb;
   int1e.init_mem(); 
   int2e.sorb = sorb; 
   int2e.init_mem(); 
   int i,j,k,l;
   Tm eri;
   while(!istrm.eof()){
      line.clear();	    
      std::getline(istrm,line);
      std::istringstream is(line);
      is >> i >> j >> k >> l >> eri;
      if(i*j == 0 && k*l == 0){
         ecore = realpart(eri);
      }else if(i*j != 0 && k*l == 0){
	 int1e.set(i-1, j-1, eri);
      }else if(i*j != 0 && k*l != 0){
	 int2e.set(i-1, j-1, k-1, l-1, eri);
      }
   }
   istrm.close();
   int2e.initQ();
   auto t1 = global::get_time();
   std::cout << "timing for integral::load_integral : " << std::setprecision(2) 
	     << global::get_duration(t1-t0) << " s" << std::endl;
}

} // integral

#endif
