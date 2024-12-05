#ifndef TOOLS_H
#define TOOLS_H

#include <cmath>
#include <vector>
#include <string>
#include <numeric>      // std::iota
#include <algorithm>    // std::stable_sort
#include <tuple>
#include <cassert>
#include <iostream>
#include <iomanip>
#include <random>
#include <chrono>
#include <complex>
#include <time.h>
 
namespace tools{

inline void license(){
   std::cout << std::endl;	
   std::cout << "======================================================================" << std::endl;
   std::cout << "     FOCUS: a platform for exploring FermiOniC qUantum Simulation     " << std::endl;
   std::cout << "   							               " << std::endl;
   std::cout << "       :!!!!!!!!!    ^!!!7!:      :!7!!~~!:!!!!!:  ~!!!. :!!!!7:      " << std::endl;  
   std::cout << "       .^G@P^^~?BJ ~GG!::~Y#P^  ^P#?^:^7G@^:5&#~.  ^J#! 7P:.:?&5      " << std::endl;
   std::cout << "         5@Y  !J :~&@^     ?&@^^B@?     .?. ?&B     :P  5&J7~^~~      " << std::endl;
   std::cout << "         5@G!?BB  ?&&:     :B@??&@^         ?&B     :P  .?5GB##P^     " << std::endl;
   std::cout << "         5@Y  :!  ^B@J     !@#:^B@Y      ^: ?&#     ~P  JJ   .~@5     " << std::endl;
   std::cout << "       .~G@P^:     ^5#5~^^7GY:  ^5&P7~~!?J: :5&5!~~!Y^  Y&5!^^7G^     " << std::endl;
   std::cout << "       :~~~~~:       .~!!~^.      .~77!~.     :!7!~^    :::~!!^       " << std::endl;
   std::cout << "   							               " << std::endl;
   std::cout << "                    Authors: Zhendong Li @ BNU2019   		       " << std::endl;
   std::cout << "======================================================================" << std::endl;
   std::cout << std::endl;
   // https://en.cppreference.com/w/c/chrono/time
   time_t now = time(NULL);
   std::cout << "TIME = " << ctime(&now) << std::endl; 
   // https://stackoverflow.com/questions/44038428/include-git-commit-hash-and-or-branch-name-in-c-c-source
   std::cout << "GIT_HASH = " << GIT_HASH << std::endl;
}

inline void finish(const std::string msg){
   time_t now = time(NULL);
   std::cout << "\nCongrats! " << msg 
	     << " calculation finishes successfully. "
	     << ctime(&now) 
	     << std::endl; 
}

inline void exit(const std::string msg){
   std::cout << "\n" << msg << std::endl;
   std::exit(1);
}

// print
const std::string line_separator(70,'-');
extern const std::string line_separator;
const std::string line_separator2(70,'=');
extern const std::string line_separator2;

template <typename Tm>
void print_vector(const std::vector<Tm>& vec, const std::string name){
   std::cout << " " << name << "=";
   for(const auto& k : vec) std::cout << " " << k;
   std::cout << std::endl;
}	
template <typename Tm>
void print_vector(const std::vector<Tm>& vec, const std::string name, const int outprec){
   std::cout << " " << name << "=";
   std::cout << std::scientific << std::setprecision(outprec);
   for(const auto& k : vec) std::cout << " " << k;
   std::cout << std::endl;
}	

template <typename Tm>
std::vector<Tm> combine_vector(const std::vector<Tm>& v1, 
		    	       const std::vector<Tm>& v2){
   std::vector<Tm> v12 = v1;
   v12.insert(v12.end(), v2.begin(), v2.end());
   return v12;
}

template <typename Tm, typename Km>
bool is_in_vector(const std::vector<Tm>& v, const Km key){
   auto it = std::find(v.begin(), v.end(), key);
   return (it != v.end());
}

// type information
template <typename Tm>
constexpr bool is_complex(){ return false; }
template <>
constexpr bool is_complex<std::complex<double>>(){ return true; }

// memory in MB/GB
template <typename Tm>
inline double sizeMB(size_t sz){ return sz*sizeof(Tm)/std::pow(1024.0,2); }
template <typename Tm>
inline double sizeGB(size_t sz){ return sz*sizeof(Tm)/std::pow(1024.0,3); }

// conjugte
inline double conjugate(const double x){ return x; }
inline std::complex<double> conjugate(const std::complex<double> x){ return std::conj(x); }

// timing 
inline std::chrono::system_clock::time_point get_time(){
   return std::chrono::system_clock::now();
}
template<typename T>
double get_duration(T t){
   return std::chrono::duration<double, std::nano>(t).count()*1.e-9; 
}
template<typename T>
void timing(const std::string msg, const T& t0, const T& t1){
   std::cout << "----- TIMING FOR " << msg << " : " << std::scientific << std::setprecision(3) 
	     << get_duration(t1-t0) << std::defaultfloat << " S -----" << std::endl;
}

// random
extern std::seed_seq seeds;
extern std::default_random_engine generator;

// compress symmetric (i,j) [i>j] pair:
// 10[0]
// 20[1]  21[2]
// 30[3]  31[4]  32[5]
// 40[6]  41[7]  42[8]  43[9]
// 50[10] 51[11] 52[12] 53[13] 54[14] = 5*6/2 - 1 
inline size_t canonical_pair0(const size_t i, const size_t j){
   assert(i != j);
   return std::max(i,j)*(std::max(i,j)-1)/2 + std::min(i,j);
}

inline double sgn_pair0(const size_t i, const size_t j){
   assert(i != j);
   return i>j? 1 : -1;
}

inline std::pair<size_t,size_t> inverse_pair0(const size_t ij){
   size_t i = floor(sqrt(2.0*(ij+1))+0.5);
   size_t j = ij-i*(i-1)/2;
   assert(canonical_pair0(i,j)==ij);
   return std::make_pair(i,j);
}

// compress symmetric (i,j) [i>=j] pair:
// 00[0]
// 10[1] 11[2]
// 20[3] 21[4] 22[5]
inline size_t canonical_pair(const size_t i, const size_t j){
   return std::max(i,j)*(std::max(i,j)+1)/2 + std::min(i,j);
}

inline std::pair<size_t,size_t> inverse_pair(const size_t ij){
   size_t i = floor(sqrt(2.0*(ij+1))-0.5);
   size_t j = ij-i*(i+1)/2;
   assert(canonical_pair(i,j)==ij);
   return std::make_pair(i,j);
}

// canonical triple
inline size_t canonical_triple0(const size_t i, const size_t j, const size_t k){
   assert(i != j and i != k and j != k);
   size_t a, b, c;
   if(i > j and j>k){
      a=i; b=j; c=k;
   }else if(i > k and k>j){
      a=i; b=k; c=j;
   }else if(j > i and i>k){
      a=j; b=i; c=k;
   }else if(j > k and k>i){
      a=j; b=k; c=i;
   }else if(k > i and i>j){
      a=k; b=i; c=j;
   }else if(k > j and j>i){
      a=k; b=j; c=i;
   }else{
      std::cout << "error: no such case in canonical_triple0! i,j,k=" << i << "," << j << "," << k << std::endl;
      std::exit(1);
   }
   return a*(a-1)*(a-2)/6 + b*(b-1)/2 + c;
}

// canonical triple
inline double sgn_triple0(const size_t i, const size_t j, const size_t k){
   assert(i != j and i != k and j != k);
   double sgn = 1.0;
   if(i > j and j>k){ // 123->123
      sgn = 1.0;
   }else if(i > k and k>j){ // 132->123
      sgn = -1.0;
   }else if(j > i and i>k){ // 213->123
      sgn = -1.0;
   }else if(j > k and k>i){ // 231->123
      sgn = 1.0;
   }else if(k > i and i>j){ // 312->123
      sgn = 1.0;
   }else if(k > j and j>i){ // 321->123
      sgn = -1.0;
   }else{
      std::cout << "error: no such case in sgn_triple0! i,j,k=" << i << "," << j << "," << k << std::endl;
      std::exit(1);
   }
   return sgn;
}

inline std::tuple<size_t,size_t,size_t> inverse_triple0(const size_t ijk){
   size_t imin = std::floor(std::pow(6*ijk,1.0/3.0));
   size_t i;
   for(size_t i0=imin; i0<=imin+2; i0++){
      if(ijk >= i0*(i0-1)*(i0-2)/6){
         i = i0; 
      }else{
         break;  
      }
   }
   size_t i3 = i*(i-1)*(i-2)/6;
   size_t jk = ijk - i3;
   auto pr = inverse_pair0(jk);
   size_t j = pr.first;
   size_t k = pr.second;
   assert(canonical_triple0(i,j,k)==ijk);
   return std::make_tuple(i,j,k);
}

// sort_index 
// pointer version
// template do not need implementation but only header
template <typename T>
std::vector<int> sort_index(const int size, const T* v, const int iop=0){
   std::vector<int> idx(size);
   std::iota(idx.begin(), idx.end(), 0);
   // return index for sorting: iop=0, min; =1, max first;
   if(iop == 0){
      std::stable_sort(idx.begin(), idx.end(),
           	  [&v](int i1, int i2) {return v[i1] < v[i2];});
   }else{
      std::stable_sort(idx.begin(), idx.end(),
        	  [&v](int i1, int i2) {return v[i1] > v[i2];});
   }
   return idx;
}
// vector version
template <typename T>
std::vector<int> sort_index(const std::vector<T>& v, const int iop=0){
   return sort_index(v.size(), v.data(), iop);
}

// sort by absolute value
template <typename T>
std::vector<int> sort_index_abs(const int size, const T* v, const int iop=0){
   std::vector<int> idx(size);
   std::iota(idx.begin(), idx.end(), 0);
   if(iop == 0){
      std::stable_sort(idx.begin(), idx.end(),
        	  [&v](int i1, int i2) {return std::abs(v[i1]) < std::abs(v[i2]);});
   }else{
      std::stable_sort(idx.begin(), idx.end(),
        	  [&v](int i1, int i2) {return std::abs(v[i1]) > std::abs(v[i2]);});
   }
   return idx;
}
// vector version
template <typename T>
std::vector<int> sort_index_abs(const std::vector<T>& v, const int iop=0){
   return sort_index_abs(v.size(), v.data(), iop);
}

} // tools

#endif
