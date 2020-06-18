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

using namespace std;

namespace tools{

// useful functions
void license();

// print
const std::string line_separator(70,'-');
extern const std::string line_separator;
const std::string line_separator2(70,'=');
extern const std::string line_separator2;

double mem_size(size_t sz, const int fac=8);

// --- timing ---
std::chrono::high_resolution_clock::time_point get_time();

template<typename T>
double get_duration(T t){
   return std::chrono::duration_cast<std::chrono::milliseconds>(t).count()*0.001; 
}

// --- random --- 
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

// pointer version
// template do not need implementation but only header
template <typename T>
vector<int> sort_index(const int size, const T* v, const int iop=0){
   vector<int> idx(size);
   iota(idx.begin(), idx.end(), 0);
   // return index for sorting: iop=0, min; =1, max first;
   if(iop == 0){
      stable_sort(idx.begin(), idx.end(),
           	  [&v](int i1, int i2) {return v[i1] < v[i2];});
   }else{
      stable_sort(idx.begin(), idx.end(),
        	  [&v](int i1, int i2) {return v[i1] > v[i2];});
   }
   return idx;
}

// vector version
template <typename T>
vector<int> sort_index(const vector<T>& v, const int iop=0){
   return sort_index(v.size(), v.data(), iop);
}

// sort by absolute value
template <typename T>
vector<int> sort_index_abs(const int size, const T* v, const int iop=0){
   vector<int> idx(size);
   iota(idx.begin(), idx.end(), 0);
   if(iop == 0){
      stable_sort(idx.begin(), idx.end(),
        	  [&v](int i1, int i2) {return abs(v[i1]) < abs(v[i2]);});
   }else{
      stable_sort(idx.begin(), idx.end(),
        	  [&v](int i1, int i2) {return abs(v[i1]) > abs(v[i2]);});
   }
   return idx;
}

template <typename T>
vector<int> sort_index_abs(const vector<T>& v, const int iop=0){
   return sort_index_abs(v.size(), v.data(), iop);
}

// permutations
class perm{
   public:
      perm(const int k): size(k){
         image.resize(k);
	 std::iota(image.begin(), image.end(), 0);
      }
      perm(std::vector<int>& order){
	 size = order.size();
	 image.resize(size);
	 std::copy(order.begin(),order.end(),image.begin());
      }
      void shuffle(){
	 std::shuffle(image.begin(), image.end(), generator);
      }
      // to spin orbital sites
      std::vector<int> to_image2(){
	 std::vector<int> image2(size*2);
	 for(int i=0; i<size; i++){
	    image2[2*i] = 2*image[i];
	    image2[2*i+1] = 2*image[i]+1;
	 }
	 return image2;
      }
      friend std::ostream& operator <<(std::ostream& os, const perm& p);
   public:
      int size;
      std::vector<int> image;
};

} // tools

#endif
