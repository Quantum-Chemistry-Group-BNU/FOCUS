#ifndef TOOLS_H
#define TOOLS_H

#include <cmath>
#include <vector>
#include <numeric>      // std::iota
#include <algorithm>    // std::stable_sort
#include <tuple>
#include <cassert>

using namespace std;

namespace tools{

// compress symmetric (i,j) [i>j] pair:
// 10[0]
// 20[1] 21[2]
// 30[3] 31[4] 32[5]
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

// return index for sorting 
// template do not need implementation but only header
// pointer version
template <typename T>
vector<int> sort_index(const int size, const T* v, const int iop=0){
   vector<int> idx(size);
   iota(idx.begin(), idx.end(), 0);
   if(iop == 0){
      stable_sort(idx.begin(), idx.end(),
           	  [&v](int i1, int i2) {return v[i1] > v[i2];});
   }else{
      stable_sort(idx.begin(), idx.end(),
        	  [&v](int i1, int i2) {return v[i1] < v[i2];});
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
        	  [&v](int i1, int i2) {return abs(v[i1]) > abs(v[i2]);});
   }else{
      stable_sort(idx.begin(), idx.end(),
        	  [&v](int i1, int i2) {return abs(v[i1]) < abs(v[i2]);});
   }
   return idx;
}

template <typename T>
vector<int> sort_index_abs(const vector<T>& v, const int iop=0){
   return sort_index_abs(v.size(), v.data(), iop);
}

} // tools

#endif
