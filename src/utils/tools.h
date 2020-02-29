#ifndef TOOLS_H
#define TOOLS_H

#include <cmath>
#include <vector>
#include <numeric>      // std::iota
#include <algorithm>    // std::stable_sort

using namespace std;

namespace tools{

// compress symmetric (i,j) [i>j] pair:
// 10[0]
// 20[1] 21[2]
// 30[3] 31[4] 32[5]
inline int canonical_pair0(const int i, const int j){
   if(i==j){
      return -1;
   }else{
      return std::max(i,j)*(std::max(i,j)-1)/2 + std::min(i,j);
   }
}

inline void inverse_pair0(const int ij, int& i, int& j){
   i = floor(sqrt(2.0*(ij+1))+0.5);
   j = ij-i*(i-1)/2;
}

// compress symmetric (i,j) [i>=j] pair:
// 00[0]
// 10[1] 11[2]
// 20[3] 21[4] 22[5]
inline int canonical_pair(const int i, const int j){
   return std::max(i,j)*(std::max(i,j)+1)/2 + std::min(i,j);
}

inline void inverse_pair(const int ij, int& i, int& j){
   i = floor(sqrt(2.0*(ij+1))-0.5);
   j = ij-i*(i+1)/2;
}

// return index for sorting 
// template do not need implementation but only header
template <typename T>
vector<int> sort_index(const vector<T>& v){
  vector<int> idx(v.size());
  iota(idx.begin(), idx.end(), 0);
  stable_sort(idx.begin(), idx.end(),
       	      [&v](int i1, int i2) {return v[i1] > v[i2];});
  return idx;
}

template <typename T>
vector<int> sort_index_abs(const vector<T>& v){
  vector<int> idx(v.size());
  iota(idx.begin(), idx.end(), 0);
  stable_sort(idx.begin(), idx.end(),
       	      [&v](int i1, int i2) {return abs(v[i1]) > abs(v[i2]);});
  return idx;
}

}

#endif
