#ifndef INPUT_H
#define INPUT_H

#include <vector>
#include <string>
#include <set> 

using namespace std;

struct schedule{
public:
   void readInput(string fname="input.dat");
public:
   int nelec;
   int norb;
   int nseed;
   set<vector<int>> detSeeds;
   int nroots;
   string integralFile;
};

#endif
