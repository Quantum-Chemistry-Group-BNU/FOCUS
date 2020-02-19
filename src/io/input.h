#ifndef INPUT_H
#define INPUT_H

#include <vector>
#include <string>
#include <set> 

using namespace std;

namespace input{

struct schedule{
public:
   int nelec;
   int norb;
   int nseed;
   set<vector<int>> det_seeds;
   int nroots;
   string integral_file;
};
   
void read_input(schedule& schd, string fname="input.dat");

}

#endif
