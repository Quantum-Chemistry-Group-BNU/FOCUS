#ifndef INPUT_H
#define INPUT_H

#include <vector>
#include <string>
#include <set> 

namespace input{

struct schedule{
public:
   int nelec;
   int norb;
   int nseed;
   std::set<std::vector<int>> det_seeds;
   int nroots;
   std::string integral_file;
};
   
void read_input(schedule& schd, std::string fname="input.dat");

}

#endif
