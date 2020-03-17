#ifndef INPUT_H
#define INPUT_H

#include <vector>
#include <string>
#include <set> 

namespace input{

struct schedule{
public:
   int nelec;
   int nroots;
   std::string integral_file;
   std::set<std::set<int>> det_seeds;
   int nseeds;
   int maxiter;
   std::vector<double> eps1;
   double deltaE;
   double dvdson;
};
   
void read_input(schedule& schd, std::string fname="input.dat");

} // input

#endif
