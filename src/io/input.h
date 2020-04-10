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
   int integral_type;
   std::set<std::set<int>> det_seeds;
   int nseeds;
   int miniter;
   int maxiter;
   bool flip;
   double eps0; // for initial guess
   std::vector<double> eps1;
   double deltaE;
   double dvdson;
   bool ifpt2;
   double eps2;
   bool ciload;
};
   
void read_input(schedule& schd, std::string fname="input.dat");

} // input

#endif
