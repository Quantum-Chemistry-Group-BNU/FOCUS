#ifndef INPUT_H
#define INPUT_H

#include <vector>
#include <string>
#include <set> 

namespace input{

struct schedule{
public:
   void create_scratch();
   void remove_scratch();
public:
   std::string scratch;
   int nelec;
   int twoms;
   int nroots;
   std::string integral_file;
   int integral_type;
   // initial dets
   std::set<std::set<int>> det_seeds;
   int nseeds;
   bool flip;
   // for initial guess
   double eps0; 
   std::vector<double> eps1;
   // sci
   int miniter;
   int maxiter;
   double deltaE;
   // dvdson
   double crit_v;
   int maxcycle;
   // pt2
   bool ifpt2;
   double eps2;
   // io
   bool ciload;
   bool combload;
   // comb
   std::string topology_file;
   int maxdets;
   double thresh_proj;
   double thresh_ortho;
   // sweep
   int dmax;
   int dots;
   int maxsweep;
};
   
void read_input(schedule& schd, std::string fname="input.dat");

} // input

#endif
