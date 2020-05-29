#ifndef INPUT_H
#define INPUT_H

#include <vector>
#include <string>
#include <set> 

namespace input{

struct sweep_ctrl{
   int isweep;
   int dots;
   int dcut;
   double eps;
   double noise; 
};

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
   // --- SCI --- 
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
   // --- CombTNS --- 
   // comb
   std::string topology_file;
   int maxdets;
   double thresh_proj;
   double thresh_ortho;
   // sweep
   int maxsweep;
   std::vector<sweep_ctrl> combsweep;
};
   
void read_input(schedule& schd, std::string fname="input.dat");

void init_combsweep(const int maxsweep,
		    std::vector<sweep_ctrl>& combsweep);

} // input

#endif
