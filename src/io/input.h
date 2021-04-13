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
   // --- Generic ---
   std::string scratch;
   int nelec;
   int twoms;
   int nroots;
   std::string integral_file;
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
   int cisolver;
   int maxcycle;
   double crit_v;
   // pt2
   bool ifpt2;
   double eps2;
   // io
   bool ciload;
   // --- CombTNS --- 
   // comb
   std::string topology_file;
   // sci
   int maxdets;
   double thresh_proj;
   double thresh_ortho;
   // sweep
   int maxsweep;
   std::vector<sweep_ctrl> combsweep;
   // io
   bool combload;
};
   
void read(schedule& schd, std::string fname="input.dat");

void combsweep_init(const int maxsweep,
		    std::vector<sweep_ctrl>& combsweep);

void combsweep_print(const int maxsweep,
		     const std::vector<sweep_ctrl>& combsweep);

void combsweep_print(const sweep_ctrl& ctrl);

} // input

#endif
