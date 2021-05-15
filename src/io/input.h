#ifndef INPUT_H
#define INPUT_H

#include <iostream>
#include <vector>
#include <string>
#include <set> 
#include <sstream> // istringstream

namespace input{

//
// SCI
//
struct params_sci{
public:
   void read(std::ifstream& istrm);
   void print() const;
public:
   bool run = false;
   int nroots = 1;
   // initial dets
   std::set<std::set<int>> det_seeds;
   int nseeds = 0;
   bool flip = false;
   // for initial guess
   double eps0 = 1.e-2; 
   std::vector<double> eps1;   
   // sci
   int miniter = 0;
   int maxiter = 0;
   double deltaE = 1.e-10;
   // dvdson
   int cisolver = 1;
   int maxcycle = 100;
   double crit_v = 1.e-6;
   // pt2
   bool ifpt2 = false;
   double eps2 = 1.e-8;
   // io
   bool load = false;
   std::string ci_file = "ci.info"; 
};

//
// CTNS
//
struct params_sweep{
public:
   inline void print() const{
      std::cout << "sweep parameters: isweep=" << isweep
         	<< " dots=" << dots << " dcut=" << dcut
         	<< " eps=" << eps << " noise=" << noise
         	<< std::endl;
   }
public:
   int isweep;
   int dots;
   int dcut;
   double eps;
   double noise; 
};

struct params_ctns{
public:
   void read(std::ifstream& istrm);
   void print() const;
public:
   bool run = false;
   std::string kind;
   std::string task = "opt";
   std::string topology_file = "TOPOLOGY";
   // conversion of sci 
   int maxdets = 10000;
   double thresh_proj = 1.e-16;
   double thresh_ortho = 1.e-8;
   // sweep
   int nroots = 1; // this can be smaller than nroots in CI 
   bool guess = true;
   int inoise = 2;
   int maxsweep;
   std::vector<params_sweep> ctrls;
   // dvdson
   int cisolver = 1;
   int maxcycle = 100;
   // io
   bool load = false;
   std::string rcanon_file = "rcanon.info"; 
};

//
// General
//
struct schedule{
public:
   void create_scratch();
   void remove_scratch();
   void read(std::string fname="input.dat");
   void print() const;
public:
   // --- Generic ---
   std::string scratch = ".";
   int dtype = 0;
   int nelec = 0;
   int twoms = 0;
   std::string integral_file = "mole.info";
   // --- Methods --- 
   params_sci sci;
   params_ctns ctns;
};

} // input

#endif
