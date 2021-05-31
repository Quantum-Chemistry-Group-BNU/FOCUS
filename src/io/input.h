#ifndef INPUT_H
#define INPUT_H

#include <iostream>
#include <vector>
#include <string>
#include <set> 
#include <sstream> // istringstream
#include "../core/serialization.h"
#ifndef SERIAL
#include <boost/mpi.hpp>
#endif

namespace input{

//
// SCI
//
struct params_sci{
private:
   // serialize
   friend class boost::serialization::access;
   template<class Archive>
   void serialize(Archive & ar, const unsigned int version){
      ar & run & nroots & det_seeds & nseeds & flip
	 & eps0 & eps1 & miniter & maxiter & deltaE
         & cisolver & maxcycle & crit_v & ifpt2 & eps2
	 & load & ci_file;
   }
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
private:
   // serialize
   friend class boost::serialization::access;
   template<class Archive>
   void serialize(Archive & ar, const unsigned int version){
      ar & isweep & dots & dcut & eps & noise;
   }
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
private:
   // serialize
   friend class boost::serialization::access;
   template<class Archive>
   void serialize(Archive & ar, const unsigned int version){
      ar & verbose & run & kind & task & topology_file & maxdets & thresh_proj & thresh_ortho
	 & nroots & guess & inoise & maxsweep & ctrls & cisolver & maxcycle
         & load & rcanon_file; 
   }
public:
   void read(std::ifstream& istrm);
   void print() const;
public:
   bool run = false;
   std::string kind;
   std::string task = "check"; // default
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
   // debug level
   int verbose = 0; 
};

//
// General
//
struct schedule{
private:
   // serialize
   friend class boost::serialization::access;
   template<class Archive>
   void serialize(Archive & ar, const unsigned int version){
      ar & scratch & dtype & nelec & twoms & integral_file
         & sci & ctns;
   }
public:
   void create_scratch(const bool debug=true);
   void remove_scratch(const bool debug=true);
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
   // --- MPI ---
#ifndef SERIAL
   boost::mpi::communicator world;
#endif
};

} // input

#endif
