#ifndef INPUT_CI_H
#define INPUT_CI_H

#include <iostream>
#include <vector>
#include <string>
#include <set> 
#include <sstream> // istringstream

#include "../core/serialization.h"

namespace input{

   // SCI
   struct params_ci{
      private:
         // serialize
         friend class boost::serialization::access;
         template<class Archive>
            void serialize(Archive & ar, const unsigned int version){
               ar & run & nroots & det_seeds & nseeds & flip
                  & eps0 & eps1 & miniter & maxiter & deltaE & checkms
                  & cisolver & maxcycle & crit_v & ifpt2 & eps2 & iroot & jroot
                  & load & ci_file & cthrd & ifanalysis & rdm & init;
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
         // selection threshold |HAI*CI|>eps for initial guess
         double eps0 = 1.e-2;
         // selection threshold |HAI*CI|>eps for iteration in SCI
         std::vector<double> eps1;   
         // sci
         int miniter = 0;
         int maxiter = 0;
         double deltaE = 1.e-10;
         bool checkms = false;
         // dvdson
         int cisolver = 1;
         int maxcycle = 100;
         double crit_v = 1.e-4;
         // pt2
         bool ifpt2 = false;
         double eps2 = 1.e-8;
         int iroot = 0;
         int jroot = 0;
         // io
         bool load = false;
         std::string ci_file = "ci.info"; 
         // print
         double cthrd = 1.e-1;
         bool ifanalysis = false;
         // rdm
         bool rdm = false;
         // initial
         std::string init = "";
   };

} // input

#endif
