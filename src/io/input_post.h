#ifndef INPUT_POSTMPS_H
#define INPUT_POSTMPS_H

#include <iostream>
#include <vector>
#include <string>
#include <set> 
#include <sstream> // istringstream

#include "../core/serialization.h"

namespace input{

   struct params_post{
      private:
         // serialize
         friend class boost::serialization::access;
         template<class Archive>
            void serialize(Archive & ar, const unsigned int version){
               ar & run & qkind & topology_file & verbose
                  & task_dumpbin & task_ovlp & task_cicoeff & task_sdiag & task_expect 
                  & task_s2proj & task_es2proj
                  & bra & ket & opname & integral_file
                  & iroot & nsample & ndetprt & twos & eps2; 
            }
      public:
         void read(std::ifstream& istrm);
         void print() const;
      public:
         bool run = false;
         std::string qkind;
         std::string topology_file = "TOPOLOGY";
         // debug level
         int verbose = 0;
         // tasks
         bool task_dumpbin = false;
         bool task_ovlp = false;
         bool task_cicoeff = false;
         bool task_sdiag = false;
         bool task_expect = false;
         bool task_s2proj = false;
         bool task_es2proj = false;
         std::vector<int> bra;
         std::vector<int> ket;
         std::string opname = "";
         std::string integral_file = "mole.info";
         int iroot = 0;
         int nsample = 10000;
         int ndetprt = 10;
         int twos = 0;
         double eps2 = 1.e-10;
   };

} // input

#endif
