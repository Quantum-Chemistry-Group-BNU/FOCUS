#ifndef INPUT_POSTMPS_H
#define INPUT_POSTMPS_H

#include <iostream>
#include <vector>
#include <string>
#include <set> 
#include <sstream> // istringstream

#include "../core/serialization.h"

namespace input{

   struct params_postmps{
      private:
         // serialize
         friend class boost::serialization::access;
         template<class Archive>
            void serialize(Archive & ar, const unsigned int version){
               ar & run & qkind & topology_file & verbose
                  & task_ovlp & task_cicoeff & task_sdiag & task_expect 
                  & bra & ket
                  & iroot & nsample & ndetprt & eps2; 
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
         bool task_ovlp = false;
         bool task_cicoeff = false;
         bool task_sdiag = false;
         bool task_expect = false;
         std::vector<int> bra;
         std::vector<int> ket;
         int iroot = 0;
         int nsample = 10000;
         int ndetprt = 10;
         double eps2 = 1.e-10;
   };

} // input

#endif
