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
               ar & run & qkind & topology_file & verbose; 
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
   };

} // input

#endif
