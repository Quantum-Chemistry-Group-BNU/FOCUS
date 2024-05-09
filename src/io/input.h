#ifndef INPUT_H
#define INPUT_H

#include <iostream>
#include <vector>
#include <string>
#include <set> 
#include <sstream> // istringstream

#include "../core/serialization.h"

#include "input_sci.h"
#include "input_ctns.h"
#include "input_post.h"
#include "input_vmc.h"

#ifndef SERIAL
#include <boost/mpi.hpp>
#endif

namespace input{

   const bool debug_input = false;
   extern const bool debug_input;

   // General
   struct schedule{
      private:
         // serialize
         friend class boost::serialization::access;
         template<class Archive>
            void serialize(Archive & ar, const unsigned int version){
               ar & scratch & dtype & nelec & twoms & twos & integral_file 
                  & perfcomm 
                  & sci & ctns & post & vmc;
            }
      public:
         void read(std::string fname="input.dat");
         void print() const;
      public:
         // --- Generic ---
         std::string scratch = ".";
         int dtype = 0;
         int nelec = 0;
         int twoms = 0;
         int twos = 0;
         std::string integral_file = "mole.info";
         int perfcomm = 0;
         // --- Methods --- 
         params_sci sci;
         params_ctns ctns;
         params_post post;
         params_vmc vmc;
         // --- MPI ---
#ifndef SERIAL
         boost::mpi::communicator world;
#endif
   };

} // input

#endif
