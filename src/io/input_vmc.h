#ifndef INPUT_VMC_H
#define INPUT_VMC_H

#include <iostream>
#include <vector>
#include <string>
#include <set> 
#include <sstream> // istringstream

#include "../core/serialization.h"

namespace input{

   struct params_vmc{
      private:
         // serialize
         friend class boost::serialization::access;
         template<class Archive>
            void serialize(Archive & ar, const unsigned int version){
               ar & run & ansatz & nhiden & iscale 
                  & exactopt & nsample & maxiter 
                  & optimizer & lr & history & wf_load & wf_file;
            }
      public:
         void read(std::ifstream& istrm);
         void print() const;
      public:
         bool run = false;
         std::string ansatz = "irbm";
         int nhiden = 0;
         double iscale = 1.e-3;
         bool exactopt = false;
         int nsample = 10000;
         int maxiter = 1000;
         std::string optimizer = "kfac";
         double lr = 1.e-2;
         std::string history  = "vmc_his.bin";
         bool wf_load = false;
         std::string wf_file = "vmc.info";
   };

} // input

#endif
