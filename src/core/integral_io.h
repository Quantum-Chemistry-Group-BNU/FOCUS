#ifndef INTEGRAL_IO_H
#define INTEGRAL_IO_H

#include "integral.h"
#include <iostream>
#include <fstream>
#include <sstream>

namespace integral{

   // load integrals from mole.info 
   template <typename Tm>
      void load(two_body<Tm>& int2e, 
            one_body<Tm>& int1e, 
            double& ecore, 
            const std::string fname){
         auto t0 = tools::get_time();
         std::cout << "\nintegral::load fname = " << fname << std::endl; 

         std::ifstream istrm(fname);
         if(!istrm){
            std::cout << "failed to open " << fname << std::endl;
            exit(1);
         }
         // parse MOLEINFO file
         int sorb;
         std::string line;
         while(!istrm.eof()){
            line.clear();	    
            std::getline(istrm,line);
            if(line.empty() || line[0]=='#'){
               continue; // skip empty and comments
            }else{
               sorb = std::stoi(line);
               break;
            }
         }
         // load integrals
         int1e.sorb = sorb;
         int1e.init_mem(); 
         int2e.sorb = sorb; 
         int2e.init_mem(); 
         std::cout << " sorb = " << sorb << std::endl;
         std::cout << " size(int1e) = " << int1e.size() << ":" 
            << tools::sizeMB<Tm>(int1e.size()) << "MB:"
            << tools::sizeGB<Tm>(int1e.size()) << "GB"
            << std::endl;
         std::cout << " size(int2e) = " << int2e.size() << ":" 
            << tools::sizeMB<Tm>(int2e.size()) << "MB:"
            << tools::sizeGB<Tm>(int2e.size()) << "GB"
            << std::endl; 

         // read
         int i,j,k,l;
         Tm eri;
         while(!istrm.eof()){
            line.clear();	    
            std::getline(istrm,line);
            if(line.empty() || line[0]=='#'){
               continue; // skip empty and comments
            }else{
               std::istringstream is(line);
               is >> i >> j >> k >> l >> eri; // read quadruple and integral
               if(i*j == 0 && k*l == 0){
                  std::cout << " ecore = " << eri << std::endl;
                  ecore = std::real(eri);
               }else if(i*j != 0 && k*l == 0){
                  int1e.set(i-1, j-1, eri);
               }else if(i*j != 0 && k*l != 0){
                  int2e.set(i-1, j-1, k-1, l-1, eri);
               }
            }
         }
         istrm.close();
         // compute Qij
         int2e.initQ();

         auto t1 = tools::get_time();
         tools::timing("integral::load", t0, t1);
      }

   // load integrals from mole.info 
   template <typename Tm>
      void save(const two_body<Tm>& int2e, 
            const one_body<Tm>& int1e, 
            const double& ecore, 
            const std::string fname,
            const double thresh=1.e-16){
         auto t0 = tools::get_time();
         std::cout << "integral::save fname = " << fname 
            << " with thresh=" << thresh << std::endl;

         std::ofstream ostrm(fname);
         if(!ostrm){
            std::cout << "failed to open " << fname << std::endl;
            exit(1);
         }
         // sorb
         int sorb = int2e.sorb;
         ostrm << sorb << std::endl;
         ostrm << std::defaultfloat << std::setprecision(18);
         // int2e
         std::string line;
         for(int i=0; i<sorb; i++){
            for(int j=0; j<i; j++){
               for(int k=0; k<=i; k++){
                  int lmax = (k==i)? j+1 : k;
                  for(int l=0; l<lmax; l++){
                     Tm val = int2e.get(i,j,k,l);
                     if(abs(val)<thresh) continue;
                     ostrm << i+1 << " " << j+1 << " " << k+1 << " " << l+1 << " "
                       << val << std::endl;
                  }
               }
            }
         }
         // int1e
         for(int i=0; i<sorb; i++){
            for(int j=0; j<sorb; j++){
               Tm val = int1e.get(i,j);
               if(abs(val)<thresh) continue;
               ostrm << i+1 << " " << j+1 << " 0 0 " 
                  << val << std::endl; 
            }
         }
         // ecore
         ostrm << "0 0 0 0 " << ecore << std::endl;
         ostrm.close();

         auto t1 = tools::get_time();
         tools::timing("integral::save", t0, t1);
      }

} // integral

#endif
