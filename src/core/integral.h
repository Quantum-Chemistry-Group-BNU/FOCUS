#ifndef INTEGRAL_H
#define INTEGRAL_H

#include <cassert>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <algorithm>
#include "tools.h"
#include "serialization.h"
#ifndef SERIAL
#include "mpi_wrapper.h"
#endif

namespace integral{

   // one-electron integral h[i,j] [full in column-major storage]	
   template <typename Tm>
      struct one_body{
         private:
            // serialize
            friend class boost::serialization::access;	   
            template <class Archive>
               void serialize(Archive & ar, const unsigned int version){
                  ar & sorb & data;
               }
         public:
            one_body(){}
            one_body(const int _sorb){
               sorb = _sorb;
               this->init_mem();
               this->set_zero();
            }
            void init_mem(){
               assert(sorb > 0);
               data.resize(sorb*sorb);
            }
            size_t size() const{ return data.size(); }
            Tm get(const size_t i, const size_t j) const{
               return data[j*sorb+i];
            }
            void set(const size_t i, const size_t j, const Tm val){
               data[j*sorb+i] = val; // column-major storage
            }
            void set_real(){
               transform(data.begin(), data.end(), data.begin(),
                     [](const Tm& x){ return std::real(x); }); 
            }
            void set_zero(){
               memset(data.data(), 0, data.size()*sizeof(Tm));
            }
            void print() const{
               std::cout << "integral::one_body sorb=" << sorb << std::endl;
               for(int i=0; i<sorb; i++){
                  for(int j=0; j<sorb; j++){
                     std::cout << i << " " << j << " " 
                        << std::setprecision(12) << get(i,j) << std::endl; 
                  }
               }
            }
         public:
            int sorb = 0;
         private:
            std::vector<Tm> data; // Oij = <i|O|j>	
      };

   // two-electron integral <ij||kl> [packed i>j, k>l, (ij)>(kl)]
   // diagonal term Qij = <ij||ij> (i>j)
   template <typename Tm>
      struct two_body{
         public:
            two_body(){}
            two_body(const int _sorb){
               sorb = _sorb;
               this->init_mem();
               this->set_zero();
            }
            void init_mem(){
               assert(sorb > 0);
               size_t pair = sorb*(sorb-1)/2;
               data.resize(pair*(pair+1)/2,0.0);
               Q.resize(pair);
            }
            size_t size() const{ return data.size()+Q.size(); }
            // return <ij||kl> from packed storage
            Tm get(const size_t i, const size_t j, 
                  const size_t k, const size_t l) const{
               if((i == j) || (k==l)) return 0.0;
               size_t ij = i>j? i*(i-1)/2+j : j*(j-1)/2+i;
               size_t kl = k>l? k*(k-1)/2+l : l*(l-1)/2+k;
               double sgn = 1;
               sgn = i>j? sgn : -sgn;
               sgn = k>l? sgn : -sgn;
               Tm val;
               if(ij >= kl){
                  size_t ijkl = ij*(ij+1)/2+kl;
                  val = sgn*data[ijkl];
               }else{
                  size_t ijkl = kl*(kl+1)/2+ij;
                  val = sgn*tools::conjugate(data[ijkl]);
               }
               return val;
            }
            // save <ij||kl>
            void set(const size_t i, const size_t j, 
                  const size_t k, const size_t l, 
                  const Tm val){
               if((i == j) || (k==l)) return;
               size_t ij = i>j? i*(i-1)/2+j : j*(j-1)/2+i;
               size_t kl = k>l? k*(k-1)/2+l : l*(l-1)/2+k;
               double sgn = 1;
               sgn = i>j? sgn : -sgn;
               sgn = k>l? sgn : -sgn;
               if(ij >= kl){
                  size_t ijkl = ij*(ij+1)/2+kl;
                  data[ijkl] = sgn*val;
               }else{
                  size_t ijkl = kl*(kl+1)/2+ij;
                  data[ijkl] = sgn*tools::conjugate(val);
               }
            }
            // set all integrals to real parts
            void set_real(){
               transform(data.begin(), data.end(), data.begin(),
                     [](const Tm& x){ return std::real(x); }); 
            }
            // set all integrals to zero
            void set_zero(){
               memset(data.data(), 0, data.size()*sizeof(Tm));
               memset(Q.data(), Q.size(), Q.size()*sizeof(Tm));
            }
            void print() const{
               std::cout << "integral::two_body sorb=" << sorb << std::endl;
               for(int i=0; i<sorb; i++){
                  for(int j=0; j<i; j++){
                     for(int k=0; k<sorb; k++){
                        for(int l=0; l<k; l++){
                           std::cout << i << " " << j << " " << k << " " << l << " " 
                              << std::setprecision(12) << get(i,j,k,l) << std::endl;
                        }
                     }
                  }
               }
               // Qij
               std::cout << "Qij:" << std::endl;
               for(int i=0; i<sorb; i++){
                  for(int j=0; j<i; j++){
                     int ij = i*(i-1)/2+j;
                     std::cout << i << " " << j << " " << Q[ij] << std::endl;
                  }
               }
            }
            // Q related functions
            void initQ(){
               for(int i=0; i<sorb; i++){
                  for(int j=0; j<i; j++){
                     int ij = i*(i-1)/2+j;
                     Q[ij] = std::real(get(i,j,i,j));
                  }
               }
            }
            double getQ(const size_t i, const size_t j) const{
               if(i == j) return 0.0;
               int ij = (i>j)? i*(i-1)/2+j : j*(j-1)/2+i;
               return Q[ij];
            }
         public:
            int sorb = 0; // no. of spin orbitals
         public:
            std::vector<Tm> data; // <ij||kl>
            std::vector<double> Q;  // Qij=<ij||ij> 
      };

   // load integrals from mole.info 
   template <typename Tm>
      void load(two_body<Tm>& int2e, one_body<Tm>& int1e, double& ecore, const std::string fname){
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
         tools::timing("integral::load_integral", t0, t1);
      }

} // integral

#ifndef SERIAL

namespace mpi_wrapper{

   // int2e: assuming int2e.data is large
   template <typename Tm>
      void broadcast(const boost::mpi::communicator & comm, integral::two_body<Tm>& int2e, int root){
         int rank = comm.rank();
         boost::mpi::broadcast(comm, int2e.sorb, root);
         if(rank != root) int2e.init_mem();
         boost::mpi::broadcast(comm, int2e.Q, root);
         broadcast(comm, int2e.data.data(), int2e.data.size(), root);
      }

} // mpi_wrapper

#endif

#endif
