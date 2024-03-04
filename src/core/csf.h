#ifndef CSF_H
#define CSF_H

#include <iostream>
#include "binom.h"
#include "onstate.h"
#include "onspace.h"

namespace fock{

   // configuration state function (CSF) [in right caonical form]
   struct csfstate{
      public:
         // constructor
         csfstate(){}
         csfstate(const int k){ 
            onstate _repr(2*k);
            repr = _repr;  
         }
         csfstate(const std::string& s){
            onstate _repr(s, 1);
            repr = _repr;
            // consistency check
            auto tsinter = intermediate_tsarray();
            for(const auto& ts : tsinter){
               if(ts < 0){
                  std::cout << "error: not a valid csf as ts<0 in s=" << s << std::endl;
                  tools::print_vector(tsinter,"tsinter");
                  exit(1);
               }
            }
         }
         // comparison used in map [see rcanon_Sdiag]
         bool operator <(const csfstate& state) const{ return repr<state.repr; }  
         int nelec() const{ return repr.nelec(); }
         int twos() const{ return repr.nelec_a()-repr.nelec_b(); }
         int norb() const{ return repr.norb(); }
         int norb_single() const{ return repr.norb_single(); }
         int dvec(const int k) const{ return repr[2*k+1]*2+repr[2*k]; } // step value
         int nvec(const int k) const{ return repr[2*k+1]+repr[2*k]; }
         std::vector<int> intermediate_narray() const;
         std::vector<int> intermediate_tsarray() const;
         std::vector<int> orbs_single() const{
            int ks = norb();
            std::vector<int> orbs_os(ks);
            int n = 0;
            for(int k=0; k<ks; k++){
               int dval = dvec(k);
               if(dval==1 || dval==2){
                  orbs_os[n] = k; // open-shell orbital index
                  n += 1;
               }
            }
            orbs_os.resize(n);
            return orbs_os;
         }
         // to determinants
         double det_coeff(const onstate& state) const;
         // by default the M=S is used.
         std::pair<onspace,std::vector<double>> to_det() const;
         // sample
         std::pair<onstate,double> random() const;
         double Sdiag_sample(const int nsample=10000, const int nprt=10) const;
         double Sdiag_exact() const;
         // print
         std::string to_string() const{ return repr.to_string2(true); }
         friend std::ostream& operator <<(std::ostream& os, const csfstate& state);
      public:
         onstate repr; // internal representation - step vector |d> (d=0,1,2,3)
   };

   // --- CSF space ---
   using csfspace = std::vector<csfstate>;

   // csf space dimsion by Weyl-Paldus formula
   inline size_t dim_csf_space(const int k, const int n, const int ts){
      size_t dim = (n%2!=ts%2)? 0 : (ts+1)*fock::binom(k+1,(n-ts)/2)*fock::binom(k+1,(n+ts)/2+1)/(k+1);
      return dim;
   }

   // generate all CSF with (N,S) via a iterative procedure
   csfspace get_csf_space(const int k, const int n, const int ts);

} // fock

#endif
