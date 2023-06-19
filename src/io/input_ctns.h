#ifndef INPUT_CTNS_H
#define INPUT_CTNS_H

#include <iostream>
#include <vector>
#include <string>
#include <set> 
#include <sstream> // istringstream

#include "../core/serialization.h"

namespace input{

   // CTNS
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
               ar & run & qkind & topology_file & verbose
                  & task_init & task_sdiag & task_ham & task_opt & task_vmc 
                  & restart_sweep & restart_bond & timestamp
                  & maxdets & thresh_proj & thresh_ortho & rdm_svd 
                  & nroots & guess & dbranch & maxsweep & maxbond & ctrls
                  & alg_hvec & alg_hinter & alg_hcoper 
                  & alg_renorm & alg_rinter & alg_rcoper & alg_decim 
                  & ifdist1 & ifdistc & save_formulae & sort_formulae & save_mmtask 
                  & batchhvec & batchrenorm & batchmem
                  & cisolver & maxcycle & nbuff & damping & precond
                  & rcanon_load & rcanon_file 
                  & iomode & async_fetch & async_save & async_remove & async_tocpu & ifnccl
                  & iroot & nsample & ndetprt; 
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
         // task
         bool task_init = false;
         bool task_sdiag = false;
         bool task_ham = false;
         bool task_opt = false;
         bool task_vmc = false;
         // restart
         int restart_sweep = 0;
         int restart_bond = 0;
         bool timestamp = false; 
         // conversion of sci 
         int maxdets = 10000;
         double thresh_proj = 1.e-16;
         double thresh_ortho = 1.e-8;
         double rdm_svd = 1.5;
         // sweep
         int nroots = 1; // this can be smaller than nroots in CI 
         int guess = 1;
         int dbranch = 0;
         int maxsweep = 0;
         int maxbond = -1;
         std::vector<params_sweep> ctrls;
         // algorithm
         int alg_hvec = 3;
         int alg_hinter = 0;
         int alg_hcoper = 0;
         int alg_renorm = 2;
         int alg_rinter = 0;
         int alg_rcoper = 0;
         int alg_decim = 1;
         bool ifdist1 = false;
         bool ifdistc = false;
         bool save_formulae = false;
         bool sort_formulae = false;
         bool save_mmtask = false;
         std::tuple<int,int,int> batchhvec = {-1,-1,-1};
         std::tuple<int,int,int> batchrenorm = {-1,-1,-1};
         double batchmem = 10; // GB
         // dvdson
         int cisolver = 1;
         int maxcycle = 30;
         int nbuff = 4; // should be greater than 2
         double damping = 1.e-10;
         bool precond = true;
         // io
         bool rcanon_load = false;
         std::string rcanon_file = "rcanon_ci.info";
         // oper_pool
         int iomode = 0;
         bool async_fetch = false;
         bool async_save = false;
         bool async_remove = false;
         bool async_tocpu = false;
         bool ifnccl = false;
         // sampling
         int iroot = 0;
         int nsample = 10000;
         int ndetprt = 10;
   };

} // input

#endif
