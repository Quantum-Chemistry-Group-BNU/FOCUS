#ifndef INPUT_CTNS_H
#define INPUT_CTNS_H

#include <iostream>
#include <vector>
#include <string>
#include <set> 
#include <sstream> // istringstream

#include "../core/serialization.h"

namespace input{

   // orbital optimization 
   struct params_oodmrg{
      private:
         // serialize
         friend class boost::serialization::access;
         template<class Archive>
            void serialize(Archive & ar, const unsigned int version){
               ar & iprt & maxiter & dfac & macroiter & microiter & alpha & thrdopt 
                  & thrdloc & nptloc & acceptall;
            }
       public:
         inline void print() const{
            std::cout << "oodmrg parameters:" << std::endl;
            std::cout << " iprt=" << iprt << std::endl;
            std::cout << " maxiter=" << maxiter << std::endl;
            std::cout << " dfac=" << dfac << std::endl;
            std::cout << " macroiter=" << macroiter << std::endl;
            std::cout << " microiter=" << microiter << std::endl;
            std::cout << " alpha=" << alpha << std::endl;
            std::cout << " thrdopt=" << thrdopt << std::endl;
            std::cout << " thrdloc=" << thrdloc << std::endl;
            std::cout << " nptloc=" << nptloc << std::endl;
            std::cout << " acceptall=" << acceptall << std::endl;
         }
      public:
         int iprt = 1;
         int maxiter = 10;
         int dfac = 2;
         int macroiter = 5;
         int microiter = 50;
         double alpha = 0.5;
         double thrdopt = 1.e-4;
         double thrdloc = 1.e-6;
         double nptloc = 20;
         bool acceptall = false; 
   };

   // sweep
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

   // CTNS
   struct params_ctns{
      private:
         // serialize
         friend class boost::serialization::access;
         template<class Archive>
            void serialize(Archive & ar, const unsigned int version){
               ar & run & qkind & topology_file & verbose
                  & task_init & task_sdiag & task_ham & task_opt & task_oodmrg
                  & task_vmc & task_expand & task_tononsu2 & task_prop
                  & task_schmidt & schmidt_file 
                  & restart_sweep & restart_bond & timestamp
                  & maxdets & thresh_proj & thresh_ortho & rdm_svd & outprec 
                  & nroots & guess & dbranch & maxsweep & maxbond & ctrls
                  & alg_hvec & alg_hinter & alg_hcoper 
                  & alg_renorm & alg_rinter & alg_rcoper & alg_decim & notrunc 
                  & ifdist1 & ifdistc & save_formulae & sort_formulae & save_mmtask 
                  & batchhvec & batchrenorm & batchmem
                  & cisolver & maxcycle & nbuff & damping & precond
                  & rcanon_file & rcanon2_file
                  & iomode & async_fetch & async_save & async_remove & ifnccl
                  & iroot & jroot & nsample & nprt & pthrd
                  & tosu2 & thresh_tosu2 & singlet 
                  & diagcheck
                  & savebin 
                  & fromnosym
                  & ooparams
                  & inputconf & saveconfs & loadconfs
                  & alg_rdm;
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
         bool task_init = false; // only perform initialization (converting to CTNS)
         bool task_sdiag = false; // compute sdiag via sampling
         bool task_ham = false; // compute Hij and Sij
         bool task_opt = false; // DMRG sweep optimization
         bool task_oodmrg = false; // orbital optimization
         bool task_vmc = false; 
         bool task_expand = false; // expand into FCI vectors
         bool task_tononsu2 = false; // convert SU2-MPS to non-SU2-MPS
         std::vector<int> task_prop; // mps properties
         bool task_schmidt = false; // compute schmidt values
         std::string schmidt_file = "schmidt_values"; 
         // restart
         int restart_sweep = 0;
         int restart_bond = 0;
         bool timestamp = false; 
         // conversion of sci 
         int maxdets = 10000;
         double thresh_proj = 1.e-14;
         double thresh_ortho = 1.e-8;
         double rdm_svd = 1.5;
         int outprec = 12;
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
         int alg_rdm = 0;
         int alg_decim = 1;
         bool notrunc = false;
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
         std::string rcanon_file = "";
         std::string rcanon2_file = "";
         // oper_pool
         int iomode = 0;
         bool async_fetch = false;
         bool async_save = false;
         bool async_remove = false;
         bool ifnccl = false;
         // sampling
         int iroot = 0;
         int jroot = 0;
         int nsample = 5000;
         int nprt = 10;
         double pthrd = -1.e-2;
         // su2 symmetry
         bool tosu2 = false;
         double thresh_tosu2 = 1.e-8;
         bool singlet = false; // singlet embedding
         // gpu
         bool diagcheck = false;
         // rdm and entropy
         bool savebin = false;
         // load from MPS without symmetry
         bool fromnosym = false;
         // orbital optimization
         params_oodmrg ooparams;
         // configurations
         std::string inputconf = "";
         std::string saveconfs = "";
         std::string loadconfs = "";
   };

} // input

#endif
