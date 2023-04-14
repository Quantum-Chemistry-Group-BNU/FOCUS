#ifndef INPUT_H
#define INPUT_H

#include <iostream>
#include <vector>
#include <string>
#include <set> 
#include <sstream> // istringstream

#include "../core/serialization.h"

#ifndef SERIAL
#include <boost/mpi.hpp>
#endif

namespace input{

   // SCI
   struct params_sci{
      private:
         // serialize
         friend class boost::serialization::access;
         template<class Archive>
            void serialize(Archive & ar, const unsigned int version){
               ar & run & nroots & det_seeds & nseeds & flip
                  & eps0 & eps1 & miniter & maxiter & deltaE
                  & cisolver & maxcycle & crit_v & ifpt2 & eps2 & iroot
                  & load & ci_file & cthrd;
            }
      public:
         void read(std::ifstream& istrm);
         void print() const;
      public:
         bool run = false;
         int nroots = 1;
         // initial dets
         std::set<std::set<int>> det_seeds;
         int nseeds = 0;
         bool flip = false;
         // selection threshold |HAI*CI|>eps for initial guess
         double eps0 = 1.e-2;
         // selection threshold |HAI*CI|>eps for iteration in SCI
         std::vector<double> eps1;   
         // sci
         int miniter = 0;
         int maxiter = 0;
         double deltaE = 1.e-10;
         // dvdson
         int cisolver = 1;
         int maxcycle = 100;
         double crit_v = 1.e-4;
         // pt2
         bool ifpt2 = false;
         double eps2 = 1.e-8;
         int iroot = 0;
         // io
         bool load = false;
         std::string ci_file = "ci.info"; 
         // print
         double cthrd = 1.e-2;
   };

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
                  & task_init & task_sdiag & task_ham & task_opt
                  & restart_sweep & restart_bond & timestamp
                  & maxdets & thresh_proj & thresh_ortho & rdm_svd & omp_decim
                  & nroots & guess & dbranch & maxsweep & maxbond & ctrls
                  & alg_hvec & alg_hinter & alg_renorm & alg_rinter 
                  & ifdist1 & save_formulae & sort_formulae & save_mmtask 
                  & mmorder & batchsize 
                  & cisolver & maxcycle & nbuff & damping
                  & rcanon_load & rcanon_file 
                  & iomode & async_fetch & async_save & async_remove
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
         // restart
         int restart_sweep = 0;
         int restart_bond = 0;
         bool timestamp = false; 
         // conversion of sci 
         int maxdets = 10000;
         double thresh_proj = 1.e-16;
         double thresh_ortho = 1.e-8;
         double rdm_svd = 1.5;
         bool omp_decim = false;
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
         int alg_renorm = 2;
         int alg_rinter = 0;
         bool ifdist1 = false;
         bool save_formulae = false;
         bool sort_formulae = false;
         bool save_mmtask = false;
         int mmorder = 0;
         int batchsize = 10000;
         // dvdson
         int cisolver = 1;
         int maxcycle = 30;
         int nbuff = 4; // should be greater than 2
         double damping = 1.e-10;
         // io
         bool rcanon_load = false;
         std::string rcanon_file = "rcanon_ci.info";
         // oper_pool
         int iomode = 0;
         bool async_fetch = false;
         bool async_save = false;
         bool async_remove = false;
         // sampling
         int iroot = 0;
         int nsample = 1.e5;
         int ndetprt = 10;
   };

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

   // General
   struct schedule{
      private:
         // serialize
         friend class boost::serialization::access;
         template<class Archive>
            void serialize(Archive & ar, const unsigned int version){
               ar & scratch & dtype & nelec & twoms & integral_file
                  & sci & ctns & vmc;
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
         std::string integral_file = "mole.info";
         // --- Methods --- 
         params_sci sci;
         params_ctns ctns;
         params_vmc vmc;
         // --- MPI ---
#ifndef SERIAL
         boost::mpi::communicator world;
#endif
   };

} // input

#endif
