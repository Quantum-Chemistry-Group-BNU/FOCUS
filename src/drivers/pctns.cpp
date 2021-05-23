#include <iostream>
#include <iomanip>
#include <string>
#include "../io/input.h"
#include "../ci/ci_header.h"
#include "../ctns/ctns_header.h"

using namespace std;
using namespace fock;

template <typename Km>  
int PCTNS(const input::schedule& schd){
   int rank = 0; 
#ifndef SERIAL
   rank = schd.world.rank();
#endif
   // consistency check for dtype
   using Tm = typename Km::dtype;
   if((schd.dtype == 1) != tools::is_complex<Tm>()){
      tools::exit("error: inconsistent dtype in CTNS!");
   }
   // read integral
   integral::two_body<Tm> int2e;
   integral::one_body<Tm> int1e;
   double ecore;
   if(rank == 0) integral::load(int2e, int1e, ecore, schd.integral_file);
#ifndef SERIAL
   boost::mpi::broadcast(schd.world, int1e, 0);
   boost::mpi::broadcast(schd.world, int2e, 0);
   boost::mpi::broadcast(schd.world, ecore, 0);
#endif
   // -- CTNS --- 
   ctns::comb<Km> icomb;
   if(rank == 0){
      // dealing with topology 
      icomb.topo.read(schd.ctns.topology_file);
      icomb.topo.print();
      // initialize RCF 
      const bool ifortho = true; 
      auto rcanon_file = schd.scratch+"/"+schd.ctns.rcanon_file;
      if(!schd.ctns.load){
         // from SCI wavefunction
         onspace sci_space;
         vector<double> es;
         vector<vector<Tm>> vs;
         auto ci_file = schd.scratch+"/"+schd.sci.ci_file;	   
         fci::ci_load(sci_space, es, vs, ci_file);
         // truncate CI coefficients
         fci::ci_truncate(sci_space, vs, schd.ctns.maxdets, ifortho);
         ctns::rcanon_init(icomb, sci_space, vs, schd.ctns.thresh_proj);
         ctns::rcanon_save(icomb, rcanon_file);
      }else{
         ctns::rcanon_load(icomb, rcanon_file);
      }
      ctns::rcanon_check(icomb, schd.ctns.thresh_ortho, ifortho);
   }
#ifndef SERIAL
   boost::mpi::broadcast(schd.world, icomb, 0);
   icomb.world = schd.world;
#endif
   // optimization from current RCF
   if(schd.ctns.task == "opt"){
      ctns::sweep_opt(icomb, int2e, int1e, ecore, schd);
      if(rank == 0){
         auto rcanon_file = schd.scratch+"/rcanon_new.info"; 
         ctns::rcanon_save(icomb, rcanon_file);
      }
   }else if(schd.ctns.task == "ham"){
      auto Hij = ctns::get_Hmat(icomb, int2e, int1e, ecore, schd.scratch);
      if(rank == 0){
         Hij.print("Hij",8);
         auto Sij = ctns::get_Smat(icomb);
         Sij.print("Sij");
      }
   }
   return 0;	
}

int main(int argc, char *argv[]){
   int rank = 0; 
#ifndef SERIAL
   // setup MPI environment 
   boost::mpi::environment env{argc, argv};
   boost::mpi::communicator world;
   rank = world.rank();
#endif
   if(rank == 0) tools::license();
   // read input
   string fname;
   if(argc == 1){
      tools::exit("error: no input file is given!");
   }else{
      fname = argv[1];
      cout << "\ninput file = " << fname << endl;
   }
   input::schedule schd;
   if(rank == 0) schd.read(fname);
#ifndef SERIAL
   boost::mpi::broadcast(world, schd, 0);
   schd.world = world;
#endif
   // setup scratch directory
   if(rank > 0) schd.scratch += to_string(rank);
   schd.create_scratch();
   // we will use Tm to control Hnr/Hrel 
   int info = 0;
   if(schd.ctns.kind == "rN"){
      info = PCTNS<ctns::kind::rN>(schd);
   }else if(schd.ctns.kind == "rNSz"){
      info = PCTNS<ctns::kind::rNSz>(schd);
   }else if(schd.ctns.kind == "cN"){
      info = PCTNS<ctns::kind::cN>(schd);
   }else if(schd.ctns.kind == "cNSz"){
      info = PCTNS<ctns::kind::cNSz>(schd);
   }else if(schd.ctns.kind == "cNK"){
      info = PCTNS<ctns::kind::cNK>(schd);
   }else{
      tools::exit("error: no such kind for ctns!");
   } // kind
   if(rank > 0) schd.remove_scratch();
   return info;
}
