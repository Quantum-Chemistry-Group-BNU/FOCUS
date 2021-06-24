#include <iostream>
#include "../io/input.h"
#include "../ci/ci_header.h"

using namespace std;
using namespace fock;

template <typename Tm>  
int SCI(const input::schedule& schd){
   // read integral
   integral::two_body<Tm> int2e;
   integral::one_body<Tm> int1e;
   double ecore;
   integral::load(int2e, int1e, ecore, schd.integral_file);
   // SCI
   int nroot = schd.sci.nroots;
   onspace sci_space;
   vector<double> es(nroot, 0.0);
   vector<vector<Tm>> vs(nroot);
   auto ci_file = schd.scratch+"/"+schd.sci.ci_file;
   if(!schd.sci.load){
      fci::sparse_hamiltonian<Tm> sparseH;
      sci::ci_solver(schd, sparseH, es, vs, sci_space, int2e, int1e, ecore);
      fci::ci_save(sci_space, es, vs, ci_file);
   }else{
      fci::ci_load(sci_space, es, vs, ci_file);
   }
   for(int i=0; i<nroot; i++){
      coeff_population(sci_space, vs[i]);
   }
   // pt2 for single root
   if(schd.sci.ifpt2){
      int istate = schd.sci.istate;
      assert(istate < nroot);
      sci::pt2_solver(schd, es[istate], vs[istate], sci_space, int2e, int1e, ecore);
   }
   return 0;
}

int main(int argc, char *argv[]){
   tools::license();
   // read input
   string fname;
   if(argc == 1){
      tools::exit("error: no input file is given!");
   }else{
      fname = argv[1];
      cout << "\ninput file = " << fname << endl;
   }
   input::schedule schd;
   schd.read(fname);
   schd.create_scratch();
   // we will use Tm to control Hnr/Hrel 
   int info = 0;
   if(schd.dtype == 0){
      info = SCI<double>(schd);
   }else if(schd.dtype == 1){
      info = SCI<complex<double>>(schd);
   }
   return info;
}
