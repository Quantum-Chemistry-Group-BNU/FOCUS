#include <iostream>
#include "../io/input.h"
#include "../ci/fci.h"
#include "../ci/fci_rdm.h"
#include "../ci/sci.h"
#include "../ci/sci_pt2.h"

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
   vector<double> es(nroot, 0.0);
   onspace sci_space;
   vector<vector<Tm>> vs(nroot);
   if(!schd.sci.load){
      fci::sparse_hamiltonian<Tm> sparseH;
      sci::ci_solver(schd, sparseH, es, vs, sci_space, int2e, int1e, ecore);
      // pt2 for single root
      if(schd.sci.ifpt2){
         sci::pt2_solver(schd, es[0], vs[0], sci_space, int2e, int1e, ecore);
      }
      fci::ci_save(sci_space, vs);
   }else{
      fci::ci_load(sci_space, vs);
   }
   for(int i=0; i<nroot; i++){
      coeff_population(sci_space, vs[i]);
   }
   return 0;
}

int main(int argc, char *argv[]){
   tools::license();
   // read input
   string fname;
   if(argc == 0){
      cout << "error: no input file is given!" << endl;
      exit(1);
   }else{
      fname = argv[1];
      cout << "\ninput file = " << fname << endl;
   }
   input::schedule schd;
   schd.read(fname);
   // we will use Tm to control Hnr/Hrel 
   int info = 0;
   if(schd.dtype == 0){
      info = SCI<double>(schd);
   }else if(schd.dtype == 1){
      info = SCI<complex<double>>(schd);
   }
   return info;
}
