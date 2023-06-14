#include <iostream>
#include "io/io.h"
#include "io/input.h"
#include "ci/ci_header.h"

using namespace std;
using namespace fock;

template <typename Tm>  
void FCI(const input::schedule& schd){
   // read integral
   integral::two_body<Tm> int2e;
   integral::one_body<Tm> int1e;
   double ecore;
   integral::load(int2e, int1e, ecore, schd.integral_file);
   // FCI 
   onspace fci_space;
   if(tools::is_complex<Tm>()){
      fci_space = get_fci_space(int1e.sorb, schd.nelec);
   }else{
      int na = (schd.nelec + schd.twoms)/2;
      int nb = (schd.nelec - schd.twoms)/2;
      fci_space = get_fci_space(int1e.sorb/2, na, nb);
   }
   int nroots = schd.sci.nroots;
   int dim = fci_space.size();
   vector<double> es(nroots);
   linalg::matrix<Tm> vs(dim, nroots);
   auto ci_file = schd.scratch+"/"+schd.sci.ci_file;
   fci::sparse_hamiltonian<Tm> sparseH;
   fci::ci_solver(sparseH, es, vs, fci_space, int2e, int1e, ecore);
   fci::ci_save(fci_space, es, vs, ci_file);
   sparseH.dump(schd.scratch+"/sparseH.bin");
   for(int i=0; i<nroots; i++){
      std::cout << "\nstate " << i << " energy = "  
                << std::setprecision(12) << es[i] 
                << std::endl;
      std::vector<Tm> vi(vs.col(i), vs.col(i)+dim);
      coeff_population(fci_space, vi, schd.sci.cthrd);
   }
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
   io::create_scratch(schd.scratch);
   // we will use Tm to control Hnr/Hrel 
   if(schd.dtype == 0){
      FCI<double>(schd);
   }else if(schd.dtype == 1){
      FCI<complex<double>>(schd);
   }
   tools::finish("FCI");
   return 0;
}
