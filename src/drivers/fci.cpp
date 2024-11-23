#include <iostream>
#include "io/io.h"
#include "io/input.h"
#include "core/integral_io.h"
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
   assert(schd.sorb == int1e.sorb);
   
   // FCI 
   onspace fci_space;
   if(tools::is_complex<Tm>()){
      fci_space = get_fci_space(int1e.sorb, schd.nelec);
   }else{
      int na = (schd.nelec + schd.twom)/2;
      int nb = (schd.nelec - schd.twom)/2;
      fci_space = get_fci_space(int1e.sorb/2, na, nb);
   }
   int nroots = schd.ci.nroots;
   int dim = fci_space.size();
   vector<double> es(nroots);
   linalg::matrix<Tm> vs(dim, nroots);
   auto ci_file = schd.scratch+"/"+schd.ci.ci_file;
   fci::sparse_hamiltonian<Tm> sparseH;
   fci::ci_solver(sparseH, es, vs, fci_space, int2e, int1e, ecore);
   fci::ci_save(fci_space, es, vs, ci_file);
   sparseH.dump(schd.scratch+"/sparseH.bin");
   
   // print the ci vectors
   if(schd.ci.ifanalysis){ 
      for(int i=0; i<nroots; i++){
         std::cout << "\nstate " << i << " energy = "  
            << std::fixed << std::setprecision(12) << es[i] 
            << std::endl;
         std::vector<Tm> vi(vs.col(i), vs.col(i)+dim);
         coeff_population(fci_space, vi, schd.ci.cthrd);
      }
   }
   
   // rdm
   if(schd.ci.rdm){
      int k = int1e.sorb;
      int k2 = k*(k-1)/2;
      linalg::matrix<Tm> rdm1(k,k), rdm2(k2,k2);
      assert(schd.ci.iroot < nroots and schd.ci.jroot < nroots);
      fci::get_rdm12(fci_space, vs, schd.ci.iroot, schd.ci.jroot, int2e, int1e, ecore, rdm1, rdm2, schd.scratch);
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
