#include <iostream>
#include <iomanip>
#include "io/io.h"
#include "io/input.h"
#include "core/tools.h"
#include "core/onspace.h"
#include "core/integral.h"
#include "core/hamiltonian.h"
#include "core/matrix.h"
#include "core/linalg.h"
#include "core/analysis.h"

using namespace std;
using namespace fock;

template <typename Tm>  
void ED(const input::schedule& schd){
   std::cout << "\nExact Diagonalization:" << std::endl;
   
   // read integral
   integral::two_body<Tm> int2e;
   integral::one_body<Tm> int1e;
   double ecore;
   integral::load(int2e, int1e, ecore, schd.integral_file);
   
   // ED: define FCI space 
   onspace fci_space;
   if(schd.nelec == -1 and schd.twoms == -1){
      fci_space = get_fci_space(int1e.sorb); // Fock space
      std::cout << "\nGenerate Fock space for"
         << " k=" << int1e.sorb 
         << " dim=" << fci_space.size()
         << std::endl; 
   }else if(schd.nelec >= 0 and schd.twoms == -1){
      fci_space = get_fci_space(int1e.sorb, schd.nelec); // N-electron Hilbert space
      std::cout << "\nGenerate Hilbert space for" 
         << " (k,n)=" << int1e.sorb << "," << schd.nelec 
         << " dim=" << fci_space.size()
         << std::endl; 
   }else if(schd.nelec >= 0 and schd.twoms >= 0){
      int na = (schd.nelec + schd.twoms)/2;
      int nb = (schd.nelec - schd.twoms)/2;
      fci_space = get_fci_space(int1e.sorb/2, na, nb); // (Na,Nb)-electron Hilbert space
      std::cout << "\nGenerate Hilbert space for"
         << " (ks,na,nb)=" << int1e.sorb/2 << "," << na << "," << nb
         << " dim=" << fci_space.size()
         << std::endl;
   }

   // construct H by simple Slater-Condon rule
   auto H = get_Hmat(fci_space, int2e, int1e, ecore);
   double sdiff = H.diff_hermitian();
   std::cout << "check ||H-H.dagger||=" << std::scientific << std::setprecision(3) << sdiff
      << std::defaultfloat << std::endl;
   const double thresh = 1.e-10;
   if(sdiff > thresh){
      std::cout << "error: sdiff is greater than thresh=" << thresh << std::endl;
      exit(1);
   }
   vector<double> es(H.rows());
   auto vs(H);
   linalg::eig_solver(H, es, vs); // Hc=ce

   // print
   int nroots = schd.sci.nroots;
   int dim = fci_space.size();
   std::cout << "\nsummary of FCI energies:" << std::endl;
   for(int i=0; i<nroots; i++){
      std::cout << " state " << i << " energy = "
         << std::setprecision(12) << es[i] 
         << std::endl;
   }
   for(int i=0; i<nroots; i++){
      std::cout << "\nstate " << i << " energy = "
         << std::setprecision(12) << es[i] 
         << std::endl;
      std::vector<Tm> vi(vs.col(i), vs.col(i)+dim);
      const int iop = 0;
      coeff_population(fci_space, vi, schd.sci.cthrd, iop);
   } // i
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
      ED<double>(schd);
   }else if(schd.dtype == 1){
      ED<complex<double>>(schd);
   }
   tools::finish("ED");
   return 0;
}
