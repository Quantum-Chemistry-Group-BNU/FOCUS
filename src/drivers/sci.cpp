#include <iostream>
#include "../io/io.h"
#include "../io/input.h"
#include "../ci/ci_header.h"

using namespace std;
using namespace fock;

template <typename Tm>  
void SCI(const input::schedule& schd){
   // read integral
   integral::two_body<Tm> int2e;
   integral::one_body<Tm> int1e;
   double ecore;
   integral::load(int2e, int1e, ecore, schd.integral_file);
   // SCI
   int nroots = schd.sci.nroots;
   onspace sci_space;
   vector<double> es;
   linalg::matrix<Tm> vs;
   auto ci_file = schd.scratch+"/"+schd.sci.ci_file;
   if(!schd.sci.load){
      fci::sparse_hamiltonian<Tm> sparseH;
      sci::ci_solver(schd, sparseH, es, vs, sci_space, int2e, int1e, ecore);
      fci::ci_save(sci_space, es, vs, ci_file);
   }else{
      fci::ci_load(sci_space, es, vs, ci_file);
   }
   int dim = sci_space.size(); 
   for(int i=0; i<nroots; i++){
      std::cout << "\nstate " << i << " energy = "  
                << std::setprecision(12) << es[i] 
                << std::endl;
      std::vector<Tm> vi(vs.col(i), vs.col(i)+dim);
      coeff_population(sci_space, vi, schd.sci.cthrd);
   }
   // pt2 for single root
   if(schd.sci.ifpt2){
      int iroot = schd.sci.iroot;
      assert(iroot < nroots);
      std::vector<Tm> vi(vs.col(iroot), vs.col(iroot)+dim);
      sci::pt2_solver(schd, es[iroot], vi, sci_space, int2e, int1e, ecore);
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
      SCI<double>(schd);
   }else if(schd.dtype == 1){
      SCI<complex<double>>(schd);
   }
   tools::finish("SCI");
   return 0;
}
