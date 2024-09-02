#include <iostream>
#include "io/io.h"
#include "io/input.h"
#include "core/integral_io.h"
#include "ci/ci_header.h"

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
   // print the ci vectors
   if(schd.sci.ifanalysis){ 
      for(int i=0; i<nroots; i++){
         std::cout << "\nstate " << i << " energy = "  
            << std::setprecision(12) << es[i] 
            << std::endl;
         std::vector<Tm> vi(vs.col(i), vs.col(i)+dim);
         coeff_population(sci_space, vi, schd.sci.cthrd);
      }
   }
   // pt2 for single root
   if(schd.sci.ifpt2){
      int iroot = schd.sci.iroot;
      assert(iroot < nroots);
      std::vector<Tm> vi(vs.col(iroot), vs.col(iroot)+dim);
      sci::pt2_solver(schd, es[iroot], vi, sci_space, int2e, int1e, ecore);
   }
   // rdm
   if(schd.sci.rdm){
      // compute sparse_hamiltonian
      const bool Htype = tools::is_complex<Tm>();
      fci::sparse_hamiltonian<Tm> sparseH2;
      sparseH2.get_hamiltonian(sci_space, int2e, int1e, ecore, Htype);
      int k = int1e.sorb;
      int k2 = k*(k-1)/2;
      int iroot = schd.sci.iroot;
      int jroot = schd.sci.jroot;
      linalg::matrix<Tm> rdm2(k2,k2);
      std::vector<Tm> vi(vs.col(iroot), vs.col(iroot)+dim);
      std::vector<Tm> vj(vs.col(jroot), vs.col(jroot)+dim);
      fci::get_rdm2(sparseH2, sci_space, vi, vj, rdm2);
      auto Hij = fock::get_etot(rdm2, int2e, int1e, ecore);
      std::cout << "I,J=" << iroot << "," << jroot
         << " HIJ=" << std::fixed << std::setprecision(12) << Hij 
         << std::endl;
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
   if(!schd.sci.run){
      std::cout << "\ncheck input again, there is no task for SCI!" << std::endl;
      return 0;
   }

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
