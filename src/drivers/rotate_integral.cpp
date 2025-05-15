#include <iostream>
#include "io/io.h"
#include "io/input.h"
#include "core/integral_io.h"
#include "core/integral_rotate.h"
#include "ci/ci_header.h"

using namespace std;
using namespace fock;

template <typename Tm>  
void rotate_integral(const input::schedule& schd){

   // read integral
   integral::two_body<Tm> int2e;
   integral::one_body<Tm> int1e;
   double ecore;
   integral::load(int2e, int1e, ecore, schd.integral_file);
   assert(schd.sorb == int1e.sorb);

   // load urot matrix
   int norb = int1e.sorb/2;
   std::vector<linalg::matrix<Tm>> umat(2);
   umat[0].resize(norb,norb);
   umat[0].load_txt(schd.scratch+"/"+schd.urot_file);
   umat[1] = umat[0];
   std::cout << "\n|U^H*U-I|=" << check_orthogonality(umat[1]) << std::endl;

   // rotate integrals
   integral::one_body<Tm> int1e_new;
   integral::two_body<Tm> int2e_new;
   rotate_spatial(int1e, int1e_new, umat);
   rotate_spatial(int2e, int2e_new, umat);

   // save integrals
   std::string fname = schd.integral_file+".rotated";
   integral::save(int2e_new, int1e_new, ecore, fname);
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
      rotate_integral<double>(schd);
   }else if(schd.dtype == 1){
      rotate_integral<complex<double>>(schd);
   }
   tools::finish("rotate_integral");
   return 0;
}
