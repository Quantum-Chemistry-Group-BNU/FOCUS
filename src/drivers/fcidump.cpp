#include <iostream>
#include "io/io.h"
#include "io/input.h"
#include "core/integral_io.h"
#include "ci/ci_header.h"

using namespace std;
using namespace fock;

template <typename Tm>  
void genFCIDUMP(const input::schedule& schd){

   // read integral
   integral::two_body<Tm> int2e;
   integral::one_body<Tm> int1e;
   double ecore;
   integral::load(int2e, int1e, ecore, schd.integral_file);
   assert(schd.sorb == int1e.sorb);

   // save integrals
   std::string fname = schd.integral_file+".FCIDUMP";
   std::ofstream ostrm(fname);
   if(!ostrm){
      std::cout << "failed to open " << fname << std::endl;
      exit(1);
   }
   auto t0 = tools::get_time();
   std::cout << "\ngenFCIDUMP file = " << fname << std::endl;

   // save title
   int norb = int1e.sorb/2;
   ostrm << "&FCI NORB=  " << norb << ",NELEC=" << schd.nelec << ",MS2=" << schd.twom << std::endl;
   ostrm << " ORBSYM=";
   for(int i=0; i<norb; i++){
      ostrm << "1,";
   }
   ostrm << endl;
   ostrm << " ISYM=1," << endl;
   ostrm << "&END" << endl;

   // save int2e,int1e,ecore
   const double thresh = 1.e-16;
   ostrm << std::fixed << std::setprecision(16);
  
   // int2e[i,j,k,l] = [ij|kl]
   size_t npair = norb*(norb+1)/2;
   size_t nquad = npair*(npair+1)/2;
   size_t counter = 0;
   for(int i=0; i<norb; i++){
      for(int j=0; j<=i; j++){
         for(int k=0; k<=i; k++){
            int lmax = (k==i)? j : k;
            for(int l=0; l<=lmax; l++){
               counter += 1;
               Tm val = int2e.get(2*i,2*k+1,2*j,2*l+1); // [ij|kl] = <iAkB||jAlB>
               if(std::abs(val)<thresh) continue; 
               ostrm << val 
                  << "   " << i+1 
                  << "   " << j+1 
                  << "   " << k+1
                  << "   " << l+1
                  << std::endl;
            }
         }
      }
   }
   assert(counter == nquad);

   // int1e[i,j]
   for(int i=0; i<norb; i++){
      for(int j=0; j<=i; j++){
         Tm val = int1e.get(2*i,2*j);
         if(std::abs(val)<thresh) continue;
         ostrm << val << "   " << i+1 << "   " << j+1 << "   0   0" << std::endl; 
      }
   }

   // ecore
   ostrm << ecore << "   0   0   0   0" << std::endl;
   ostrm.close();

   auto t1 = tools::get_time();
   tools::timing("genFCIDUMP", t0, t1);
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
      genFCIDUMP<double>(schd);
   }else{
      tools::exit("error: fcidump only for dtype=0");
   }
   tools::finish("fcidump");
   return 0;
}
