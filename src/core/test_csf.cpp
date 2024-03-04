#include "csf.h"
#include "tests_core.h"

using namespace std;
using namespace fock;

int tests::test_csf(){
   cout << endl;
   cout << tools::line_separator << endl;	
   cout << "tests::test_csf" << endl;
   cout << tools::line_separator << endl;	

   csfstate state("00udu2");
   auto exp = state.Sdiag_exact();
   auto pr = state.random();
   std::cout << "random: " << pr.first << " : " << pr.second << std::endl;
   state.Sdiag_sample();

   csfstate state2("2udu00");
   auto exp2 = state2.Sdiag_exact();
   auto pr2 = state2.random();
   std::cout << "random: " << pr2.first << " : " << pr2.second << std::endl;
   state2.Sdiag_sample();

   int k=3, n=3, ts=1;
   //int k=6, n=6, ts=1;
   //int k=35, n=2, ts=0;

   std::cout << fock::dim_csf_space(k,n,ts) << std::endl;

   fock::onspace det_space = fock::get_fci_space(k,(n+ts)/2,(n-ts)/2);
   fock::csfspace csf_space = fock::get_csf_space(k,n,ts);
   size_t dim = csf_space.size();
   for(int i=0; i<dim; i++){
      std::cout << "i=" << i << " csf=" << csf_space[i] << std::endl;
      auto ninter = csf_space[i].intermediate_narray();
      tools::print_vector(ninter,"ninter");
      auto tsinter = csf_space[i].intermediate_tsarray();
      tools::print_vector(tsinter,"tsinter");

      double sum = 0.0;
      for(int j=0; j<det_space.size(); j++){
         double coeff = csf_space[i].det_coeff(det_space[j]);
         if(std::abs(coeff)<1.e-10) continue;
         std::cout << " j=" << j << " det=" << det_space[j]
            << " coeff=" << coeff << std::endl;
         sum += coeff*coeff; 
      }
      std::cout << " sum=" << std::scientific << sum << std::endl;
      if(abs(sum-1.0)>1.e-10){
         std::cout << "error: deviate from 1! sum-1.0=" << sum-1.0 << std::endl;
         exit(1);
      }

      auto detexpansion = csf_space[i].to_det();
      const auto& dets = detexpansion.first;
      const auto& coeffs = detexpansion.second;
      for(int j=0; j<dets.size(); j++){
         if(std::abs(coeffs[j])<1.e-10) continue;
         std::cout << " j=" << j << " det=" << dets[j] 
            << " coeff=" << coeffs[j] << std::endl;
      }
   }

   //fock::csfstate csf("duudud0d2uu");
   std::string str = "uuuuuddddduuuuuddddd";
   fock::csfstate csf(str);
   std::cout << "\nSTATE=" << csf << std::endl;
   auto detexpansion = csf.to_det();
   const auto& dets = detexpansion.first;
   const auto& coeffs = detexpansion.second;
   double sum = 0.0;
   for(int j=0; j<dets.size(); j++){
      if(std::abs(coeffs[j])<1.e-10) continue;
      std::cout << " j=" << j << " det=" << dets[j] 
         << " coeff=" << coeffs[j] << std::endl;
      sum += coeffs[j]*coeffs[j];
   }
   std::cout << "sum=" << sum << std::endl;

   // sample det from csf
   int nsample = 10000;
   csf.Sdiag_exact();
   csf.Sdiag_sample(nsample);

   return 0;
}
