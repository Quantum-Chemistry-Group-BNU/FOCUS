#ifndef ANALYSIS_H
#define ANALYSIS_H

#include <iomanip>
#include <vector>
#include <complex>
#include "onspace.h"

namespace fock{

template <typename Tm>	
void coeff_population(const onspace& space, 
	   	      const std::vector<Tm>& civec, 
	   	      const double thresh=1.e-2){
   std::cout << "\nfock::coeff_population dim=" << space.size() << " thresh=" << thresh << std::endl;
   std::cout << "  i-th  /  idx  /  coeff  /  rank  /  seniority  /  onstate  /  nelec" << std::endl;
   std::cout << std::setprecision(12);
   double ne = 0.0, na = 0.0, nb = 0.0;
   double pi, psum = 0.0, psum1 = 0.0, Sd = 0.0;
   const double cutoff = 1.e-12;
   std::vector<int> idx;
   idx = tools::sort_index_abs(civec, 1);
   int j = 0;
   for(const auto& i : idx){ 
      pi = std::norm(civec[i]);
      psum += pi;
      Sd += (pi < cutoff)? 0.0 : -pi*log2(pi);
      // Measurement in Z-basis 
      ne += pi*space[i].nelec();
      na += pi*space[i].nelec_a();
      nb += pi*space[i].nelec_b();
      if(std::abs(civec[i]) > thresh){ 
	 std::cout << std::setw(8) << j << " : " << std::setw(8) << i << "  ";
	 std::cout << std::fixed << std::scientific << std::setprecision(2) << std::showpos << civec[i] << std::noshowpos;
	 std::cout << std::fixed << std::setw(3) << space[i].diff_num(space[idx[0]])/2 << " " 
	           << std::fixed << std::setw(3) << space[i].norb_single() << "  "
	           << space[i] << " ("
                   << space[i].nelec() << ","
                   << space[i].nelec_a() << ","
                   << space[i].nelec_b() << ") "
                   << std::endl;
	 psum1 += pi;
	 j++;
      }
   }
   std::cout << "psum=" << psum 
	     << " psum1=" << psum1 
	     << " Sd=" << Sd
             << std::endl;
   std::cout << "<Ne,Na,Nb>=" << ne << "," 
	     << na << "," 
	     << nb << std::endl; 
}

inline double entropy(const std::vector<double>& p, 
	              const double cutoff=1.e-12){
   double psum = 0.0, ssum = 0.0;
   for(const auto& pi : p){
      if(pi < cutoff) continue;
      psum += pi;
      ssum -= pi*log2(pi);
   }
   std::cout << "fock::entropy : " 
        << std::defaultfloat << std::setprecision(12)
        << "psum=" << psum << " SvN=" << ssum << std::endl; 
   return ssum;
}

template <typename Tm>
double coeff_entropy(const std::vector<Tm>& coeff, 
	       	     const double cutoff=1.e-12){
   double psum = 0.0, ssum = 0.0;
   for(const auto& ci : coeff){
      double pi = std::norm(ci);
      if(pi < cutoff) continue;
      psum += pi;
      ssum += -pi*log2(pi);
   }
   return ssum;
}

} // fock

#endif
