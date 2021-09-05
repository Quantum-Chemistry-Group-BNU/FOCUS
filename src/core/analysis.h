#ifndef ANALYSIS_H
#define ANALYSIS_H

#include <iomanip>
#include <vector>
#include <complex>
#include "onspace.h"

namespace fock{

template <typename Tm>	
void coeff_analysis(const std::vector<Tm>& civec,
		    const double thresh=1.e-8){
   std::cout << "fock::coeff_analysis dim=" << civec.size() << std::endl;
   std::map<int,int,std::greater<int>> bucket;
   std::map<int,double> population;
   int size = civec.size();
   for(int i=0; i<size; i++){
      double aval = std::abs(civec[i]);
      if(aval > thresh){
         int n = floor(log10(aval));
         bucket[n] += 1;
	 population[n] += std::norm(civec[i]);
      }
   }
   // print statistics by magnitude
   double accum = 0.0;
   for(const auto& pr : bucket){
      int n = pr.first;
      double pop = population[n];
      accum += pop;
      std::cout << " |c[i]| in 10^" << std::showpos << n+1 << "-10^" << n << " :"
               << " pop=" << std::defaultfloat << std::noshowpos << std::fixed 
	       << std::setprecision(3) << pop
               << " accum=" << std::defaultfloat << std::noshowpos << std::fixed 
	       << std::setprecision(3) << accum
	       << " counts=" << pr.second
               << std::endl;
   }
}

template <typename Tm>	
void coeff_population(const onspace& space, 
	   	      const std::vector<Tm>& civec, 
	   	      const double thresh=1.e-2){
   std::cout << "\nfock::coeff_population dim=" << space.size() << " thresh=" << thresh << std::endl;
   std::cout << "  i-th  /  idx  /  coeff  /  pop  /  rank  /  seniority  /  onstate  /  nelec" << std::endl;
   std::cout << std::setprecision(12);
   double ne = 0.0, na = 0.0, nb = 0.0;
   double ne2 = 0.0, na2 = 0.0, nb2 = 0.0;
   double pi, psum0 = 0.0, psum = 0.0, Sd = 0.0;
   const double cutoff = 1.e-12;
   std::vector<int> idx;
   idx = tools::sort_index_abs(civec, 1);
   int j = 0;
   for(const auto& i : idx){ 
      pi = std::norm(civec[i]);
      psum0 += pi;
      Sd += (pi < cutoff)? 0.0 : -pi*log2(pi);
      // Measurement in Z-basis
      int nelec = space[i].nelec(); ne += pi*nelec; ne2 += pi*nelec*nelec;
      int nelec_a = space[i].nelec_a(); na += pi*nelec_a; na2 += pi*nelec_a*nelec_a;
      int nelec_b = space[i].nelec_b(); nb += pi*nelec_b; nb2 += pi*nelec_b*nelec_b;
      if(std::abs(civec[i]) > thresh){ 
	 std::cout << std::setw(8) << j << " : " << std::setw(8) << i << "   ";
	 std::cout << std::fixed << std::scientific << std::setprecision(2) 
		   << std::showpos << civec[i] << std::noshowpos;
	 std::cout << std::fixed << std::setw(6) << std::setprecision(2) << pi*100;
	 std::cout << std::fixed << std::setw(3) << space[i].diff_num(space[idx[0]])/2
	           << std::fixed << std::setw(3) << space[i].norb_single() << "  "
	           << space[i] << " ("
                   << space[i].nelec() << ","
                   << space[i].nelec_a() << ","
                   << space[i].nelec_b() << ") "
                   << std::endl;
	 psum += pi;
	 j++;
      }
   }
   std::cout << std::setprecision(6);
   std::cout << "psum=" << psum << " psum0=" << psum0 << " Sd=" << Sd << std::endl;
   std::cout << "<Ne>=" << ne << " std=" << std::pow(std::abs(ne2-ne*ne),0.5) << std::endl;
   std::cout << "<Na>=" << na << " std=" << std::pow(std::abs(na2-na*na),0.5) << std::endl;
   std::cout << "<Nb>=" << nb << " std=" << std::pow(std::abs(nb2-nb*nb),0.5) << std::endl;
   coeff_analysis(civec);
}
template <typename Tm>	
void coeff_population(const onspace& space, 
	   	      const Tm* civec_ptr,
	   	      const double thresh=1.e-2){
   int size = space.size();
   std::vector<Tm> civec(civec_ptr, civec_ptr+size);
   coeff_population(space, civec, thresh);
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
