#ifndef VMC_MEAN_H
#define VMC_MEAN_H

namespace vmc{

   // <X>
   template <typename Tm>
   Tm get_mean(const int nsample,
               const Tm* x){
      double fac = 1.0/nsample;
      Tm mean = 0.0;
      for(int i=0; i<nsample; i++){
         mean += x[i]*fac;
      }
      return mean;
   }

   // <XY>
   template <typename Tm>
   Tm get_mean(const int nsample,
               const Tm* x,
               const Tm* y){
      double fac = 1.0/nsample;
      Tm mean = 0.0;
      for(int i=0; i<nsample; i++){
         mean += x[i]*y[i]*fac;
      }
      return mean;
   }
   // <XY>
   inline std::complex<double> get_mean(const int nsample,
               const double* x,
               const std::complex<double>* y){
      double fac = 1.0/nsample;
      std::complex<double> mean = 0.0;
      for(int i=0; i<nsample; i++){
         mean += x[i]*y[i]*fac;
      }
      return mean;
   }

} // vmc

#endif
