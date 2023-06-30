#ifndef SPECIAL_H
#define SPECIAL_H

#include <vector>
#include <math.h>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/quadrature/gauss.hpp>

namespace fock{

   // Gamma function
   template <typename Tm>
      Tm gamma(const Tm x){
         return tgamma(x);
      }

   // https://en.wikipedia.org/wiki/Wigner_D-matrix
   inline double smalld(const double j, 
         const double mp,
         const double ms,
         const double beta){
      double r1 = j - int(j);
      double r2 = mp - int(mp);
      double r3 = ms - int(ms);
      double thresh = 1.e-10;
      // Inconsistent j,mp,ms values
      if(abs(r1-r2)>thresh || abs(r2-r3)>thresh || abs(r1-r3)>thresh){
         return 0.0;
      }
      double f1 = gamma(j+mp+1);
      double f2 = gamma(j-mp+1);
      double f3 = gamma(j+ms+1);
      double f4 = gamma(j-ms+1);
      double f1234 = sqrt(f1*f2*f3*f4);
      double cb = cos(beta/2);
      double sb = sin(beta/2);
      double val = 0.0;
      thresh = -0.1;
      for(int s=0; s<=int(2*j+0.1); s++){
         if(j+ms-s>thresh && mp-ms+s>thresh && j-mp-s>thresh){
            double sval = s;
            double sgn = pow(-1.0,mp-ms+sval);
            double a1 = 1.0/gamma(j+ms-sval+1.0);
            double a2 = 1.0/gamma(sval+1.0);
            double a3 = 1.0/gamma(mp-ms+sval+1.0);
            double a4 = 1.0/gamma(j-mp-sval+1.0);
            double cs = pow(cb,2*j+ms-mp-2.0*sval);
            double ss = pow(sb,mp-ms+2.0*sval);
            double tmp = sgn*a1*a2*a3*a4*cs*ss;
            val += tmp;
         }
      }
      val = f1234*val;
      return val;
   }

   // Gauss-Legendre quadrature
   void gen_glquad(const int n,
         std::vector<double>& xts, 
         std::vector<double>& wts){
      std::vector<double> xk(n), wk(n);
      if(n == 1){
         auto g = boost::math::quadrature::gauss<double,1>();
         auto x = g.abscissa(); 
         auto w = g.weights();
         std::copy(x.begin(), x.end(), xk.begin());
         std::copy(w.begin(), w.end(), wk.begin());
      }else if(n == 2){
         auto g = boost::math::quadrature::gauss<double,2>();
         auto x = g.abscissa(); 
         auto w = g.weights();
         std::copy(x.begin(), x.end(), xk.begin());
         std::copy(w.begin(), w.end(), wk.begin());
      }else if(n == 3){
         auto g = boost::math::quadrature::gauss<double,3>();
         auto x = g.abscissa(); 
         auto w = g.weights();
         std::copy(x.begin(), x.end(), xk.begin());
         std::copy(w.begin(), w.end(), wk.begin());
      }else if(n == 4){
         auto g = boost::math::quadrature::gauss<double,4>();
         auto x = g.abscissa(); 
         auto w = g.weights();
         std::copy(x.begin(), x.end(), xk.begin());
         std::copy(w.begin(), w.end(), wk.begin());
      }else if(n == 5){
         auto g = boost::math::quadrature::gauss<double,5>();
         auto x = g.abscissa(); 
         auto w = g.weights();
         std::copy(x.begin(), x.end(), xk.begin());
         std::copy(w.begin(), w.end(), wk.begin());
      }else if(n == 6){
         auto g = boost::math::quadrature::gauss<double,6>();
         auto x = g.abscissa(); 
         auto w = g.weights();
         std::copy(x.begin(), x.end(), xk.begin());
         std::copy(w.begin(), w.end(), wk.begin());
      }else if(n == 7){
         auto g = boost::math::quadrature::gauss<double,7>();
         auto x = g.abscissa(); 
         auto w = g.weights();
         std::copy(x.begin(), x.end(), xk.begin());
         std::copy(w.begin(), w.end(), wk.begin());
      }else if(n == 8){
         auto g = boost::math::quadrature::gauss<double,8>();
         auto x = g.abscissa(); 
         auto w = g.weights();
         std::copy(x.begin(), x.end(), xk.begin());
         std::copy(w.begin(), w.end(), wk.begin());
      }else if(n == 9){
         auto g = boost::math::quadrature::gauss<double,9>();
         auto x = g.abscissa(); 
         auto w = g.weights();
         std::copy(x.begin(), x.end(), xk.begin());
         std::copy(w.begin(), w.end(), wk.begin());
      }else if(n == 10){
         auto g = boost::math::quadrature::gauss<double,10>();
         auto x = g.abscissa(); 
         auto w = g.weights();
         std::copy(x.begin(), x.end(), xk.begin());
         std::copy(w.begin(), w.end(), wk.begin());
      }else if(n == 11){
         auto g = boost::math::quadrature::gauss<double,11>();
         auto x = g.abscissa(); 
         auto w = g.weights();
         std::copy(x.begin(), x.end(), xk.begin());
         std::copy(w.begin(), w.end(), wk.begin());
      }else if(n == 12){
         auto g = boost::math::quadrature::gauss<double,12>();
         auto x = g.abscissa(); 
         auto w = g.weights();
         std::copy(x.begin(), x.end(), xk.begin());
         std::copy(w.begin(), w.end(), wk.begin());
      }else if(n == 13){
         auto g = boost::math::quadrature::gauss<double,13>();
         auto x = g.abscissa(); 
         auto w = g.weights();
         std::copy(x.begin(), x.end(), xk.begin());
         std::copy(w.begin(), w.end(), wk.begin());
      }else if(n == 14){
         auto g = boost::math::quadrature::gauss<double,14>();
         auto x = g.abscissa(); 
         auto w = g.weights();
         std::copy(x.begin(), x.end(), xk.begin());
         std::copy(w.begin(), w.end(), wk.begin());
      }else if(n == 15){
         auto g = boost::math::quadrature::gauss<double,15>();
         auto x = g.abscissa(); 
         auto w = g.weights();
         std::copy(x.begin(), x.end(), xk.begin());
         std::copy(w.begin(), w.end(), wk.begin());
      }else{
         std::cout << "error: n is not supported! n=" << n << std::endl;
         exit(1);
      }
      xts.resize(n);
      wts.resize(n);
      int nh = n/2;
      if(n%2 == 0){
         for(int i=0; i<nh; i++){
            xts[i] = -xk[nh-1-i]; wts[i] = wk[nh-1-i];
            xts[i+nh] = xk[i]; wts[i+nh] = wk[i];
         }
      }else{
         for(int i=0; i<nh; i++){
            xts[i] = -xk[nh-i]; wts[i] = wk[nh-i];
            xts[i+nh+1] = xk[i+1]; wts[i+nh+1] = wk[i+1];
         }
         xts[nh] = xk[0]; wts[nh] = wk[0]; 
      }
   }

   void gen_s2quad(const int k, const int n, 
         const double s, const double sz, 
         std::vector<double>& xts, 
         std::vector<double>& wts){
      const bool debug = true;
      // https://pubs.acs.org/doi/epdf/10.1021/acs.jctc.7b00270
      int omega = (n<=k)? n : 2*k-n; 
      int ng = ceil( ((omega+2*s)/2.0 + 1)/2 );
      gen_glquad(ng, xts, wts);
      for(int i=0; i<ng; i++){
         xts[i] = acos(xts[i]);
         double fac = smalld(s, sz, sz, xts[i]);
         wts[i] = (s+0.5)*fac*wts[i];
      }
      // debug
      if(debug){
         std::cout << "fock::gen_s2quad: k,n=" << k << "," << n
            << " s,sz=" << s << "," << sz << std::endl;
         std::cout << " no. of gauss-legendre quadrature points=" << ng << std::endl;
         tools::print_vector(xts, "xts");
         tools::print_vector(wts, "wts");
      }
   }

} // fock

#endif
