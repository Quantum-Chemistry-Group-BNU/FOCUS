#ifndef SWEEP_DATA_H
#define SWEEP_DATA_H

#include <vector>
#include "../io/input.h"

namespace ctns{

// timing
struct dot_timing{
   void print(){
      std::cout << " T(load)  = " << std::scientific << std::setprecision(2) << dt0 << " S"
   	        << "  per = " << std::defaultfloat << dt0/dt*100 
		<< "  per(accum) = " << dt0/dt*100 
		<< std::endl;
      std::cout << " T(hdiag) = " << std::scientific << std::setprecision(2) << dt1 << " S"
   	        << "  per = " << std::defaultfloat << dt1/dt*100 
		<< "  per(accum) = " << (dt0+dt1)/dt*100 
		<< std::endl;
      std::cout << " T(dvdsn) = " << std::scientific << std::setprecision(2) << dt2 << " S"
   	        << "  per = " << std::defaultfloat << dt2/dt*100 
		<< "  per(accum) = " << (dt0+dt1+dt2)/dt*100 
		<< std::endl;
      std::cout << " T(decim) = " << std::scientific << std::setprecision(2) << dt3 << " S"
   	        << "  per = " << std::defaultfloat << dt3/dt*100 
		<< "  per(accum) = " << (dt0+dt1+dt2+dt3)/dt*100 
		<< std::endl;
      std::cout << " T(guess) = " << std::scientific << std::setprecision(2) << dt4 << " S"
   	        << "  per = " << std::defaultfloat << dt4/dt*100 
		<< "  per(accum) = " << (dt0+dt1+dt2+dt3+dt4)/dt*100 
		<< std::endl;
      std::cout << " T(renrm) = " << std::scientific << std::setprecision(2) << dt5 << " S"
   	        << "  per = " << std::defaultfloat << dt5/dt*100 
		<< "  per(accum) = " << (dt0+dt1+dt2+dt3+dt4+dt5)/dt*100 
		<< std::endl;
      std::cout << " T(save)  = " << std::scientific << std::setprecision(2) << dt6 << " S"
   	        << "  per = " << std::defaultfloat << dt6/dt*100 
		<< "  per(accum) = " << (dt0+dt1+dt2+dt3+dt4+dt5+dt6)/dt*100 
		<< std::endl;
   }
   void analysis(){
      dt  = tools::get_duration(t1-t0); // total
      dt0 = tools::get_duration(ta-t0); // t(procs)
      dt1 = tools::get_duration(tb-ta); // t(hdiag)
      dt2 = tools::get_duration(tc-tb); // t(dvdsn)
      dt3 = tools::get_duration(td-tc); // t(decim)
      dt4 = tools::get_duration(te-td); // t(guess)
      dt5 = tools::get_duration(tf-te); // t(renrm)
      dt6 = tools::get_duration(t1-tf); // t(save)
      std::cout << "##### timing_local: " << std::scientific << std::setprecision(2)
                << dt << " S #####" << std::endl;
      this->print();
   }
   void accumulate(const dot_timing& timer){
      dt  += timer.dt;
      dt0 += timer.dt0;
      dt1 += timer.dt1;
      dt2 += timer.dt2;
      dt3 += timer.dt3;
      dt4 += timer.dt4;
      dt5 += timer.dt5;
      dt6 += timer.dt6;
      std::cout << "##### timing_global: " << std::scientific << std::setprecision(2) 
                << dt << " S #####" << std::endl;
      this->print();
   }
public:
   using Tm = std::chrono::high_resolution_clock::time_point;
   Tm t0;
   Tm ta; // ta-t0: t(load) 
   Tm tb; // tb-ta: t(hdiag)
   Tm tc; // tc-ta: t(dvdson)
   Tm td; // td-tc: t(decim)
   Tm te; // te-td: t(guess)
   Tm tf; // tf-te: t(renrm)
   Tm t1; // t1-tf: t(save)
   double dt=0, dt0=0, dt1=0, dt2=0, dt3=0, dt4=0, dt5=0, dt6=0;
};

// computed results at a given dot	
struct dot_result{
   std::vector<double> eopt; // eopt[nroots]
   double dwt;
   int deff;
   int nmvp;
};

struct sweep_data{
   // constructor
   sweep_data(const std::vector<directed_bond>& sweep_seq,
              const int _nroots,
              const bool _guess,
              const int _maxsweep,
              const std::vector<input::params_sweep>& _ctrls,
              const int _dbranch,
	      const double _rdm_vs_svd){
      seq = sweep_seq;
      seqsize = sweep_seq.size();
      guess = _guess;
      nroots = _nroots;
      maxsweep = _maxsweep;
      ctrls = _ctrls;
      dbranch = _dbranch;
      rdm_vs_svd = _rdm_vs_svd;
      // sweep results
      opt_result.resize(maxsweep);
      opt_timing.resize(maxsweep);
      for(int i=0; i<maxsweep; i++){
	 opt_result[i].resize(seqsize);
	 opt_timing[i].resize(seqsize);
	 for(int j=0; j<seqsize; j++){
	    opt_result[i][j].eopt.resize(nroots);
	 }
      }
      min_result.resize(maxsweep);
      t_total.resize(maxsweep);
   }
   // print control parameters
   void print_ctrls(const int isweep) const{
      const auto& ctrl = ctrls[isweep];
      ctrl.print();
   }
   // print optimized energies
   void print_eopt(const int isweep, const int ibond) const{
      const auto& eopt = opt_result[isweep][ibond].eopt;
      int dots = ctrls[isweep].dots;
      for(int i=0; i<nroots; i++){
         std::cout << " optimized energies:"
		   << " isweep=" << isweep 
		   << " dots=" << dots
		   << " ibond=" << ibond 
                   << " e[" << i << "]=" << std::defaultfloat << std::setprecision(12) << eopt[i]
		   << std::endl;
      } // i
   }
   // summary for a single sweep
   void summary(const int isweep);
public:
   bool guess;
   int seqsize, nroots, maxsweep, dbranch;
   double rdm_vs_svd;
   std::vector<directed_bond> seq; // sweep bond sequence 
   std::vector<input::params_sweep> ctrls; // control parameters
   // energies
   std::vector<std::vector<dot_result>> opt_result; // (maxsweep,seqsize) 
   std::vector<dot_result> min_result;
   // timing
   std::vector<std::vector<dot_timing>> opt_timing;
   dot_timing timing_global;
   std::vector<double> t_total; 
};

// analysis of the current sweep (eopt,dwt,deff) and timing
inline void sweep_data::summary(const int isweep){
   std::cout << "\n" << tools::line_separator2 << std::endl;
   std::cout << "sweep_data::summary isweep=" << isweep 
             << " dbranch=" << dbranch << std::endl;
   std::cout << tools::line_separator << std::endl;
   print_ctrls(isweep);
   // print results for each dot in a single sweep
   std::vector<double> emean(seqsize,0.0);
   int nmvp = 0;
   for(int ibond=0; ibond<seqsize; ibond++){
      auto dbond = seq[ibond];
      auto p0 = dbond.p0;
      auto p1 = dbond.p1;
      auto forward = dbond.forward;
      auto p = dbond.p;
      std::cout << " ibond=" << ibond << " bond=" << p0 << "-" << p1 
                << " forward=" << forward
                << " deff=" << opt_result[isweep][ibond].deff
                << " dwt=" << std::showpos << std::scientific << std::setprecision(3)
	        << opt_result[isweep][ibond].dwt << std::noshowpos
	        << " nmvp=" << opt_result[isweep][ibond].nmvp;
      nmvp += opt_result[isweep][ibond].nmvp;      
      // print energy
      std::cout << std::defaultfloat << std::setprecision(12);
      const auto& eopt = opt_result[isweep][ibond].eopt;
      for(int j=0; j<nroots; j++){ 
         std::cout << " e[" << j << "]=" << eopt[j];
         emean[ibond] += eopt[j]; 
      } // jstate
      emean[ibond] /= nroots;
      std::cout << std::endl;
   }
   // find the minimal energy
   auto pos = std::min_element(emean.begin(), emean.end());
   int mbond = std::distance(emean.begin(), pos);
   min_result[isweep] = opt_result[isweep][mbond];
   min_result[isweep].nmvp = nmvp;
   std::cout << "minimal energies at ibond=" << mbond << std::endl;
   const auto& eopt = min_result[isweep].eopt; 
   for(int i=0; i<nroots; i++){
      std::cout << " sweep energies:"
                << " isweep=" << isweep 
	        << " dots=" << ctrls[isweep].dots
		<< " dcut=" << ctrls[isweep].dcut
		<< " deff=" << opt_result[isweep][mbond].deff
		<< " dwt=" << std::showpos << std::scientific << std::setprecision(3)
		<< opt_result[isweep][mbond].dwt << std::noshowpos
                << " e[" << i << "]=" << std::defaultfloat << std::setprecision(12) << eopt[i]
                << std::endl;
   } // i
   
   // print all previous optimized results - sweep_data
   std::cout << tools::line_separator << std::endl;
   std::cout << "summary of sweep optimization up to isweep=" << isweep
             << " dbranch=" << dbranch << std::endl;
   std::cout << "schedule: isweep, dots, dcut, eps, noise | nmvp | TIMING/S | Tav/S | Taccum/S" << std::endl;
   std::cout << std::scientific << std::setprecision(2);
   // print previous ctrl parameters
   double taccum = 0.0;
   for(int jsweep=0; jsweep<=isweep; jsweep++){
      auto& ctrl = ctrls[jsweep];
      taccum += t_total[jsweep];
      nmvp = min_result[jsweep].nmvp;
      std::cout << std::setw(10) << jsweep 
                << "  " << ctrl.dots 
                << "  " << ctrl.dcut 
                << "  " << ctrl.eps 
 		<< "  " << ctrl.noise << " | " 
		<< nmvp << " | " 
           	<< t_total[jsweep] << " | " 
		<< (t_total[jsweep]/nmvp) << " | " 
	        << taccum << std::endl;
   } // jsweep
   std::cout << "results: isweep, dwt, energies (delta_e)" << std::endl;
   const auto& eopt_isweep = min_result[isweep].eopt;
   for(int jsweep=0; jsweep<=isweep; jsweep++){
      const auto& dwt = min_result[jsweep].dwt;
      const auto& eopt_jsweep = min_result[jsweep].eopt;
      std::cout << std::setw(10) << jsweep << " "
      	        << std::showpos << std::scientific << std::setprecision(2) << dwt
                << std::noshowpos << std::defaultfloat << std::setprecision(12);
      for(int j=0; j<nroots; j++){ 
         std::cout << " e[" << j << "]=" 
     	           << std::defaultfloat << std::setprecision(12) << eopt_jsweep[j] << " ("
                   << std::scientific << std::setprecision(2) << eopt_jsweep[j]-eopt_isweep[j] << ")";
      } // jstate
      std::cout << std::endl;
   } // jsweep
   std::cout << tools::line_separator2 << "\n" << std::endl;
}

} // ctns

#endif
