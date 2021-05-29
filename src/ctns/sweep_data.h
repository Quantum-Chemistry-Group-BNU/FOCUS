#ifndef SWEEP_DATA_H
#define SWEEP_DATA_H

#include "../io/input.h"
#include <vector>

namespace ctns{

// computed results at a given dot	
struct dot_result{
   std::vector<double> eopt; // eopt[nstates]
   double dwt;
   int deff;
   int nmvp;
};

// timing
struct dot_timing{
   void analysis(){
      double dt0 = tools::get_duration(ta-t0); // t(procs)
      double dt1 = tools::get_duration(tb-ta); // t(hdiag)
      double dt2 = tools::get_duration(tc-tb); // t(dvdsn)
      double dt3 = tools::get_duration(td-tc); // t(decim)
      double dt4 = tools::get_duration(te-td); // t(renrm)
      double dt5 = tools::get_duration(t1-te); // t(save)
      double dt  = tools::get_duration(t1-t0); // total
      std::cout << " t(load)  = " << std::scientific << std::setprecision(2) << dt0 << " s"
   	        << "  per = " << std::defaultfloat << dt0/dt*100 
		<< "  per(accum) = " << dt0/dt*100 
		<< std::endl;
      std::cout << " t(hdiag) = " << std::scientific << std::setprecision(2) << dt1 << " s"
   	        << "  per = " << std::defaultfloat << dt1/dt*100 
		<< "  per(accum) = " << (dt0+dt1)/dt*100 
		<< std::endl;
      std::cout << " t(dvdsn) = " << std::scientific << std::setprecision(2) << dt2 << " s"
   	        << "  per = " << std::defaultfloat << dt2/dt*100 
		<< "  per(accum) = " << (dt0+dt1+dt2)/dt*100 
		<< std::endl;
      std::cout << " t(decim) = " << std::scientific << std::setprecision(2) << dt3 << " s"
   	        << "  per = " << std::defaultfloat << dt3/dt*100 
		<< "  per(accum) = " << (dt0+dt1+dt2+dt3)/dt*100 
		<< std::endl;
      std::cout << " t(renrm) = " << std::scientific << std::setprecision(2) << dt4 << " s"
   	        << "  per = " << std::defaultfloat << dt4/dt*100 
		<< "  per(accum) = " << (dt0+dt1+dt2+dt3+dt4)/dt*100 
		<< std::endl;
      std::cout << " t(save)  = " << std::scientific << std::setprecision(2) << dt5 << " s"
   	        << "  per = " << std::defaultfloat << dt5/dt*100 
		<< "  per(accum) = " << (dt0+dt1+dt2+dt3+dt4+dt5)/dt*100 
		<< std::endl;
   }
public:
   using tm = std::chrono::high_resolution_clock::time_point;
   tm t0;
   tm ta; // ta-t0: t(load) 
   tm tb; // tb-ta: t(hdiag)
   tm tc; // tc-ta: t(dvdson)
   tm td; // td-tc: t(decim)
   tm te; // te-td: t(renrm)
   tm t1; // t1-te: t(save)
};

struct sweep_data{
   // constructor
   sweep_data(const std::vector<directed_bond>& sweep_seq, 
	      const int _nstates,
	      const bool _guess,
	      const int _inoise,
	      const int _maxsweep,
	      const std::vector<input::params_sweep>& _ctrls){
      seq = sweep_seq; 
      seqsize = sweep_seq.size();
      guess = _guess;
      nstates = _nstates;
      inoise = _inoise;
      maxsweep = _maxsweep;
      ctrls = _ctrls;
      // sweep results
      opt_result.resize(maxsweep);
      opt_timing.resize(maxsweep);
      for(int i=0; i<maxsweep; i++){
	 opt_result[i].resize(seqsize);
	 opt_timing[i].resize(seqsize);
	 for(int j=0; j<seqsize; j++){
	    opt_result[i][j].eopt.resize(nstates);
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
      for(int i=0; i<nstates; i++){
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
   int seqsize, nstates, maxsweep, inoise; 
   std::vector<directed_bond> seq; // sweep bond sequence 
   std::vector<input::params_sweep> ctrls; // control parameters
   // energies
   std::vector<std::vector<dot_result>> opt_result; // (maxsweep,seqsize) 
   std::vector<dot_result> min_result;
   // timing
   std::vector<std::vector<dot_timing>> opt_timing;
   std::vector<double> t_total; 
};

// analysis of the current sweep (eopt,dwt,deff) and timing
inline void sweep_data::summary(const int isweep){
   std::cout << "\n" << tools::line_separator2 << std::endl;
   std::cout << "sweep_data::summary isweep=" << isweep << std::endl;
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
                << " dwt=" << std::showpos << std::scientific << std::setprecision(2) 
	        << opt_result[isweep][ibond].dwt << std::noshowpos
	        << " nmvp=" << opt_result[isweep][ibond].nmvp;
      nmvp += opt_result[isweep][ibond].nmvp;      
      // print energy
      std::cout << std::defaultfloat << std::setprecision(12);
      const auto& eopt = opt_result[isweep][ibond].eopt;
      for(int j=0; j<nstates; j++){ 
         std::cout << " e[" << j << "]=" << eopt[j];
         emean[ibond] += eopt[j]; 
      } // jstate
      emean[ibond] /= nstates;
      std::cout << std::endl;
   }
   // find the minimal energy
   auto pos = std::min_element(emean.begin(), emean.end());
   int mbond = std::distance(emean.begin(), pos);
   min_result[isweep] = opt_result[isweep][mbond];
   min_result[isweep].nmvp = nmvp;
   std::cout << "minimal energies at ibond=" << mbond << "  " 
             << "timing for sweep: " << std::setprecision(2) << t_total[isweep] << " s" 
	     << std::endl;
   const auto& eopt = min_result[isweep].eopt; 
   for(int i=0; i<nstates; i++){
      std::cout << " sweep energies:"
                << " isweep=" << isweep 
	        << " dots=" << ctrls[isweep].dots
		<< " dcut=" << ctrls[isweep].dcut
		<< " deff=" << opt_result[isweep][mbond].deff
		<< " dwts=" << std::showpos << std::scientific << std::setprecision(2) 
		<< opt_result[isweep][mbond].dwt << std::noshowpos
                << " e[" << i << "]=" << std::defaultfloat << std::setprecision(12) << eopt[i]
                << std::endl;
   } // i
   
   // print all previous optimized results - sweep_data
   std::cout << tools::line_separator << std::endl;
   std::cout << "summary of sweep optimization up to isweep=" << isweep << std::endl;
   std::cout << "schedule: isweep, dots, dcut, eps, noise | nmvp | timing/s | tav/s | taccum/s" << std::endl;
   std::cout << std::scientific << std::setprecision(2);
   // print previous ctrl parameters
   double taccum = 0.0;
   for(int jsweep=0; jsweep<=isweep; jsweep++){
      auto& ctrl = ctrls[jsweep];
      taccum += t_total[jsweep];
      nmvp = min_result[jsweep].nmvp;
      std::cout << std::setw(10) << jsweep << " "
           	<< ctrl.dots << " " << ctrl.dcut << " "
           	<< ctrl.eps  << " " << ctrl.noise << " | " 
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
      for(int j=0; j<nstates; j++){ 
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
