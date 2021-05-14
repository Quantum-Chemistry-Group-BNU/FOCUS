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
};

// timing
struct dot_timing{
   void analysis(){
      double dt0 = tools::get_duration(ta-t0); // t(procs)
      double dt1 = tools::get_duration(tb-ta); // t(hdiag)
      double dt2 = tools::get_duration(tc-tb); // t(dvdsn)
      double dt3 = tools::get_duration(td-tc); // t(decim)
      double dt4 = tools::get_duration(t1-td); // t(renrm)
      double dt  = tools::get_duration(t1-t0); // total
      std::cout << " t(procs) = " << std::scientific << std::setprecision(2) << dt0 << " s"
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
   }
public:
   using tm = std::chrono::high_resolution_clock::time_point;
   tm t0;
   tm ta; // ta-t0: t(procs) 
   tm tb; // tb-ta: t(hdiag)
   tm tc; // tc-ta: t(dvdson)
   tm td; // td-tc: t(decim)
   tm t1; // t1-td: t(renrm)
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
   // print
   void print_ctrls(const int isweep) const{
      const auto& ctrl = ctrls[isweep];
      ctrl.print();
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
void sweep_data::summary(const int isweep){
   std::cout << "\n" << tools::line_separator << std::endl;
   std::cout << "sweep_data::summary isweep=" << isweep << std::endl;
   std::cout << tools::line_separator << std::endl;
   print_ctrls(isweep);
   // print results for each dot in a single sweep
   std::vector<double> emean(seqsize,0.0);
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
	        << opt_result[isweep][ibond].dwt << std::noshowpos;
      // print energy
      std::cout << std::defaultfloat << std::setprecision(12);
      const auto& eopt = opt_result[isweep][ibond].eopt;
      for(int j=0; j<nstates; j++){ 
         std::cout << " e[" << j << "]=" << eopt[j];
         emean[ibond] += eopt[j]; 
      }
      emean[ibond] /= nstates;
      std::cout << std::endl;
   }
   // find the minimal energy
   auto pos = std::min_element(emean.begin(), emean.end());
   auto minpos = std::distance(emean.begin(), pos);
   min_result[isweep] = opt_result[isweep][minpos];
   std::cout << "min energies at pos=" << minpos << std::endl;
   std::cout << "timing for sweep: " << std::setprecision(2) << t_total[isweep] << " s" << std::endl; 
   std::cout << tools::line_separator << std::endl;
   
   // print all previous optimized results - sweep_data
   std::cout << "summary of sweep optimization up to isweep=" << isweep << std::endl;
   std::cout << "schedule: iter, dots, dcut, eps, noise | timing/s (taccum/s)" << std::endl;
   std::cout << std::scientific << std::setprecision(2);
   // print previous ctrl parameters
   double taccum = 0.0;
   for(int jsweep=0; jsweep<=isweep; jsweep++){
      auto& ctrl = ctrls[jsweep];
      taccum += t_total[jsweep];
      std::cout << " " << jsweep << " "
           	<< ctrl.dots << " " << ctrl.dcut << " "
           	<< ctrl.eps  << " " << ctrl.noise << " | "
           	<< t_total[jsweep] << " (" << taccum << ")" << std::endl;
   } // jsweep
   std::cout << "results: iter, dwt, energies (delta_e)" << std::endl;
   const auto& eopt_isweep = min_result[isweep].eopt;
   for(int jsweep=0; jsweep<=isweep; jsweep++){
      const auto& dwt = min_result[jsweep].dwt;
      const auto& eopt_jsweep = min_result[jsweep].eopt;
      std::cout << std::setw(3) << jsweep << " ";
      std::cout << std::showpos << std::scientific << std::setprecision(2) << dwt;
      std::cout << std::noshowpos << std::defaultfloat << std::setprecision(12);
      for(int j=0; j<nstates; j++){ 
         std::cout << " e" << j << ":" 
     	      << std::defaultfloat << std::setprecision(12) << eopt_jsweep[j] << " ("
              << std::scientific << std::setprecision(2) << eopt_jsweep[j]-eopt_isweep[j] << ")";
      } // jstate
      std::cout << std::endl;
   } // jsweep
   std::cout << tools::line_separator << "\n" << std::endl;
}

} // ctns

#endif
