#ifndef SWEEP_DATA_H
#define SWEEP_DATA_H

#include <vector>
#include "../io/input.h"

namespace ctns{

    // memory
    struct dot_memory{
        void display(){
            const double toGB = 1.0/std::pow(1024.0,3);
            tot = comb + oper + dvdson + hvec + renorm;
            std::cout << "===== CPUmem(GB): tot=" << tot*toGB
                      << " (comb,oper,dvdson,hvec,renorm)=" 
                      << comb*toGB << ","
                      << oper*toGB << ","
                      << dvdson*toGB << ","
                      << hvec*toGB << ","
		      << renorm*toGB
                      << " =====" << std::endl;
        }
        public:
            size_t comb = 0;
            size_t oper = 0;
            size_t dvdson = 0;
            size_t hvec = 0;
	    size_t renorm = 0;
            size_t tot = 0;
    };

    // timing
    struct dot_timing{
        void print_part(const std::string key,
                const double dtkey,
                const double dtacc) const{
            std::cout << " T(" << std::setw(5) << key << ") = " 
                << std::scientific << std::setprecision(2) << dtkey << " S"
                << "  per = " << std::setw(4) << std::defaultfloat << dtkey/dt*100 
                << "  per(accum) = " << dtacc/dt*100 
                << std::endl;
        }
        void print(const std::string msg) const{
            std::cout << "##### " << msg << ": " 
                << std::scientific << std::setprecision(2) << dt
                << " S #####" 
                << std::endl;
            double dtacc = dt0;
            this->print_part("fetch", dt0, dtacc); dtacc += dt1; 
            this->print_part("hdiag", dt1, dtacc); dtacc += dt2;
            this->print_part("dvdsn", dt2, dtacc); dtacc += dt3;
            this->print_part("decim", dt3, dtacc); dtacc += dt4;
            this->print_part("guess", dt4, dtacc); dtacc += dt5;
            this->print_part("renrm", dt5, dtacc); dtacc += dt6;
            this->print_part("save" , dt6, dtacc);
        }
        void analysis(const std::string msg,
                const bool debug=true){
            dt  = tools::get_duration(t1-t0); // total
            dt0 = tools::get_duration(ta-t0); // t(fetch)
            dt1 = tools::get_duration(tb-ta); // t(hdiag)
            dt2 = tools::get_duration(tc-tb); // t(dvdsn)
            dt3 = tools::get_duration(td-tc); // t(decim)
            dt4 = tools::get_duration(te-td); // t(guess)
            dt5 = tools::get_duration(tf-te); // t(renrm)
            dt6 = tools::get_duration(t1-tf); // t(save)
            if(debug) this->print(msg);
        }
        void accumulate(const dot_timing& timer,
                const std::string msg,
                const bool debug=true){
            dt  += timer.dt;
            dt0 += timer.dt0;
            dt1 += timer.dt1;
            dt2 += timer.dt2;
            dt3 += timer.dt3;
            dt4 += timer.dt4;
            dt5 += timer.dt5;
            dt6 += timer.dt6;
            if(debug) this->print(msg);
        }
        public:
        using Tm = std::chrono::high_resolution_clock::time_point;
        Tm t0;
        Tm ta; // ta-t0: t(fetch) 
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
                const int _maxsweep,
                const std::vector<input::params_sweep>& _ctrls){
            seq = sweep_seq;
            seqsize = sweep_seq.size();
            nroots = _nroots;
            maxsweep = _maxsweep;
            ctrls = _ctrls;
            // sweep results
            timing_sweep.resize(maxsweep);
            opt_result.resize(maxsweep);
            opt_memory.resize(maxsweep);
            opt_timing.resize(maxsweep);
            for(int i=0; i<maxsweep; i++){
                opt_result[i].resize(seqsize);
                opt_memory[i].resize(seqsize);
                opt_timing[i].resize(seqsize);
                for(int j=0; j<seqsize; j++){
                    opt_result[i][j].eopt.resize(nroots);
                }
            }
            min_result.resize(maxsweep);
            t_total.resize(maxsweep);
            t_kernel_total.resize(maxsweep);
            t_reduction_total.resize(maxsweep);
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
                std::cout << "optimized energies:"
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
        int seqsize, nroots, maxsweep;
        std::vector<directed_bond> seq; // sweep bond sequence 
        std::vector<input::params_sweep> ctrls; // control parameters
        // energies
        std::vector<std::vector<dot_result>> opt_result; // (maxsweep,seqsize) 
        std::vector<dot_result> min_result;
        // memory
        std::vector<std::vector<dot_memory>> opt_memory;
        // timing
        std::vector<std::vector<dot_timing>> opt_timing;
        std::vector<dot_timing> timing_sweep;
        std::vector<double> t_total; 
        std::vector<double> t_kernel_total; 
        std::vector<double> t_reduction_total; 
    };

    // analysis of the current sweep (eopt,dwt,deff) and timing
    inline void sweep_data::summary(const int isweep){
        std::cout << "\n" << tools::line_separator2 << std::endl;
        std::cout << "sweep_data::summary isweep=" << isweep << std::endl; 
        std::cout << tools::line_separator << std::endl;
        print_ctrls(isweep);

        // print results for each dot in a single sweep
        std::vector<double> eav(seqsize,0.0);
        std::vector<double> dwt(seqsize,0.0);
        int nmvp = 0;
        for(int ibond=0; ibond<seqsize; ibond++){
            const auto& dbond = seq[ibond];
            const auto& p0 = dbond.p0;
            const auto& p1 = dbond.p1;
            const auto& forward = dbond.forward;
            std::cout << " ibond=" << ibond 
                << " bond=" << p0 << "-" << p1 
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
                eav[ibond] += eopt[j]; 
            } // jstate
            eav[ibond] /= nroots;
            dwt[ibond] = opt_result[isweep][ibond].dwt;
            std::cout << std::endl;
        }
        // find the min,max energy and discard weights
        auto eav_ptr = std::minmax_element(eav.begin(), eav.end());
        int pos_eav_min = std::distance(eav.begin(),eav_ptr.first);
        int pos_eav_max = std::distance(eav.begin(),eav_ptr.second);
        auto dwt_ptr = std::minmax_element(dwt.begin(), dwt.end());
        int pos_dwt_min = std::distance(dwt.begin(),dwt_ptr.first);
        int pos_dwt_max = std::distance(dwt.begin(),dwt_ptr.second);
        std::cout << " eav_max=" << *eav_ptr.second
                  << " [ibond=" << pos_eav_max << "] "
                  << " dwt_min=" << *dwt_ptr.first 
                  << " [ibond=" << pos_dwt_min << "]"
                  << std::endl;
        std::cout << " eav_min=" << *eav_ptr.first 
                  << " [ibond=" << pos_eav_min << "] "
                  << " dwt_max=" << *dwt_ptr.second
                  << " [ibond=" << pos_dwt_max << "]"
                  << std::endl;
        std::cout << " eav_diff=" << *eav_ptr.second - *eav_ptr.first << std::endl;
        // minimal energy   
        min_result[isweep] = opt_result[isweep][pos_eav_min];
        min_result[isweep].nmvp = nmvp;
        const auto& eopt = min_result[isweep].eopt; 
        for(int i=0; i<nroots; i++){
            std::cout << " sweep energies:"
                << " isweep=" << isweep 
                << " dots=" << ctrls[isweep].dots
                << " dcut=" << ctrls[isweep].dcut
                << " deff=" << opt_result[isweep][pos_eav_min].deff
                << " dwt=" << std::showpos << std::scientific << std::setprecision(3)
                << opt_result[isweep][pos_eav_min].dwt << std::noshowpos
                << " e[" << i << "]=" << std::defaultfloat << std::setprecision(12) << eopt[i]
                << std::endl;
        } // i

        // print all previous optimized results - sweep_data
        std::cout << tools::line_separator << std::endl;
        std::cout << "summary of sweep optimization up to isweep=" << isweep << std::endl;
        std::cout << "schedule: isweep, dots, dcut, eps, noise | nmvp | Tsweep/S | Tav/S | Taccum/S| t_kernel/S | t_reduction/S " << std::endl;
        std::cout << std::scientific << std::setprecision(2);
        // print previous ctrl parameters
        double taccum = 0.0;
        for(int jsweep=0; jsweep<=isweep; jsweep++){
            const auto& ctrl = ctrls[jsweep];
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
                << taccum << " | "
                << t_kernel_total[jsweep] << " | " 
                << t_reduction_total[jsweep] 
                << std::endl;
        } // jsweep
        std::cout << "results: isweep, dcut, dwt, energies (delta_e)" << std::endl;
        const auto& eopt_isweep = min_result[isweep].eopt;
        for(int jsweep=0; jsweep<=isweep; jsweep++){
            const auto& ctrl = ctrls[jsweep];
            const auto& dwt = min_result[jsweep].dwt;
            const auto& eopt_jsweep = min_result[jsweep].eopt;
            std::cout << std::setw(10) << jsweep
                << std::setw(8) << ctrl.dcut << " "
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
