#ifndef SWEEP_DATA_H
#define SWEEP_DATA_H

#include <vector>
#include "../io/input.h"
#include "../core/serialization.h"

namespace ctns{

   // timing
   struct dot_timing{
      void print_part(const std::string key,
            const double dtkey,
            const double dttot,
            double& dtacc) const{
         dtacc += dtkey;
         std::cout << " T(" << std::setw(5) << key << ") = " 
            << std::scientific << std::setprecision(3) << dtkey << " S"
            << "  per = " << std::setw(4) << std::defaultfloat << dtkey/(dttot+eps)*100 
            << "  per(accum) = " << dtacc/(dttot+eps)*100 
            << std::endl;
      }
      void print(const std::string msg) const{
         std::cout << "##### " << msg << ": " 
            << std::scientific << std::setprecision(3) << dt
            << " S #####" 
            << std::endl;
         double dtacc = 0.0;
         this->print_part(msg+": fetch", dt0, dt, dtacc); 
         this->print_part(msg+": hdiag", dt1, dt, dtacc);
         this->print_part(msg+": dvdsn", dt2, dt, dtacc);
         this->print_part(msg+": decim", dt3, dt, dtacc);
         this->print_part(msg+": guess", dt4, dt, dtacc);
         this->print_part(msg+": renrm", dt5, dt, dtacc);
         this->print_part(msg+": save " , dt6, dt, dtacc);

         double tdvdsn = dtb0 + dtb1 + dtb2 + dtb3 + dtb4 + dtb5 + dtb6 + dtb7 + dtb8 + dtb9;
         std::cout << "Detailed decomposition of T(dvdsn) = " 
            << std::scientific << std::setprecision(3) << dt2 << " S"
            << "  T(sum) = " << tdvdsn << " S  per = " << tdvdsn/(dt2+eps)*100
            << std::endl;
         dtacc = 0.0;
         this->print_part(msg+": preprocess                  ", dtb0, tdvdsn, dtacc);
         this->print_part(msg+": symbolic_formulae           ", dtb1, tdvdsn, dtacc);
         this->print_part(msg+": hintermediate init          ", dtb2, tdvdsn, dtacc);
         this->print_part(msg+": preprocess_hformulae_Hxlist ", dtb3, tdvdsn, dtacc);
         this->print_part(msg+": hmmtasks init               ", dtb4, tdvdsn, dtacc);
         this->print_part(msg+": initial guess for dvdson    ", dtb5, tdvdsn, dtacc);
         this->print_part(msg+": dvdson solver [Hx_batchGPU] ", dtb6, tdvdsn, dtacc);
         this->print_part(msg+": dvdson solver [comm(gpu)]   ", dtb7, tdvdsn, dtacc);
         this->print_part(msg+": dvdson solver [comm(cpu)]   ", dtb8, tdvdsn, dtacc);
         this->print_part(msg+": dvdson solver [rest part]   ", dtb9, tdvdsn, dtacc);

         double trenrm = dtfa + dtf0 + dtf1 + dtf2 + dtf3 + dtf4 + dtf5 + dtf6 + dtf7 + dtf8 + dtf9 + dtf10 + dtf11 + dtf12 + dtfb;
         std::cout << "Detailed decomposition of T(renrm) = " 
            << std::scientific << std::setprecision(3) << dt5 << " S"
            << "  T(sum) = " << trenrm << " S  per = " << trenrm/(dt5+eps)*100
            << std::endl;
         dtacc = 0.0;
         this->print_part(msg+": before oper_renorm           ", dtfa, trenrm, dtacc);
         this->print_part(msg+": qops init                    ", dtf0, trenrm, dtacc);
         this->print_part(msg+": site memcpy cpu2gpu          ", dtf1, trenrm, dtacc);
         this->print_part(msg+": symbolic_formulae_renorm     ", dtf2, trenrm, dtacc);
         this->print_part(msg+": rintermediate init           ", dtf3, trenrm, dtacc);
         this->print_part(msg+": rintermediates memcpy cpu2gpu", dtf4, trenrm, dtacc);
         this->print_part(msg+": preprocess_formulae_Rlist2   ", dtf5, trenrm, dtacc);
         this->print_part(msg+": rmmtasks init                ", dtf6, trenrm, dtacc);
         this->print_part(msg+": qops memset                  ", dtf7, trenrm, dtacc);
         this->print_part(msg+": preprocess_renorm_batchGPU   ", dtf8, trenrm, dtacc);
         this->print_part(msg+": reduction of opS & opH [nccl]", dtf9, trenrm, dtacc);
         this->print_part(msg+": qops memcpy gpu2cpu          ", dtf10, trenrm, dtacc);
         this->print_part(msg+": deallocate gpu memory        ", dtf11, trenrm, dtacc);
         this->print_part(msg+": reduction of opS & opH [comm]", dtf12, trenrm, dtacc);
         this->print_part(msg+": after oper_renorm            ", dtfb, trenrm, dtacc);
      }
      void analysis(const std::string msg,
            const bool debug=true){
         dt  = tools::get_duration(t1-t0); 
         dt0 = tools::get_duration(ta-t0); 
         dt1 = tools::get_duration(tb-ta); 
         dt2 = tools::get_duration(tc-tb); 
         dt3 = tools::get_duration(td-tc); 
         dt4 = tools::get_duration(te-td); 
         dt5 = tools::get_duration(tf-te); 
         dt6 = tools::get_duration(t1-tf); 

         // decomposition of dt2 into different parts
         dtb0 = tools::get_duration(tb1-tb); 
         dtb1 = tools::get_duration(tb2-tb1); 
         dtb2 = tools::get_duration(tb3-tb2); 
         dtb3 = tools::get_duration(tb4-tb3); 
         dtb4 = tools::get_duration(tb5-tb4); 
         // dtb5-9 are obtained in sweep_twodot_local.h

         // decomposition of dt5 into different parts
         dtfa = tools::get_duration(tf0-te); 
         dtf0 = tools::get_duration(tf1-tf0); 
         dtf1 = tools::get_duration(tf2-tf1); 
         dtf2 = tools::get_duration(tf3-tf2); 
         dtf3 = tools::get_duration(tf4-tf3); 
         dtf4 = tools::get_duration(tf5-tf4); 
         dtf5 = tools::get_duration(tf6-tf5); 
         dtf6 = tools::get_duration(tf7-tf6); 
         dtf7 = tools::get_duration(tf8-tf7); 
         dtf8 = tools::get_duration(tf9-tf8); 
         dtf9 = tools::get_duration(tf10-tf9); 
         dtf10 = tools::get_duration(tf11-tf10); 
         dtf11 = tools::get_duration(tf12-tf11); 
         dtf12 = tools::get_duration(tf13-tf12); 
         dtfb = tools::get_duration(tf-tf13); 

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

         // decomposition of dt2 into different parts
         dtb0 += timer.dtb0;
         dtb1 += timer.dtb1; 
         dtb2 += timer.dtb2; 
         dtb3 += timer.dtb3; 
         dtb4 += timer.dtb4; 
         dtb5 += timer.dtb5; 
         dtb6 += timer.dtb6; 
         dtb7 += timer.dtb7; 
         dtb8 += timer.dtb8; 
         dtb9 += timer.dtb9; 

         // decomposition of dt5 into different parts
         dtfa += timer.dtfa; 
         dtf0 += timer.dtf0; 
         dtf1 += timer.dtf1; 
         dtf2 += timer.dtf2; 
         dtf3 += timer.dtf3; 
         dtf4 += timer.dtf4; 
         dtf5 += timer.dtf5; 
         dtf6 += timer.dtf6; 
         dtf7 += timer.dtf7; 
         dtf8 += timer.dtf8; 
         dtf9 += timer.dtf9; 
         dtf10 += timer.dtf10;
         dtf11 += timer.dtf11;
         dtf12 += timer.dtf12;
         dtfb += timer.dtfb; 

         if(debug) this->print(msg);
      }
      public:
      using Tm = std::chrono::high_resolution_clock::time_point;
      const double eps = 1.e-20;
      Tm t0;
      Tm ta; // fetch
      Tm tb; // hdiag
      Tm tc; // dvdsn
      Tm td; // decim
      Tm te; // guess
      Tm tf; // renrm
      Tm t1; // save
      double dt=0, dt0=0, dt1=0, dt2=0, dt3=0, dt4=0, dt5=0, dt6=0;
      // decomposition of dt2 into different parts
      Tm tb1; 
      Tm tb2; 
      Tm tb3; 
      Tm tb4; 
      Tm tb5; 
      double dtb0=0, dtb1=0, dtb2=0, dtb3=0, dtb4=0, dtb5=0, dtb6=0, dtb7=0, dtb8=0, dtb9=0;
      Tm tf0;
      Tm tf1; // qops init
      Tm tf2; // qops_dict memcpy cpu2gpu   
      Tm tf3; // symbolic_formulae_renorm        
      Tm tf4; // rintermediate init           
      Tm tf5; // rintermediates memcpy cpu2gpu
      Tm tf6; // preprocess_formulae_Rlist2   
      Tm tf7; // rmmtasks init                
      Tm tf8; // qops memset
      Tm tf9; // preprocess_renorm_batchGPU   
      Tm tf10; // reduction [nccl] 
      Tm tf11; // qops memcpy gpu2cpu
      Tm tf12; // deallocate gpu memory
      Tm tf13; // reduction 
      double dtfa=0, dtf0=0, dtf1=0, dtf2=0, dtf3=0, dtf4=0, dtf5=0, dtf6=0, dtf7=0, dtf8=0, dtf9=0, dtf10=0, dtf11=0, dtf12=0, dtfb=0;
   };

   // computed results at a given dot	
   struct dot_result{
      private:
         // serialize
         friend class boost::serialization::access;
         template<class Archive>
            void serialize(Archive & ar, const unsigned int version){
               ar & eopt & dwt & deff & nmvp;
            }
      public:
         void print() const{
            int nroots = eopt.size();
            for(int i=0; i<nroots; i++){
               std::cout << "optimized energies:"
                  << " e[" << i << "]=" << std::defaultfloat << std::setprecision(12) << eopt[i]
                  << std::endl;
            } // i
            std::cout << " deff=" << deff
               << " dwt=" << std::showpos << std::scientific << std::setprecision(3) << dwt
               << std::noshowpos << " nmvp=" << nmvp
               << std::endl;
         }    
      public:
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
            const int _restart_sweep,
            const std::vector<input::params_sweep>& _ctrls){
         seq = sweep_seq;
         seqsize = sweep_seq.size();
         nroots = _nroots;
         maxsweep = _maxsweep;
         restart_sweep = _restart_sweep;
         ctrls = _ctrls;
         // sweep results
         timing_sweep.resize(maxsweep);
         opt_result.resize(maxsweep);
         opt_timing.resize(maxsweep);
         for(int i=0; i<maxsweep; i++){
            opt_result[i].resize(seqsize);
            opt_timing[i].resize(seqsize);
            for(int j=0; j<seqsize; j++){
               opt_result[i][j].eopt.resize(nroots, 0.0);
            }
         }
         min_result.resize(maxsweep);
         t_total.resize(maxsweep, 0);
         t_inter.resize(maxsweep, 0);
         t_gemm.resize(maxsweep, 0);
         t_red.resize(maxsweep, 0);
      }
      // print control parameters
      void print_ctrls(const int isweep) const{ 
         const auto& ctrl = ctrls[isweep];
         ctrl.print();
      }
      // print optimized energies
      void print_eopt(const int isweep, const int ibond) const{
         const auto& eopt = opt_result[isweep][ibond].eopt;
         const auto& nmvp = opt_result[isweep][ibond].nmvp;
         int dots = ctrls[isweep].dots;
         for(int i=0; i<nroots; i++){
            std::cout << "optimized energies:"
               << " isweep=" << isweep 
               << " ibond=" << ibond 
               << " dots=" << dots
               << " e[" << i << "]=" << std::defaultfloat << std::setprecision(12) << eopt[i]
               << " nmvp=" << nmvp
               << std::endl;
         } // i
      }
      // summary for a single sweep
      void summary(const int isweep, const int mpisize);
      // helps
      double get_eminlast(const int iroot=0) const{
         return min_result[maxsweep-1].eopt[iroot];
      }

      public:
      int seqsize, nroots, maxsweep, restart_sweep;
      std::vector<directed_bond> seq; // sweep bond sequence 
      std::vector<input::params_sweep> ctrls; // control parameters
                                              // energies
      std::vector<std::vector<dot_result>> opt_result; // (maxsweep,seqsize) 
      std::vector<dot_result> min_result;
      // timing
      std::vector<std::vector<dot_timing>> opt_timing;
      std::vector<dot_timing> timing_sweep;
      std::vector<double> t_total; 
      std::vector<double> t_inter; 
      std::vector<double> t_gemm; 
      std::vector<double> t_red;
   };

   // analysis of the current sweep (eopt,dwt,deff) and timing
   inline void sweep_data::summary(const int isweep, const int mpisize){
      int maxthreads = 1;
#ifdef _OPENMP
      maxthreads = omp_get_max_threads();
#endif
      std::cout << "\n" << tools::line_separator2 << std::endl;
      std::cout << "sweep_data::summary isweep=" << isweep 
         << " mpisize=" << mpisize 
         << " maxthreads=" << maxthreads
         << std::endl; 
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
         std::cout << " t=" << std::setprecision(3) << opt_timing[isweep][ibond].dt;
         eav[ibond] /= nroots;
         dwt[ibond] = opt_result[isweep][ibond].dwt;
         std::cout << std::endl;
      }
      // find the min,max energy and discard weights
      auto eav_ptr = std::minmax_element(eav.begin(), eav.end());
      int pos_eav_min = std::distance(eav.begin(), eav_ptr.first);
      int pos_eav_max = std::distance(eav.begin(), eav_ptr.second);
      auto dwt_ptr = std::minmax_element(dwt.begin(), dwt.end());
      int pos_dwt_min = std::distance(dwt.begin(), dwt_ptr.first);
      int pos_dwt_max = std::distance(dwt.begin(), dwt_ptr.second);
      std::cout << " eav_max=" << std::defaultfloat << std::setprecision(12) << *eav_ptr.second
         << " [ibond=" << pos_eav_max << "] "
         << " dwt_min=" << std::scientific << std::setprecision(3) << *dwt_ptr.first 
         << " [ibond=" << pos_dwt_min << "]"
         << std::endl;
      std::cout << " eav_min=" << std::defaultfloat << std::setprecision(12) << *eav_ptr.first 
         << " [ibond=" << pos_eav_min << "] "
         << " dwt_max=" << std::scientific << std::setprecision(3) << *dwt_ptr.second
         << " [ibond=" << pos_dwt_max << "]"
         << std::endl;
      std::cout << " eav_diff=" << std::defaultfloat << std::setprecision(12)
         << (*eav_ptr.second - *eav_ptr.first) << std::endl;
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
      std::cout << "summary of sweep optimization up to isweep=" << isweep 
         << " mpisize=" << mpisize 
         << " maxthreads=" << maxthreads
         << std::endl;
      std::cout << "schedule: isweep, dots, dcut, eps, noise | nmvp | Tsweep/S | Taccum/S | t_inter/S t_gemm/S t_red/S t_blas/S per" << std::endl;
      // print previous ctrl parameters
      double taccum=0.0;
      for(int jsweep=0; jsweep<=isweep; jsweep++){
         if(jsweep < restart_sweep) continue;
         const auto& ctrl = ctrls[jsweep];
         taccum += t_total[jsweep];
         double tblas = t_inter[jsweep] + t_gemm[jsweep] + t_red[jsweep];
         nmvp = min_result[jsweep].nmvp;
         std::cout << std::setw(13) << jsweep 
            << std::setw(3) << ctrl.dots 
            << std::setw(8) << ctrl.dcut 
            << std::scientific << std::setprecision(3)
            << " " << ctrl.eps 
            << " " << ctrl.noise << " | " 
            << nmvp << " | " 
            << t_total[jsweep] << " | "  // single sweep
            << taccum << " | "           // total time
            << t_inter[jsweep] << " "    // intemediates [gemv]
            << t_gemm[jsweep] << " "     // kernel [gemm]
            << t_red[jsweep] << " "      // reduction [gemv]
            << tblas << " "              // gemv+gemm
            << std::defaultfloat << tblas/t_total[jsweep]*100 
            << std::endl;
      } // jsweep
      std::cout << "results: isweep, dcut, dwt, energies (delta_e)" << std::endl;
      const auto& eopt_isweep = min_result[isweep].eopt;
      for(int jsweep=0; jsweep<=isweep; jsweep++){
         if(jsweep < restart_sweep) continue;
         const auto& ctrl = ctrls[jsweep];
         const auto& dwt = min_result[jsweep].dwt;
         const auto& eopt_jsweep = min_result[jsweep].eopt;
         std::cout << std::setw(13) << jsweep
            << std::setw(8) << ctrl.dcut << " "
            << std::showpos << std::scientific << std::setprecision(3) << dwt
            << std::noshowpos << std::defaultfloat << std::setprecision(12);
         for(int j=0; j<nroots; j++){ 
            std::cout << " e[" << j << "]=" 
               << std::defaultfloat << std::setprecision(12) << eopt_jsweep[j] << " ("
               << std::scientific << std::setprecision(3) << eopt_jsweep[j]-eopt_isweep[j] << ")";
         } // jstate
         std::cout << std::endl;
      } // jsweep
      std::cout << tools::line_separator2 << std::endl;
   }

} // ctns

#endif
