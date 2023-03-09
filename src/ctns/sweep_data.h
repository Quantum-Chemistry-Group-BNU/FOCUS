#ifndef SWEEP_DATA_H
#define SWEEP_DATA_H

#include <vector>
#include "../io/input.h"
#include "../core/serialization.h"

namespace ctns{

   // memory monitor
   struct dot_CPUmem{
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
            << "  per = " << std::setw(4) << std::defaultfloat << dtkey/(dt+eps)*100 
            << "  per(accum) = " << dtacc/(dt+eps)*100 
            << std::endl;
      }
      void print_part_dmrg(const std::string key,
            const double dtkey
            ) const{
         std::cout << " T(" << std::setw(5) << key << ") = " 
            << std::scientific << std::setprecision(2) << dtkey << " S"
            << "  per = " << std::setw(4) << std::defaultfloat << dtkey/(dt2+eps)*100 
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

         this->print_part_dmrg("preprocess_op_wf           ",dtb1 );  
         this->print_part_dmrg("symbolic_formulae_twodot   ",dtb2 ); 
         this->print_part_dmrg("preprocess_formulae_Hxlist2",dtb3 ); 
         this->print_part_dmrg("verbose1_debug_Hxlst2      ",dtb4 ); 
         this->print_part_dmrg("op_lrc1c2_cpumem_host2GPU  ",dtb5 ); 
         this->print_part_dmrg("inter_cpumem_host2GPU      ",dtb6 ); 
         this->print_part_dmrg("batchsize_compute          ",dtb7 );
         this->print_part_dmrg("generate_mmtasks           ",dtb8 ); 
         this->print_part_dmrg("save_mmtasks               ",dtb9 ); 
         this->print_part_dmrg("preprocess_Hx_batchGPU     ",dtb10);
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
                                           // decomposition of dt2 into different parts
         dtb1 = tools::get_duration(tb1-tb); // tb1-tb : t(preprocess_op_wf           )
         dtb2 = tools::get_duration(tb2-tb1);// tb2-tb1: t(symbolic_formulae_twodot   ) 
         dtb3 = tools::get_duration(tb3-tb2);// tb3-tb2: t(preprocess_formulae_Hxlist2)
         dtb4 = tools::get_duration(tb4-tb3);// tb4-tb3: t(verbose1_debug_Hxlst2      )
         dtb5 = tools::get_duration(tb5-tb4);// tb5-tb4: t(op_lrc1c2_cpumem_host2GPU  )
         dtb6 = tools::get_duration(tb6-tb5);// tb6-tb5: t(inter_cpumem_host2GPU      )
         dtb7 = tools::get_duration(tb7-tb6);// tb7-tb6: t(batchsize_compute          )
         dtb8 = tools::get_duration(tb8-tb7);// tb8-tb7: t(generate_mmtasks           )
         dtb9 = tools::get_duration(tb9-tb8);// tb9-tb8: t(save_mmtasks               )
         dtb10 =tools::get_duration(tb10-tb9);//tb10-tb9:t(preprocess_Hx_batchGPU     ) 
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

         dtb1  += timer.dtb1; 
         dtb2  += timer.dtb2; 
         dtb3  += timer.dtb3; 
         dtb4  += timer.dtb4; 
         dtb5  += timer.dtb5; 
         dtb6  += timer.dtb6; 
         dtb7  += timer.dtb7; 
         dtb8  += timer.dtb8; 
         dtb9  += timer.dtb9; 
         dtb10 += timer.dtb10;
         if(debug) this->print(msg);
      }
      public:
      using Tm = std::chrono::high_resolution_clock::time_point;
      const double eps = 1.e-20;
      Tm t0;
      Tm ta; // ta-t0: t(fetch) 
      Tm tb; // tb-ta: t(hdiag)
      Tm tc; // tc-ta: t(dvdson)
      Tm td; // td-tc: t(decim)
      Tm te; // te-td: t(guess)
      Tm tf; // tf-te: t(renrm)
      Tm t1; // t1-tf: t(save)
      double dt=0, dt0=0, dt1=0, dt2=0, dt3=0, dt4=0, dt5=0, dt6=0;

      Tm tb1; // tb1-tb : t(preprocess_op_wf)
      Tm tb2; // tb2-tb1: t(prepare_GPU) 
      Tm tb3; // tb3-tb2: t(symbolic_formulae)
      Tm tb4; // tb4-tb3: t(preprocess_formulae)
      Tm tb5; // tb5-tb4: t(debug_Hxlst2)
      Tm tb6; // tb6-tb5: t(GPU_malloc_opertot)
      Tm tb7; // tb7-tb6: t(cpumem_copy)
      Tm tb8; // tb8-tb7: t(task_init)
      Tm tb9; // tb9-tb8: t(GPU_malloc_dev_workspace)
      Tm tb10;// tb10-tb9: t(preprocess_Hx_batchGPU)
      double dtb1=0, dtb2=0, dtb3=0, dtb4=0, dtb5=0, dtb6=0, dtb7=0, dtb8=0, dtb9=0, dtb10=0;
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
         opt_CPUmem.resize(maxsweep);
         opt_timing.resize(maxsweep);
         for(int i=0; i<maxsweep; i++){
            opt_result[i].resize(seqsize);
            opt_CPUmem[i].resize(seqsize);
            opt_timing[i].resize(seqsize);
            for(int j=0; j<seqsize; j++){
               opt_result[i][j].eopt.resize(nroots, 0.0);
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
      std::vector<double> t_kernel_total; 
      std::vector<double> t_reduction_total; 
      // memory
      std::vector<std::vector<dot_CPUmem>> opt_CPUmem;
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
      int pos_eav_min = std::distance(eav.begin(), eav_ptr.first);
      int pos_eav_max = std::distance(eav.begin(), eav_ptr.second);
      auto dwt_ptr = std::minmax_element(dwt.begin(), dwt.end());
      int pos_dwt_min = std::distance(dwt.begin(), dwt_ptr.first);
      int pos_dwt_max = std::distance(dwt.begin(), dwt_ptr.second);
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
         if(jsweep < restart_sweep) continue;
         const auto& ctrl = ctrls[jsweep];
         taccum += t_total[jsweep];
         nmvp = min_result[jsweep].nmvp;
         std::cout << std::setw(13) << jsweep 
            << std::setw(3) << ctrl.dots 
            << std::setw(8) << ctrl.dcut 
            << " " << ctrl.eps 
            << " " << ctrl.noise << " | " 
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
         if(jsweep < restart_sweep) continue;
         const auto& ctrl = ctrls[jsweep];
         const auto& dwt = min_result[jsweep].dwt;
         const auto& eopt_jsweep = min_result[jsweep].eopt;
         std::cout << std::setw(13) << jsweep
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
