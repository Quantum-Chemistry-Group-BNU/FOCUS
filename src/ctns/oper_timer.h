#ifndef OPER_TIMER_H
#define OPER_TIMER_H

#include "../core/tools.h"
#include <boost/timer/timer.hpp>

namespace ctns{

   struct batch_timing{
      public:
         // GPU_kernel
         void start(const int _nd){
            nd = _nd;
            tHx.resize(nd);
            memset(tHx.data(), 0, nd*sizeof(double));
            cHx.resize(nd);
            memset(cHx.data(), 0, nd*sizeof(double));
            if(Hx_counter == 0){
               tHx_tot.resize(nd);
               memset(tHx_tot.data(), 0, nd*sizeof(double));
               cHx_tot.resize(nd);
               memset(cHx_tot.data(), 0, nd*sizeof(double));
            }
            Hx_counter += 1;
         }
         void print(std::string name, 
               const std::vector<double>& tdata, 
               const std::vector<double>& cost,
               double tinter){
            double tgemm = 0.0;
            for(int i=0; i<nd-1; i++){
               tgemm += tdata[i]; 
            }
            double tot = tinter + tgemm + tdata[nd-1];
            std::cout << "--- TIMING for " << name << " :"
               << " tinter=" << tinter 
               << " tgemm=" << tgemm 
               << " treduction=" << tdata[nd-1]
               << " tot=" << tot << " ---" 
               << std::endl;
            double cgemm = 0.0;
            for(int i=0; i<nd-1; i++){
               std::cout << " " << name << " counter=" << Hx_counter << " i=" << i << " t=" << tdata[i] 
                  << " per=" << tdata[i]/tgemm*100 
                  << " cost=" << cost[i]
                  << " flops=" << cost[i]/tdata[i]
                  << std::endl;
               cgemm += cost[i];
            }
            std::cout << " total GEMM cost=" << cgemm << " flops=" << cgemm/tgemm << std::endl;
         }
         void analysis(std::string name){
            // accumulate
            for(int i=0; i<nd; i++){
               tHx_tot[i] += tHx[i];
               cHx_tot[i] += cHx[i];
            }
            tInter_tot += tInter;
            this->print(name, tHx, cHx, tInter);
            this->print(name+"_tot", tHx_tot, cHx_tot, tInter_tot);
         }
      public:
         int nd = 0;
         int Hx_counter = 0;
         std::vector<double> tHx;
         std::vector<double> tHx_tot;
         std::vector<double> cHx;
         std::vector<double> cHx_tot;
         double tInter = 0.0;
         double tInter_tot = 0.0;
   };

   struct oper_timing{
      public:
         void analysis(){
            std::cout << "----- oper_timing -----" << std::endl;
            boost::timer::cpu_times elapsed = timer.elapsed();
            double cputime = (elapsed.user+elapsed.system)/1.0e9;
            std::cout << std::scientific << std::setprecision(2)
               << " user=" << elapsed.user/1.0e9 << " S"
               << " system=" << elapsed.system/1.0e9 << " S"
               << " wall=" << elapsed.wall/1.0e9 << " S"
               << " ratio=" << (cputime/(elapsed.wall/1.0e9))
               << std::endl;
            double tot = tC + tA + tB + tH + tS + tP + tQ;
            std::cout << " opxwf=" << tot << " S"
               << " per=" << tot/cputime*1.0e2 
               << std::endl; 
            std::cout << "  opC: t=" << tC << " S" << " per=" << tC/tot*100
               << " n=" << nC << " tav=" << tC/nC << " S" 
               << std::endl;
            std::cout << "  opA: t=" << tA << " S" << " per=" << tA/tot*100 
               << " n=" << nA << " tav=" << tA/nA << " S" 
               << std::endl;
            std::cout << "  opB: t=" << tB << " S" << " per=" << tB/tot*100 
               << " n=" << nB << " tav=" << tB/nB << " S" 
               << std::endl;
            std::cout << "  opH: t=" << tH << " S" << " per=" << tH/tot*100
               << " n=" << nH << " tav=" << tH/nH << " S" 
               << std::endl;
            std::cout << "  opS: t=" << tS << " S" << " per=" << tS/tot*100 
               << " n=" << nS << " tav=" << tS/nS << " S" 
               << std::endl;
            std::cout << "  opP: t=" << tP << " S" << " per=" << tP/tot*100
               << " n=" << nP << " tav=" << tP/nP << " S" 
               << std::endl;
            std::cout << "  opQ: t=" << tQ << " S" << " per=" << tQ/tot*100 
               << " n=" << nQ << " tav=" << tQ/nQ << " S" 
               << std::endl;
            std::cout << " totHx=" << (tHxInit+tHxCalc+tHxFinl) << " S" 
               << " Init=" << tHxInit << " S"
               << " Calc=" << tHxCalc << " S"
               << " Finl=" << tHxFinl << " S"
               << std::endl; 
         }
         void sigma_analysis(){ sigma.analysis("sigma"); }
         void renorm_analysis(){ renorm.analysis("renorm"); }
         void start(){
            timer.start();
            nC=0; nA=0; nB=0; nH=0; nS=0; nP=0; nQ=0;
            tC=0.0; tA=0.0; tB=0.0; tH=0.0; tS=0.0; tP=0.0; tQ=0.0;
            tHxInit=0.0; tHxCalc=0.0; tHxFinl=0.0;
         }
         void sigma_start(){ sigma.start(9); }
         void renorm_start(){ renorm.start(8); }
      public:
         boost::timer::cpu_timer timer;
         // opxwf
         int nC=0, nA=0, nB=0, nH=0, nS=0, nP=0, nQ=0;
         double tC=0.0, tA=0.0, tB=0.0, tH=0.0, tS=0.0, tP=0.0, tQ=0.0;
         // Hx
         double tHxInit=0.0, tHxCalc=0.0, tHxFinl=0.0;
         // GPU_kernel
         batch_timing sigma;
         batch_timing renorm;
   };

   extern oper_timing oper_timer;

} // ctns

#endif
