#ifndef SYMBOLIC_KERNEL_RENORM_H
#define SYMBOLIC_KERNEL_RENORM_H

#ifdef _OPENMP
#include <omp.h>
#endif

#include "symbolic_formulae_renorm.h"

namespace ctns{

   template <typename Tm> 
      stensor3<Tm> symbolic_renorm_single(const std::string& block1,
            const std::string& block2,
            const oper_dictmap<Tm>& qops_dict,
            const char key,
            const symbolic_task<Tm>& formulae,
            const stensor3<Tm>& wf,
            const bool skipId){
         const bool debug = false;
         stensor3<Tm> Hwf;
         for(int it=0; it<formulae.size(); it++){
            const auto& HTerm = formulae.tasks[it];
            stensor3<Tm> opxwf;
            bool applied = false;
            for(int idx=HTerm.size()-1; idx>=0; idx--){
               const auto& sop = HTerm.terms[idx];
               int len = sop.size();
               auto wt0 = sop.sums[0].first;
               auto sop0 = sop.sums[0].second;
               // we assume the rest of terms have the same label/dagger/parity
               auto block  = sop0.block;
               char label  = sop0.label;
               bool dagger = sop0.dagger;
               bool parity = sop0.parity;
               int  index0 = sop0.index;
               int  nbar0  = sop0.nbar;
               if(debug){
                  std::cout << " idx=" << idx
                     << " len=" << len
                     << " block=" << block
                     << " label=" << label
                     << " dagger=" << dagger
                     << " parity=" << parity
                     << " index0=" << index0 
                     << std::endl;
               }
               if(skipId and label == 'I') continue;
               const auto& qops = qops_dict.at(block);

               // form opsum = wt0*op0 + wt1*op1 + ...
               const auto& op0 = qops(label).at(index0);
               if(dagger) wt0 = tools::conjugate(wt0);
               auto optmp = wt0*((nbar0==0)? op0 : op0.K(nbar0));      
               for(int k=1; k<len; k++){
                  auto wtk = sop.sums[k].first;
                  auto sopk = sop.sums[k].second;
                  int indexk = sopk.index;
                  int nbark  = sopk.nbar;
                  const auto& opk = qops(label).at(indexk);
                  if(dagger) wtk = tools::conjugate(wtk);
                  optmp += wtk*((nbark==0)? opk : opk.K(nbark));
               } // k
               if(dagger) linalg::xconj(optmp.size(), optmp.data());
               
               // opN*|wf>
               if(!applied){
                  opxwf = contract_opxwf(block, wf, optmp, dagger); // optmp[block]*|wf>
                  applied = true;
               }else{
                  opxwf = contract_opxwf(block,opxwf,optmp,dagger); // optmp[block]*|opxwf>
               }
               
               // impose antisymmetry by adding fermionic signs here
               if(block == block2 and parity){ 
                  if(block1 == "l"){ // lc or lr
                     opxwf.row_signed();
                  }else if(block1 == "c"){
                     opxwf.mid_signed();
                  }
               }
            } // idx
            assert(applied); // ZL@20240910 must be applied, otherwise the formula is empty
            if(it == 0) Hwf.init(opxwf.info);
            linalg::xaxpy(Hwf.size(), 1.0, opxwf.data(), Hwf.data());
         } // it
         return Hwf;
      }

   template <typename Tm>
      void symbolic_kernel_renorm(const std::string superblock,
            const renorm_tasks<Tm>& rtasks,
            const stensor3su2<Tm>& site,
            const stensor3su2<Tm>& site2,
            const opersu2_dict<Tm>& qops1,
            const opersu2_dict<Tm>& qops2,
            opersu2_dict<Tm>& qops,
            const bool skipId,
            const bool ifdist1,
            const int verbose){
         std::cout << "error: no implementation of symbolic_kernel_renorm for su2!" << std::endl;
         exit(1);
      }
   template <typename Tm>
      void symbolic_kernel_renorm(const std::string superblock,
            const renorm_tasks<Tm>& rtasks,
            const stensor3<Tm>& site,
            const stensor3<Tm>& site2,
            const oper_dict<Tm>& qops1,
            const oper_dict<Tm>& qops2,
            oper_dict<Tm>& qops,
            const bool skipId,
            const bool ifdist1,
            const int verbose){
         if(qops.mpirank==0 and verbose>1){
            std::cout << "ctns::symbolic_kernel_renorm"
               << " skipId=" << skipId
               << " size(formulae)=" << rtasks.size() 
               << std::endl;
         }

         // ZL@20240406: initialize opS & opH in the case of ifdist1,
         // because otherwise they may not be touched for large mpisize
         if(ifdist1){
            for(const auto& key : "SH"){
               for(const auto& pr : qops(key)){
                  int index = pr.first;
                  auto& op = pr.second;
                  memset(op._data, 0, op.size()*sizeof(Tm));
               }
            }
         }

         // perform renormalization
         const std::string block1 = superblock.substr(0,1);
         const std::string block2 = superblock.substr(1,2);
         const oper_dictmap<Tm> qops_dict = {{block1,qops1},
            {block2,qops2}};
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
         for(int i=0; i<rtasks.size(); i++){
            const auto& task = rtasks.op_tasks[i];
            auto key = std::get<0>(task);
            auto index = std::get<1>(task);
            auto formula = std::get<2>(task);
            auto size = formula.size();
            if(verbose>2){
               std::cout << "rank=" << qops.mpirank 
                  << " idx=" << i 
                  << " op=" << key
                  << " index=" << index
                  << " size=" << size
                  << std::endl;
               formula.display("formula", 1);
            }
            if(size == 0) continue;
            auto opxwf = symbolic_renorm_single(block1, block2, qops_dict, key, formula, site2, skipId);
            auto op = contract_qt3_qt3(superblock, site, opxwf);
            if(key == 'H') op += op.H();
            if(key == 'H' && qops.ifkr) op += op.K();
            linalg::xcopy(op.size(), op.data(), qops(key)[index].data());
         } // i
      }

} // ctns

#endif
