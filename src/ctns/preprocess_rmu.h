#ifndef PREPROCESS_RMU_H
#define PREPROCESS_RMU_H

#include "preprocess_header.h"
#include "preprocess_rlist.h"

namespace ctns{

   // R[mu] = coeff*O1*O2
   template <typename Tm>
      struct Rmu_ptr{
         public:
            bool empty() const{ return terms==0; }
            bool identity(const int i) const{ return loc[i]==-1; }
            void init(const bool ifDirect,
                  const int k, const int it,
                  const symbolic_task<Tm>& R_formulae,
                  const oper_dictmap<Tm>& qops_dict,
                  const rintermediates<Tm>& rinter,
                  const std::map<std::string,int>& oploc);
            // onedot
            void gen_Rlist2(const int alg_rcoper,
                  Tm** opaddr, 
                  const std::string superblock,
                  const qinfo3<Tm>& site_info, 
                  Rlist2<Tm>& Rlst2,
                  size_t& blksize,
                  size_t& blksize0,
                  double& cost,
                  const bool ifdagger) const;
         public:
            qinfo2<Tm>* rinfo;
            size_t offrop = 0;
            bool parity[3] = {false,false,false};
            bool dagger[3] = {false,false,false};
            qinfo2<Tm>* info[3] = {nullptr,nullptr,nullptr};
            int loc[3] = {-1,-1,-1};
            size_t off[3] = {0,0,0};
            int terms = 0, cterms = 0; // terms corresponds to 'c' used for alg_rcoper=1
            Tm coeff = 1.0, coeffH = 1.0;
            // intermediates [direct] -> we assume each rmu contains only one intermediates
            int posInter = -1, lenInter = -1;
            size_t offInter = 0, ldaInter = 0;
      };

   template <typename Tm>
      void Rmu_ptr<Tm>::init(const bool ifDirect,
            const int k, const int it,
            const symbolic_task<Tm>& R_formulae,
            const oper_dictmap<Tm>& qops_dict,
            const rintermediates<Tm>& rinter,
            const std::map<std::string,int>& oploc){
         const auto& RTerm = R_formulae.tasks[it];
         terms = RTerm.size();
         for(int idx=terms-1; idx>=0; idx--){
            const auto& sop = RTerm.terms[idx];
            const auto& sop0 = sop.sums[0].second;
            const auto& par = sop0.parity;
            const auto& dag = sop0.dagger;
            const auto& block = sop0.block;
            const auto& label = sop0.label;
            const auto& index0 = sop0.index;
            const auto& qops = qops_dict.at(block); 
            const auto& op0 = qops(label).at(index0);
            int pos = oploc.at(block); 
            parity[pos] = par;
            dagger[pos] = dag;
            info[pos] = const_cast<qinfo2<Tm>*>(&op0.info);
            if(block[0]=='c') cterms += 1;
            int len = sop.size();
            if(len == 1){
               coeff *= sop.sums[0].first;
               loc[pos] = pos;
               off[pos] = qops._offset.at(std::make_pair(label,index0)); // qops
            }else{
               loc[pos] = locInter;
               if(!ifDirect){
                  off[pos] = rinter._offset.at(std::make_tuple(k,it,idx)); // intermediates
               }else{
                  off[pos] = qops._offset.at(std::make_pair(label,index0)); // fake intermediates
                  posInter = pos;
                  lenInter = len;
                  offInter = rinter._offset.at(std::make_tuple(k,it,idx)); // alpha
                  const auto& sop1 = sop.sums[1].second; // used for determine LDA
                  const auto& index1 = sop1.index;
                  const auto& op1 = qops(label).at(index1);
                  //ldaInter = std::distance(op0._data, op1._data); // Ca & Cb can be of different dimes for isym=2
                  ldaInter = qops._offset.at(std::make_pair(label,index1)) - qops._offset.at(std::make_pair(label,index0));
               }
            }
         } // idx
         coeffH = tools::conjugate(coeff)*RTerm.Hsign(); 
      }

   // lc: O[bc,bc'] = psi*[br,bc,bm] sigma[br,bc',bm] (Ol^dagger0[br,br'] Oc^dagger1[bm,bm']) psi[br',bc',bm']
   // cr: O[br,br'] = psi*[br,bc,bm] sigma[br',bc,bm] (Oc^dagger0[bm,bm'] Or^dagger1[bc,bc']) psi[br',bc',bm']
   // lr: O[bm,bm'] = psi*[br,bc,bm] sigma[br,bc,bm'] (Ol^dagger0[br,br'] Or^dagger1[bc,bc']) psi[br',bc',bm']
   template <typename Tm>
      void Rmu_ptr<Tm>::gen_Rlist2(const int alg_rcoper,
            Tm** opaddr,
            const std::string superblock,
            const qinfo3<Tm>& site_info,
            Rlist2<Tm>& Rlst2,
            size_t& blksize,
            size_t& blksize0,
            double& cost,
            const bool ifdagger) const{
         if(this->empty()) return;
         int bo[3], bi[3], bi2[3];
         // psi[br',bc',bm']
         for(int i=0; i<site_info._nnzaddr.size(); i++){
            int idx = site_info._nnzaddr[i];
            site_info._addr_unpack(idx,bi[0],bi[1],bi[2]);
            Rblock<Tm> Rblk(terms,cterms,alg_rcoper);
            Rblk.offin = site_info._offset[idx]-1;
            Rblk.dimin[0] = site_info.qrow.get_dim(bi[0]); // br
            Rblk.dimin[1] = site_info.qcol.get_dim(bi[1]); // bc
            Rblk.dimin[2] = site_info.qmid.get_dim(bi[2]); // bm
            // finding the corresponding operator blocks given {bo[0],bo[1],bo[2]}
            bool symAllowed = true;
            Tm coeff_coper = 1.0;
            for(int k=0; k<3; k++){ // l,r,c
               Rblk.dagger[k] = dagger[k]^ifdagger;
               if(this->identity(k)){
                  bo[k] = bi[k];
                  Rblk.dimout[k] = Rblk.dimin[k];
               }else{
                  bool iftrans = dagger[k]^ifdagger;
                  bo[k] = iftrans? info[k]->_br2bc[bi[k]] : info[k]->_bc2br[bi[k]];
                  if(bo[k] == -1){
                     symAllowed = false;
                     break;
                  }else{
                     int jdx = iftrans? info[k]->_addr(bi[k],bo[k]) : info[k]->_addr(bo[k],bi[k]);
                     assert(info[k]->_offset[jdx] != 0);
                     Rblk.loc[k] = loc[k];
                     Rblk.off[k] = off[k]+(info[k]->_offset[jdx]-1);
                     Rblk.dimout[k] = iftrans? info[k]->qcol.get_dim(bo[k]) : info[k]->qrow.get_dim(bo[k]);
                     // special treatment of op[c] for NSz symmetry
                     if(k >= 2 && alg_rcoper == 1){
                        assert(k == loc[k]); // op[c] cannot be intermediates
                        Tm coper = *(opaddr[loc[k]] + Rblk.off[k]);
                        coeff_coper *= Rblk.dagger[k]? tools::conjugate(coper) : coper;
                        if(std::abs(coeff_coper)<thresh_coper){
                           symAllowed = false;
                           break;
                        }
                     }
                  }
               }
            }
            if(!symAllowed) continue;
            // Additional information for psi*[br,bc,bm]
            // lc: O[bc,bc'] = psi*[br,bc,bm] sigma[br,bc',bm] 
            // cr: O[br,br'] = psi*[br,bc,bm] sigma[br',bc,bm] 
            // lr: O[bm,bm'] = psi*[br,bc,bm] sigma[br,bc,bm']
            bi2[0] = bo[0];
            bi2[1] = bo[1];
            bi2[2] = bo[2];
            Rblk.dimin2[0] = Rblk.dimout[0];
            Rblk.dimin2[1] = Rblk.dimout[1];
            Rblk.dimin2[2] = Rblk.dimout[2];
            // compute sign due to parity
            int icase = 0;
            Rblk.coeff = (ifdagger? coeffH : coeff)*coeff_coper;
            if(superblock == "lc"){
               // OlOc|lc> = Ol|l> * Oc|c> (-1)^{p(Oc)*p(l)}
               icase = 1;
               int p0 = site_info.qrow.get_parity(bi[0]);
               if(parity[2] && p0==1) Rblk.coeff *= -1.0;
            }else if(superblock == "cr"){ 
               // OcOr|cr> = Oc|c> * Or|r> (-1)^{p(Or)*p(c)}
               icase = 0;
               int p0 = site_info.qmid.get_parity(bi[2]);
               if(parity[1] && p0==1) Rblk.coeff *= -1.0;
            }else if(superblock == "lr"){
               // OlOr|lr> = Ol|l> * Or|r> (-1)^{p(Or)*p(l)} 
               icase = 2;
               int p0 = site_info.qrow.get_parity(bi[0]);
               if(parity[1] && p0==1) Rblk.coeff *= -1.0;
            }
            Rblk.icase = icase;
            bi2[icase] = rinfo->_bc2br[bo[icase]];
            if(bi2[icase] == -1) continue;
            size_t offin2 = site_info._offset[site_info._addr(bi2[0],bi2[1],bi2[2])];
            if(offin2 == 0) continue;
            Rblk.offin2 = offin2-1;
            Rblk.dimin2[icase] = rinfo->qrow.get_dim(bi2[icase]);
            size_t offop = rinfo->_offset[rinfo->_addr(bi2[icase],bo[icase])];
            if(offop == 0) continue;
            Rblk.offrop = offrop+offop-1; // add global offset
            Rblk.size = Rblk.dimin2[icase]*Rblk.dimout[icase];
            Rblk.setup();
            blksize = std::max(blksize, Rblk.blksize);
            cost += Rblk.cost;
            // Intermediates
            if(posInter != -1){
               Rblk.posInter = posInter;
               Rblk.lenInter = lenInter;
               Rblk.offInter = offInter;
               Rblk.ldaInter = ldaInter;
               blksize0 = std::max(blksize0, Rblk.dimout[posInter]*Rblk.dimin[posInter]);
            }
            Rlst2[bi2[icase]].push_back(Rblk);
         } // i
      }

} // ctns

#endif
