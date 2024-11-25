#ifndef PREPROCESS_RMU_H
#define PREPROCESS_RMU_H

#include "preprocess_header.h"
#include "preprocess_rlist.h"

namespace ctns{

   // R[mu] = coeff*O1*O2
   template <bool ifab, typename Tm>
      struct Rmu_ptr{
         public:
            bool empty() const{ return terms==0; }
            bool skip(const int i) const{ return loc[i]==-1; }
            void init(const bool ifDirect,
                  const int k, const int it,
                  const symbolic_task<Tm>& R_formulae,
                  const qoper_dictmap<ifab,Tm>& qops_dict,
                  const rintermediates<ifab,Tm>& rinter,
                  const std::map<std::string,int>& oploc,
                  const bool skipId);
            // onedot
            template <bool y=ifab, std::enable_if_t<y,int> = 0> 
            void gen_Rlist2(const int alg_rcoper,
                  Tm** opaddr, 
                  const std::string superblock,
                  const qinfo3type<ifab,Tm>& site_info, 
                  const qinfo3type<ifab,Tm>& site2_info,
                  Rlist2<Tm>& Rlst2,
                  size_t& blksize,
                  size_t& blksize0,
                  double& cost,
                  const bool ifdagger) const;
            template <bool y=ifab, std::enable_if_t<!y,int> = 0> 
            void gen_Rlist2(const int alg_rcoper,
                  Tm** opaddr, 
                  const std::string superblock,
                  const qinfo3type<ifab,Tm>& site_info, 
                  const qinfo3type<ifab,Tm>& site2_info, 
                  Rlist2<Tm>& Rlst2,
                  size_t& blksize,
                  size_t& blksize0,
                  double& cost,
                  const bool ifdagger) const;
         public:
            qinfo2type<ifab,Tm>* rinfo; // ptr for the output operator
            size_t offrop = 0;
            bool parity[3] = {false,false,false};
            bool dagger[3] = {false,false,false};
            qinfo2type<ifab,Tm>* info[3] = {nullptr,nullptr,nullptr};
            int loc[3] = {-1,-1,-1};
            size_t off[3] = {0,0,0};
            int terms = 0, cterms = 0; // terms corresponds to 'c' used for alg_rcoper=1
            Tm coeff = 1.0, coeffH = 1.0;
            // intermediates [direct] -> we assume each rmu contains only one intermediates
            int posInter = -1, lenInter = -1;
            size_t offInter = 0, ldaInter = 0;
            // intermediate spins {S1,S2,S12}
            int tspins[3] = {-1,-1,-1};
      };

   template <bool ifab, typename Tm>
      void Rmu_ptr<ifab,Tm>::init(const bool ifDirect,
            const int k, const int it,
            const symbolic_task<Tm>& R_formulae,
            const qoper_dictmap<ifab,Tm>& qops_dict,
            const rintermediates<ifab,Tm>& rinter,
            const std::map<std::string,int>& oploc,
            const bool skipId){
         const auto& RTerm = R_formulae.tasks[it];
         for(int idx=RTerm.size()-1; idx>=0; idx--){
            const auto& sop = RTerm.terms[idx];
            const auto& sop0 = sop.sums[0].second;
            const auto& par = sop0.parity;
            const auto& dag = sop0.dagger;
            const auto& block = sop0.block;
            const auto& label = sop0.label;
            if(label == 'I' and skipId){
               assert(sop.size() == 1);
               coeff *= sop.sums[0].first;
               continue; // as we add 'I' into formula, we may need to skip it if skipId=true
            }
            terms += 1;
            const auto& index0 = sop0.index;
            const auto& qops = qops_dict.at(block); 
            const auto& op0 = qops(label).at(index0);
            int pos = oploc.at(block); 
            parity[pos] = par;
            dagger[pos] = dag;
            info[pos] = const_cast<qinfo2type<ifab,Tm>*>(&op0.info);
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
                  ldaInter = qops._offset.at(std::make_pair(label,index1)) - qops._offset.at(std::make_pair(label,index0));
               }
            }
         } // idx
         coeffH = tools::conjugate(coeff)*RTerm.Hsign();
         // intermediate spins
         if(RTerm.ispins.size() != 0){
            assert(RTerm.ispins.size() == 1);
            const auto& ispin = RTerm.ispins[0];
            tspins[0] = std::get<0>(ispin);
            tspins[1] = std::get<1>(ispin);
            tspins[2] = std::get<2>(ispin);
         }
      }

   // Abelian case
   // lc: O[bc,bc'] = psi*[br,bc,bm] sigma[br,bc',bm] (Ol^dagger0[br,br'] Oc^dagger1[bm,bm'] psi2[br',bc',bm'])
   // cr: O[br,br'] = psi*[br,bc,bm] sigma[br',bc,bm] (Oc^dagger0[bm,bm'] Or^dagger1[bc,bc'] psi2[br',bc',bm'])
   // lr: O[bm,bm'] = psi*[br,bc,bm] sigma[br,bc,bm'] (Ol^dagger0[br,br'] Or^dagger1[bc,bc'] psi2[br',bc',bm'])
   template <bool ifab, typename Tm>
      template <bool y, std::enable_if_t<y,int>>
      void Rmu_ptr<ifab,Tm>::gen_Rlist2(const int alg_rcoper,
            Tm** opaddr,
            const std::string superblock,
            const qinfo3type<ifab,Tm>& site_info,
            const qinfo3type<ifab,Tm>& site2_info,
            Rlist2<Tm>& Rlst2,
            size_t& blksize,
            size_t& blksize0,
            double& cost,
            const bool ifdagger) const{
         if(this->empty()) return;
         int k1, k2, k3;
         if(superblock == "lc"){
            // OlOc|lc> = Ol|l> * Oc|c> (-1)^{p(Oc)*p(l)}
            k1 = 0;
            k2 = 2;
            k3 = 1; 
         }else if(superblock == "cr"){
            // OcOr|cr> = Oc|c> * Or|r> (-1)^{p(Or)*p(c)}
            k1 = 2;
            k2 = 1;
            k3 = 0;
         }else if(superblock == "lr"){
            // OlOr|lr> = Ol|l> * Or|r> (-1)^{p(Or)*p(l)} 
            k1 = 0;
            k2 = 1;
            k3 = 2;
         }else{
            std::cout << "error: not supported yet for gen_Rlist2 with superblock=" 
               << superblock << std::endl;
            exit(1);
         }
         int bi[3];  // psi2[br',bc',bm']
         int bo[3];  // sigma
         int bi2[3]; // psi*[br,bc,bm]
         // loop over psi2[br',bc',bm']
         for(int i=0; i<site2_info._nnzaddr.size(); i++){
            int idx = site2_info._nnzaddr[i];
            site2_info._addr_unpack(idx,bi[0],bi[1],bi[2]);
            Rblock<Tm> Rblk(terms,cterms,alg_rcoper);
            Rblk.icase = k3;
            Rblk.offin = site2_info._offset[idx]-1;
            Rblk.dimin[0] = site2_info.qrow.get_dim(bi[0]); // br
            Rblk.dimin[1] = site2_info.qcol.get_dim(bi[1]); // bc
            Rblk.dimin[2] = site2_info.qmid.get_dim(bi[2]); // bm
            // finding the corresponding operator blocks: {bo[0],bo[1],bo[2]}
            bool symAllowed = true;
            Tm coeff_coper = 1.0;
            for(int k=0; k<3; k++){ // l,r,c
               if(this->skip(k)){
                  bo[k] = bi[k];
                  Rblk.dimout[k] = Rblk.dimin[k];
               }else{
                  Rblk.dagger[k] = dagger[k]^ifdagger;
                  bool iftrans = dagger[k]^ifdagger;
                  bo[k] = iftrans? info[k]->_br2bc[bi[k]] : info[k]->_bc2br[bi[k]]; // out sector
                  if(bo[k] == -1){
                     symAllowed = false;
                     break;
                  }else{
                     // setup the location of data
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
            // Setup additional information for psi*[br,bc,bm]
            // lc: O[bc,bc'] = psi*[br,bc,bm] sigma[br,bc',bm] 
            // cr: O[br,br'] = psi*[br,bc,bm] sigma[br',bc,bm] 
            // lr: O[bm,bm'] = psi*[br,bc,bm] sigma[br,bc,bm']
            // For Abelian case: bi2 and dimin2 are uniquely determined.
            bi2[k1] = bo[k1];
            bi2[k2] = bo[k2];
            bi2[k3] = rinfo->_bc2br[bo[k3]];
            if(bi2[k3] == -1) continue;
            size_t offop = rinfo->get_offset(bi2[k3],bo[k3]);
            if(offop == 0) continue;
            size_t offin2 = site_info.get_offset(bi2[0],bi2[1],bi2[2]);
            if(offin2 == 0) continue;
            Rblk.offin2 = offin2-1;
            Rblk.offrop = offrop+offop-1; // add global offset
            // compute sign due to parity
            Rblk.coeff = (ifdagger? coeffH : coeff)*coeff_coper;
            if(superblock == "lc"){
               // OlOc|lc> = Ol|l> * Oc|c> (-1)^{p(Oc)*p(l)}
               int p0 = site2_info.qrow.get_parity(bi[0]);
               if(parity[2] && p0==1) Rblk.coeff *= -1.0;
            }else if(superblock == "cr"){ 
               // OcOr|cr> = Oc|c> * Or|r> (-1)^{p(Or)*p(c)}
               int p0 = site2_info.qmid.get_parity(bi[2]);
               if(parity[1] && p0==1) Rblk.coeff *= -1.0;
            }else if(superblock == "lr"){
               // OlOr|lr> = Ol|l> * Or|r> (-1)^{p(Or)*p(l)} 
               int p0 = site2_info.qrow.get_parity(bi[0]);
               if(parity[1] && p0==1) Rblk.coeff *= -1.0;
            }
            // setup dimensions
            Rblk.dimin2[k1] = Rblk.dimout[k1];
            Rblk.dimin2[k2] = Rblk.dimout[k2];
            Rblk.dimin2[k3] = rinfo->qrow.get_dim(bi2[k3]);
            Rblk.size = Rblk.dimin2[k3]*Rblk.dimout[k3];
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
            Rlst2[bi2[k3]*rinfo->_cols+bo[k3]].push_back(Rblk);
         } // i
      }

   // Non-Abelian case: just work for MPS
   // lc: O[bc,bc'] = psi*[br,bc,bm] sigma[br,bc',bm] (Ol^dagger0[br,br'] Oc^dagger1[bm,bm'] psi2[br',bc',bm'])
   // cr: O[br,br'] = psi*[br,bc,bm] sigma[br',bc,bm] (Oc^dagger0[bm,bm'] Or^dagger1[bc,bc'] psi2[br',bc',bm'])
   template <bool ifab, typename Tm>
      template <bool y, std::enable_if_t<!y,int>>
      void Rmu_ptr<ifab,Tm>::gen_Rlist2(const int alg_rcoper,
            Tm** opaddr,
            const std::string superblock,
            const qinfo3type<ifab,Tm>& site_info,
            const qinfo3type<ifab,Tm>& site2_info,
            Rlist2<Tm>& Rlst2,
            size_t& blksize,
            size_t& blksize0,
            double& cost,
            const bool ifdagger) const{
         if(this->empty()) return;
         const std::map<int,const qbond*> qmap = {{0,&site_info.qrow},
                                                  {1,&site_info.qcol},
                                                  {2,&site_info.qmid}};
         const std::map<int,const qbond*> qmap2 = {{0,&site2_info.qrow},
                                                   {1,&site2_info.qcol},
                                                   {2,&site2_info.qmid}};
         int k1, k2, k3;
         if(superblock == "lc"){
            // OlOc|lc> = Ol|l> * Oc|c> (-1)^{p(Oc)*p(l)}
            k1 = 0;
            k2 = 2;
            k3 = 1; 
         }else if(superblock == "cr"){
            // OcOr|cr> = Oc|c> * Or|r> (-1)^{p(Or)*p(c)}
            k1 = 2;
            k2 = 1;
            k3 = 0;
         }else{
            std::cout << "error: not supported yet for gen_Rlist2 with superblock=" 
               << superblock << std::endl;
            exit(1);
         }
         // loop over psi2[br',bc',bm']
         int bi[3], ts12;
         for(int i=0; i<site2_info._nnzaddr.size(); i++){
            auto key = site2_info._nnzaddr[i];
            bi[0] = std::get<0>(key);
            bi[1] = std::get<1>(key);
            bi[2] = std::get<2>(key);
            ts12  = std::get<3>(key);
            const auto& bo1vec = (this->skip(k1))? std::vector<int>({bi[k1]}) : 
               (dagger[k1]^ifdagger? info[k1]->_br2bc[bi[k1]] : info[k1]->_bc2br[bi[k1]]);
            const auto& bo2vec = (this->skip(k2))? std::vector<int>({bi[k2]}) : 
               (dagger[k2]^ifdagger? info[k2]->_br2bc[bi[k2]] : info[k2]->_bc2br[bi[k2]]);
            for(const auto& bo1 : bo1vec){
               for(const auto& bo2 : bo2vec){
                  int bo[3]; // sigma
                  bo[k1] = bo1;
                  bo[k2] = bo2;
                  bo[k3] = bi[k3]; 
                  // Additional information for psi*[br,bc,bm]
                  // lc: O[bc,bc'] = psi*[br,bc,bm] sigma[br,bc',bm] 
                  // cr: O[br,br'] = psi*[br,bc,bm] sigma[br',bc,bm] 
                  // lr: O[bm,bm'] = psi*[br,bc,bm] sigma[br,bc,bm']
                  int bi2[3]; // psi*
                  bi2[k1] = bo[k1];
                  bi2[k2] = bo[k2];
                  for(int b3=0; b3<qmap.at(k3)->size(); b3++){
                     bi2[k3] = b3;
                     // check symmetry for output and psi*
                     size_t offop = rinfo->get_offset(bi2[k3],bo[k3]);
                     if(offop == 0) continue;
                     int ts12p = (qmap.at(k3)->get_sym(b3)).ts();
                     size_t offin2 = site_info.get_offset(bi2[0],bi2[1],bi2[2],ts12p);
                     if(offin2 == 0) continue;
                     // determine the block
                     Rblock<Tm> Rblk(terms,cterms,alg_rcoper);
                     Rblk.icase = k3;
                     Rblk.offin  = site2_info.get_offset(bi[0],bi[1],bi[2],ts12)-1;
                     Rblk.dimin[0] = site2_info.qrow.get_dim(bi[0]); // br
                     Rblk.dimin[1] = site2_info.qcol.get_dim(bi[1]); // bc
                     Rblk.dimin[2] = site2_info.qmid.get_dim(bi[2]); // bm
                     Rblk.offin2 = offin2-1;
                     Rblk.offrop = offrop+offop-1; // add global offset
                     // update Rblk.dagger/loc/off
                     Tm coeff_coper = 1.0;
                     bool skip = false;
                     for(int k=0; k<3; k++){
                        if(this->skip(k)){
                           Rblk.dimout[k] = Rblk.dimin[k]; 
                        }else{
                           Rblk.dagger[k] = dagger[k]^ifdagger;
                           Rblk.loc[k] = loc[k];
                           size_t offset = Rblk.dagger[k]? info[k]->get_offset(bi[k],bo[k]) :
                              info[k]->get_offset(bo[k],bi[k]);
                           assert(offset != 0);
                           Rblk.off[k] = off[k]+offset-1;
                           Rblk.dimout[k] = Rblk.dagger[k]? info[k]->qcol.get_dim(bo[k]) : info[k]->qrow.get_dim(bo[k]);
                           // su2 case: sgn from bar{bar{Ts}} = (-1)^2s Ts
                           if(dagger[k] && ifdagger) coeff_coper *= parity[k]? -1.0 : 1.0;
                           if(k >= 2 && alg_rcoper == 1){
                              assert(k == loc[k]); // op[c] cannot be intermediates
                              Tm coper = *(opaddr[loc[k]] + Rblk.off[k]);
                              coeff_coper *= Rblk.dagger[k]? tools::conjugate(coper) : coper;
                              if(std::abs(coeff_coper)<thresh_coper){
                                 skip = true;
                                 break;
                              }
                           }
                        }
                     } // k
                     if(skip) continue;
                     // sign factors due to spin
                     // <S1p|O1|S1><S2p|O2|S2>
                     int ts1p = (qmap.at(k1)->get_sym(bo[k1])).ts();
                     int ts2p = (qmap.at(k2)->get_sym(bo[k2])).ts();
                     int ts1  = (qmap2.at(k1)->get_sym(bi[k1])).ts();
                     int ts2  = (qmap2.at(k2)->get_sym(bi[k2])).ts();
                     coeff_coper *= std::sqrt((ts1p+1.0)*(ts2p+1.0)*(ts12+1.0)*(tspins[2]+1.0))*
                                    fock::wigner9j(ts1p,ts2p,ts12p,ts1,ts2,ts12,tspins[0],tspins[1],tspins[2]);
                     if(std::abs(coeff_coper)<thresh_coper) continue;
                     // sign from adjoint
                     if(!this->skip(k1) && dagger[k1]^ifdagger){
                        int ts = tspins[0] + ts1p - ts1;
                        coeff_coper *= ((ts/2)%2==0? 1.0 : -1.0)*std::sqrt((ts1+1.0)/(ts1p+1.0));
                     }
                     if(!this->skip(k2) && dagger[k2]^ifdagger){
                        int ts = tspins[1] + ts2p - ts2;
                        coeff_coper *= ((ts/2)%2==0? 1.0 : -1.0)*std::sqrt((ts2+1.0)/(ts2p+1.0));
                     }
                     // compute sign due to parity
                     Rblk.coeff = (ifdagger? coeffH : coeff)*coeff_coper;
                     if(k3 == 1){
                        // OlOc|lc> = Ol|l> * Oc|c> (-1)^{p(Oc)*p(l)}
                        int p0 = site2_info.qrow.get_parity(bi[0]);
                        if(parity[2] && p0==1) Rblk.coeff *= -1.0;
                     }else if(k3 == 0){ 
                        // OcOr|cr> = Oc|c> * Or|r> (-1)^{p(Or)*p(c)}
                        int p0 = site2_info.qmid.get_parity(bi[2]);
                        if(parity[1] && p0==1) Rblk.coeff *= -1.0;
                     }else if(k3 == 2){
                        // OlOr|lr> = Ol|l> * Or|r> (-1)^{p(Or)*p(l)} 
                        int p0 = site2_info.qrow.get_parity(bi[0]);
                        if(parity[1] && p0==1) Rblk.coeff *= -1.0;
                     }
                     // setup dimensions
                     Rblk.dimin2[0] = site_info.qrow.get_dim(bi2[0]);
                     Rblk.dimin2[1] = site_info.qcol.get_dim(bi2[1]);
                     Rblk.dimin2[2] = site_info.qmid.get_dim(bi2[2]);
                     Rblk.size = Rblk.dimin2[k3]*Rblk.dimout[k3];
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
                     Rlst2[bi2[k3]*rinfo->_cols+bo[k3]].push_back(Rblk);
                  } // b3
               } // b2
            } // b1
         } // i
      }

} // ctns

#endif
