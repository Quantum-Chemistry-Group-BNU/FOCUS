#ifndef PREPROCESS_HMU_H
#define PREPROCESS_HMU_H

#include "preprocess_header.h"
#include "preprocess_hxlist.h"

namespace ctns{

   // H[mu] = coeff*Ol*Or*Oc (onedot)
   //       = coeff*Ol*Or*Oc1*Oc2 (twodot) 
   template <bool ifab, typename Tm>
      struct Hmu_ptr{
         public:
            bool empty() const{ return terms==0; }
            bool identity(const int i) const{ return loc[i]==-1; }
            void init(const bool ifDirect, 
                  const int it,
                  const symbolic_task<Tm>& H_formulae,
                  const qoper_dictmap<ifab,Tm>& qops_dict,
                  const hintermediates<ifab,Tm>& hinter,
                  const std::map<std::string,int>& oploc);
            // onedot
            template <bool y=ifab, std::enable_if_t<y,int> = 0> 
               void gen_Hxlist2(const int alg_hcoper,
                     Tm** opaddr,
                     const qinfo3type<ifab,Tm>& wf_info, 
                     Hxlist2<Tm>& Hxlst2,
                     size_t& blksize,
                     size_t& blksize0,
                     double& cost,
                     const bool ifdagger) const;
            template <bool y=ifab, std::enable_if_t<!y,int> = 0> 
               void gen_Hxlist2(const int alg_hcoper,
                     Tm** opaddr,
                     const qinfo3type<ifab,Tm>& wf_info, 
                     Hxlist2<Tm>& Hxlst2,
                     size_t& blksize,
                     size_t& blksize0,
                     double& cost,
                     const bool ifdagger) const;
            // twodot
            template <bool y=ifab, std::enable_if_t<y,int> = 0> 
               void gen_Hxlist2(const int alg_hcoper,
                     Tm** opaddr,
                     const qinfo4type<ifab,Tm>& wf_info, 
                     Hxlist2<Tm>& Hxlst2,
                     size_t& blksize,
                     size_t& blksize0,
                     double& cost,
                     const bool ifdagger) const;
            template <bool y=ifab, std::enable_if_t<!y,int> = 0> 
               void gen_Hxlist2(const int alg_hcoper,
                     Tm** opaddr,
                     const qinfo4type<ifab,Tm>& wf_info, 
                     Hxlist2<Tm>& Hxlst2,
                     size_t& blksize,
                     size_t& blksize0,
                     double& cost,
                     const bool ifdagger) const;
         public:
            bool parity[4] = {false,false,false,false};
            bool dagger[4] = {false,false,false,false};
            qinfo2type<ifab,Tm>* info[4] = {nullptr,nullptr,nullptr,nullptr};
            int loc[4] = {-1,-1,-1,-1};
            size_t off[4] = {0,0,0,0};
            int terms = 0, cterms = 0; // terms corresponds to 'c' used for alg_hcoper=1
            Tm coeff = 1.0, coeffH = 1.0;
            // intermediates [direct] -> we assume each hmu contains only one intermediates
            int posInter = -1, lenInter = -1;
            size_t offInter = 0, ldaInter = 0;
            // intermediate spins:
            // twodot: {{S1,S2,S12},{S3,S4,S34},{S12,S34,S1234}} ((lc1)(c2r))
            // onedot: {{S1,S2,S12},{S12,S3,S123}} ((lc)r)
            //         {{S2,S3,S23},{S1,S23,S123}} (l(cr)}
            int tspins[9] = {-1,-1,-1,
               -1,-1,-1,
               -1,-1,-1};
      };

   template <bool ifab, typename Tm>
      void Hmu_ptr<ifab,Tm>::init(const bool ifDirect,
            const int it,
            const symbolic_task<Tm>& H_formulae,
            const qoper_dictmap<ifab,Tm>& qops_dict,
            const hintermediates<ifab,Tm>& hinter,
            const std::map<std::string,int>& oploc){
         const auto& HTerm = H_formulae.tasks[it];
         for(int idx=HTerm.size()-1; idx>=0; idx--){
            const auto& sop = HTerm.terms[idx];
            const auto& sop0 = sop.sums[0].second;
            const auto& par = sop0.parity;
            const auto& dag = sop0.dagger;
            const auto& block = sop0.block;
            const auto& label = sop0.label;
            if(label == 'I'){
               assert(sop.size() == 1);
               coeff *= sop.sums[0].first;
               continue; // for su2 case, we add 'I' into formula
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
                  off[pos] = hinter._offset.at(std::make_pair(it,idx)); // intermediates
               }else{
                  off[pos] = qops._offset.at(std::make_pair(label,index0)); // fake intermediates
                  posInter = pos;
                  lenInter = len; 
                  offInter = hinter._offset.at(std::make_pair(it,idx)); // alpha
                  const auto& sop1 = sop.sums[1].second; // used for determine LDA
                  const auto& index1 = sop1.index;
                  const auto& op1 = qops(label).at(index1);
                  //ldaInter = std::distance(op0._data, op1._data); // Ca & Cb can be of different dimes for isym=2
                  ldaInter = qops._offset.at(std::make_pair(label,index1)) - qops._offset.at(std::make_pair(label,index0));
               }
            }
         } // idx
         coeffH = tools::conjugate(coeff)*HTerm.Hsign();
         // intermediate spins
         if(HTerm.ispins.size() != 0){
            assert(HTerm.ispins.size() == 3 or HTerm.ispins.size() == 2); // twodot or onedot
            int j = 0;
            for(int i=0; i<HTerm.ispins.size(); i++){
               const auto& ispin = HTerm.ispins[i];
               tspins[j] = std::get<0>(ispin);
               tspins[j+1] = std::get<1>(ispin);
               tspins[j+2] = std::get<2>(ispin);
               j += 3;
            }
         }
      }

   // onedot: 
   // sigma[br,bc,bm] = Ol^dagger0[br,br'] Or^dagger1[bc,bc'] 
   //            Oc1^dagger2[bm,bm'] 
   // 		     wf[br',bc',bm',bv']
   template <bool ifab, typename Tm>
      template <bool y, std::enable_if_t<y,int>>
      void Hmu_ptr<ifab,Tm>::gen_Hxlist2(const int alg_hcoper,
            Tm** opaddr,
            const qinfo3type<ifab,Tm>& wf_info,
            Hxlist2<Tm>& Hxlst2,
            size_t& blksize,
            size_t& blksize0,
            double& cost,
            const bool ifdagger) const{
         if(this->empty()) return;
         int bo[3], bi[3];
         for(int i=0; i<wf_info._nnzaddr.size(); i++){
            int idx = wf_info._nnzaddr[i];
            wf_info._addr_unpack(idx,bo[0],bo[1],bo[2]);
            Hxblock<Tm> Hxblk(3,terms,cterms,alg_hcoper);
            Hxblk.offout = wf_info._offset[idx]-1;
            Hxblk.dimout[0] = wf_info.qrow.get_dim(bo[0]);
            Hxblk.dimout[1] = wf_info.qcol.get_dim(bo[1]);
            Hxblk.dimout[2] = wf_info.qmid.get_dim(bo[2]);
            Hxblk.size = Hxblk.dimout[0]*Hxblk.dimout[1]*Hxblk.dimout[2];
            // finding the corresponding operator blocks given {bo[0],bo[1],bo[2],bo[3]}
            bool symAllowed = true;
            Tm coeff_coper = 1.0;
            for(int k=0; k<3; k++){ // l,r,c
               if(this->identity(k)){
                  // identity operator
                  bi[k] = bo[k];
               }else{
                  // not identity
                  Hxblk.dagger[k] = dagger[k]^ifdagger;
                  bool iftrans = dagger[k]^ifdagger;
                  bi[k] = iftrans? info[k]->_bc2br[bo[k]] : info[k]->_br2bc[bo[k]];
                  if(bi[k] == -1){
                     symAllowed = false;
                     break;
                  }else{
                     int jdx = iftrans? info[k]->_addr(bi[k],bo[k]) : info[k]->_addr(bo[k],bi[k]);
                     assert(info[k]->_offset[jdx] != 0);
                     Hxblk.loc[k] = loc[k];
                     Hxblk.off[k] = off[k]+(info[k]->_offset[jdx]-1);
                     // special treatment of op[c] for NSz symmetry
                     if(k >= 2 && ((alg_hcoper==1 && terms>cterms) || alg_hcoper==2)){
                        assert(k == loc[k]); // op[c] cannot be intermediates
                        Tm coper = *(opaddr[loc[k]] + Hxblk.off[k]);
                        coeff_coper *= Hxblk.dagger[k]? tools::conjugate(coper) : coper;
                        if(std::abs(coeff_coper)<thresh_coper){
                           symAllowed = false;
                           break;
                        }
                     }
                  }
               }
            }
            if(!symAllowed) continue;
            size_t offin = wf_info.get_offset(bi[0],bi[1],bi[2]);
            if(offin == 0) continue; // in case of no matching contractions
            Hxblk.offin = offin-1;
            Hxblk.dimin[0] = wf_info.qrow.get_dim(bi[0]);
            Hxblk.dimin[1] = wf_info.qcol.get_dim(bi[1]);
            Hxblk.dimin[2] = wf_info.qmid.get_dim(bi[2]);
            // compute sign due to parity
            Hxblk.coeff = (ifdagger? coeffH : coeff)*coeff_coper;
            int pl = wf_info.qrow.get_parity(bi[0]);
            int pc = wf_info.qmid.get_parity(bi[2]);
            if(parity[1] && (pl+pc)%2==1) Hxblk.coeff *= -1.0; // Or: Or|lcr> = (-1)^{pl+pc}|lc>*Or|r>
            if(parity[2] && pl%2==1) Hxblk.coeff *= -1.0; // Oc
            Hxblk.setup();
            blksize = std::max(blksize, Hxblk.blksize);
            cost += Hxblk.cost;
            // Intermediates
            if(posInter != -1){
               Hxblk.posInter = posInter;
               Hxblk.lenInter = lenInter;
               Hxblk.offInter = offInter;
               Hxblk.ldaInter = ldaInter;
               blksize0 = std::max(blksize0, Hxblk.dimout[posInter]*Hxblk.dimin[posInter]);
            }
            Hxlst2[i].push_back(Hxblk);
         } // i
      }

   // su2 case:
   template <bool ifab, typename Tm>
      template <bool y, std::enable_if_t<!y,int>>
      void Hmu_ptr<ifab,Tm>::gen_Hxlist2(const int alg_hcoper,
            Tm** opaddr,
            const qinfo3type<ifab,Tm>& wf_info,
            Hxlist2<Tm>& Hxlst2,
            size_t& blksize,
            size_t& blksize0,
            double& cost,
            const bool ifdagger) const{
         if(this->empty()) return;
         // sigma[br',bc',bm']
         int bo[3], tstot;
         tstot = wf_info.sym.ts();
         if(wf_info.couple == CRcouple){
            // l|cr: CRcouple
            int tscrp;
            for(int i=0; i<wf_info._nnzaddr.size(); i++){
               auto key = wf_info._nnzaddr[i];
               bo[0] = std::get<0>(key); // br
               bo[1] = std::get<1>(key); // bc
               bo[2] = std::get<2>(key); // bm
               tscrp = std::get<3>(key); // tscr
               size_t offout = wf_info.get_offset(bo[0],bo[1],bo[2],tscrp);
               assert(offout > 0);
               const auto& bi0vec = this->identity(0)? std::vector<int>({bo[0]}) :
                  (dagger[0]^ifdagger? info[0]->_bc2br[bo[0]] : info[0]->_br2bc[bo[0]]);
               const auto& bi1vec = this->identity(1)? std::vector<int>({bo[1]}) :
                  (dagger[1]^ifdagger? info[1]->_bc2br[bo[1]] : info[1]->_br2bc[bo[1]]);
               const auto& bi2vec = this->identity(2)? std::vector<int>({bo[2]}) :
                  (dagger[2]^ifdagger? info[2]->_bc2br[bo[2]] : info[2]->_br2bc[bo[2]]);
               for(const auto& bi0 : bi0vec){
                  for(const auto& bi1 : bi1vec){
                     for(const auto& bi2 : bi2vec){
                        int bi[3]; // wf
                        bi[0] = bi0;
                        bi[1] = bi1;
                        bi[2] = bi2;
                        // setup Scr
                        int tslp = wf_info.qrow.get_sym(bo[0]).ts(); // l
                        int tsrp = wf_info.qcol.get_sym(bo[1]).ts(); // r
                        int tscp = wf_info.qmid.get_sym(bo[2]).ts(); // c
                        int tsl  = wf_info.qrow.get_sym(bi[0]).ts();
                        int tsr  = wf_info.qcol.get_sym(bi[1]).ts();
                        int tsc  = wf_info.qmid.get_sym(bi[2]).ts();
                        for(int tscr=std::abs(tsc-tsr); tscr<=tsc+tsr; tscr+=2){
                           size_t offin = wf_info.get_offset(bi[0],bi[1],bi[2],tscr);
                           if(offin == 0) continue;
                           // setup block
                           Hxblock<Tm> Hxblk(3,terms,cterms,alg_hcoper);
                           Hxblk.offin = offin-1;
                           Hxblk.offout = offout-1;
                           // update Hxblk.dagger/loc/off
                           Tm coeff_coper = 1.0;
                           bool skip = false;
                           for(int k=0; k<3; k++){ // l,r,c
                              if(this->identity(k)) continue;
                              Hxblk.dagger[k] = dagger[k]^ifdagger; // (O1^d1)^d = O1^(d^d1)
                              Hxblk.loc[k] = loc[k];
                              size_t offset = Hxblk.dagger[k]? info[k]->get_offset(bi[k],bo[k]) : 
                                 info[k]->get_offset(bo[k],bi[k]);
                              assert(offset != 0);
                              Hxblk.off[k] = off[k]+offset-1;
                              // sgn from bar{bar{Ts}} = (-1)^2s Ts
                              if(dagger[k] && ifdagger) coeff_coper *= parity[k]? -1.0 : 1.0;
                              // special treatment of op[c2/c1] for NS symmetry
                              if(k >= 2 && ((alg_hcoper==1 && terms>cterms) || alg_hcoper==2)){ 
                                 assert(k == loc[k]); // op[c] cannot be intermediates
                                 Tm coper = *(opaddr[loc[k]] + Hxblk.off[k]);
                                 coeff_coper *= Hxblk.dagger[k]? tools::conjugate(coper) : coper;
                                 if(std::abs(coeff_coper)<thresh_coper){
                                    skip = true;
                                    break;
                                 }
                              }
                           } // k
                           if(skip) continue;
                           // sign factors due to spin: l|cr
                           // (<Slp|Ol|Sl>)(<Scp|Oc|Sc><Srp|Or|Sr>)[ScrpScr])[Stot]
                           coeff_coper *= std::sqrt((tslp+1.0)*(tscrp+1.0)*(tstot+1.0)*(tspins[5]+1.0))*
                              fock::wigner9j(tslp,tscrp,tstot,tsl,tscr,tstot,tspins[3],tspins[4],tspins[5])*
                              std::sqrt((tscp+1.0)*(tsrp+1.0)*(tscr+1.0)*(tspins[2]+1.0))*
                              fock::wigner9j(tscp,tsrp,tscrp,tsc,tsr,tscr,tspins[0],tspins[1],tspins[2]);
                           if(std::abs(coeff_coper)<thresh_coper) continue;
                           // sign from adjoint: l|cr
                           // tspins = (Sc,Sr,Scr),(Sl,Scr,Stot)
                           // <Slp|Ol|Sl>
                           if(!this->identity(0) && dagger[0]^ifdagger){
                              int ts = tspins[3] + tslp - tsl;
                              coeff_coper *= ((ts/2)%2==0? 1.0 : -1.0)*std::sqrt((tsl+1.0)/(tslp+1.0));
                           }
                           // <Srp|Or|Sr> 
                           if(!this->identity(1) && dagger[1]^ifdagger){
                              int ts = tspins[1] + tsrp - tsr;
                              coeff_coper *= ((ts/2)%2==0? 1.0 : -1.0)*std::sqrt((tsr+1.0)/(tsrp+1.0));
                           }
                           // <Scp|Oc|Sc>
                           if(!this->identity(2) && dagger[2]^ifdagger){
                              int ts = tspins[0] + tscp - tsc;
                              coeff_coper *= ((ts/2)%2==0? 1.0 : -1.0)*std::sqrt((tsc+1.0)/(tscp+1.0));
                           }
                           // compute sign due to parity
                           Hxblk.coeff = (ifdagger? coeffH : coeff)*coeff_coper;
                           int pl = wf_info.qrow.get_parity(bi[0]);
                           int pc = wf_info.qmid.get_parity(bi[2]);
                           if(parity[1] && (pl+pc)%2==1) Hxblk.coeff *= -1.0; // Or
                           if(parity[2] && pl%2==1) Hxblk.coeff *= -1.0;  // Oc
                           // setup dimsions
                           Hxblk.dimin[0] = wf_info.qrow.get_dim(bi[0]);
                           Hxblk.dimin[1] = wf_info.qcol.get_dim(bi[1]);
                           Hxblk.dimin[2] = wf_info.qmid.get_dim(bi[2]);
                           Hxblk.dimout[0] = wf_info.qrow.get_dim(bo[0]);
                           Hxblk.dimout[1] = wf_info.qcol.get_dim(bo[1]);
                           Hxblk.dimout[2] = wf_info.qmid.get_dim(bo[2]);
                           Hxblk.size = Hxblk.dimout[0]*Hxblk.dimout[1]*Hxblk.dimout[2];
                           Hxblk.setup();
                           blksize = std::max(blksize, Hxblk.blksize);
                           cost += Hxblk.cost;
                           // Intermediates
                           if(posInter != -1){
                              Hxblk.posInter = posInter;
                              Hxblk.lenInter = lenInter;
                              Hxblk.offInter = offInter;
                              Hxblk.ldaInter = ldaInter;
                              blksize0 = std::max(blksize0, Hxblk.dimout[posInter]*Hxblk.dimin[posInter]);
                           }
                           Hxlst2[i].push_back(Hxblk);
                        } // tscr
                     } // bi2
                  } // bi1
               } // bi0
            } // i
         }else{
            // lc|r: LCcouple
            int tslcp;
            for(int i=0; i<wf_info._nnzaddr.size(); i++){
               auto key = wf_info._nnzaddr[i];
               bo[0] = std::get<0>(key); // br
               bo[1] = std::get<1>(key); // bc
               bo[2] = std::get<2>(key); // bm
               tslcp = std::get<3>(key); // tslc
               size_t offout = wf_info.get_offset(bo[0],bo[1],bo[2],tslcp);
               assert(offout > 0);
               const auto& bi0vec = this->identity(0)? std::vector<int>({bo[0]}) :
                  (dagger[0]^ifdagger? info[0]->_bc2br[bo[0]] : info[0]->_br2bc[bo[0]]);
               const auto& bi1vec = this->identity(1)? std::vector<int>({bo[1]}) :
                  (dagger[1]^ifdagger? info[1]->_bc2br[bo[1]] : info[1]->_br2bc[bo[1]]);
               const auto& bi2vec = this->identity(2)? std::vector<int>({bo[2]}) :
                  (dagger[2]^ifdagger? info[2]->_bc2br[bo[2]] : info[2]->_br2bc[bo[2]]);
               for(const auto& bi0 : bi0vec){
                  for(const auto& bi1 : bi1vec){
                     for(const auto& bi2 : bi2vec){
                        int bi[3]; // wf
                        bi[0] = bi0;
                        bi[1] = bi1;
                        bi[2] = bi2;
                        // setup Slc
                        int tslp = wf_info.qrow.get_sym(bo[0]).ts(); // l
                        int tsrp = wf_info.qcol.get_sym(bo[1]).ts(); // r
                        int tscp = wf_info.qmid.get_sym(bo[2]).ts(); // c
                        int tsl  = wf_info.qrow.get_sym(bi[0]).ts();
                        int tsr  = wf_info.qcol.get_sym(bi[1]).ts();
                        int tsc  = wf_info.qmid.get_sym(bi[2]).ts();
                        for(int tslc=std::abs(tsl-tsc); tslc<=tsl+tsc; tslc+=2){
                           size_t offin = wf_info.get_offset(bi[0],bi[1],bi[2],tslc);
                           if(offin == 0) continue;
                           // setup block
                           Hxblock<Tm> Hxblk(3,terms,cterms,alg_hcoper);
                           Hxblk.offin = offin-1;
                           Hxblk.offout = offout-1;
                           // update Hxblk.dagger/loc/off
                           Tm coeff_coper = 1.0;
                           bool skip = false;
                           for(int k=0; k<3; k++){ // l,r,c
                              if(this->identity(k)) continue;
                              Hxblk.dagger[k] = dagger[k]^ifdagger; // (O1^d1)^d = O1^(d^d1)
                              Hxblk.loc[k] = loc[k];
                              size_t offset = Hxblk.dagger[k]? info[k]->get_offset(bi[k],bo[k]) : 
                                 info[k]->get_offset(bo[k],bi[k]);
                              assert(offset != 0);
                              Hxblk.off[k] = off[k]+offset-1;
                              // sgn from bar{bar{Ts}} = (-1)^2s Ts
                              if(dagger[k] && ifdagger) coeff_coper *= parity[k]? -1.0 : 1.0;
                              // special treatment of op[c2/c1] for NS symmetry
                              if(k >= 2 && ((alg_hcoper==1 && terms>cterms) || alg_hcoper==2)){ 
                                 assert(k == loc[k]); // op[c] cannot be intermediates
                                 Tm coper = *(opaddr[loc[k]] + Hxblk.off[k]);
                                 coeff_coper *= Hxblk.dagger[k]? tools::conjugate(coper) : coper;
                                 if(std::abs(coeff_coper)<thresh_coper){
                                    skip = true;
                                    break;
                                 }
                              }
                           } // k
                           if(skip) continue;
                           // sign factors due to spin: lc|r
                           // ((<Slp|Ol|Sl><Scp|Oc1|Sc>)[Slcp,Slc](<Srp|Or|Sr>))[Stot]
                           coeff_coper *= std::sqrt((tslcp+1.0)*(tsrp+1.0)*(tstot+1.0)*(tspins[5]+1.0))*
                              fock::wigner9j(tslcp,tsrp,tstot,tslc,tsr,tstot,tspins[3],tspins[4],tspins[5])*
                              std::sqrt((tslp+1.0)*(tscp+1.0)*(tslc+1.0)*(tspins[2]+1.0))*
                              fock::wigner9j(tslp,tscp,tslcp,tsl,tsc,tslc,tspins[0],tspins[1],tspins[2]);
                           if(std::abs(coeff_coper)<thresh_coper) continue;
                           // sign from adjoint: lc|r
                           // tspins = (Sl,Sc,Slc),(Slc,Sr,Stot)
                           // <Slp|Ol|Sl>
                           if(!this->identity(0) && dagger[0]^ifdagger){
                              int ts = tspins[0] + tslp - tsl;
                              coeff_coper *= ((ts/2)%2==0? 1.0 : -1.0)*std::sqrt((tsl+1.0)/(tslp+1.0));
                           }
                           // <Srp|Or|Sr> 
                           if(!this->identity(1) && dagger[1]^ifdagger){
                              int ts = tspins[4] + tsrp - tsr;
                              coeff_coper *= ((ts/2)%2==0? 1.0 : -1.0)*std::sqrt((tsr+1.0)/(tsrp+1.0));
                           }
                           // <Scp|Oc|Sc>
                           if(!this->identity(2) && dagger[2]^ifdagger){
                              int ts = tspins[1] + tscp - tsc;
                              coeff_coper *= ((ts/2)%2==0? 1.0 : -1.0)*std::sqrt((tsc+1.0)/(tscp+1.0));
                           }
                           // compute sign due to parity
                           Hxblk.coeff = (ifdagger? coeffH : coeff)*coeff_coper;
                           int pl = wf_info.qrow.get_parity(bi[0]);
                           int pc = wf_info.qmid.get_parity(bi[2]);
                           if(parity[1] && (pl+pc)%2==1) Hxblk.coeff *= -1.0; // Or
                           if(parity[2] && pl%2==1) Hxblk.coeff *= -1.0; // Oc1
                           // setup dimsions
                           Hxblk.dimin[0] = wf_info.qrow.get_dim(bi[0]);
                           Hxblk.dimin[1] = wf_info.qcol.get_dim(bi[1]);
                           Hxblk.dimin[2] = wf_info.qmid.get_dim(bi[2]);
                           Hxblk.dimout[0] = wf_info.qrow.get_dim(bo[0]);
                           Hxblk.dimout[1] = wf_info.qcol.get_dim(bo[1]);
                           Hxblk.dimout[2] = wf_info.qmid.get_dim(bo[2]);
                           Hxblk.size = Hxblk.dimout[0]*Hxblk.dimout[1]*Hxblk.dimout[2];
                           Hxblk.setup();
                           blksize = std::max(blksize, Hxblk.blksize);
                           cost += Hxblk.cost;
                           // Intermediates
                           if(posInter != -1){
                              Hxblk.posInter = posInter;
                              Hxblk.lenInter = lenInter;
                              Hxblk.offInter = offInter;
                              Hxblk.ldaInter = ldaInter;
                              blksize0 = std::max(blksize0, Hxblk.dimout[posInter]*Hxblk.dimin[posInter]);
                           }
                           Hxlst2[i].push_back(Hxblk);
                        } // tslc
                     } // bi2
                  } // bi1
               } // bi0
            } // i
         } // couple
      }

   // twodot:
   // sigma[br',bc',bm',bv'] = Ol^dagger0[br',br] Or^dagger1[bc',bc] 
   // 			Oc1^dagger2[bm',bm] Oc2^dagger3[bv',bv] 
   // 			wf[br,bc,bm,bv]
   template <bool ifab, typename Tm>
      template <bool y, std::enable_if_t<y,int>>
      void Hmu_ptr<ifab,Tm>::gen_Hxlist2(const int alg_hcoper,
            Tm** opaddr,
            const qinfo4type<ifab,Tm>& wf_info,
            Hxlist2<Tm>& Hxlst2,
            size_t& blksize,
            size_t& blksize0,
            double& cost,
            const bool ifdagger) const{
         if(this->empty()) return;
         int bo[4], bi[4];
         for(int i=0; i<wf_info._nnzaddr.size(); i++){
            int idx = wf_info._nnzaddr[i];
            wf_info._addr_unpack(idx,bo[0],bo[1],bo[2],bo[3]);
            Hxblock<Tm> Hxblk(4,terms,cterms,alg_hcoper);
            Hxblk.offout = wf_info._offset[idx]-1;
            Hxblk.dimout[0] = wf_info.qrow.get_dim(bo[0]);
            Hxblk.dimout[1] = wf_info.qcol.get_dim(bo[1]);
            Hxblk.dimout[2] = wf_info.qmid.get_dim(bo[2]);
            Hxblk.dimout[3] = wf_info.qver.get_dim(bo[3]);
            Hxblk.size = Hxblk.dimout[0]*Hxblk.dimout[1]*Hxblk.dimout[2]*Hxblk.dimout[3];
            // finding the corresponding operator blocks given {bo[0],bo[1],bo[2],bo[3]}
            bool symAllowed = true;
            Tm coeff_coper = 1.0;
            for(int k=0; k<4; k++){ // l,r,c1,c2
               if(this->identity(k)){ 
                  // identity operator
                  bi[k] = bo[k];
               }else{
                  // not identity
                  Hxblk.dagger[k] = dagger[k]^ifdagger; // (O1^d1)^d = O1^(d^d1)
                  bool iftrans = dagger[k]^ifdagger;
                  bi[k] = iftrans? info[k]->_bc2br[bo[k]] : info[k]->_br2bc[bo[k]];
                  if(bi[k] == -1){
                     symAllowed = false;
                     break;
                  }else{
                     int jdx = iftrans? info[k]->_addr(bi[k],bo[k]) : info[k]->_addr(bo[k],bi[k]);
                     assert(info[k]->_offset[jdx] != 0);
                     Hxblk.loc[k] = loc[k];
                     Hxblk.off[k] = off[k]+(info[k]->_offset[jdx]-1);
                     // special treatment of op[c2/c1] for NSz symmetry
                     if(k >= 2 && ((alg_hcoper==1 && terms>cterms) || alg_hcoper==2)){ 
                        assert(k == loc[k]); // op[c] cannot be intermediates
                        Tm coper = *(opaddr[loc[k]] + Hxblk.off[k]);
                        coeff_coper *= Hxblk.dagger[k]? tools::conjugate(coper) : coper;
                        if(std::abs(coeff_coper)<thresh_coper){
                           symAllowed = false;
                           break;
                        }
                     }
                  }
               }
            }
            if(!symAllowed) continue;
            size_t offin = wf_info.get_offset(bi[0],bi[1],bi[2],bi[3]);
            if(offin == 0) continue; // in case of no matching contractions
            Hxblk.offin = offin-1;
            Hxblk.dimin[0] = wf_info.qrow.get_dim(bi[0]);
            Hxblk.dimin[1] = wf_info.qcol.get_dim(bi[1]);
            Hxblk.dimin[2] = wf_info.qmid.get_dim(bi[2]);
            Hxblk.dimin[3] = wf_info.qver.get_dim(bi[3]);
            // compute sign due to parity
            Hxblk.coeff = (ifdagger? coeffH : coeff)*coeff_coper;
            int pl  = wf_info.qrow.get_parity(bi[0]);
            int pc1 = wf_info.qmid.get_parity(bi[2]);
            int pc2 = wf_info.qver.get_parity(bi[3]);
            if(parity[1] && (pl+pc1+pc2)%2==1) Hxblk.coeff *= -1.0; // Or
            if(parity[3] && (pl+pc1)%2==1) Hxblk.coeff *= -1.0;  // Oc2
            if(parity[2] && pl%2==1) Hxblk.coeff *= -1.0; // Oc1
            Hxblk.setup();
            blksize = std::max(blksize, Hxblk.blksize);
            cost += Hxblk.cost;
            // Intermediates
            if(posInter != -1){
               Hxblk.posInter = posInter;
               Hxblk.lenInter = lenInter;
               Hxblk.offInter = offInter;
               Hxblk.ldaInter = ldaInter;
               blksize0 = std::max(blksize0, Hxblk.dimout[posInter]*Hxblk.dimin[posInter]);
            }
            Hxlst2[i].push_back(Hxblk);
         } // i
      }

   // su2 case:
   // sigma[br',bc',bm',bv'] = Ol^dagger0[br',br] Or^dagger1[bc',bc] 
   // 			Oc1^dagger2[bm',bm] Oc2^dagger3[bv',bv] 
   // 			wf[br,bc,bm,bv]
   template <bool ifab, typename Tm>
      template <bool y, std::enable_if_t<!y,int>>
      void Hmu_ptr<ifab,Tm>::gen_Hxlist2(const int alg_hcoper,       
            Tm** opaddr,
            const qinfo4type<ifab,Tm>& wf_info,
            Hxlist2<Tm>& Hxlst2,
            size_t& blksize,
            size_t& blksize0,
            double& cost,
            const bool ifdagger) const{
         if(this->empty()) return;
         // sigma[br',bc',bm',bv']
         int bo[4], tslc1p, tsc2rp, tstot;
         tstot = wf_info.sym.ts();
         for(int i=0; i<wf_info._nnzaddr.size(); i++){
            auto key = wf_info._nnzaddr[i];
            bo[0] = std::get<0>(key);
            bo[1] = std::get<1>(key);
            bo[2] = std::get<2>(key);
            bo[3] = std::get<3>(key);
            tslc1p = std::get<4>(key);
            tsc2rp = std::get<5>(key);
            size_t offout = wf_info.get_offset(bo[0],bo[1],bo[2],bo[3],tslc1p,tsc2rp);
            assert(offout > 0);
            const auto& bi0vec = this->identity(0)? std::vector<int>({bo[0]}) :
               (dagger[0]^ifdagger? info[0]->_bc2br[bo[0]] : info[0]->_br2bc[bo[0]]);
            const auto& bi1vec = this->identity(1)? std::vector<int>({bo[1]}) :
               (dagger[1]^ifdagger? info[1]->_bc2br[bo[1]] : info[1]->_br2bc[bo[1]]);
            const auto& bi2vec = this->identity(2)? std::vector<int>({bo[2]}) :
               (dagger[2]^ifdagger? info[2]->_bc2br[bo[2]] : info[2]->_br2bc[bo[2]]);
            const auto& bi3vec = this->identity(3)? std::vector<int>({bo[3]}) :
               (dagger[3]^ifdagger? info[3]->_bc2br[bo[3]] : info[3]->_br2bc[bo[3]]);
            for(const auto& bi0 : bi0vec){
               for(const auto& bi1 : bi1vec){
                  for(const auto& bi2 : bi2vec){
                     for(const auto& bi3 : bi3vec){
                        int bi[4]; // wf
                        bi[0] = bi0;
                        bi[1] = bi1;
                        bi[2] = bi2;
                        bi[3] = bi3;
                        // setup Slc1,Sc2r
                        int tslp  = wf_info.qrow.get_sym(bo[0]).ts(); // l
                        int tsrp  = wf_info.qcol.get_sym(bo[1]).ts(); // r
                        int tsc1p = wf_info.qmid.get_sym(bo[2]).ts(); // c1
                        int tsc2p = wf_info.qver.get_sym(bo[3]).ts(); // c2
                        int tsl   = wf_info.qrow.get_sym(bi[0]).ts();
                        int tsr   = wf_info.qcol.get_sym(bi[1]).ts();
                        int tsc1  = wf_info.qmid.get_sym(bi[2]).ts();
                        int tsc2  = wf_info.qver.get_sym(bi[3]).ts();
                        for(int tslc1=std::abs(tsl-tsc1); tslc1<=tsl+tsc1; tslc1+=2){
                           for(int tsc2r=std::abs(tsc2-tsr); tsc2r<=tsc2+tsr; tsc2r+=2){
                              size_t offin = wf_info.get_offset(bi[0],bi[1],bi[2],bi[3],tslc1,tsc2r);
                              if(offin == 0) continue;
                              // setup block
                              Hxblock<Tm> Hxblk(4,terms,cterms,alg_hcoper);
                              Hxblk.offin = offin-1;
                              Hxblk.offout = offout-1;
                              // update Hxblk.dagger/loc/off
                              Tm coeff_coper = 1.0;
                              bool skip = false;
                              for(int k=0; k<4; k++){ // l,r,c1,c2
                                 if(this->identity(k)) continue;
                                 Hxblk.dagger[k] = dagger[k]^ifdagger; // (O1^d1)^d = O1^(d^d1)
                                 Hxblk.loc[k] = loc[k];
                                 size_t offset = Hxblk.dagger[k]? info[k]->get_offset(bi[k],bo[k]) : 
                                    info[k]->get_offset(bo[k],bi[k]);
                                 assert(offset != 0);
                                 Hxblk.off[k] = off[k]+offset-1;
                                 // sgn from bar{bar{Ts}} = (-1)^2s Ts
                                 if(dagger[k] && ifdagger) coeff_coper *= parity[k]? -1.0 : 1.0;
                                 // special treatment of op[c2/c1] for NS symmetry
                                 if(k >= 2 && ((alg_hcoper==1 && terms>cterms) || alg_hcoper==2)){ 
                                    assert(k == loc[k]); // op[c] cannot be intermediates
                                    Tm coper = *(opaddr[loc[k]] + Hxblk.off[k]);
                                    coeff_coper *= Hxblk.dagger[k]? tools::conjugate(coper) : coper;
                                    if(std::abs(coeff_coper)<thresh_coper){
                                       skip = true;
                                       break;
                                    }
                                 }
                              } // k
                              if(skip) continue;
                              // sign factors due to spin
                              // ((<Slp|Ol|Sl><Sc1p|Oc1|Sc1>)[Slc1p,Slc1](<Sc2p|Oc2|Sc2><Srp|Or|Sr>)[Sc2rpSc2r])[Stot]
                              coeff_coper *= std::sqrt((tslc1p+1.0)*(tsc2rp+1.0)*(tstot+1.0)*(tspins[8]+1.0))*
                                 fock::wigner9j(tslc1p,tsc2rp,tstot,tslc1,tsc2r,tstot,tspins[6],tspins[7],tspins[8])*
                                 std::sqrt((tslp+1.0)*(tsc1p+1.0)*(tslc1+1.0)*(tspins[2]+1.0))*
                                 fock::wigner9j(tslp,tsc1p,tslc1p,tsl,tsc1,tslc1,tspins[0],tspins[1],tspins[2])*
                                 std::sqrt((tsc2p+1.0)*(tsrp+1.0)*(tsc2r+1.0)*(tspins[5]+1.0))*
                                 fock::wigner9j(tsc2p,tsrp,tsc2rp,tsc2,tsr,tsc2r,tspins[3],tspins[4],tspins[5]);
                              if(std::abs(coeff_coper)<thresh_coper) continue;
                              // sign from adjoint
                              // tspins = (Sl,Sc1,Slc1)(Sc2,Sr,Sc2r),(Slc1,Sc2r,Stot)
                              // <Slp|Ol|Sl>
                              if(!this->identity(0) && dagger[0]^ifdagger){
                                 int ts = tspins[0] + tslp - tsl;
                                 coeff_coper *= ((ts/2)%2==0? 1.0 : -1.0)*std::sqrt((tsl+1.0)/(tslp+1.0));
                              }
                              // <Srp|Or|Sr> 
                              if(!this->identity(1) && dagger[1]^ifdagger){
                                 int ts = tspins[4] + tsrp - tsr;
                                 coeff_coper *= ((ts/2)%2==0? 1.0 : -1.0)*std::sqrt((tsr+1.0)/(tsrp+1.0));
                              }
                              // <Sc1p|Oc1|Sc1>
                              if(!this->identity(2) && dagger[2]^ifdagger){
                                 int ts = tspins[1] + tsc1p - tsc1;
                                 coeff_coper *= ((ts/2)%2==0? 1.0 : -1.0)*std::sqrt((tsc1+1.0)/(tsc1p+1.0));
                              }
                              // <Sc2p|Oc2|Sc2>
                              if(!this->identity(3) && dagger[3]^ifdagger){
                                 int ts = tspins[3] + tsc2p - tsc2;
                                 coeff_coper *= ((ts/2)%2==0? 1.0 : -1.0)*std::sqrt((tsc2+1.0)/(tsc2p+1.0));
                              }
                              // compute sign due to parity
                              Hxblk.coeff = (ifdagger? coeffH : coeff)*coeff_coper;
                              int pl  = wf_info.qrow.get_parity(bi[0]);
                              int pc1 = wf_info.qmid.get_parity(bi[2]);
                              int pc2 = wf_info.qver.get_parity(bi[3]);
                              if(parity[1] && (pl+pc1+pc2)%2==1) Hxblk.coeff *= -1.0; // Or
                              if(parity[3] && (pl+pc1)%2==1) Hxblk.coeff *= -1.0;  // Oc2
                              if(parity[2] && pl%2==1) Hxblk.coeff *= -1.0; // Oc1
                              // setup dimsions
                              Hxblk.dimin[0] = wf_info.qrow.get_dim(bi[0]);
                              Hxblk.dimin[1] = wf_info.qcol.get_dim(bi[1]);
                              Hxblk.dimin[2] = wf_info.qmid.get_dim(bi[2]);
                              Hxblk.dimin[3] = wf_info.qver.get_dim(bi[3]);
                              Hxblk.dimout[0] = wf_info.qrow.get_dim(bo[0]);
                              Hxblk.dimout[1] = wf_info.qcol.get_dim(bo[1]);
                              Hxblk.dimout[2] = wf_info.qmid.get_dim(bo[2]);
                              Hxblk.dimout[3] = wf_info.qver.get_dim(bo[3]);
                              Hxblk.size = Hxblk.dimout[0]*Hxblk.dimout[1]*Hxblk.dimout[2]*Hxblk.dimout[3];
                              Hxblk.setup();
                              blksize = std::max(blksize, Hxblk.blksize);
                              cost += Hxblk.cost;
                              // Intermediates
                              if(posInter != -1){
                                 Hxblk.posInter = posInter;
                                 Hxblk.lenInter = lenInter;
                                 Hxblk.offInter = offInter;
                                 Hxblk.ldaInter = ldaInter;
                                 blksize0 = std::max(blksize0, Hxblk.dimout[posInter]*Hxblk.dimin[posInter]);
                              }
                              Hxlst2[i].push_back(Hxblk);
                           } // tsc2r
                        } // tslc1
                     } // b3
                  } // b2
               } // b1
            } // b0
         } // i
      }

} // ctns

#endif
