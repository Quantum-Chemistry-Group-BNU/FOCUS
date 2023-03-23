#ifndef PREPROCESS_HMU_H
#define PREPROCESS_HMU_H

#include "preprocess_header.h"
#include "preprocess_hxlist.h"

namespace ctns{

   // H[mu] = coeff*Ol*Or*Oc (onedot)
   //       = coeff*Ol*Or*Oc1*Oc2 (twodot) 
   template <typename Tm>
      struct Hmu_ptr{
         public:
            bool identity(const int i) const{ return loc[i]==-1; }
            void init(const int it,
                  const symbolic_task<Tm>& H_formulae,
                  const oper_dictmap<Tm>& qops_dict,
                  const hintermediates<Tm>& hinter,
                  const std::map<std::string,int>& oploc);
            // onedot
            void gen_Hxlist(const qinfo3<Tm>& wf_info, 
                  Hxlist<Tm>& Hxlst,
                  size_t& blksize,
                  double& cost,
                  const bool ifdagger) const;
            // twodot
            void gen_Hxlist(const qinfo4<Tm>& wf_info, 
                  Hxlist<Tm>& Hxlst,
                  size_t& blksize,
                  double& cost,
                  const bool ifdagger) const;
            // twodot
            void gen_Hxlist2(const qinfo4<Tm>& wf_info, 
                  Hxlist2<Tm>& Hxlst2,
                  size_t& blksize,
                  double& cost,
                  const bool ifdagger) const;
            // --- Direct version withou hinter --- 
            void initDirect(const int it,
                  const symbolic_task<Tm>& H_formulae,
                  const oper_dictmap<Tm>& qops_dict,
                  const std::map<std::string,int>& oploc);
            void gen_Hxlist2Direct(const qinfo4<Tm>& wf_info, 
                  Hxlist2<Tm>& Hxlst2,
                  size_t& blksize,
                  size_t& blksize0,
                  double& cost,
                  const bool ifdagger) const;
         public:
            bool parity[4] = {false,false,false,false};
            bool dagger[4] = {false,false,false,false};
            qinfo2<Tm>* info[4] = {nullptr,nullptr,nullptr,nullptr};
            int loc[4] = {-1,-1,-1,-1};
            size_t off[4] = {0,0,0,0};
            int terms = 0;
            Tm coeff = 1.0, coeffH = 1.0;
            // intermediates [direct] -> we assume each hmu contains only one intermediates
            int posInter = -1;
            size_t ldaInter = 0;
            std::vector<Tm> alpha_vec;
      };

   template <typename Tm>
      void Hmu_ptr<Tm>::init(const int it,
            const symbolic_task<Tm>& H_formulae,
            const oper_dictmap<Tm>& qops_dict,
            const hintermediates<Tm>& hinter,
            const std::map<std::string,int>& oploc){
         const auto& HTerm = H_formulae.tasks[it];
         terms = HTerm.size();
         for(int idx=terms-1; idx>=0; idx--){
            const auto& sop = HTerm.terms[idx];
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
            if(sop.size() == 1){
               coeff *= sop.sums[0].first;
               loc[pos] = pos;
               off[pos] = qops._offset.at(std::make_pair(label,index0)); // qops
            }else{
               loc[pos] = locInter;
               off[pos] = hinter._offset.at(std::make_pair(it,idx)); // intermediates
            }
         } // idx
         coeffH = coeff*HTerm.Hsign(); 
      }

   // sigma[br,bc,bm] = Ol^dagger0[br,br'] Or^dagger1[bc,bc'] Oc1^dagger2[bm,bm'] 
   // 		     wf[br',bc',bm',bv']
   template <typename Tm>
      void Hmu_ptr<Tm>::gen_Hxlist(const qinfo3<Tm>& wf_info,
            Hxlist<Tm>& Hxlst,
            size_t& blksize,
            double& cost,
            const bool ifdagger) const{
         int bo[3], bi[3];
         for(int i=0; i<wf_info._nnzaddr.size(); i++){
            int idx = wf_info._nnzaddr[i];
            wf_info._addr_unpack(idx,bo[0],bo[1],bo[2]);
            Hxblock<Tm> Hxblk(3,terms);
            Hxblk.offout = wf_info._offset[idx]-1;
            Hxblk.dimout[0] = wf_info.qrow.get_dim(bo[0]);
            Hxblk.dimout[1] = wf_info.qcol.get_dim(bo[1]);
            Hxblk.dimout[2] = wf_info.qmid.get_dim(bo[2]);
            Hxblk.size = Hxblk.dimout[0]*Hxblk.dimout[1]*Hxblk.dimout[2];
            // finding the corresponding operator blocks given {bo[0],bo[1],bo[2],bo[3]}
            bool symAllowed = true;
            for(int k=0; k<3; k++){
               Hxblk.dagger[k] = dagger[k]^ifdagger;
               if(this->identity(k)){
                  bi[k] = bo[k];
               }else{
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
                  }
               }
            }
            if(!symAllowed) continue;
            size_t offin = wf_info._offset[wf_info._addr(bi[0],bi[1],bi[2])];
            if(offin == 0) continue; // in case of no matching contractions
            Hxblk.offin = offin-1;
            Hxblk.dimin[0] = wf_info.qrow.get_dim(bi[0]);
            Hxblk.dimin[1] = wf_info.qcol.get_dim(bi[1]);
            Hxblk.dimin[2] = wf_info.qmid.get_dim(bi[2]);
            // compute sign due to parity
            Hxblk.coeff = ifdagger? coeffH : coeff;
            int pl = wf_info.qrow.get_parity(bi[0]);
            int pc = wf_info.qmid.get_parity(bi[2]);
            if(parity[1] && (pl+pc)%2==1) Hxblk.coeff *= -1.0; // Or: Or|lcr> = (-1)^{pl+pc}|lc>*Or|r>
            if(parity[2] && pl%2==1) Hxblk.coeff *= -1.0; // Oc
            Hxblk.setup();
            blksize = std::max(blksize, Hxblk.blksize);
            cost += Hxblk.cost;
            Hxlst.push_back(Hxblk);
         } // i
      }

   // sigma[br,bc,bm,bv] = Ol^dagger0[br,br'] Or^dagger1[bc,bc'] 
   // 			Oc1^dagger2[bm,bm'] Oc2^dagger3[bv,bv'] 
   // 			wf[br',bc',bm',bv']
   template <typename Tm>
      void Hmu_ptr<Tm>::gen_Hxlist(const qinfo4<Tm>& wf_info,
            Hxlist<Tm>& Hxlst,
            size_t& blksize,
            double& cost,
            const bool ifdagger) const{
         int bo[4], bi[4];
         for(int i=0; i<wf_info._nnzaddr.size(); i++){
            int idx = wf_info._nnzaddr[i];
            wf_info._addr_unpack(idx,bo[0],bo[1],bo[2],bo[3]);
            Hxblock<Tm> Hxblk(4,terms);
            Hxblk.offout = wf_info._offset[idx]-1;
            Hxblk.dimout[0] = wf_info.qrow.get_dim(bo[0]);
            Hxblk.dimout[1] = wf_info.qcol.get_dim(bo[1]);
            Hxblk.dimout[2] = wf_info.qmid.get_dim(bo[2]);
            Hxblk.dimout[3] = wf_info.qver.get_dim(bo[3]);
            Hxblk.size = Hxblk.dimout[0]*Hxblk.dimout[1]*Hxblk.dimout[2]*Hxblk.dimout[3];
            // finding the corresponding operator blocks given {bo[0],bo[1],bo[2],bo[3]}
            bool symAllowed = true;
            for(int k=0; k<4; k++){
               Hxblk.dagger[k] = dagger[k]^ifdagger;
               if(this->identity(k)){ // identity operator
                  bi[k] = bo[k];
               }else{
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
                  }
               }
            }
            if(!symAllowed) continue;
            size_t offin = wf_info._offset[wf_info._addr(bi[0],bi[1],bi[2],bi[3])];
            if(offin == 0) continue; // in case of no matching contractions
            Hxblk.offin = offin-1;
            Hxblk.dimin[0] = wf_info.qrow.get_dim(bi[0]);
            Hxblk.dimin[1] = wf_info.qcol.get_dim(bi[1]);
            Hxblk.dimin[2] = wf_info.qmid.get_dim(bi[2]);
            Hxblk.dimin[3] = wf_info.qver.get_dim(bi[3]);
            // compute sign due to parity
            Hxblk.coeff = ifdagger? coeffH : coeff;
            int pl  = wf_info.qrow.get_parity(bi[0]);
            int pc1 = wf_info.qmid.get_parity(bi[2]);
            int pc2 = wf_info.qver.get_parity(bi[3]);
            if(parity[1] && (pl+pc1+pc2)%2==1) Hxblk.coeff *= -1.0; // Or
            if(parity[3] && (pl+pc1)%2==1) Hxblk.coeff *= -1.0;  // Oc2
            if(parity[2] && pl%2==1) Hxblk.coeff *= -1.0; // Oc1
            Hxblk.setup();
            blksize = std::max(blksize, Hxblk.blksize);
            cost += Hxblk.cost;
            Hxlst.push_back(Hxblk);
         } // i
      }

   // sigma[br,bc,bm,bv] = Ol^dagger0[br,br'] Or^dagger1[bc,bc'] 
   // 			Oc1^dagger2[bm,bm'] Oc2^dagger3[bv,bv'] 
   // 			wf[br',bc',bm',bv']
   template <typename Tm>
      void Hmu_ptr<Tm>::gen_Hxlist2(const qinfo4<Tm>& wf_info,
            Hxlist2<Tm>& Hxlst2,
            size_t& blksize,
            double& cost,
            const bool ifdagger) const{
         int bo[4], bi[4];
         for(int i=0; i<wf_info._nnzaddr.size(); i++){
            int idx = wf_info._nnzaddr[i];
            wf_info._addr_unpack(idx,bo[0],bo[1],bo[2],bo[3]);
            Hxblock<Tm> Hxblk(4,terms);
            Hxblk.offout = wf_info._offset[idx]-1;
            Hxblk.dimout[0] = wf_info.qrow.get_dim(bo[0]);
            Hxblk.dimout[1] = wf_info.qcol.get_dim(bo[1]);
            Hxblk.dimout[2] = wf_info.qmid.get_dim(bo[2]);
            Hxblk.dimout[3] = wf_info.qver.get_dim(bo[3]);
            Hxblk.size = Hxblk.dimout[0]*Hxblk.dimout[1]*Hxblk.dimout[2]*Hxblk.dimout[3];
            // finding the corresponding operator blocks given {bo[0],bo[1],bo[2],bo[3]}
            bool symAllowed = true;
            for(int k=0; k<4; k++){
               Hxblk.dagger[k] = dagger[k]^ifdagger;
               if(this->identity(k)){ // identity operator
                  bi[k] = bo[k];
               }else{
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
                  }
               }
            }
            if(!symAllowed) continue;
            size_t offin = wf_info._offset[wf_info._addr(bi[0],bi[1],bi[2],bi[3])];
            if(offin == 0) continue; // in case of no matching contractions
            Hxblk.offin = offin-1;
            Hxblk.dimin[0] = wf_info.qrow.get_dim(bi[0]);
            Hxblk.dimin[1] = wf_info.qcol.get_dim(bi[1]);
            Hxblk.dimin[2] = wf_info.qmid.get_dim(bi[2]);
            Hxblk.dimin[3] = wf_info.qver.get_dim(bi[3]);
            // compute sign due to parity
            Hxblk.coeff = ifdagger? coeffH : coeff;
            int pl  = wf_info.qrow.get_parity(bi[0]);
            int pc1 = wf_info.qmid.get_parity(bi[2]);
            int pc2 = wf_info.qver.get_parity(bi[3]);
            if(parity[1] && (pl+pc1+pc2)%2==1) Hxblk.coeff *= -1.0; // Or
            if(parity[3] && (pl+pc1)%2==1) Hxblk.coeff *= -1.0;  // Oc2
            if(parity[2] && pl%2==1) Hxblk.coeff *= -1.0; // Oc1
            Hxblk.setup();
            blksize = std::max(blksize, Hxblk.blksize);
            cost += Hxblk.cost;
            Hxlst2[i].push_back(Hxblk);
         } // i
      }

   // --- Direct version ---
   template <typename Tm>
      void Hmu_ptr<Tm>::initDirect(const int it,
            const symbolic_task<Tm>& H_formulae,
            const oper_dictmap<Tm>& qops_dict,
            const std::map<std::string,int>& oploc){
         const auto& HTerm = H_formulae.tasks[it];
         terms = HTerm.size();
         for(int idx=terms-1; idx>=0; idx--){
            const auto& sop = HTerm.terms[idx];
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
            int len = sop.size();
            if(len == 1){
               coeff *= sop.sums[0].first;
               loc[pos] = pos;
               off[pos] = qops._offset.at(std::make_pair(label,index0)); // qops
            }else{
               loc[pos] = locInter;
               off[pos] = qops._offset.at(std::make_pair(label,index0)); // fake intermediates
               posInter = pos;
               alpha_vec.resize(len);
               for(int k=0; k<len; k++){
                  auto wtk = sop.sums[k].first;
                  alpha_vec[k] = dagger? tools::conjugate(wtk) : wtk;
               }
               const auto& sop1 = sop.sums[1].second; // used for determine LDA
               const auto& index1 = sop1.index;
               const auto& op1 = qops(label).at(index1);
               ldaInter = std::distance(op0._data, op1._data); // Ca & Cb can be of different dimes for isym=2
            }
         } // idx
         coeffH = coeff*HTerm.Hsign(); 
      }

   // sigma[br,bc,bm,bv] = Ol^dagger0[br,br'] Or^dagger1[bc,bc'] 
   // 			Oc1^dagger2[bm,bm'] Oc2^dagger3[bv,bv'] 
   // 			wf[br',bc',bm',bv']
   template <typename Tm>
      void Hmu_ptr<Tm>::gen_Hxlist2Direct(const qinfo4<Tm>& wf_info,
            Hxlist2<Tm>& Hxlst2,
            size_t& blksize,
            size_t& blksize0,
            double& cost,
            const bool ifdagger) const{
         int bo[4], bi[4];
         for(int i=0; i<wf_info._nnzaddr.size(); i++){
            int idx = wf_info._nnzaddr[i];
            wf_info._addr_unpack(idx,bo[0],bo[1],bo[2],bo[3]);
            Hxblock<Tm> Hxblk(4,terms);
            Hxblk.offout = wf_info._offset[idx]-1;
            Hxblk.dimout[0] = wf_info.qrow.get_dim(bo[0]);
            Hxblk.dimout[1] = wf_info.qcol.get_dim(bo[1]);
            Hxblk.dimout[2] = wf_info.qmid.get_dim(bo[2]);
            Hxblk.dimout[3] = wf_info.qver.get_dim(bo[3]);
            Hxblk.size = Hxblk.dimout[0]*Hxblk.dimout[1]*Hxblk.dimout[2]*Hxblk.dimout[3];
            // finding the corresponding operator blocks given {bo[0],bo[1],bo[2],bo[3]}
            bool symAllowed = true;
            for(int k=0; k<4; k++){
               Hxblk.dagger[k] = dagger[k]^ifdagger;
               if(this->identity(k)){ // identity operator
                  bi[k] = bo[k];
               }else{
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

                  }
               }
            }
            if(!symAllowed) continue;
            size_t offin = wf_info._offset[wf_info._addr(bi[0],bi[1],bi[2],bi[3])];
            if(offin == 0) continue; // in case of no matching contractions
            Hxblk.offin = offin-1;
            Hxblk.dimin[0] = wf_info.qrow.get_dim(bi[0]);
            Hxblk.dimin[1] = wf_info.qcol.get_dim(bi[1]);
            Hxblk.dimin[2] = wf_info.qmid.get_dim(bi[2]);
            Hxblk.dimin[3] = wf_info.qver.get_dim(bi[3]);
            // compute sign due to parity
            Hxblk.coeff = ifdagger? coeffH : coeff;
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
               Hxblk.ldaInter= ldaInter;
               Hxblk.alpha_vec = alpha_vec;
               blksize0 = std::max(blksize0, Hxblk.dimout[posInter]*Hxblk.dimin[posInter]);
            }
            Hxlst2[i].push_back(Hxblk);
         } // i
      }

} // ctns

#endif
