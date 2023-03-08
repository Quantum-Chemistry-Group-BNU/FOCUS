#ifndef STENSOR3_H
#define STENSOR3_H

#include "../../core/serialization.h"
#include "../../core/matrix.h"

namespace ctns{

   template <typename Tm>
      struct stensor2;
   template <typename Tm>
      struct stensor4;

   const bool debug_stensor3 = false;
   extern const bool debug_stensor3;

   template <typename Tm>
      struct stensor3{
         private:
            friend class boost::serialization::access;	   
            template <class Archive>
               void save(Archive & ar, const unsigned int version) const{
                  ar & own & info;
                  if(own){
                     for(size_t i=0; i<info._size; i++){
                        ar & _data[i];
                     }
                  }
               }
            template <class Archive>
               void load(Archive & ar, const unsigned int version){
                  ar & own & info;
                  if(own){
                     _data = new Tm[info._size];
                     for(size_t i=0; i<info._size; i++){
                        ar & _data[i];
                     }
                  }
               }
            BOOST_SERIALIZATION_SPLIT_MEMBER()
               // memory allocation
               void allocate(){
                  _data = new Tm[info._size];
                  memset(_data, 0, info._size*sizeof(Tm));
               }
         public:
            // --- GENERAL FUNCTIONS ---
            // constructors
            stensor3(){}
            void init(const qsym& _sym, const qbond& _qrow, const qbond& _qcol, const qbond& _qmid,
                  const direction3 _dir={0,1,1}, const bool _own=true){
               info.init(_sym, _qrow, _qcol, _qmid, _dir);
               own = _own;
               if(own) this->allocate();
            }
            stensor3(const qsym& _sym, const qbond& _qrow, const qbond& _qcol, const qbond& _qmid,
                  const direction3 _dir={0,1,1}, const bool _own=true){
               this->init(_sym, _qrow, _qcol, _qmid, _dir, _own);
            }
            // simple constructor from qinfo
            void init(const qinfo3<Tm>& _info, const bool _own=true){
               info = _info;
               own = _own;
               if(own) this->allocate();
            }
            stensor3(const qinfo3<Tm>& _info, const bool _own=true){
               this->init(_info, _own);
            }
            // used to for setup ptr, if own=false
            void setup_data(Tm* data){
               assert(own == false);
               _data = data;
            }
            // desctructors
            ~stensor3(){ 
               if(own) delete[] _data; 
            }
            // copy constructor
            stensor3(const stensor3& st){;
               if(debug_stensor3) std::cout << "stensor3: copy constructor - st.own=" << st.own << std::endl;   
               own = st.own;
               info = st.info;
               if(st.own){
                  _data = new Tm[info._size];
                  linalg::xcopy(info._size, st._data, _data);
               }else{
                  // shalow copy of the wrapper in case st.own = false;
                  // needs to be here for direct manipulations of data in xaxpy
                  _data = st._data;
               }
            }
            // copy assignment
            stensor3& operator =(const stensor3& st) = delete;
            /*
            // copy assignment
            stensor3& operator =(const stensor3& st){
            std::cout << "stensor3: copy assignment" << std::endl;    
            exit(1); 
            if(this != &st){
            info = st.info;
            delete[] _data;
            _data = new Tm[info._size];
            linalg::xcopy(info._size, st._data, _data);
            info.setup_data(_data);
            }
            return *this;
            }
            */
            // move constructor
            stensor3(stensor3&& st){
               if(debug_stensor3) std::cout << "stensor3: move constructor - st.own=" << st.own << std::endl;    
               assert(own == true);
               own = st.own; 
               info = std::move(st.info);
               _data = st._data;
               st._data = nullptr;
            }
            // move assignment
            stensor3& operator =(stensor3&& st){
               if(debug_stensor3) std::cout << "stensor3: move assignment - st.own=" << st.own << std::endl;
               assert(own == true && st.own == true); 
               if(this != &st){
                  info = std::move(st.info);
                  delete[] _data;
                  _data = st._data;
                  st._data = nullptr;	    
               }
               return *this;
            }
            // helpers
            int rows() const{ return info._rows; }
            int cols() const{ return info._cols; }
            int mids() const{ return info._mids; }
            bool dir_row() const{ return std::get<0>(info.dir); } 
            bool dir_col() const{ return std::get<1>(info.dir); }
            bool dir_mid() const{ return std::get<2>(info.dir); } 
            size_t size() const{ return info._size; }
            Tm* data() const{ return _data; }
            // access
            dtensor3<Tm> operator()(const int br, const int bc, const int bm) const{
               return info(br,bc,bm,_data);
            }
            Tm* start_ptr(const int br, const int bc, const int bm) const{
               size_t off = info._offset[info._addr(br,bc,bm)];
               return (off==0)? nullptr : _data+off-1;
            }
            // print
            void print(const std::string name, const int level=0) const;
            // simple arithmetic operations
            stensor3<Tm>& operator *=(const Tm fac){
               linalg::xscal(info._size, fac, _data);
               return *this;
            }
            stensor3<Tm>& operator +=(const stensor3<Tm>& st){
               assert(info == st.info);
               linalg::xaxpy(info._size, 1.0, st.data(), _data);
               return *this;
            }
            stensor3<Tm>& operator -=(const stensor3<Tm>& st){
               assert(info == st.info);
               linalg::xaxpy(info._size, -1.0, st.data(), _data);
               return *this;
            }
            // algebra
            friend stensor3<Tm> operator +(const stensor3<Tm>& qta, const stensor3<Tm>& qtb){
               assert(qta.info == qtb.info); 
               stensor3<Tm> qt(qta.info);
               linalg::xcopy(qt.info._size, qta._data, qt._data);
               qt += qtb;
               return qt;
            }
            friend stensor3<Tm> operator -(const stensor3<Tm>& qta, const stensor3<Tm>& qtb){
               assert(qta.info == qtb.info); 
               stensor3<Tm> qt(qta.info);
               linalg::xcopy(qt.info._size, qta._data, qt._data);
               qt -= qtb;
               return qt;
            }
            friend stensor3<Tm> operator *(const Tm fac, const stensor3<Tm>& qta){
               stensor3<Tm> qt(qta.info);
               linalg::xaxpy(qt.info._size, fac, qta._data, qt._data);
               return qt;
            }
            friend stensor3<Tm> operator *(const stensor3<Tm>& qt, const Tm fac){
               return fac*qt;
            }
            double normF() const{ return linalg::xnrm2(info._size, _data); }
            void set_zero(){ memset(_data, 0, info._size*sizeof(Tm)); }
            // --- SPECIFIC FUNCTIONS ---
            // fix middle index (bm,im) - bm-th block, im-idx - composite index!
            stensor2<Tm> fix_mid(const std::pair<int,int> mdx) const;
            // deal with fermionic sign in fermionic direct product
            void mid_signed(const double fac=1.0); // wf[lcr](-1)^{p(c)}
      void row_signed(const double fac=1.0); // wf[lcr](-1)^{p(l)}
      void cntr_signed(const std::string block);
      void permCR_signed(); // wf[lcr]->wf[lcr]*(-1)^{p[c]*p[r]}
                            // ZL20210413: application of time-reversal operation
      stensor3<Tm> K(const int nbar=0) const;
      // for sweep algorithm
      void from_array(const Tm* array){
         linalg::xcopy(info._size, array, _data);
      }
      void to_array(Tm* array) const{
         linalg::xcopy(info._size, _data, array);
      }
      // for decimation
      qproduct dpt_lc() const{ return qmerge(info.qrow, info.qmid); }
      qproduct dpt_cr() const{ return qmerge(info.qmid, info.qcol); }
      qproduct dpt_lr() const{ return qmerge(info.qrow, info.qcol); }
      // reshape: merge wf3[l,r,c]
      stensor2<Tm> merge_lc() const{ // wf2[lc,r] 
         auto qprod = dpt_lc();
         return merge_qt3_qt2_lc(*this, qprod.first, qprod.second);
      }
      stensor2<Tm> merge_cr() const{ // wf2[l,cr]
         auto qprod = dpt_cr(); 
         return merge_qt3_qt2_cr(*this, qprod.first, qprod.second);
      }
      stensor2<Tm> merge_lr() const{ // wf2[lr,c]
         auto qprod = dpt_lr();  
         return merge_qt3_qt2_lr(*this, qprod.first, qprod.second);
      }
      // reshape: split
      // wf3[lc1,r,c2] -> wf4[l,r,c1,c2]
      stensor4<Tm> split_lc1(const qbond& qlx, const qbond& qc1) const{
         auto dpt = qmerge(qlx, qc1).second;
         return split_qt4_qt3_lc1(*this, qlx, qc1, dpt);
      }
      // wf3[l,c2r,c1] -> wf4[l,r,c1,c2]
      stensor4<Tm> split_c2r(const qbond& qc2, const qbond& qrx) const{
         auto dpt = qmerge(qc2, qrx).second;
         return split_qt4_qt3_c2r(*this, qc2, qrx, dpt); 
      }
      // wf3[l,r,c1c2] -> wf4[l,r,c1,c2]
      stensor4<Tm> split_c1c2(const qbond& qc1, const qbond& qc2) const{
         auto dpt = qmerge(qc1, qc2).second;     
         return split_qt4_qt3_c1c2(*this, qc1, qc2, dpt);
      }
      // ZL@20221207 dump
      void dump(std::ofstream& ofs) const;
         public:
      bool own = true; // whether the object owns its data
      Tm* _data = nullptr;
      qinfo3<Tm> info;
      };

   template <typename Tm>
      void stensor3<Tm>::dump(std::ofstream& ofs) const{
         info.dump(ofs);
         ofs.write((char*)(_data), sizeof(Tm)*info._size);
      }

   template <typename Tm>
      void stensor3<Tm>::print(const std::string name, const int level) const{
         std::cout << "stensor3: " << name << " own=" << own << " _data=" << _data << std::endl; 
         info.print(name);
         int br, bc, bm;
         for(int i=0; i<info._nnzaddr.size(); i++){
            int idx = info._nnzaddr[i];
            info._addr_unpack(idx,br,bc,bm);
            const auto blk = (*this)(br,bc,bm);
            if(level >= 1){
               std::cout << "i=" << i << " idx=" << idx << " block[" 
                  << info.qrow.get_sym(br) << "," 
                  << info.qcol.get_sym(bc) << "," 
                  << info.qmid.get_sym(bm) << "]" 
                  << " dim0,dim1,dim2=(" 
                  << blk.dim0 << "," 
                  << blk.dim1 << ","
                  << blk.dim2 << ")" 
                  << " size=" << blk._size 
                  << std::endl; 
               if(level >= 2) blk.print("blk_"+std::to_string(idx));
            } // level>=1
         } // i
      }

   // fix middle index (bm,im) - bm-th block, im-idx - composite index!
   // A(l,r) = B[m](l,r)
   template <typename Tm>
      stensor2<Tm> stensor3<Tm>::fix_mid(const std::pair<int,int> mdx) const{
         int bm = mdx.first, im = mdx.second;   
         assert(im < info.qmid.get_dim(bm));
         auto symIn = std::get<2>(info.dir) ? info.sym-info.qmid.get_sym(bm) : 
            info.sym+info.qmid.get_sym(bm);
         stensor2<Tm> qt2(symIn, info.qrow, info.qcol, 
               {std::get<0>(info.dir), std::get<1>(info.dir)});
         for(int br=0; br<info._rows; br++){
            for(int bc=0; bc<info._cols; bc++){
               const auto blk3 = (*this)(br,bc,bm);
               if(blk3.empty()) continue;
               auto blk2 = qt2(br,bc);
               linalg::xcopy(blk2.size(), blk3.get(im).data(), blk2.data()); 
            } // bc
         } // br
         return qt2;
      }

   // deal with fermionic sign in fermionic direct product
   // wf[lcr](-1)^{p(c)}
   template <typename Tm>
      void mid_signed(const qinfo3<Tm>& info, Tm* data, const double fac=1.0){
         int br, bc, bm;
         for(int i=0; i<info._nnzaddr.size(); i++){
            int idx = info._nnzaddr[i];
            info._addr_unpack(idx,br,bc,bm);
            auto blk3 = info(br,bc,bm,data);
            double fac2 = (info.qmid.get_parity(bm)==0)? fac : -fac;
            linalg::xscal(blk3.size(), fac2, blk3.data());  
         }
      }
   template <typename Tm>
      void stensor3<Tm>::mid_signed(const double fac){
         ctns::mid_signed(info, _data, fac);
      }

   // wf[lcr](-1)^{p(l)}
   template <typename Tm>
      void row_signed(const qinfo3<Tm>& info, Tm* data, const double fac=1.0){
         int br, bc, bm;
         for(int i=0; i<info._nnzaddr.size(); i++){
            int idx = info._nnzaddr[i];
            info._addr_unpack(idx,br,bc,bm);
            auto blk3 = info(br,bc,bm,data); 
            double fac2 = (info.qrow.get_parity(br)==0)? fac : -fac;
            linalg::xscal(blk3.size(), fac2, blk3.data());  
         }
      }
   template <typename Tm>
      void stensor3<Tm>::row_signed(const double fac){
         ctns::row_signed(info, _data, fac);
      }

   template <typename Tm>
      void cntr_signed(const std::string block, const qinfo3<Tm>& info, Tm* data){
         if(block == "r"){
            int br, bc, bm;
            for(int i=0; i<info._nnzaddr.size(); i++){
               int idx = info._nnzaddr[i];
               info._addr_unpack(idx,br,bc,bm);
               auto blk3 = info(br,bc,bm,data); 
               // (-1)^{p(l)+p(c)}wf[l,c,r]
               int pt = info.qrow.get_parity(br) 
                  + info.qmid.get_parity(bm);
               if(pt%2 == 1) linalg::xscal(blk3.size(), -1.0, blk3.data());
            } // i
         }else if(block == "c"){
            int br, bc, bm;
            for(int i=0; i<info._nnzaddr.size(); i++){
               int idx = info._nnzaddr[i];
               info._addr_unpack(idx,br,bc,bm);
               auto blk3 = info(br,bc,bm,data); 
               // (-1)^{p(l)}wf[l,c,r]
               int pt = info.qrow.get_parity(br);
               if(pt%2 == 1) linalg::xscal(blk3.size(), -1.0, blk3.data());
            } // i
         } // block
      }
   template <typename Tm>
      void stensor3<Tm>::cntr_signed(const std::string block){
         ctns::cntr_signed(block, info, _data);
      }

   // Generate the sign for wf[lcr]|lcr> = wf3[lcr]|lrc> 
   // with wf3[lcr] = wf[lcr]*(-1)^{p[c]*p[r]}|lrc>
   // which is later used for wf3[l,c,r] <-> wf2[lr,c] (merge_lr)
   template <typename Tm>
      void stensor3<Tm>::permCR_signed(){
         int br, bc, bm;
         for(int i=0; i<info._nnzaddr.size(); i++){
            int idx = info._nnzaddr[i];
            info._addr_unpack(idx,br,bc,bm);
            auto blk3 = (*this)(br,bc,bm); 
            if(info.qmid.get_parity(bm)*info.qcol.get_parity(bc) == 1){
               linalg::xscal(blk3.size(), -1.0, blk3.data());
            }
         }
      }

   template <typename Tm>
      stensor3<Tm> stensor3<Tm>::K(const int nbar) const{
         const double fpo = (nbar%2==0)? 1.0 : -1.0;
         // the symmetry is flipped
         stensor3<Tm> qt3(info.sym.flip(), info.qrow, info.qcol, info.qmid, info.dir);
         int br, bc, bm;
         for(int i=0; i<info._nnzaddr.size(); i++){
            int idx = info._nnzaddr[i];
            info._addr_unpack(idx,br,bc,bm);
            auto blk3 = qt3(br,bc,bm); 
            // kramers
            const auto blkk = (*this)(br,bc,bm);
            int pt_r = info.qrow.get_parity(br);
            int pt_c = info.qcol.get_parity(bc);
            int pt_m = info.qmid.get_parity(bm);
            int mdim = info.qmid.get_dim(bm);
            // qt3[c](l,r) = blk[bar{c}](bar{l},bar{r})^*
            if(pt_m == 0){
               // c[e]
               for(int im=0; im<mdim; im++){
                  auto mat = blkk.get(im).time_reversal(pt_r, pt_c);
                  linalg::xaxpy(mat.size(), fpo, mat.data(), blk3.get(im).data());
               }
            }else{
               assert(mdim%2 == 0);
               int mdim2 = mdim/2;
               // c[o],c[\bar{o}]
               for(int im=0; im<mdim2; im++){
                  auto mat = blkk.get(im+mdim2).time_reversal(pt_r, pt_c);
                  linalg::xaxpy(mat.size(), fpo, mat.data(), blk3.get(im).data());
               }
               for(int im=0; im<mdim2; im++){
                  auto mat = blkk.get(im).time_reversal(pt_r, pt_c);
                  linalg::xaxpy(mat.size(), -fpo, mat.data(), blk3.get(im+mdim2).data());
               }
            } // pm
         } // i
         return qt3;
      }

} // ctns

#endif
