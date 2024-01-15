#ifndef QTENSOR4_H
#define QTENSOR4_H

#include <type_traits>
#include "../core/serialization.h"
#include "../core/matrix.h"
#include "qinfo4.h"
#include "qinfo4su2.h"
#include "qnum_qdpt.h"

namespace ctns{

   template <bool ifab, typename Tm>
      struct qtensor2;
   template <bool ifab, typename Tm>
      struct qtensor3;

   const bool debug_qtensor4 = false;
   extern const bool debug_qtensor4;

   template <bool ifab, typename Tm>
      struct qtensor4{
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
            qtensor4(){}
            void init(const qsym& _sym, 
                  const qbond& _qrow, const qbond& _qcol, 
                  const qbond& _qmid, const qbond& _qver,
                  const bool _own=true){
               info.init(_sym, _qrow, _qcol, _qmid, _qver);
               own = _own;
               if(own) this->allocate();
            }
            qtensor4(const qsym& _sym, 
                  const qbond& _qrow, const qbond& _qcol, 
                  const qbond& _qmid, const qbond& _qver,
                  const bool _own=true){
               this->init(_sym, _qrow, _qcol, _qmid, _qver, _own);
            }
            // simple constructor from qinfo
            void init(const qinfo4<Tm>& _info, const bool _own=true){
               info = _info;
               own = _own;
               if(own) this->allocate();
            }
            qtensor4(const qinfo4<Tm>& _info, const bool _own=true){
               this->init(_info, _own);
            }
            // used to for setup ptr, if own=false
            void setup_data(Tm* data){
               assert(own == false);
               _data = data;
            }
            // desctructors
            ~qtensor4(){ 
               if(own) delete[] _data; 
            }
            // copy constructor
            qtensor4(const qtensor4& st){
               if(debug_qtensor4) std::cout << "qtensor4: copy constructor - st.own=" << st.own << std::endl;   
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
            qtensor4& operator =(const qtensor4& st) = delete;
            // move constructor
            qtensor4(qtensor4&& st){
               if(debug_qtensor4) std::cout << "qtensor4: move constructor - st.own=" << st.own << std::endl;
               assert(own == true);
               own = st.own; 
               info = std::move(st.info);
               _data = st._data;
               st._data = nullptr;
            }
            // move assignment
            qtensor4& operator =(qtensor4&& st){
               if(debug_qtensor4) std::cout << "qtensor4: move assignment - st.own=" << st.own << std::endl;     
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
            int vers() const{ return info._vers; }
            bool dir_row() const{ return true; } 
            bool dir_col() const{ return true; }
            bool dir_mid() const{ return true; } 
            bool dir_ver() const{ return true; }
            std::tuple<int,int,int,int> get_shape() const{
               return std::make_tuple(
                     info.qrow.get_dimAll(),
                     info.qcol.get_dimAll(),
                     info.qmid.get_dimAll(),
                     info.qver.get_dimAll()
                     );
            }
            size_t size() const{ return info._size; }
            Tm* data() const{ return _data; }
            // simple arithmetic operations
            qtensor4& operator *=(const Tm fac){
               linalg::xscal(info._size, fac, _data);
               return *this;
            }
            qtensor4& operator +=(const qtensor4& st){
               assert(info == st.info);
               linalg::xaxpy(info._size, 1.0, st.data(), _data);
               return *this;
            }
            qtensor4& operator -=(const qtensor4& st){
               assert(info == st.info);
               linalg::xaxpy(info._size, -1.0, st.data(), _data);
               return *this;
            }
            // algebra
            friend qtensor4 operator +(const qtensor4& qta, const qtensor4& qtb){
               assert(qta.info == qtb.info); 
               qtensor4 qt(qta.info);
               linalg::xcopy(qt.info._size, qta._data, qt._data);
               qt += qtb;
               return qt;
            }
            friend qtensor4 operator -(const qtensor4& qta, const qtensor4& qtb){
               assert(qta.info == qtb.info); 
               qtensor4 qt(qta.info);
               linalg::xcopy(qt.info._size, qta._data, qt._data);
               qt -= qtb;
               return qt;
            }
            friend qtensor4 operator *(const Tm fac, const qtensor4& qta){
               qtensor4 qt(qta.info);
               linalg::xaxpy(qt.info._size, fac, qta._data, qt._data);
               return qt;
            }
            friend qtensor4 operator *(const qtensor4& qt, const Tm fac){
               return fac*qt;
            }
            double normF() const{ return linalg::xnrm2(info._size, _data); }
            void set_zero(){ memset(_data, 0, info._size*sizeof(Tm)); }

            // --- SPECIFIC FUNCTIONS ---
            // access
            template <bool y=ifab, std::enable_if_t<y,int> = 0>
               dtensor4<Tm> operator()(const int br, const int bc, 
                     const int bm, const int bv) const{ 
                  return info(br,bc,bm,bv,_data);
               }
            template <bool y=ifab, std::enable_if_t<y,int> = 0>
               Tm* start_ptr(const int br, const int bc, 
                     const int bm, const int bv) const{
                  size_t off = info._offset[info._addr(br,bc,bm,bv)];
                  return (off==0)? nullptr : _data+off-1;
               }
            // print
            template <bool y=ifab, std::enable_if_t<y,int> = 0>
               void print(const std::string name, const int level=0) const;
            // deal with fermionic sign in fermionic direct product            
            template <bool y=ifab, std::enable_if_t<y,int> = 0>
               void cntr_signed(const std::string block);
            // wf[lc1c2r]->wf[lc1c2r]*(-1)^{(p[c1]+p[c2])*p[r]} 
            template <bool y=ifab, std::enable_if_t<y,int> = 0>
               void permCR_signed(); 
            // ZL20210413: application of time-reversal operation
            template <bool y=ifab, std::enable_if_t<y,int> = 0>
               qtensor4<ifab,Tm> K(const int nbar=0) const;
            // for sweep algorithm
            template <bool y=ifab, std::enable_if_t<y,int> = 0>
               void from_array(const Tm* array){
                  linalg::xcopy(info._size, array, _data);
               }
            template <bool y=ifab, std::enable_if_t<y,int> = 0>
               void to_array(Tm* array) const{
                  linalg::xcopy(info._size, _data, array);
               }
            // for decimation
            template <bool y=ifab, std::enable_if_t<y,int> = 0>
               qproduct dpt_lc1()  const{ return qmerge(info.qrow, info.qmid); };
            template <bool y=ifab, std::enable_if_t<y,int> = 0>
               qproduct dpt_c2r()  const{ return qmerge(info.qver, info.qcol); };
            template <bool y=ifab, std::enable_if_t<y,int> = 0>
               qproduct dpt_c1c2() const{ return qmerge(info.qmid, info.qver); };
            template <bool y=ifab, std::enable_if_t<y,int> = 0>
               qproduct dpt_lr()   const{ return qmerge(info.qrow, info.qcol); };
            // reshape: merge wf4[l,r,c1,c2]
            template <bool y=ifab, std::enable_if_t<y,int> = 0>
               qtensor3<ifab,Tm> merge_lc1() const{ // wf3[lc1,r,c2] 
                  auto qprod = dpt_lc1();
                  return merge_qt4_qt3_lc1(*this, qprod.first, qprod.second);
               }
            template <bool y=ifab, std::enable_if_t<y,int> = 0>
               qtensor3<ifab,Tm> merge_c2r() const{ // wf3[l,c2r,c1]
                  auto qprod = dpt_c2r();
                  return merge_qt4_qt3_c2r(*this, qprod.first, qprod.second);
               } 
            template <bool y=ifab, std::enable_if_t<y,int> = 0>
               qtensor3<ifab,Tm> merge_c1c2() const{ // wf3[l,r,c1c2]
                  auto qprod = dpt_c1c2();
                  return merge_qt4_qt3_c1c2(*this, qprod.first, qprod.second);
               }
            // shorthand function
            // wf4[l,r,c1,c2] => wf3[lc1,r,c2] => wf2[lc1,c2r]
            template <bool y=ifab, std::enable_if_t<y,int> = 0>
               qtensor2<ifab,Tm> merge_lc1_c2r() const{
                  return (this->merge_lc1()).merge_cr();
               }
            // wf4[l,r,c1,c2] => wf3[l,r,c1c2] => wf2[lr,c1c2]
            template <bool y=ifab, std::enable_if_t<y,int> = 0>
               qtensor2<ifab,Tm> merge_lr_c1c2() const{
                  return (this->merge_c1c2()).merge_lr();
               }

            // --- SPECIFIC FUNCTIONS : non-abelian case ---

         public:
            bool own = true; // whether the object owns its data
            Tm* _data = nullptr;
            typename std::conditional<ifab, qinfo4<Tm>, qinfo4su2<Tm>>::type info;
      };

   template <typename Tm>
      using stensor4 = qtensor4<true,Tm>;
   template <typename Tm>
      using stensor4su2 = qtensor4<false,Tm>;

} // ctns

#endif
