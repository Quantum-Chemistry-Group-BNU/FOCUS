#ifndef QTENSOR3_H
#define QTENSOR3_H

#include <type_traits>
#include "../core/serialization.h"
#include "../core/matrix.h"
#include "qinfo3.h"
#include "qinfo3su2.h"
#include "qnum_qdpt.h"

#ifndef SERIAL
#include "../core/mpi_wrapper.h"
#endif

namespace ctns{

   template <bool ifab, typename Tm>
      struct qtensor2;
   template <bool ifab, typename Tm>
      struct qtensor4;

   const bool debug_qtensor3 = false;
   extern const bool debug_qtensor3;

   template <bool ifab, typename Tm>
      struct qtensor3{
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
            qtensor3(){}
            void init(const qsym& _sym, const qbond& _qrow, const qbond& _qcol, const qbond& _qmid,
                  const direction3 _dir={0,1,1}, const bool _own=true){
               info.init(_sym, _qrow, _qcol, _qmid, _dir);
               own = _own;
               if(own) this->allocate();
            }
            qtensor3(const qsym& _sym, const qbond& _qrow, const qbond& _qcol, const qbond& _qmid,
                  const direction3 _dir={0,1,1}, const bool _own=true){
               this->init(_sym, _qrow, _qcol, _qmid, _dir, _own);
            }
            // simple constructor from qinfo
            void init(const qinfo3<Tm>& _info, const bool _own=true){
               info = _info;
               own = _own;
               if(own) this->allocate();
            }
            qtensor3(const qinfo3<Tm>& _info, const bool _own=true){
               this->init(_info, _own);
            }
            // used to for setup ptr, if own=false
            void setup_data(Tm* data){
               assert(own == false);
               _data = data;
            }
            // desctructors
            ~qtensor3(){ 
               if(own) delete[] _data; 
            }
            // copy constructor
            qtensor3(const qtensor3& st){;
               if(debug_qtensor3) std::cout << "qtensor3: copy constructor - st.own=" << st.own << std::endl;   
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
            qtensor3& operator =(const qtensor3& st) = delete;
            // move constructor
            qtensor3(qtensor3&& st){
               if(debug_qtensor3) std::cout << "qtensor3: move constructor - st.own=" << st.own << std::endl;    
               assert(own == true);
               own = st.own; 
               info = std::move(st.info);
               _data = st._data;
               st._data = nullptr;
            }
            // move assignment
            qtensor3& operator =(qtensor3&& st){
               if(debug_qtensor3) std::cout << "qtensor3: move assignment - st.own=" << st.own << std::endl;
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
            std::tuple<int,int,int> get_shape() const{
               return std::make_tuple(
                     info.qrow.get_dimAll(),
                     info.qcol.get_dimAll(),
                     info.qmid.get_dimAll()
                     );
            }
            size_t size() const{ return info._size; }
            Tm* data() const{ return _data; }
            // simple arithmetic operations
            qtensor3& operator *=(const Tm fac){
               linalg::xscal(info._size, fac, _data);
               return *this;
            }
            qtensor3& operator +=(const qtensor3& st){
               assert(info == st.info);
               linalg::xaxpy(info._size, 1.0, st.data(), _data);
               return *this;
            }
            qtensor3& operator -=(const qtensor3& st){
               assert(info == st.info);
               linalg::xaxpy(info._size, -1.0, st.data(), _data);
               return *this;
            }
            // algebra
            friend qtensor3 operator +(const qtensor3& qta, const qtensor3& qtb){
               assert(qta.info == qtb.info); 
               qtensor3 qt(qta.info);
               linalg::xcopy(qt.info._size, qta._data, qt._data);
               qt += qtb;
               return qt;
            }
            friend qtensor3 operator -(const qtensor3& qta, const qtensor3& qtb){
               assert(qta.info == qtb.info); 
               qtensor3 qt(qta.info);
               linalg::xcopy(qt.info._size, qta._data, qt._data);
               qt -= qtb;
               return qt;
            }
            friend qtensor3 operator *(const Tm fac, const qtensor3& qta){
               qtensor3 qt(qta.info);
               linalg::xaxpy(qt.info._size, fac, qta._data, qt._data);
               return qt;
            }
            friend qtensor3 operator *(const qtensor3& qt, const Tm fac){
               return fac*qt;
            }
            double normF() const{ return linalg::xnrm2(info._size, _data); }
            void set_zero(){ memset(_data, 0, info._size*sizeof(Tm)); }

            // --- SPECIFIC FUNCTIONS : abelian case ---
            // access
            template <bool y=ifab, std::enable_if_t<y,int> = 0>
               dtensor3<Tm> operator()(const int br, const int bc, const int bm) const{
                  return info(br,bc,bm,_data);
               }
            template <bool y=ifab, std::enable_if_t<y,int> = 0>
               Tm* start_ptr(const int br, const int bc, const int bm) const{
                  size_t off = info._offset[info._addr(br,bc,bm)];
                  return (off==0)? nullptr : _data+off-1;
               }
            // print
            template <bool y=ifab, std::enable_if_t<y,int> = 0>
               void print(const std::string name, const int level=0) const;
            // fix middle index (bm,im) - bm-th block, im-idx - composite index!
            template <bool y=ifab, std::enable_if_t<y,int> = 0>
               qtensor2<ifab,Tm> fix_mid(const std::pair<int,int> mdx) const;
            // deal with fermionic sign in fermionic direct product
            // wf[lcr](-1)^{p(c)}
            template <bool y=ifab, std::enable_if_t<y,int> = 0>
               void mid_signed(const double fac=1.0); 
            // wf[lcr](-1)^{p(l)}
            template <bool y=ifab, std::enable_if_t<y,int> = 0>
               void row_signed(const double fac=1.0); 
            template <bool y=ifab, std::enable_if_t<y,int> = 0>
               void cntr_signed(const std::string block);
            // wf[lcr]->wf[lcr]*(-1)^{p[c]*p[r]}
            template <bool y=ifab, std::enable_if_t<y,int> = 0>
               void permCR_signed(); 
            // ZL20210413: application of time-reversal operation
            template <bool y=ifab, std::enable_if_t<y,int> = 0>
               qtensor3<ifab,Tm> K(const int nbar=0) const;
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
               qproduct dpt_lc() const{ return qmerge(info.qrow, info.qmid); }
            template <bool y=ifab, std::enable_if_t<y,int> = 0>
               qproduct dpt_cr() const{ return qmerge(info.qmid, info.qcol); }
            template <bool y=ifab, std::enable_if_t<y,int> = 0>
               qproduct dpt_lr() const{ return qmerge(info.qrow, info.qcol); }
            // reshape: merge wf3[l,r,c]
            template <bool y=ifab, std::enable_if_t<y,int> = 0>
               qtensor2<ifab,Tm> merge_lc() const{ // wf2[lc,r] 
                  auto qprod = dpt_lc();
                  return merge_qt3_qt2_lc(*this, qprod.first, qprod.second);
               }
            template <bool y=ifab, std::enable_if_t<y,int> = 0>
               qtensor2<ifab,Tm> merge_cr() const{ // wf2[l,cr]
                  auto qprod = dpt_cr(); 
                  return merge_qt3_qt2_cr(*this, qprod.first, qprod.second);
               }
            template <bool y=ifab, std::enable_if_t<y,int> = 0>
               qtensor2<ifab,Tm> merge_lr() const{ // wf2[lr,c]
                  auto qprod = dpt_lr();  
                  return merge_qt3_qt2_lr(*this, qprod.first, qprod.second);
               }
            // reshape: split
            // wf3[lc1,r,c2] -> wf4[l,r,c1,c2]
            template <bool y=ifab, std::enable_if_t<y,int> = 0>
               qtensor4<ifab,Tm> split_lc1(const qbond& qlx, const qbond& qc1) const{
                  auto dpt = qmerge(qlx, qc1).second;
                  return split_qt4_qt3_lc1(*this, qlx, qc1, dpt);
               }
            // wf3[l,c2r,c1] -> wf4[l,r,c1,c2]
            template <bool y=ifab, std::enable_if_t<y,int> = 0>
               qtensor4<ifab,Tm> split_c2r(const qbond& qc2, const qbond& qrx) const{
                  auto dpt = qmerge(qc2, qrx).second;
                  return split_qt4_qt3_c2r(*this, qc2, qrx, dpt); 
               }
            // wf3[l,r,c1c2] -> wf4[l,r,c1,c2]
            template <bool y=ifab, std::enable_if_t<y,int> = 0>
               qtensor4<ifab,Tm> split_c1c2(const qbond& qc1, const qbond& qc2) const{
                  auto dpt = qmerge(qc1, qc2).second;     
                  return split_qt4_qt3_c1c2(*this, qc1, qc2, dpt);
               }
            // ZL@20221207 dump
            template <bool y=ifab, std::enable_if_t<y,int> = 0>
               void dump(std::ofstream& ofs) const;

            // --- SPECIFIC FUNCTIONS : non-abelian case ---

         public:
            bool own = true; // whether the object owns its data
            Tm* _data = nullptr;
            typename std::conditional<ifab, qinfo3<Tm>, qinfo3su2<Tm>>::type info;
      };

   template <typename Tm>
      using stensor3 = qtensor3<true,Tm>;
   template <typename Tm>
      using stensor3su2 = qtensor3<false,Tm>;

} // ctns

#ifndef SERIAL

namespace mpi_wrapper{

   // qtensor3
   template <bool ifab, typename Tm>
      void broadcast(const boost::mpi::communicator & comm, ctns::qtensor3<ifab,Tm>& qt3, int root){
         boost::mpi::broadcast(comm, qt3.own, root);
         boost::mpi::broadcast(comm, qt3.info, root);
         int rank = comm.rank();
         if(rank != root && qt3.own) qt3._data = new Tm[qt3.info._size];
         size_t chunksize = get_chunksize<Tm>();
         size_t size = qt3.info._size; 
         for(size_t offset=0; offset<size; offset+=chunksize){
            size_t len = std::min(chunksize, size-offset);
            boost::mpi::broadcast(comm, qt3._data+offset, len, root); 
         }
      }

} // mpi_wrapper

#endif

#endif
