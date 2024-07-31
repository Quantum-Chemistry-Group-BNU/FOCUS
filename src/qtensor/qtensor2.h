#ifndef QTENSOR2_H
#define QTENSOR2_H

#include <type_traits>
#include "../core/serialization.h"
#include "../core/matrix.h"
#include "../core/linalg.h"
#include "qinfo2.h"
#include "qinfo2su2.h"
#include "qnum_qdpt.h"

#ifndef SERIAL
#include "../core/mpi_wrapper.h"
#endif

namespace ctns{

   template <bool ifab, typename Tm>
      struct qtensor3;
   template <bool ifab, typename Tm>
      struct qtensor4;

   const bool debug_qtensor2 = false;
   extern const bool debug_qtensor2; 

   template <bool ifab, typename Tm>
      using qinfo2type = typename std::conditional<ifab, qinfo2<Tm>, qinfo2su2<Tm>>::type;

   template <bool ifab, typename Tm>
      struct qtensor2{
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
            qtensor2(){};
            void init(const qsym& _sym, const qbond& _qrow, const qbond& _qcol, 
                  const direction2 _dir=dir_OPER, const bool _own=true){
               info.init(_sym, _qrow, _qcol, _dir);
               own = _own;
               if(own) this->allocate();
            }
            qtensor2(const qsym& _sym, const qbond& _qrow, const qbond& _qcol, 
                  const direction2 _dir=dir_OPER, const bool _own=true){
               this->init(_sym, _qrow, _qcol, _dir, _own);
            }
            // simple constructor from qinfo
            void init(const qinfo2type<ifab,Tm>& _info, const bool _own=true){
               info = _info;
               own = _own;
               if(own) this->allocate();
            }
            qtensor2(const qinfo2type<ifab,Tm>& _info, const bool _own=true){
               this->init(_info, _own);
            }
            // used to for setup ptr, if own=false
            void setup_data(Tm* data){
               assert(own == false);
               _data = data;
            }
            // desctructors
            ~qtensor2(){ 
               if(own) delete[] _data; 
            }
            // copy constructor 
            qtensor2(const qtensor2& st){
               if(debug_qtensor2) std::cout << "qtensor2: copy constructor - st.own=" << st.own << std::endl;   
               //assert(st.own == false);
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
            qtensor2& operator =(const qtensor2& st) = delete;
            // move constructor
            qtensor2(qtensor2&& st){
               if(debug_qtensor2) std::cout << "qtensor2: move constructor - st.own=" << st.own << std::endl;     
               assert(own == true);
               own = st.own;
               info = std::move(st.info);
               _data = st._data;
               st._data = nullptr;
            }
            // move assignment
            qtensor2& operator =(qtensor2&& st){
               if(debug_qtensor2) std::cout << "qtensor2: move assignment - st.own=" << st.own << std::endl;    
               // only move if the data is owned by the object, 
               // otherwise data needs to be copied explicitly!
               // e.g., linalg::xcopy(info._size, st._data, _data);
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
            bool dir_row() const{ return std::get<0>(info.dir); } 
            bool dir_col() const{ return std::get<1>(info.dir); }
            std::tuple<int,int> get_shape() const{
               return std::make_tuple(
                     info.qrow.get_dimAll(),
                     info.qcol.get_dimAll()
                     );
            }
            size_t size() const{ return info._size; }
            Tm* data() const{ return _data; }
            // inplace conjugate
            void _conj(){
               linalg::xconj(info._size, _data);
            }
            // conjugate
            qtensor2 conj() const{
               qtensor2 qt2(info);
               linalg::xcopy(info._size, _data, qt2._data);
               linalg::xconj(info._size, qt2._data);
               return qt2; 
            }
            // simple arithmetic operations
            qtensor2& operator *=(const Tm fac){
               linalg::xscal(info._size, fac, _data);
               return *this;
            }
            qtensor2& operator +=(const qtensor2& st){
               assert(info == st.info);
               linalg::xaxpy(info._size, 1.0, st.data(), _data);
               return *this;
            }
            qtensor2& operator -=(const qtensor2& st){
               assert(info == st.info);
               linalg::xaxpy(info._size, -1.0, st.data(), _data);
               return *this;
            }
            qtensor2 operator -() const{
               qtensor2 st(info);
               linalg::xaxpy(info._size, -1.0, _data, st._data);
               return st;
            }
            // algebra
            friend qtensor2 operator +(const qtensor2& qta, const qtensor2& qtb){
               assert(qta.info == qtb.info); 
               qtensor2 qt(qta.info);
               linalg::xcopy(qt.info._size, qta._data, qt._data);
               qt += qtb;
               return qt;
            }
            friend qtensor2 operator -(const qtensor2& qta, const qtensor2& qtb){
               assert(qta.info == qtb.info); 
               qtensor2 qt(qta.info);
               linalg::xcopy(qt.info._size, qta._data, qt._data);
               qt -= qtb;
               return qt;
            }
            friend qtensor2 operator *(const Tm fac, const qtensor2& qta){
               qtensor2 qt(qta.info);
               linalg::xaxpy(qt.info._size, fac, qta._data, qt._data);
               return qt;
            }
            friend qtensor2 operator *(const qtensor2& qt, const Tm fac){
               return fac*qt;
            }
            double normF() const{ return linalg::xnrm2(info._size, _data); }
            void set_zero(){ memset(_data, 0, info._size*sizeof(Tm)); }
            // for sweep algorithm
            void add_noise(const double noise){
               auto rand = linalg::random_matrix<Tm>(info._size,1);
               linalg::xaxpy(info._size, noise, rand.data(), _data);
            }
            // check whether <l|o|r> is a faithful rep for o=I
            double check_identityMatrix(const double thresh_ortho, const bool debug=false) const;
            // from/to dense matrix: assign block to proper place
            void from_matrix(const linalg::matrix<Tm>& mat); 
            linalg::matrix<Tm> to_matrix() const;
            // algebra
            qtensor2<ifab,Tm> dot(const qtensor2<ifab,Tm>& qt) const{ 
               return contract_qt2_qt2(*this, qt); 
            }

            // --- SPECIFIC FUNCTIONS : abelian case ---
            // access
            template <bool y=ifab, std::enable_if_t<y,int> = 0>
               dtensor2<Tm> operator()(const int br, const int bc) const{
                  return info(br,bc,_data);
               }
            template <bool y=ifab, std::enable_if_t<y,int> = 0>
               Tm* start_ptr(const int br, const int bc) const{
                  size_t off = info.get_offset(br,bc);
                  return (off==0)? nullptr : _data+off-1;
               }
            // print [comes latter than access]
            template <bool y=ifab, std::enable_if_t<y,int> = 0>
               void print(const std::string name, const int level=0) const;
            // ZL20200531: Permute the line of diagrams, while maintaining their directions
            // 	     This does not change the tensor, but just permute order of index
            //         
            //         i --<--*--<-- j => j -->--*-->-- i
            //
            template <bool y=ifab, std::enable_if_t<y,int> = 0>
               qtensor2<ifab,Tm> P() const;
            // ZL20200531: This is used in taking Hermitian conjugate of operators.
            // 	     If row/col is permuted while dir fixed, this effectively changes 
            // 	     the direction of lines in diagrams
            //         
            //         i --<--*--<-- j => j --<--*'--<-- i
            //
            template <bool y=ifab, std::enable_if_t<y,int> = 0>
               qtensor2<ifab,Tm> H() const;
            // ZL20210401: generate matrix representation for Kramers paired operators
            template <bool y=ifab, std::enable_if_t<y,int> = 0>
               qtensor2<ifab,Tm> K(const int nbar=0) const;
            // reshape: split into qtensor3
            // wf2[lc,r] -> wf3[l,r,c]
            template <bool y=ifab, std::enable_if_t<y,int> = 0>
               qtensor3<ifab,Tm> split_lc(const qbond& qlx, const qbond& qcx) const{
                  auto dpt = qmerge(qlx, qcx).second;
                  return split_qt3_qt2_lc(*this, qlx, qcx, dpt);
               }
            // wf2[l,cr] -> wf3[l,r,c]
            template <bool y=ifab, std::enable_if_t<y,int> = 0>
               qtensor3<ifab,Tm> split_cr(const qbond& qcx, const qbond& qrx) const{
                  auto dpt = qmerge(qcx, qrx).second;
                  return split_qt3_qt2_cr(*this, qcx, qrx, dpt);
               }
            // wf2[lr,c] -> wf3[l,r,c]
            template <bool y=ifab, std::enable_if_t<y,int> = 0>
               qtensor3<ifab,Tm> split_lr(const qbond& qlx, const qbond& qrx) const{
                  auto dpt = qmerge(qlx, qrx).second;
                  return split_qt3_qt2_lr(*this, qlx, qrx, dpt);
               }
            // shorthand function
            // wf2[lr,c1c2] => wf3[l,r,c1c2] => wf4[l,r,c1,c2] 
            template <bool y=ifab, std::enable_if_t<y,int> = 0>
               qtensor4<ifab,Tm> split_lr_c1c2(const qbond& qlx, const qbond& qrx, 
                     const qbond& qc1, const qbond& qc2) const{
                  return (this->split_lr(qlx, qrx)).split_c1c2(qc1, qc2);
               }
            template <bool y=ifab, std::enable_if_t<y,int> = 0>
               qtensor2<ifab,Tm> align_qrow(const qbond& qrow2) const;

            // --- SPECIFIC FUNCTIONS : non-abelian case ---
            // access
            template <bool y=ifab, std::enable_if_t<!y,int> = 0>
               dtensor2<Tm> operator()(const int br, const int bc) const{
                  return info(br,bc,_data);
               }
            template <bool y=ifab, std::enable_if_t<!y,int> = 0>
               Tm* start_ptr(const int br, const int bc) const{
                  size_t off = info.get_offset(br,bc);
                  return (off==0)? nullptr : _data+off-1;
               }
            // print
            template <bool y=ifab, std::enable_if_t<!y,int> = 0>
               void print(const std::string name, const int level=0) const;
            // ZL20200531: Permute the line of diagrams, while maintaining their directions
            // 	     This does not change the tensor, but just permute order of index
            //         
            //         i --<--*--<-- j => j -->--*-->-- i
            //
            template <bool y=ifab, std::enable_if_t<!y,int> = 0>
               qtensor2<ifab,Tm> P() const;
            // return adjoint tensor
            template <bool y=ifab, std::enable_if_t<!y,int> = 0>
               qtensor2<ifab,Tm> H(const bool adjoint=false) const;
            // reshape: split into qtensor3
            // wf2[lc,r] -> wf3[l,r,c]
            template <bool y=ifab, std::enable_if_t<!y,int> = 0>
               qtensor3<ifab,Tm> split_lc(const qbond& qlx, const qbond& qcx) const{
                  auto dpt = qmerge_su2(qlx, qcx).second;
                  return split_qt3_qt2_lc(*this, qlx, qcx, dpt);
               }
            // wf2[l,cr] -> wf3[l,r,c]
            template <bool y=ifab, std::enable_if_t<!y,int> = 0>
               qtensor3<ifab,Tm> split_cr(const qbond& qcx, const qbond& qrx) const{
                  auto dpt = qmerge_su2(qcx, qrx).second;
                  return split_qt3_qt2_cr(*this, qcx, qrx, dpt);
               }
            // wf2[lr,c] -> wf3[l,r,c]
            template <bool y=ifab, std::enable_if_t<!y,int> = 0>
               qtensor3<ifab,Tm> split_lr(const qbond& qlx, const qbond& qrx) const{
                  std::cout << "error: split_lr is not implemented for su2 case!" << std::endl;
                  exit(1);
               }
            // shorthand function
            // wf2[lr,c1c2] => wf3[l,r,c1c2] => wf4[l,r,c1,c2] 
            template <bool y=ifab, std::enable_if_t<!y,int> = 0>
               qtensor4<ifab,Tm> split_lr_c1c2(const qbond& qlx, const qbond& qrx, 
                     const qbond& qc1, const qbond& qc2) const{
                  std::cout << "error: split_lr_c1c2 is not implemented for su2 case!" << std::endl;
                  exit(1);
               }
            template <bool y=ifab, std::enable_if_t<!y,int> = 0>
               qtensor2<ifab,Tm> align_qrow(const qbond& qrow2) const;

         public:
            bool own = true; // whether the object owns its data
            Tm* _data = nullptr;
            qinfo2type<ifab,Tm> info;
      };

   template <typename Tm>
      using stensor2 = qtensor2<true,Tm>;
   template <typename Tm>
      using stensor2su2 = qtensor2<false,Tm>;

   template <bool ifab, typename Tm>
      double qtensor2<ifab,Tm>::check_identityMatrix(const double thresh_ortho, const bool debug) const{
         if(debug) std::cout << "qtensor2::check_identityMatrix thresh_ortho=" << thresh_ortho << std::endl;
         double maxdiff = -1.0;
         for(int br=0; br<info._rows; br++){
            for(int bc=0; bc<info._cols; bc++){
               const auto blk = (*this)(br,bc);
               if(blk.empty()) continue;
               if(br != bc){
                  std::string msg = "error: not a block-diagonal matrix! br,bc=";
                  tools::exit(msg+std::to_string(br)+","+std::to_string(bc));
               }
               auto qr = info.qrow.get_sym(br);
               int ndim = info.qrow.get_dim(br);
               double diff = (blk.to_matrix() - linalg::identity_matrix<Tm>(ndim)).normF();
               maxdiff = std::max(diff,maxdiff);
               if(debug || (!debug && diff > thresh_ortho)){ 
                  std::cout << " br=" << br << " qr=" << qr << " ndim=" << ndim 
                     << " |Sr-Id|_F=" << diff << std::endl;
               }
               if(diff > thresh_ortho){
                  blk.print("diagonal block");
                  tools::exit("error: not an identity matrix!"); 
               }
            } // bc
         } // br
         return maxdiff;
      }

   template <bool ifab, typename Tm>
      linalg::matrix<Tm> qtensor2<ifab,Tm>::to_matrix() const{
         int m = info.qrow.get_dimAll();
         int n = info.qcol.get_dimAll();
         linalg::matrix<Tm> mat(m,n);
         // assign block to proper place
         auto roff = info.qrow.get_offset();
         auto coff = info.qcol.get_offset();
         for(int br=0; br<info._rows; br++){
            int offr = roff[br];		 
            for(int bc=0; bc<info._cols; bc++){
               int offc = coff[bc];
               const auto blk = (*this)(br,bc);
               if(blk.empty()) continue;
               for(int ic=0; ic<blk.dim1; ic++){
                  for(int ir=0; ir<blk.dim0; ir++){
                     mat(offr+ir,offc+ic) = blk(ir,ic);
                  } // ir
               } // ic
            } // bc
         } // br
         return mat;
      }

   // from dense matrix: assign block to proper place
   template <bool ifab, typename Tm>
      void qtensor2<ifab,Tm>::from_matrix(const linalg::matrix<Tm>& mat){
         auto roff = info.qrow.get_offset();
         auto coff = info.qcol.get_offset();
         for(int br=0; br<info._rows; br++){
            int offr = roff[br];		 
            for(int bc=0; bc<info._cols; bc++){
               int offc = coff[bc];
               auto blk = (*this)(br,bc);
               if(blk.empty()) continue;
               for(int ic=0; ic<blk.dim1; ic++){
                  for(int ir=0; ir<blk.dim0; ir++){
                     blk(ir,ic) = mat(offr+ir,offc+ic);
                  } // ir
               } // ic
            } // bc
         } // br
      }

} // ctns

#ifndef SERIAL

namespace mpi_wrapper{

   // qtensor2
   template <bool ifab, typename Tm>
      void broadcast(const boost::mpi::communicator & comm, ctns::qtensor2<ifab,Tm>& qt2, int root){
         boost::mpi::broadcast(comm, qt2.own, root);
         boost::mpi::broadcast(comm, qt2.info, root);
         int rank = comm.rank();
         if(rank != root && qt2.own) qt2._data = new Tm[qt2.info._size];
         size_t chunksize = get_chunksize<Tm>();
         size_t size = qt2.info._size; 
         for(size_t offset=0; offset<size; offset+=chunksize){
            size_t len = std::min(chunksize, size-offset);
            boost::mpi::broadcast(comm, qt2._data+offset, len, root); 
         }
      }

} // mpi_wrapper

#endif

#endif
