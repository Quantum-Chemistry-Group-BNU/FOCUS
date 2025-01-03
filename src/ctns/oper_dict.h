#ifndef OPER_DICT_H
#define OPER_DICT_H

#include <map>
#include "../core/serialization.h"
#include "../core/integral.h"
#include "../qtensor/qtensor.h"
#include "oper_index.h"
#ifdef GPU
#include "../gpu/gpu_env.h"
#endif

namespace ctns{

   // --- oper_dict: container for operators --- 

   template <bool ifab, typename Tm>
      using qoper_map = std::map<int,qtensor2<ifab,Tm>>; // index to operator
   template <typename Tm>
      using oper_map = std::map<int,qtensor2<true,Tm>>;
   template <typename Tm>
      using opersu2_map = std::map<int,qtensor2<false,Tm>>;

   template <bool ifab, typename Tm>
      struct qoper_dict{
         private:
            // IO of data will be handled in oper_io.h
            friend class boost::serialization::access;	   
            template <class Archive>
               void serialize(Archive & ar, const unsigned int version){
                  ar & sorb & isym & ifkr & qbra & qket 
                     & cindex & krest & oplist 
                     & mpisize & mpirank & ifdist2 & ifdists
                     & ifhermi;
               }
         public:
            // constructor
            qoper_dict(){}
            ~qoper_dict(){ 
               delete[] _data;
#ifdef GPU
               GPUmem.deallocate(_dev_data, _size*sizeof(Tm));
#endif
            }
            void clear(){
               delete[] _data;
               _data = nullptr;
            }
            void clear_gpu(){
#ifdef GPU
               GPUmem.deallocate(_dev_data, _size*sizeof(Tm));
#endif
               _dev_data = nullptr;
            }
            bool avail_cpu() const{ return _data != nullptr; }
            bool avail_gpu() const{ return _dev_data != nullptr; }
#ifdef GPU
            void allocate_gpu(const bool ifmemset=false){
               assert(!this->avail_gpu());
               //--- ZL@2024/12/15 memory check --- 
               size_t avail, total;
               CUDA_CHECK(cudaMemGetInfo(&avail, &total));
               size_t size_bytes = _size*sizeof(Tm);
               if(size_bytes > avail){
                  std::cout << "error in qoper_dict.allocate_gpu: insufficient memory!"
                     << " rank=" << mpirank 
                     << " size=" << size_bytes/1024.0/1024.0/1024.0
                     << " avail=" << avail/1024.0/1024.0/1024.0 
                     << " total=" << total/1024.0/1024.0/1024.0
                     << std::endl;
                  exit(1);
               }
               //--- end of gpu memory check ---
               _dev_data = (Tm*)GPUmem.allocate(size_bytes);
               if(ifmemset) GPUmem.memset(_dev_data, size_bytes);
            }
            void to_gpu(){
               assert(_dev_data != nullptr && _data != nullptr);
               GPUmem.to_gpu(_dev_data, _data, _size*sizeof(Tm));
            }
            void to_cpu(){
               assert(_dev_data != nullptr);
               if(!this->avail_cpu()){
                  this->allocate_cpu();
                  this->setup_data(); // assign pointer for each operator
               }
               GPUmem.to_cpu(_data, _dev_data, _size*sizeof(Tm));
            }
#endif
            // copy
            qoper_dict(const qoper_dict& op_dict) = delete;
            qoper_dict& operator =(const qoper_dict& op_dict) = delete;
            // move
            qoper_dict(qoper_dict&& op_dict) = delete;
            // move assignment
            qoper_dict& operator =(qoper_dict&& st){
               if(this != &st){
                  sorb = st.sorb;
                  isym = st.isym;
                  ifkr = st.ifkr;
                  qbra = std::move(st.qbra);
                  qket = std::move(st.qket);
                  cindex = std::move(st.cindex);
                  krest = std::move(st.krest);
                  oplist = std::move(st.oplist);
                  mpisize = st.mpisize;
                  mpirank = st.mpirank;
                  ifdist2 = st.ifdist2;
                  ifdists = st.ifdists;
                  ifhermi = st.ifhermi;
                  _offset = std::move(st._offset);
                  _opdict = std::move(st._opdict);
                  _indexmap = std::move(st._indexmap);
                  _sizes = std::move(st._sizes);
                  _opoff = std::move(st._opoff);
                  _opsize = st._opsize;
                  delete[] _data;
                  _data = st._data;
                  st._data = nullptr;
#ifdef GPU
                  if(_dev_data != nullptr) GPUmem.deallocate(_dev_data, _size*sizeof(Tm));
#endif
                  _dev_data = st._dev_data;
                  st._dev_data = nullptr;
                  // put it after deallocate to make the deallocate correct!
                  _size = st._size;
               }
               return *this;
            }
            // initialize _opdict, _size, _opsize
            void setup_opdict(const bool debug=false);
            // setup the mapping to physical address
            void setup_data();
            // allocate cpu memory
            void allocate_cpu(const bool ifmemset=false){
               assert(!this->avail_cpu());
               _data = new Tm[_size];
               if(ifmemset) memset(_data, 0, _size*sizeof(Tm));
            }
            // initialization
            void init(const bool ifmemset=false){
               this->setup_opdict();
               this->allocate_cpu(ifmemset);
               this->setup_data(); // assign pointer for each operator
            }
            // stored operators
            std::vector<int> oper_index_op(const char key) const;
            // symmetry of op
            qsym get_qsym_op(const char key, const int idx) const;
            // access
            const qoper_map<ifab,Tm>& operator()(const char key) const{
               return _opdict.at(key);
            }
            qoper_map<ifab,Tm>& operator()(const char key){
               return _opdict[key];      
            }
            // check existence
            bool ifexist(const char key) const{
               return _opdict.find(key) != _opdict.end();
            }
            // no. of operators for certain type
            size_t num_ops(const char key) const{
               if(!this->ifexist(key)){
                  return 0;
               }else{
                  return _opdict.at(key).size();
               }
            }
            size_t size_ops(const char key) const{
               if(!this->ifexist(key)){
                  return 0;
               }else{
                  return _sizes.at(key);
               }
            }
            Tm* ptr_ops(const char key) const{
               if(this->num_ops(key) == 0){
                  return nullptr;
               }else{
                  return _data + _opoff.at(key);
               }
            }
#ifdef GPU
            Tm* ptr_ops_gpu(const char key) const{
               if(this->num_ops(key) == 0){
                  return nullptr;
               }else{
                  return _dev_data + _opoff.at(key);
               }
            }
#endif
            // return qindexmap for certain type of op
            const qindexmap& indexmap(const char key) const{
               assert(this->num_ops(key) > 0);
               return _indexmap.at(key);
            }
            // helpers
            size_t size() const{ return _size; };
            size_t opsize() const{ return _opsize; };
            void print(const std::string name, const int level=0) const;
         public:
            int sorb = 0;
            int isym = 0;
            bool ifkr = false;
            qbond qbra, qket;
            std::vector<int> cindex;
            std::vector<int> krest;
            std::string oplist;
            int mpisize = 1;
            int mpirank = 0;
            bool ifdist2 = false; // whether distribute two-index object
            bool ifdists = false; // whether distribute opS
            bool ifhermi = true; // whether to use hermicity to reduce no. of B/N
            //private:
            std::map<std::pair<char,int>,size_t> _offset;
            std::map<char,qoper_map<ifab,Tm>> _opdict;
            std::map<char,qindexmap> _indexmap; 
            std::map<char,size_t> _sizes;
            std::map<char,size_t> _opoff;
            size_t _size = 0, _opsize = 0;
            Tm* _data = nullptr;
            Tm* _dev_data = nullptr;
      };
   template <typename Tm>
      using oper_dict = qoper_dict<true,Tm>;
   template <typename Tm>
      using opersu2_dict = qoper_dict<false,Tm>;

   template <bool ifab, typename Tm>
      using qoper_dictmap = std::map<std::string,const qoper_dict<ifab,Tm>&>; // for sigma2
   template <typename Tm>
      using oper_dictmap = qoper_dictmap<true,Tm>;
   template <typename Tm>
      using opersu2_dictmap = qoper_dictmap<false,Tm>;

   // helpers
   template <bool ifab, typename Tm>
      void qoper_dict<ifab,Tm>::print(const std::string name, const int level) const{
         std::cout << " " << name << ": oplist=" << oplist;
         // count no. of operators in each class
         std::string opseq = "ICSHABPQTFDMN";
         std::map<char,int> exist;
         std::string s = " nops=";
         for(const auto& key : opseq){
            if(_opdict.find(key) != _opdict.end()){ 
               s += key;
               s += ":"+std::to_string(_opdict.at(key).size())+ " ";
               exist[key] = _opdict.at(key).size(); // size of dictionary
            }else{
               exist[key] = 0;
            }
         }
         std::cout << s << std::endl;
         // memory information
         std::cout << "        dim(bra,ket)=" << qbra.get_dimAll() << "," << qket.get_dimAll() 
            << " size(op)=" << _opsize << ":" << tools::sizeMB<Tm>(_opsize) << "MB" 
            << " size(tot)=" << _size 
            << ":" << tools::sizeMB<Tm>(_size) << "MB" 
            << ":" << tools::sizeGB<Tm>(_size) << "GB" 
            << std::endl;
         // display the sizes of each class of operators
         std::cout << "        per(mem): " << std::defaultfloat << std::setprecision(3);
         for(const auto& key : opseq){
            if(exist[key] == 0) continue;
            std::cout << key << ":" << double(_sizes.at(key))/_size*100 << " ";
         }
         std::cout << std::endl;
         // print each operator
         if(level > 0){
            std::cout << " sorb=" << sorb << " isym=" << isym << " ifkr=" << ifkr 
               << " size[cindex]=" << cindex.size()
               << " size[krest]=" << krest.size() << std::endl;
            tools::print_vector(cindex, "cindex"); 
            tools::print_vector(krest, "krest"); 
            qbra.print("qbra");
            qket.print("qket");
            if(level > 1){
               for(const auto& key : opseq){
                  if(exist[key] == 0) continue;
                  std::cout << " list of op" << key << ": ";
                  auto op_index = this->oper_index_op(key);
                  for(int idx : op_index){
                     if(key == 'H' || key == 'I' || key == 'C' || key == 'D' || key == 'S' || key == 'T'){
                        std::cout << idx << " ";
                     }else{
                        auto pq = oper_unpack(idx);
                        std::cout << idx << ":(" << pq.first << "," << pq.second << ") ";
                     }
                  }
                  std::cout << std::endl;
                  // print operator matrix
                  if(level > 2){
                     for(int idx : op_index){
                        const auto& op = _opdict.at(key).at(idx);
                        std::string opname(1,key);
                        opname += "("+std::to_string(idx)+")";
                        op.to_matrix().print("op"+opname);
                     }
                  } // level>2
               } // key
            } // level>1	 
         } // level>0
      }

   // array of indices for operator with key
   template <bool ifab, typename Tm>
      std::vector<int> qoper_dict<ifab,Tm>::oper_index_op(const char key) const{
         std::vector<int> index;
         if(key == 'C' || key == 'D'){
            index = cindex;
         }else if(key == 'H' || key == 'I'){
            index.push_back(0);
         }else if(key == 'S'){
            index = oper_index_opS(krest, ifkr);
            // distribute opS
            if(ifdists && mpisize > 1){
               index = distribute1vec(ifkr, mpisize, index, mpirank);
            }
         }else if(key == 'A' || key == 'B' || key == 'P' || key == 'Q' || key == 'M' || key == 'N'){
            if(key == 'A' || key == 'M'){
               index = oper_index_opA(cindex, ifkr, isym);
            }else if(key == 'B' || key == 'N'){
               index = oper_index_opB(cindex, ifkr, isym, ifhermi);
            }else if(key == 'P'){
               index = oper_index_opP(krest, ifkr, isym);
            }else if(key == 'Q'){
               index = oper_index_opQ(krest, ifkr, isym);
            }
            // distribute two index operators
            if(ifdist2 && mpisize > 1){
               index = distribute2vec(key, ifkr, mpisize, index, mpirank, sorb); 
            }
         // ZL@20240905: just for dot operators
         }else if(key == 'F'){
            index = {cindex[0]};
         }else if(key == 'T'){
            index = cindex;
         }
         return index;
      }

   // get qsym of operator
   inline qsym get_qsym_op(const char key, const short isym, const int idx){
      qsym sym_op;
      if(key == 'C'){
         sym_op = get_qsym_opC(isym, idx);
      }else if(key == 'D'){
         sym_op = get_qsym_opD(isym, idx);
      }else if(key == 'A'){
         auto pr = oper_unpack(idx);
         sym_op = get_qsym_opA(isym, pr.first, pr.second);
      }else if(key == 'B'){
         auto pr = oper_unpack(idx);
         sym_op = get_qsym_opB(isym, pr.first, pr.second);
      }else if(key == 'M'){
         auto pr = oper_unpack(idx);
         sym_op = get_qsym_opM(isym, pr.first, pr.second);
      }else if(key == 'N'){
         auto pr = oper_unpack(idx);
         sym_op = get_qsym_opN(isym, pr.first, pr.second);
      }else if(key == 'H' || key == 'I'){
         sym_op = qsym(isym,0,0);	   
      }else if(key == 'S'){
         sym_op = get_qsym_opS(isym, idx);
      }else if(key == 'P'){
         auto pr = oper_unpack(idx);
         sym_op = get_qsym_opP(isym, pr.first, pr.second);
      }else if(key == 'Q'){
         auto pr = oper_unpack(idx);
         sym_op = get_qsym_opQ(isym, pr.first, pr.second);
         //
         // ZL@20240905: just for dot operators
         //
      }else if(key == 'F'){
         sym_op = qsym(isym,0,0);
      }else if(key == 'T'){
         sym_op = get_qsym_opT(isym, idx);
      }
      return sym_op;
   } 
   template <bool ifab, typename Tm>
      qsym qoper_dict<ifab,Tm>::get_qsym_op(const char key, const int idx) const{
         return ctns::get_qsym_op(key, isym, idx);
      }

   // initialize _opdict, _size, _opsize
   template <bool ifab, typename Tm>
      void qoper_dict<ifab,Tm>::setup_opdict(const bool debug){
         if(debug){
            std::cout << "ctns::qoper_dict::setup_opdict oplist=" 
               << oplist << std::endl;
         }
         // count the size
         _size = 0;
         // loop over different types of operators
         for(const auto& key : oplist){
            _sizes[key] = 0;
            _opdict[key] = qoper_map<ifab,Tm>(); // initialize an empty dictionary
            if(debug) std::cout << " allocate_memory for op" << key << ":";
            // loop over indices
            auto op_index = this->oper_index_op(key);
            if(op_index.size() > 0) _opoff[key] = _size;
            for(int idx : op_index){
               auto sym_op = this->get_qsym_op(key,idx);
               // only compute size 
               _opdict[key][idx].init(sym_op, qbra, qket, {1,0}, false);
               _indexmap[key][sym_op].push_back(idx);
               size_t sz = _opdict[key][idx].size();
               _offset[std::make_pair(key,idx)] = _size;
               _size += sz;
               _sizes[key] += sz;
               _opsize = std::max(_opsize, sz);
            }
            if(debug){
               std::cout << " nop=" << op_index.size()
                  << " size=" << _sizes[key] 
                  << " sizeMB=" << tools::sizeMB<Tm>(_sizes[key]) 
                  << std::endl;
            }
         }
         if(debug){
            std::cout << "total size=" << _size 
               << " sizeMB=" << tools::sizeMB<Tm>(_size) 
               << std::endl;
         }
      }

   // setup the mapping to physical address for each operator
   template <bool ifab, typename Tm>
      void qoper_dict<ifab,Tm>::setup_data(){
         for(const auto& key : oplist){
            auto op_index = this->oper_index_op(key); // use the same order in setup_opdict for storage 
            for(int idx : op_index){
               auto& op = _opdict[key][idx];
               size_t off = _offset[std::make_pair(key,idx)];
               assert(op.own == false);
               op.setup_data( _data+off );
            }
         }
      }

} // ctns

#endif
