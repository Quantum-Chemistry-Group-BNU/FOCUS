#ifndef OPER_DICT_H
#define OPER_DICT_H

#include <map>
#include "../core/serialization.h"
#include "../core/integral.h"
#include "qtensor/qtensor.h"
#include "oper_index.h"
#ifdef GPU
#include "../gpu/gpu_env.h"
#endif

namespace ctns{

   // --- oper_dict: container for operators --- 

   template <typename Tm>
      using oper_map = std::map<int,stensor2<Tm>>; // index to operator

   template <typename Tm>
      struct oper_dict{
         private:
            // IO of data will be handled in oper_io.h
            friend class boost::serialization::access;	   
            template <class Archive>
               void serialize(Archive & ar, const unsigned int version){
                  ar & sorb & isym & ifkr & qbra & qket 
                     & cindex & krest & oplist 
                     & mpisize & mpirank & ifdist2;
               }
         public:
            // constructor
            oper_dict(){}
            ~oper_dict(){ 
               delete[] _data;
#ifdef GPU
               GPUmem.deallocate(_dev_data, _size*sizeof(Tm));
#endif
            }
            void clear(){
               delete[] _data;
               _data = nullptr;
            }
            bool avail_cpu() const{ return _data != nullptr; }
            bool avail_gpu() const{ return _dev_data != nullptr; }
#ifdef GPU
            void allocate_gpu(const bool ifmemset=false){
               _dev_data = (Tm*)GPUmem.allocate(_size*sizeof(Tm));
               if(ifmemset) GPUmem.memset(_dev_data, _size*sizeof(Tm));
            }
            void to_gpu(){
               assert(_dev_data != nullptr && _data != nullptr);
               GPUmem.to_gpu(_dev_data, _data, _size*sizeof(Tm));
            }
            void to_cpu(){
               assert(_dev_data != nullptr && _data != nullptr);
               GPUmem.to_cpu(_data, _dev_data, _size*sizeof(Tm));
            }
#endif
            // copy
            oper_dict(const oper_dict& op_dict) = delete;
            oper_dict& operator =(const oper_dict& op_dict) = delete;
            // move
            oper_dict(oper_dict&& op_dict) = delete;
            oper_dict& operator =(oper_dict&& op_dict) = delete;
            // initialize _opdict, _size, _opsize
            void _setup_opdict(const bool debug=false);
            // setup the mapping to physical address
            void _setup_data();
            // allocate memory
            void allocate(const bool ifmemset=false){
               this->_setup_opdict();
               _data = new Tm[_size];
               this->_setup_data(); // assign pointer for each operator
               if(ifmemset) memset(_data, 0, _size*sizeof(Tm));
            }
            // stored operators
            std::vector<int> oper_index_op(const char key) const;
            // symmetry of op
            qsym get_qsym_op(const char key, const int idx) const;
            // access
            const oper_map<Tm>& operator()(const char key) const{
               return _opdict.at(key);
            }
            oper_map<Tm>& operator()(const char key){
               return _opdict[key];      
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
            //private:
            std::map<std::pair<char,int>,size_t> _offset;
            std::map<char,oper_map<Tm>> _opdict;
            size_t _size = 0, _opsize = 0;
            Tm* _data = nullptr;
            Tm* _dev_data = nullptr;
      };
   template <typename Tm>
      using oper_dictmap = std::map<std::string,const oper_dict<Tm>&>; // for sigma2

   // helpers
   template <typename Tm>
      void oper_dict<Tm>::print(const std::string name, const int level) const{
         std::cout << " " << name << ": oplist=" << oplist;
         // count no. of operators in each class
         std::string opseq = "CABPQSH";
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
                  std::cout << " " << key << ": ";
                  auto op_index = this->oper_index_op(key);
                  for(int idx : op_index){
                     if(key == 'H' || key == 'C' || key == 'S'){
                        std::cout << "(" << idx << ") ";
                     }else{
                        auto pq = oper_unpack(idx);
                        std::cout << "(" << pq.first << "," << pq.second << ") ";
                     }
                  }
                  std::cout << std::endl;
               }
            } // level>1	 
         } // level>0
      }

   // array of indices for operator with key
   template <typename Tm>
      std::vector<int> oper_dict<Tm>::oper_index_op(const char key) const{
         std::vector<int> index;
         if(key == 'C'){
            index = cindex;
         }else if(key == 'H'){
            index.push_back(0);
         }else if(key == 'S'){
            index = oper_index_opS(krest, ifkr);
         }else if(key == 'A' || key == 'B' || key == 'P' || key == 'Q'){
            std::vector<int> index2;
            if(key == 'A'){
               index2 = oper_index_opA(cindex, ifkr);
            }else if(key == 'B'){
               index2 = oper_index_opB(cindex, ifkr);
            }else if(key == 'P'){
               index2 = oper_index_opP(krest, ifkr);
            }else if(key == 'Q'){
               index2 = oper_index_opQ(krest, ifkr);
            }
            // distribute two index operators
            if(ifdist2 && mpisize > 1){
               index = distribute2vec(key, ifkr, mpisize, index2, mpirank, sorb); 
            }else{
               index = std::move(index2);
            }
         }
         return index;
      }

   // get qsym of operator
   inline qsym get_qsym_op(const char key, const short isym, const int idx){
      qsym sym_op;
      if(key == 'C'){
         sym_op = get_qsym_opC(isym, idx);
      }else if(key == 'A'){
         auto pr = oper_unpack(idx);
         sym_op = get_qsym_opA(isym, pr.first, pr.second);
      }else if(key == 'B'){
         auto pr = oper_unpack(idx);
         sym_op = get_qsym_opB(isym, pr.first, pr.second);
      }else if(key == 'H'){
         sym_op = qsym(isym,0,0);	   
      }else if(key == 'S'){
         sym_op = get_qsym_opS(isym, idx);
      }else if(key == 'P'){
         auto pr = oper_unpack(idx);
         sym_op = get_qsym_opP(isym, pr.first, pr.second);
      }else if(key == 'Q'){
         auto pr = oper_unpack(idx);
         sym_op = get_qsym_opQ(isym, pr.first, pr.second);
      }
      return sym_op;
   } 
   template <typename Tm>
      qsym oper_dict<Tm>::get_qsym_op(const char key, const int idx) const{
         return ctns::get_qsym_op(key, isym, idx);
      }

   // initialize _opdict, _size, _opsize
   template <typename Tm>
      void oper_dict<Tm>::_setup_opdict(const bool debug){
         if(debug){
            std::cout << "ctns::oper_dict<Tm>:_setup_opdict oplist=" 
               << oplist << std::endl;
         }
         // count the size
         _size = 0;
         std::map<char,size_t> sizes;
         // loop over different types of operators
         for(const auto& key : oplist){
            sizes[key] = 0;
            _opdict[key] = oper_map<Tm>(); // initialize an empty dictionary
            if(debug) std::cout << " allocate_memory for op" << key << ":";
            // loop over indices
            auto op_index = this->oper_index_op(key);
            for(int idx : op_index){
               auto sym_op = this->get_qsym_op(key,idx);
               // only compute size 
               _opdict[key][idx].init(sym_op, qbra, qket, {1,0}, false);
               size_t sz = _opdict[key][idx].size();
               sizes[key] += sz;
               _opsize = std::max(_opsize, sz);
            }
            _size += sizes[key];
            if(debug){
               std::cout << " nop=" << op_index.size()
                  << " size=" << sizes[key] 
                  << " sizeMB=" << tools::sizeMB<Tm>(sizes[key]) 
                  << std::endl;
            }
         }
         if(debug){
            std::cout << "total size=" << _size 
               << " sizeMB=" << tools::sizeMB<Tm>(_size) 
               << std::endl;
         }
      }

   // setup the mapping to physical address
   template <typename Tm>
      void oper_dict<Tm>::_setup_data(){
         size_t off = 0;
         for(const auto& key : oplist){
            // use the same order in setup_opdict for storage 
            auto op_index = this->oper_index_op(key); 
            for(int idx : op_index){
               auto& op = _opdict[key][idx];
               assert(op.own == false);
               op.setup_data( _data+off );
               _offset[std::make_pair(key,idx)] = off; 
               off += op.size();
            }
         }
         assert(off == _size);
      }

} // ctns

#endif
