#ifndef OPER_DICT_H
#define OPER_DICT_H

#include <map>
#include "../core/serialization.h"
#include "../core/integral.h"
#include "qtensor/qtensor.h"
#include "oper_index.h"

namespace ctns{

// --- oper_dict: container for operators --- 

template <typename Tm>
using oper_map = std::map<int,stensor2<Tm>>;

template <typename Tm>
struct oper_dict{
   private:
      // serialize
      friend class boost::serialization::access;	   
      template <class Archive>
      void save(Archive & ar, const unsigned int version) const{
	 ar & isym & ifkr & qbra & qket 
	    & cindex & krest & oplist 
	    & mpisize & mpirank & ifdist2;
	 /*
         for(int i=0; i<_size; i++){
	    ar & _data[i];
	 }
	 */
      }
      template <class Archive>
      void load(Archive & ar, const unsigned int version){
	 ar & isym & ifkr & qbra & qket 
            & cindex & krest & oplist
	    & mpisize & mpirank & ifdist2;
	 //
	 // IO of data will be handled in oper_io.h
	 //
	 /*
	 this->_setup_opdict();
	 _data = new Tm[_size];
         for(int i=0; i<_size; i++){
	    ar & _data[i];
	 }
	 this->_setup_data(_data);
	 */
      }
      BOOST_SERIALIZATION_SPLIT_MEMBER()
   public:
      // initialize _opdict, _size, _opsize
      void _setup_opdict(const bool debug=false);
      // setup the mapping to physical address
      void _setup_data(Tm* data){
         size_t off = 0;
         for(const auto& key : oplist){
            for(auto& pr : _opdict[key]){
	       auto& op = pr.second;
	       assert(op.own == false);
	       op.setup_data(_data+off);
	       off += op.size();
            }
         }
	 assert(off == _size);
      }
      // constructor
      oper_dict(){}
      ~oper_dict(){ delete[] _data; }
      // copy
      oper_dict(const oper_dict& op_dict) = delete;
      oper_dict& operator =(const oper_dict& op_dict) = delete;
      // move
      oper_dict(oper_dict&& op_dict) = delete;
      oper_dict& operator =(oper_dict&& op_dict) = delete;
      // stored operators
      std::vector<int> oper_index_op(const char key) const;
      // symmetry of op
      qsym get_qsym_op(const char key, const int idx) const;
      // allocate memory
      void allocate_memory(const bool debug=false){
	 this->_setup_opdict(debug);
         _data = new Tm[_size];
         memset(_data, 0, _size*sizeof(Tm));
         this->_setup_data(_data); // assign pointer for each operator
      }
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
      std::map<char,oper_map<Tm>> _opdict;
      size_t _size = 0, _opsize = 0;
      Tm* _data = nullptr;
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
      std::cout << " isym=" << isym << " ifkr=" << ifkr 
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
            if(key == 'H' || key == 'C' || key == 'S'){
               for(const auto& op : _opdict.at(key)){
                  std::cout << "(" << op.first << ") ";
               }
            }else{
               for(const auto& op : _opdict.at(key)){
                  auto pq = oper_unpack(op.first);
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
      if(ifdist2 && mpisize > 1){ 
         for(int idx : index2){
            int iproc = distribute2(idx, mpisize);
            if(iproc == mpirank) index.push_back(idx);
         }
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

} // ctns

#endif
