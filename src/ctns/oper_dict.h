#ifndef OPER_DICT_H
#define OPER_DICT_H

#include <map>
#include "../core/serialization.h"
#include "../core/integral.h"
#include "qtensor/qtensor.h"
#include "oper_index.h"
#include "ctns_comb.h"

namespace ctns{

// --- oper_dict: container for operators --- 

template <typename Tm>
using oper_map = std::map<int,stensor2<Tm>>;

template <typename Tm>
class oper_dict{
   private:
      // serialize
      friend class boost::serialization::access;	   
      template <class Archive>
      void save(Archive & ar, const unsigned int version) const{
	 ar & isym & ifkr & qbra & qket 
            & cindex & krest & ops & _size;
         for(int i=0; i<_size; i++){
	    ar & _data[i];
	 }
      }
      template <class Archive>
      void load(Archive & ar, const unsigned int version){
	 ar & isym & ifkr & qbra & qket 
            & cindex & krest & ops & _size;
	 _data = new Tm[_size];
         for(int i=0; i<_size; i++){
	    ar & _data[i];
	 }
	 // setup the mapping to physical address
	 for(auto& pr_opmap : ops){
            for(auto& pr_op : pr_opmap.second){
	       auto& op = pr_op.second;
	       op.setup_data(_data);
	    }
	 }
      }
      BOOST_SERIALIZATION_SPLIT_MEMBER()
   public:
      // constructor
      oper_dict(){
	 _size = 0;
	 _data = nullptr;
      }
      ~oper_dict(){
	 delete[] _data;
      }
      // access
      const oper_map<Tm>& operator()(const char key) const{
         return ops.at(key);
      }
      oper_map<Tm>& operator()(const char key){
         return ops[key];      
      }
      // helpers
      void print(const std::string name, const int level=0) const;
      // construction of operators
      template <typename Km>
      void init_dot(const comb<Km>& icomb, const int kp,
		    const integral::two_body<Tm>& int2e,
		    const integral::one_body<Tm>& int1e);
      // precompute memory required
      std::vector<int> oper_index_op(const char key) const;
      qsym get_qsym_op(const char key, const int idx) const;
      void allocate_memory();
   public:
      int isym;
      bool ifkr;
      qbond qbra, qket;
      std::vector<int> cindex;
      std::vector<int> krest;
      std::string oplist;
   private:
      std::map<char,oper_map<Tm>> ops;
      size_t _size;
      Tm* _data;
};

template <typename Tm>
void oper_dict<Tm>::print(const std::string name, const int level) const{
   std::cout << name << ": oplist=" << oplist 
	     << " isym=" << isym << " ifkr=" << ifkr 
             << " size[cindex]=" << cindex.size()
	     << " size[krest]=" << krest.size() << std::endl; 
   qbra.print("qbra");
   qket.print("qket");
   // count no. of operators in each class
   std::string opseq = "CABHSPQ";
   std::map<char,int> exist;
   std::string s = " nops=";
   for(const auto& key : opseq){
      if(ops.find(key) != ops.end()){ 
         s += key;
         s += ":"+std::to_string(ops.at(key).size())+ " ";
         exist[key] = ops.at(key).size(); // size of dictionary
      }else{
         exist[key] = 0;
      }
   }
   std::cout << s << std::endl;
   // print each operator
   if(level > 0){
      for(const auto& key : opseq){
         if(exist[key] == 0) continue;
         std::cout << " " << key << ": ";
         if(key == 'H' || key == 'C' || key == 'S'){
            for(const auto& op : ops.at(key)){
               std::cout << "(" << op.first << ") ";
            }
         }else{
            for(const auto& op : ops.at(key)){
               auto pq = oper_unpack(op.first);
               std::cout << "(" << pq.first << "," << pq.second << ") ";
            }
         }
	 std::cout << std::endl;
      }
   } // level
   std::cout << "total size=" << _size << " sizeMB=" << tools::sizeMB<Tm>(_size) << std::endl; 
}

template <typename Tm>
std::vector<int> oper_dict<Tm>::oper_index_op(const char key) const{
   std::vector<int> index;
   if(key == 'C'){
      index = cindex;
   }else if(key == 'A'){
      index = oper_index_opA(cindex, ifkr);
   }else if(key == 'B'){
      index = oper_index_opB(cindex, ifkr);
   }else if(key == 'H'){
      index.push_back(0);
   }else if(key == 'S'){
      index = oper_index_opS(krest, ifkr);
   }else if(key == 'P'){
      index = oper_index_opP(krest, ifkr);
   }else if(key == 'Q'){
      index = oper_index_opQ(krest, ifkr);
   }
   return index;
}
      
template <typename Tm>
qsym oper_dict<Tm>::get_qsym_op(const char key, const int idx) const{
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
void oper_dict<Tm>::allocate_memory(){
   const bool debug = true;
   std::cout << "ctns::oper_dict<Tm>:allocate_memory for oplist=" << oplist << std::endl;
   // count the size
   _size = 0;
   std::map<char,size_t> sizes;
   for(const auto& key : oplist){
      sizes[key] = 0; 
      if(debug) std::cout << " allocate_memory for op" << key << ":";
      auto op_index = this->oper_index_op(key);
      for(int idx : op_index){
	 auto sym_op = this->get_qsym_op(key,idx);
	 ops[key][idx].init(sym_op, qbra, qket, {1,0}, false);
	 sizes[key] += ops[key][idx].size();
      }
      _size += sizes[key];
      std::cout << " size=" << sizes[key] 
	        << " sizeMB=" << tools::sizeMB<Tm>(sizes[key]) 
		<< std::endl; 
   }
   std::cout << "total size=" << _size << " sizeMB=" << tools::sizeMB<Tm>(_size) << std::endl;
   // allocate & assign pointer
   _data = new Tm[_size];
   size_t off = 0;
   for(const auto& key : oplist){
      for(auto& pr : ops.at(key)){
	 auto& op = pr.second;
	 op.setup_data(_data+off);
	 off += op.size();
      }
   }
   // check
   this->print("qops",1);
}

// init local operators on dot
// Member templates: https://en.cppreference.com/w/cpp/language/member_template
template <typename Tm> template<typename Km>
void oper_dict<Tm>::init_dot(const comb<Km>& icomb, const int kp,
                             const integral::two_body<Tm>& int2e,
                             const integral::one_body<Tm>& int1e){
   std::cout << "init_dot" << std::endl;
   // setup basic information
   isym = Km::isym;
   ifkr = qkind::is_kramers<Km>();
   cindex.push_back(2*kp);
   if(not ifkr) cindex.push_back(2*kp+1);
   // rest of spatial orbital indices
   for(int k=0; k<int1e.sorb/2; k++){
      if(k == kp) continue;
      krest.push_back(k);
   }
   auto qphys = get_qbond_phys(isym);
   qbra = qphys;
   qket = qphys;
   oplist = "CABHSPQ";
   // initialize memory
   this->allocate_memory();
}

} // ctns

#endif
