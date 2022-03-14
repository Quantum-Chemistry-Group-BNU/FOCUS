#ifndef SYMBOLIC_OPER_H
#define SYMBOLIC_OPER_H

#include "oper_dict.h"
#include <numeric>      // std::iota
#include <algorithm>    // std::stable_sort

namespace ctns{

/*
 symbolic operator: op(block,label,index,dagger,parity) 
   block=l,c1,c2,r
   label=H,A,P,B,Q,C,S
   index=integer
*/
struct symbolic_oper{
   public:
      // constructor
      symbolic_oper(){}
      symbolic_oper(const std::string _block, 
		    const char _label, 
		    const int _index, 
		    const bool _dagger=false,
		    const int _nbar=0){
	 block = _block;
         label = _label;
	 index = _index;
	 dagger = _dagger;
	 nbar = _nbar;
	 if(label == 'C' || label == 'S'){
	    parity = true;
	 }else{
	    parity = false;
	 }
      }
      // string format
      std::string to_string() const{
         std::string str = "";
         str += "op(block=" + block + ",";
	 str += " label=" + std::to_string(label) + ",";
	 str += " index=" + std::to_string(index) + ",";
	 str += " dagger=" + std::to_string(dagger) + ",";
	 str += " parity=" + std::to_string(parity) + ")";
         return str;
      }
      // print
      friend std::ostream& operator <<(std::ostream& os, const symbolic_oper& op){
         os << op.symbol();
	 if(op.label == 'H' || op.label == 'C' || op.label == 'S'){
            os << "(" << op.index << ")";
	 }else{
            auto pr = oper_unpack(op.index);
	    os << "(" << pr.first << "," << pr.second << ")";
	 }
	 if(op.nbar > 0){
	    os << ".K(" << op.nbar << ")";
	 }
         return os;
      }
      // operations
      symbolic_oper H() const{
	 return symbolic_oper(block,label,index,!dagger,nbar);
      }
      symbolic_oper K(const int nbar) const{
	 return symbolic_oper(block,label,index,dagger,nbar);
      }
      // qsym
      qsym get_qsym(const short isym) const{
         auto sym = get_qsym_op(label, isym, index);
	 return (dagger? -sym : sym);
      }
      // symbol
      std::string symbol() const{
         std::string lb(1,label);
         lb = dagger? lb+"d" : lb;
	 return  lb+"["+block+"]";
      }
   public:
      std::string block;
      char label;
      int index;
      bool dagger = false;
      bool parity = false;
      int nbar = 0; // for kramers operations
};

// sum of weighted symbolic operators: wa*opa + wb*opb + ...
template <typename Tm>
struct symbolic_sum{
   public:
      // constructor
      symbolic_sum(){}
      symbolic_sum(const symbolic_oper& sop){
         sums.push_back(std::make_pair(1.0,sop));
      }
      symbolic_sum(const symbolic_oper& sop, const Tm& wt){
         sums.push_back(std::make_pair(wt,sop));
      }
      // sum a term
      void sum(const Tm& wt, const symbolic_oper& sop){
         sums.push_back(std::make_pair(wt,sop));
      }
      // print
      friend std::ostream& operator <<(std::ostream& os, const symbolic_sum& ops){
	 int n = ops.sums.size();
	 if(n > 0){
	    os << "{" << ops.sums[0].first << "*" << ops.sums[0].second;
	    for(int i=1; i<n; i++){
               os << " + " << ops.sums[i].first << "*" << ops.sums[i].second;
	    }
	    os << "}";
	 }
         return os;
      }
      // scale by a factor
      void scale(const double fac){ 
	 for(auto& pr : sums){
	    pr.first *= fac; 	 
	 }
      }
      // operations
      symbolic_sum H() const{
	 symbolic_sum<Tm> sH;
	 int n = sums.size();
	 sH.sums.resize(n);
	 for(int i=0; i<n; i++){
	    auto wt = tools::conjugate(sums[i].first);
	    auto op = sums[i].second.H();
            sH.sums[i] = std::make_pair(wt,op);
	 }
	 return sH;
      }
      // helper
      int size() const{ return sums.size(); } 
      // cost for op*|wf>
      double cost(std::map<std::string,int>& dims) const{
	 double t = 1.e0; 
         for(const auto& pr : dims){
	    t *= pr.second;
	 }
	 std::string block = sums[0].second.block;
	 int d = dims.at(block);
	 int len = this->size();
	 return d*t + len*d*d;
      }
   public:
      std::vector<std::pair<Tm,symbolic_oper>> sums;
};

// product of operators: o[l]*o[c1], o[l]*o[c1]*o[r]
template <typename Tm>
struct symbolic_prod{
   public:
      // constructor
      symbolic_prod(){}
      symbolic_prod(const symbolic_oper& op1, const double _wt=1.0){
         terms.push_back(symbolic_sum<Tm>(op1,_wt));
      }
      symbolic_prod(const symbolic_oper& op1, 
		    const symbolic_oper& op2,
		    const double _wt=1.0){
         terms.push_back(symbolic_sum<Tm>(op1,_wt));
         terms.push_back(symbolic_sum<Tm>(op2));
      }
      symbolic_prod(const symbolic_oper& op1, 
		    const symbolic_oper& op2,
		    const symbolic_oper& op3,
		    const double _wt=1.0){
         terms.push_back(symbolic_sum<Tm>(op1,_wt));
         terms.push_back(symbolic_sum<Tm>(op2));
         terms.push_back(symbolic_sum<Tm>(op3));
      }
      // o[l]o[c1]o[c2]o[r]
      symbolic_prod(const symbolic_oper& op1, 
		    const symbolic_oper& op2,
		    const symbolic_oper& op3,
		    const symbolic_oper& op4,
		    const double _wt=1.0){
         terms.push_back(symbolic_sum<Tm>(op1,_wt));
         terms.push_back(symbolic_sum<Tm>(op2));
         terms.push_back(symbolic_sum<Tm>(op3));
         terms.push_back(symbolic_sum<Tm>(op4));
      }
      // oper & sum
      symbolic_prod(const symbolic_oper& op1,
		    const symbolic_sum<Tm>& ops2){
         terms.push_back(symbolic_sum<Tm>(op1));
	 terms.push_back(ops2);
      }
      symbolic_prod(const symbolic_sum<Tm>& ops1,
		    const symbolic_oper& op2){
         terms.push_back(ops1);
	 terms.push_back(symbolic_sum<Tm>(op2));
      }
      // print
      friend std::ostream& operator <<(std::ostream& os, const symbolic_prod& ops){
         int n = ops.terms.size();
	 if(n > 0){
	    os << ops.terms[0];
	    for(int i=1; i<n; i++){
	       os << " * " << ops.terms[i];
	    }
	 }
         return os;
      }
      // scale by a factor
      void scale(const double fac){ 
	 if(terms.size() > 0) terms[0].scale(fac);
      }
      // (o0o1o2)^H = (-1)^{p0*p1+p0*p2+p1*p2}*o0Ho1Ho2H 
      double Hsign() const{
         int n = terms.size();
	 bool parity = false;
	 for(int i=0; i<n; i++){
	    const auto& op1 = terms[i].sums[0].second;
	    for(int j=0; j<i; j++){
	       const auto& op2 = terms[j].sums[0].second; 
	       parity ^= (op1.parity & op2.parity); 
	    }
	 }
	 double sgn = parity? -1.0 : 1.0;
	 return sgn;
      }
      // operations
      symbolic_prod H() const{
	 symbolic_prod<Tm> tH;
         int n = terms.size();
	 tH.terms.resize(n);
	 for(int i=0; i<n; i++){
	    tH.terms[i] = terms[i].H();
	 }
	 tH.scale(tH.Hsign());
	 return tH;
      }
      // t1*t2 
      symbolic_prod product(const symbolic_prod& t) const{
         symbolic_prod t12;
	 t12.terms = terms;
         std::copy(t.terms.begin(), t.terms.end(), std::back_inserter(t12.terms));
	 return t12;
      }
      // helper
      int size() const{ return terms.size(); }
      // symbol
      std::string symbol() const{
	 std::string lbl;
	 for(int i=0; i<terms.size(); i++){
	    lbl += terms[i].sums[0].second.symbol();
	 }
	 return lbl;
      }
      // cost for op1*op2*...*|wf>
      double cost(std::map<std::string,int>& dims) const{
         double t = 0.e0;
	 for(int i=0; i<terms.size(); i++){
	    t += terms[i].cost(dims);
	 }
 	 return t;
      }
   public:
      std::vector<symbolic_sum<Tm>> terms;
};

// list of terms to be computed distributedly
template <typename Tm>
struct symbolic_task{
   public:
      // constructor
      symbolic_task(){}
      symbolic_task(const symbolic_prod<Tm>& t){
         tasks.push_back(t);
      }
      // append a term
      void append(const symbolic_prod<Tm>& t){
         tasks.push_back(t);
      }
      // join a task	   
      void join(const symbolic_task& st){
         std::copy(st.tasks.begin(), st.tasks.end(), std::back_inserter(tasks));
      }
      // scale by a factor
      void scale(const double fac){ 
         for(auto& task : tasks){
            task.scale(fac);	 
         }
      }
      // outer_product of two tasks
      symbolic_task outer_product(const symbolic_task& st) const{
	 symbolic_task st_new;
         for(int i=0; i<tasks.size(); i++){
            for(int j=0; j<st.tasks.size(); j++){
  	       st_new.tasks.push_back( tasks[i].product(st.tasks[j]) );
	    }
	 }
	 return st_new;
      }
      // display
      void display(const std::string& name, const int level=1) const{
         std::cout << " formulae " << name << " : size=" << tasks.size() << std::endl;
	 if(level > 0){
            for(int i=0; i<tasks.size(); i++){
	       std::cout << "  i=" << i << " " << tasks[i] << std::endl; 
	    }
	 }
      }
      // helper
      int size() const{ return tasks.size(); }
      // reorder
      void sort(std::map<std::string,int>& dims, const bool debug=false){
         std::stable_sort(tasks.begin(), tasks.end(),
        	          [&dims](const symbolic_prod<Tm>& t1,
			     	  const symbolic_prod<Tm>& t2){
			  	  double c1 = t1.cost(dims);
				  double c2 = t2.cost(dims);
			     	  return (c1>c2) || (c1==c2 && t1.symbol()>t2.symbol()); 
				  }
			 );
	 if(debug){
            for(int i=0; i<tasks.size(); i++){
	       std::cout << "i=" << i << " cost=" << tasks[i].cost(dims) 
	                 << " symbol=" << tasks[i].symbol()
	                 << " " << tasks[i] << std::endl;
	    }
	 }
      }
      // cost for (op1+op2+...+)|wf>
      double cost(std::map<std::string,int>& dims) const{
         double t = 0.e0;
	 for(int i=0; i<tasks.size(); i++){
	    t += tasks[i].cost(dims);
	 }
 	 return t;
      }
   public:
      std::vector<symbolic_prod<Tm>> tasks;
}; 

template <typename Tm>
using op_task = std::tuple<char,int,symbolic_task<Tm>>;

template <typename Tm>
struct renorm_tasks{
   public:
      // constructor	   
      renorm_tasks(){}
      // add a renorm operator
      void append(const op_task<Tm>& op){
         op_tasks.push_back(op);	      
      }
      // helper
      int size() const{ return op_tasks.size(); }
      // reorder
      void sort(std::map<std::string,int>& dims){
         for(auto& op : op_tasks){
	    auto& formulae = std::get<2>(op);
	    formulae.sort(dims);
	 }
      }
   public:
      std::vector<op_task<Tm>> op_tasks;
};

} // ctns

#endif
