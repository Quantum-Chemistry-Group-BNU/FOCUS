#ifndef SYMBOLIC_TASK_H
#define SYMBOLIC_TASK_H

#include "symbolic_oper.h"

namespace ctns{

// H = \sum_k H_k where H_k is a product of operator
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

// symbolic tasks for renormalization
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

template <typename Tm>
struct bipart_task{
   public:
     // contructor
      bipart_task(){}
   public:
      symbolic_task<Tm> lop;
      symbolic_task<Tm> rop;
};

} // ctns

#endif
