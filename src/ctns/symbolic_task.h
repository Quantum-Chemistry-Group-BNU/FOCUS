#ifndef SYMBOLIC_TASK_H
#define SYMBOLIC_TASK_H

#include "symbolic_oper.h"

namespace ctns{

   // High-level: symbolic_task, op_task/renorm_tasks, bipart_oper/bipart_task

   // H = \sum_k H_k where H_k is a product of operator
   template <typename Tm>
      struct symbolic_task{
         public:
            // constructor
            symbolic_task(){}
            symbolic_task(const symbolic_prod<Tm>& t){
               tasks.push_back(t);
               parity = t.parity;
            }
            // append a term
            void append(const symbolic_prod<Tm>& t){
               if(tasks.size() == 0) parity = t.parity;
               tasks.push_back(t);
            }
            // append a term
            void append(const symbolic_sum<Tm>& top1, 
                  const symbolic_oper& op2,
                  const bool ifdagger=false){
               if(top1.size() == 0) return;
               auto term = symbolic_prod<Tm>(top1,op2);
               if(ifdagger) term = term.H();
               this->append(term);
            }
            void append(const symbolic_oper& op1,
                  const symbolic_sum<Tm>& top2, 
                  const bool ifdagger=false){
               if(top2.size() == 0) return;
               auto term = symbolic_prod<Tm>(op1,top2);
               if(ifdagger) term = term.H();
               this->append(term);
            }
            // join a task	   
            void join(const symbolic_task& st){
               if(tasks.size() == 0) parity = st.parity;
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
               st_new.parity = parity^st.parity;
               return st_new;
            }
            // display
            void display(const std::string name, const int level=1) const{
               std::cout << " symbolic_task=" << name << " : size=" << tasks.size() << std::endl;
               if(level > 0){
                  for(int i=0; i<tasks.size(); i++){
                     std::cout << "  i=" << i
                        << "  symbol=" << tasks[i].symbol() 
                        << "  formula=" << tasks[i] 
                        << std::endl; 
                  }
               }
            }
            // helper
            int size() const{ return tasks.size(); }
            // reorder
            void sort(const std::map<std::string,int>& dims, const bool debug=false){
               std::stable_sort(tasks.begin(), tasks.end(),
                     [&dims](const symbolic_prod<Tm>& t1,
                        const symbolic_prod<Tm>& t2){
                     double c1 = t1.cost(dims);
                     double c2 = t2.cost(dims);
                     return (c1>c2) || (c1==c2 && t1.symbol()>t2.symbol()); 
                     }
                     );
               if(debug) this->display("reordered");
            }
            // cost for (op1+op2+...+)|wf>
            double cost(const std::map<std::string,int>& dims) const{
               double t = 0.e0;
               for(int i=0; i<tasks.size(); i++){
                  t += tasks[i].cost(dims);
               }
               return t;
            }
         public:
            bool parity = false;
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
            int sizetot() const{
               int sz = 0;
               for(int idx=0; idx<op_tasks.size(); idx++){
                  const auto& task = op_tasks[idx];
                  sz += std::get<2>(task).size();
               }
               return sz;
            }
            // reorder
            void sort(const std::map<std::string,int>& dims){
               // sort each operator
               for(auto& op : op_tasks){
                  auto& formulae = std::get<2>(op);
                  formulae.sort(dims);
               }
               // sort all operator
               std::stable_sort(op_tasks.begin(), op_tasks.end(),
                     [&dims](const op_task<Tm>& t1,
                        const op_task<Tm>& t2){
                     return std::get<2>(t1).cost(dims)>std::get<2>(t2).cost(dims);
                     }
                     );
            }
            // display
            void display(const std::string name, const int level=1) const{
               std::cout << " renorm_task=" << name << " : size=" << op_tasks.size() << std::endl;
               if(level > 0){
                  for(int idx=0; idx<op_tasks.size(); idx++){
                     const auto& task = op_tasks[idx];
                     const auto& label = std::get<0>(task);
                     const auto& index = std::get<1>(task);
                     const auto& stask = std::get<2>(task);
                     std::string opname(1,label);
                     opname += "[" + std::to_string(index) + "]";
                     std::cout << "idx=" << idx;
                     stask.display(opname, level);
                  }
               }
            }
         public:
            std::vector<op_task<Tm>> op_tasks;
      };

   template <typename Tm>
      struct bipart_oper{
         public:
            // contructor
            bipart_oper(const symbolic_task<Tm>& _lop,
                  const symbolic_task<Tm>& _rop,
                  const std::string _name=""){
               lop = _lop;
               rop = _rop;
               if(_name != "") name = _name;
               parity = lop.parity^rop.parity; 
            }
            bipart_oper(const char space, 
                  const symbolic_task<Tm>& _op,
                  const std::string _name=""){
               if(space == 'l') lop = _op;
               if(space == 'r') rop = _op;
               if(_name != "") name = _name;
               parity = lop.parity^rop.parity; 
            }
            // display
            void display(const int level=1) const{
               std::cout << " bipart_oper=" << name << " size(lop,rop)=" 
                  << lop.size() << "," << rop.size()
                  << std::endl;
               if(level > 0){
                  lop.display("lop", level);
                  rop.display("rop", level);
               }
            }
            // sort lop & rop
            void sort(const std::map<std::string,int>& dims){
               lop.sort(dims);
               rop.sort(dims);
            }
            // cost
            double cost(const std::map<std::string,int>& dims) const{
               return lop.cost(dims) + rop.cost(dims);
            }
            // (lop*rop)^H 
            double Hsign() const{
               return ((lop.parity & rop.parity)? -1.0 : 1.0);
            }
         public:
            bool parity;
            std::string name;
            symbolic_task<Tm> lop;
            symbolic_task<Tm> rop;
      };
   template <typename Tm>
      using bipart_task = std::vector<bipart_oper<Tm>>;

   template <typename Tm>
      void sort(bipart_task<Tm>& formulae, const std::map<std::string,int>& dims){
         // sort each operator
         for(auto& oper : formulae){
            oper.sort(dims);
         }
         // sort all operator
         std::stable_sort(formulae.begin(), formulae.end(),
               [&dims](const bipart_oper<Tm>& t1,
                  const bipart_oper<Tm>& t2){
               return t1.cost(dims)>t2.cost(dims);
               }
               );
      }

   template <typename Tm>
      void display(const bipart_task<Tm>& formulae, 
            const std::string name,
            const int level=0){
         std::cout << " bipart_task=" << name << " : size=" << formulae.size() << std::endl;
         for(int i=0; i<formulae.size(); i++){
            std::cout << " idx=" << i;
            formulae[i].display(level);
         }
      }

} // ctns

#endif
