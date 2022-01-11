#ifndef SYMBOLIC_OPER_H
#define SYMBOLIC_OPER_H

namespace ctns{

/*
 symbolic operator: op(block,label,index,dagger,parity) 
   block=l,c1,c2,r
   label=H,A,P,B,Q,a,S
   index=integer
*/
struct symbolic_oper{
   public:
      // constructor
      symbolic_oper(const std::string _block, 
		    const std::string _label, 
		    const int _index, 
		    const bool _dagger=false){
	 block = _block;
         label = _label;
	 index = _index;
	 dagger = _dagger;
	 if(label == "a" || label == "S"){
	    parity = 1;
	 }else{
	    parity = 0;
	 }
      }
      // string format
      std::string to_string() const{
         std::string str = "";
         str += "op(block=" + block + ",";
	 str += " label=" + label + ",";
	 str += " index=" + std::to_string(index) + ",";
	 str += " dagger=" + std::to_string(dagger) + ",";
	 str += " parity=" + std::to_string(parity) + ")";
         return str;
      }
      // print
      friend std::ostream& operator <<(std::ostream& os, const symbolic_oper& op){
         os << op.label;
	 if(op.dagger) os << "+";
	 os << "[" << op.block << "]";
	 if(op.label == "H" || op.label == "C" || op.label == "S"){
            os << "(" << op.index << ")";
	 }else{
            auto pr = oper_unpack(op.index);
	    os << "(" << pr.first << "," << pr.second << ")";
	 }
         return os;
      }
      // operations
      symbolic_oper H() const{
	 return symbolic_oper(block,label,index,!dagger);
      }
   public:
      std::string block, label;
      int index, parity;
      bool dagger;
};

/*
 product of operators: wt*o[l]*o[c1], wt*o[l]*o[c1]*o[r]
*/
struct symbolic_term{
   public:
      // constructor
      symbolic_term(){}
      symbolic_term(const symbolic_oper& op1, const double _wt=1.0){
         terms.push_back(op1);
	 wt = _wt;
      }
      symbolic_term(const symbolic_oper& op1, 
		    const symbolic_oper& op2,
		    const double _wt=1.0){
         terms.push_back(op1);
         terms.push_back(op2);
	 wt = _wt;
      }
      symbolic_term(const symbolic_oper& op1, 
		    const symbolic_oper& op2,
		    const symbolic_oper& op3,
		    const double _wt=1.0){
         terms.push_back(op1);
         terms.push_back(op2);
         terms.push_back(op3);
	 wt = _wt;
      }
      // o[l]o[c1]o[c2]o[r]
      symbolic_term(const symbolic_oper& op1, 
		    const symbolic_oper& op2,
		    const symbolic_oper& op3,
		    const symbolic_oper& op4,
		    const double _wt=1.0){
         terms.push_back(op1);
         terms.push_back(op2);
         terms.push_back(op3);
         terms.push_back(op4);
	 wt = _wt;
      }
      // print
      friend std::ostream& operator <<(std::ostream& os, const symbolic_term& ops){
	 os << ops.wt;
	 for(const auto& op : ops.terms){
            os << "*" << op;
	 }
         return os;
      }
      // scale by a factor
      void scale(const double fac){ wt *= fac; }
      // t1*t2 
      symbolic_term product(const symbolic_term& t) const{
         symbolic_term t12;
         t12.wt = wt*t.wt;
	 t12.terms = terms;
         std::copy(t.terms.begin(), t.terms.end(), std::back_inserter(t12.terms));
	 return t12;
      }
   public:
      std::vector<symbolic_oper> terms;	   
      double wt = 1.0;
};

/*
 list of terms to be computed distributedly
*/
struct symbolic_task{
   public:
      // append a term
      void push_back(const symbolic_term& t){
         tasks.push_back(t);
      }
      // joint a task	   
      void join(const symbolic_task& st){
         std::copy(st.tasks.begin(), st.tasks.end(), std::back_inserter(tasks));
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
      void display(const std::string& name, const int level=0) const{
         std::cout << "formulae " << name << " : size=" << tasks.size() << std::endl;
	 if(level > 0){
            for(int i=0; i<tasks.size(); i++){
	       std::cout << " i=" << i << " " << tasks[i] << std::endl; 
	    }
	 }
      }
   public:
      std::vector<symbolic_term> tasks;
}; 

} // ctns

#endif
