#include "tns_qsym.h"

using namespace std;
using namespace fock;
using namespace tns;

// --- qsym ---
string qsym::to_string() const{
   return "("+std::to_string(_ne)+","+std::to_string(_na)+")";
}

ostream& tns::operator <<(ostream& os, const qsym& sym){
   os << sym.to_string();
   return os;
}

qsym tns::operator +(const qsym& sym1, const qsym& sym2){
   qsym sym;
   sym._ne = sym1._ne + sym2._ne;
   sym._na = sym1._na + sym2._na;
   return sym;
}

qsym tns::operator -(const qsym& sym1, const qsym& sym2){
   qsym sym;
   sym._ne = sym1._ne - sym2._ne;
   sym._na = sym1._na - sym2._na;
   return sym;
}

// --- qspace ---
// total dimension
int tns::qsym_space_dim(const qsym_space& qs){
   int dim = 0;
   for(const auto& p : qs) dim += p.second;
   return dim;
}

// print
void tns::qsym_space_print(const qsym_space& qs, const string& name){ 
   cout << name 
	<< " nsym=" << qs.size() 
	<< " dim=" << qsym_space_dim(qs) 
	<< endl;
   // loop over symmetry sectors
   for(const auto& p : qs){
      auto sym = p.first;
      auto dim = p.second;
      cout << sym << ":" << dim << " ";
   }
   cout << endl;
}

// --- physical degree of freedoms  ---
// symmetry
const vector<qsym> tns::phys_sym({qsym(0,0), 
		       	          qsym(1,0), 
		      	          qsym(1,1), 
		      	          qsym(2,1)});
// states
const onspace tns::phys_space({onstate("00"),   // 0
			       onstate("10"),   // b
	  		       onstate("01"),   // a
			       onstate("11")}); // 2		  
// qsym_space
const qsym_space tns::phys_qsym_space({{qsym(0,0),1},
			               {qsym(1,0),1},
			               {qsym(1,1),1},
			               {qsym(2,1),1}});

// --- vacuum ---
const qsym_space tns::vac_qsym_space({{qsym(0,0),1}});

