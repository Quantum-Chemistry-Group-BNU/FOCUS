#include "ctns_qsym.h"

using namespace std;
using namespace ctns;

// --- qsym ---
string qsym::to_string() const{
   return "("+std::to_string(_ne)+","+std::to_string(_tm)+")";
}

ostream& ctns::operator <<(ostream& os, const qsym& sym){
   os << sym.to_string();
   return os;
}

qsym ctns::operator +(const qsym& sym1, const qsym& sym2){
   qsym sym;
   sym._ne = sym1._ne + sym2._ne;
   sym._tm = sym1._tm + sym2._tm;
   return sym;
}

qsym ctns::operator -(const qsym& sym1, const qsym& sym2){
   qsym sym;
   sym._ne = sym1._ne - sym2._ne;
   sym._tm = sym1._tm - sym2._tm;
   return sym;
}

/*
// direct product space V1*V2->V12
pair<qsym_space,qsym_dpt> ctns::qsym_space_dpt(const qsym_space& qs1, 
		         		      const qsym_space& qs2){
   qsym_space qs12;
   qsym_dpt dpt;
   // init
   for(const auto& p1 : qs1){
      auto q1 = p1.first;
      for(const auto& p2 : qs2){
	 auto q2 = p2.first;
	 qs12[q1+q2] = 0;
	 dpt[q1+q2][make_pair(q1,q2)] = make_tuple(0,0,0); // just init
      }
   }
   // form qs12 and dpt
   for(const auto& p : dpt){
      int ioff = 0;
      for(const auto& p12 : p.second){
	 auto q12 = p12.first;
	 int d1 = qs1.at(q12.first);
	 int d2 = qs2.at(q12.second);
	 qs12[p.first] += d1*d2;
	 dpt[p.first][p12.first] = make_tuple(d1,d2,ioff); // save d1,d2,offset
	 ioff += d1*d2;
      }
   }
   return make_pair(qs12,dpt);
}
*/
