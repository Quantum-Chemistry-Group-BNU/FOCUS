#include "../settings/global.h"
#include "tns_comb.h" 
#include "tns_qtensor.h"
#include "tns_oper.h"

using namespace std;
using namespace linalg;
using namespace tns;

void tns::oper_rbases(const comb& bra,
		      const comb& ket, 
		      const comb_coord& p,
	              const integral::two_body& int2e,
	              const integral::one_body& int1e,
		      const string scratch){
   cout << "\ntns::oper_rbases" << endl;
   int i = p.first, j = p.second;
   auto& rbasis0 = bra.rbases.at(p);
   auto& rbasis1 = ket.rbases.at(p); 

   for(const auto& rsec0 : rbasis0){
      rsec0.print("rsec",2);
   }
   exit(1);
}	
