#include "tns_qtensor.h"

using namespace std;
using namespace tns;

// --- rank-3 tensor ---
void qtensor3::print(const string msg, const int level){
   cout << "qtensor3: " << msg << endl;
   qsym_space_print(qspace0,"qspace0");
   qsym_space_print(qspace1,"qspace1");
   qsym_space_print(qspace,"qspace");
   if(level >= 1){
      cout << "qblocks: nblocks=" << qblocks.size() << endl;
      int nnz = 0;
      for(const auto& p : qblocks){
         auto& t = p.first;
         auto& m = p.second;
         auto sym0 = get<0>(t);
         auto sym1 = get<1>(t);
         auto sym = get<2>(t);
         if(m.size() > 0){
            nnz++;
            cout << "idx=" << nnz 
		 << " block[" << sym0 << "," << sym1 << "," << sym << "]"
                 << " size=" << m.size() 
                 << " rows,cols=(" << m[0].rows() << "," << m[0].cols() << ")" 
                 << endl; 
            if(level >= 2){
               for(int i=0; i<m.size(); i++){		 
                  m[i].print("mat"+to_string(i));
               }
            } // level=2
         }
      }
      cout << "total no. of nonzero blocks =" << nnz << endl;
   } // level=1
}
