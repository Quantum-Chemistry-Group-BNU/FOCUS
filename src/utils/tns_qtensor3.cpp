#include "tns_qtensor.h"
#include "../core/linalg.h"
#include <iostream>
#include <algorithm>

using namespace std;
using namespace linalg;
using namespace tns;

// --- rank-3 tensor ---
qtensor3::qtensor3(const qsym& sym1,
		   const qsym_space& qmid1,
		   const qsym_space& qrow1, 
		   const qsym_space& qcol1){
   sym = sym1;
   qmid = qmid1;
   qrow = qrow1;
   qcol = qcol1;
   for(const auto& pm : qmid){
      const auto& qm = pm.first;
      int mdim = pm.second; 
      for(const auto& pr : qrow){
         const auto& qr = pr.first;
         int rdim = pr.second;
         for(const auto& pc : qcol){
            const auto& qc = pc.first;
            int cdim = pc.second;
	    // initialization
	    auto key = make_tuple(qm,qr,qc);
            if(qm + qr == sym + qc){
	       vector<matrix> blk(mdim,matrix(rdim,cdim));
	       qblocks[key] = blk;
	    }else{
	       qblocks[key] = empty_block;
	    }
	 }
      }
   }
}
      
qtensor3 qtensor3::mid_signed(const double fac) const{
   qtensor3 qt3;
   qt3.sym = sym;
   qt3.qmid = qmid;
   qt3.qrow = qrow;
   qt3.qcol = qcol;
   qt3.qblocks = qblocks;
   for(const auto& pm : qmid){
      const auto& qm = pm.first;
      int mdim = pm.second; 
      double fac2 = qm.parity()*fac;
      for(const auto& pr : qrow){
         const auto& qr = pr.first;
         for(const auto& pc : qcol){
            const auto& qc = pc.first;
	    auto key = make_tuple(qm,qr,qc);
	    auto& blk = qt3.qblocks[key];
	    if(blk.size() > 0){ 
	       for(int m=0; m<mdim; m++){
	          blk[m] *= fac2;
	       }
	    }
	 } // c
      } // r
   } // m
   return qt3;
}

qtensor3 qtensor3::col_signed(const double fac) const{
   qtensor3 qt3;
   qt3.sym = sym;
   qt3.qmid = qmid;
   qt3.qrow = qrow;
   qt3.qcol = qcol;
   qt3.qblocks = qblocks;
   for(const auto& pm : qmid){
      const auto& qm = pm.first;
      int mdim = pm.second; 
      for(const auto& pr : qrow){
         const auto& qr = pr.first;
         for(const auto& pc : qcol){
            const auto& qc = pc.first;
            double fac2 = qc.parity()*fac;
	    auto key = make_tuple(qm,qr,qc);
	    auto& blk = qt3.qblocks[key];
	    if(blk.size() > 0){ 
	       for(int m=0; m<mdim; m++){
	          blk[m] *= fac2;
	       }
	    }
	 } // c
      } // r
   } // m
   return qt3;
}

void qtensor3::print(const string msg, const int level) const{
   cout << "qtensor3: " << msg << " sym=" << sym << endl;
   qsym_space_print(qmid,"qmid");
   qsym_space_print(qrow,"qrow");
   qsym_space_print(qcol,"qcol");
   if(level >= 1){
      cout << "qblocks: nblocks=" << qblocks.size() << endl;
      int nnz = 0;
      for(const auto& p : qblocks){
         auto& t = p.first;
         auto& m = p.second;
         auto sym_mid = get<0>(t);
         auto sym_row = get<1>(t);
         auto sym_col = get<2>(t);
         if(m.size() > 0){
            nnz++;
            cout << "idx=" << nnz 
		 << " block[" << sym_mid << "," << sym_row << "," << sym_col << "]"
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
      cout << "total no. of nonzero blocks=" << nnz << endl;
   } // level=1
}

double qtensor3::normF() const{
   double sum = 0.0;
   for(const auto& pm : qmid){
      const auto& qm = pm.first;
      for(const auto& pr : qrow){
	 const auto& qr = pr.first;
         for(const auto& pc : qcol){
	    const auto& qc = pc.first;
            auto key = make_tuple(qm,qr,qc);
            const auto& blk = qblocks.at(key);
            if(blk.size() > 0){
	       for(int m=0; m<pm.second; m++){
                  sum += pow(linalg::normF(blk[m]),2);
	       }
            }
         }
      }
   }
   return sqrt(sum);
}
