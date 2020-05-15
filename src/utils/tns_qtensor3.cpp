#include "../core/linalg.h"
#include "tns_qtensor.h"
#include <iostream>
#include <algorithm>

using namespace std;
using namespace linalg;
using namespace tns;

// --- rank-3 tensor ---
qtensor3::qtensor3(const qsym& sym1,
		   const qsym_space& qmid1,
		   const qsym_space& qrow1, 
		   const qsym_space& qcol1,
		   const vector<bool> dir1){
   dir = dir1;
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
            if(ifconserve(qm,qr,qc)){
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
   qt3.dir = dir;
   qt3.sym = sym;
   qt3.qmid = qmid;
   qt3.qrow = qrow;
   qt3.qcol = qcol;
   qt3.qblocks = qblocks;
   for(auto& p : qt3.qblocks){
      auto& key = p.first;
      auto& blk = p.second;
      if(blk.size() > 0){
         auto qm = get<0>(key);
         double fac2 = qm.parity()*fac;
	 for(int m=0; m<blk.size(); m++){
	    blk[m] *= fac2;
	 }
      }
   }
   return qt3;
}

qtensor3 qtensor3::col_signed(const double fac) const{
   qtensor3 qt3;
   qt3.dir = dir;
   qt3.sym = sym;
   qt3.qmid = qmid;
   qt3.qrow = qrow;
   qt3.qcol = qcol;
   qt3.qblocks = qblocks;
   for(auto& p : qt3.qblocks){
      auto& key = p.first;
      auto& blk = p.second;
      if(blk.size() > 0){
         auto qc = get<2>(key);
         double fac2 = qc.parity()*fac;
	 for(int m=0; m<blk.size(); m++){
	    blk[m] *= fac2;
	 }
      }
   }
   return qt3;
}

void qtensor3::print(const string msg, const int level) const{
   cout << "qtensor3: " << msg << " sym=" << sym;
   cout << " dir=";
   for(auto b : dir) cout << b << " ";
   cout << endl;
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
   for(const auto& p : qblocks){
      const auto& key = p.first;
      const auto& blk = p.second;
      if(blk.size() > 0){
	 for(int m=0; m<blk.size(); m++){
            sum += pow(linalg::normF(blk[m]),2);
         }
      }
   }
   return sqrt(sum);
}

qtensor3& qtensor3::operator +=(const qtensor3& qt){
   assert(dir == qt.dir); // direction must be the same
   assert(sym == qt.sym); // symmetry blocking must be the same
   for(auto& p : qblocks){
      auto& key = p.first;
      auto& blk = p.second;
      const auto& blk1 = qt.qblocks.at(key);
      assert(blk.size() == blk1.size());
      if(blk.size() > 0){
	 for(int m=0; m<blk.size(); m++){
	    blk[m] += blk1[m]; 
	 }
      }
   }
   return *this;
}

qtensor3& qtensor3::operator -=(const qtensor3& qt){
   assert(dir == qt.dir); // direction must be the same
   assert(sym == qt.sym); // symmetry blocking must be the same
   for(auto& p : qblocks){
      auto& key = p.first;
      auto& blk = p.second;
      const auto& blk1 = qt.qblocks.at(key);
      assert(blk.size() == blk1.size());
      if(blk.size() > 0){
	 for(int m=0; m<blk.size(); m++){
	    blk[m] -= blk1[m]; 
	 }
      }
   }
   return *this;
}

qtensor3 tns::operator +(const qtensor3& qta, const qtensor3& qtb){
   qtensor3 qt3 = qta;
   qt3 += qtb;
   return qt3;
}

qtensor3 tns::operator -(const qtensor3& qta, const qtensor3& qtb){
   qtensor3 qt3 = qta;
   qt3 -= qtb;
   return qt3;
}

qtensor3& qtensor3::operator *=(const double fac){
   for(auto& p : qblocks){
      auto& blk = p.second;
      if(blk.size() > 0){ 
	 for(int m=0; m<blk.size(); m++){
	    blk[m] *= fac;
	 }
      }
   }
   return *this;
}

qtensor3 tns::operator *(const double fac, const qtensor3& qt){
   qtensor3 qt3 = qt; // use default assignment constructor;
   qt3 *= fac;
   return qt3;
}

qtensor3 tns::operator *(const qtensor3& qt, const double fac){
   return fac*qt;
}

// for Davidson algorithm
void qtensor3::random(){
   for(auto& p : qblocks){
      auto& blk = p.second;
      if(blk.size() > 0){
	 int mdim = blk.size();
	 int rdim = blk[0].rows();
	 int cdim = blk[0].cols();
	 for(int m=0; m<mdim; m++){
	    blk[m] = random_matrix(rdim,cdim);
	 }
      }
   }
}

int qtensor3::get_dim() const{
   int dim = 0;
   for(const auto& p : qblocks){
      auto& blk = p.second;
      if(blk.size() > 0){
	 dim += blk.size()*blk[0].size();
      }
   }
   return dim;
}

void qtensor3::from_array(const double* array){
   int ioff = 0;
   for(auto& p : qblocks){
      auto& key = p.first;
      auto& blk = p.second;
      if(blk.size() > 0){
         int size = blk[0].size();
	 for(int m=0; m<blk.size(); m++){
            auto psta = array+ioff+m*size;
	    copy(psta, psta+size, blk[m].data());
	 }
	 ioff += blk.size()*size;
      }
   }
}

void qtensor3::to_array(double* array) const{
   int ioff = 0;
   for(auto& p : qblocks){
      auto& key = p.first;
      auto& blk = p.second;
      if(blk.size() > 0){
         int size = blk[0].size();
	 for(int m=0; m<blk.size(); m++){
            auto psta = array+ioff+m*size;
	    copy(blk[m].data(), blk[m].data()+size, psta);
	 }
	 ioff += blk.size()*size;
      }
   }
}

