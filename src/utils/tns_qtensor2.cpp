#include "tns_qtensor.h"
#include "../core/linalg.h"
#include <iostream>
#include <algorithm>

using namespace std;
using namespace linalg;
using namespace tns;

// --- rank-2 tensor ---
// constructor for operator <r|o|c>
qtensor2::qtensor2(const qsym& sym1,
		   const qsym_space& qrow1, 
		   const qsym_space& qcol1,
		   const int nindex){
   if(nindex>0) index.resize(nindex); 
   sym = sym1;
   qrow = qrow1;
   qcol = qcol1;
   for(const auto& pr : qrow){
      const auto& qr = pr.first;
      int rdim = pr.second;
      for(const auto& pc : qcol){
	 const auto& qc = pc.first;
	 int cdim = pc.second;
	 auto key = make_pair(qr,qc);
         if(qr == sym + qc){
	    qblocks[key] = matrix(rdim,cdim);
	 }else{
	    qblocks[key] = matrix();
	 }
      }
   }
}

void qtensor2::print(const string msg, const int level) const{
   cout << "qtensor2: " << msg << " sym=" << sym << " index=";
   for(int i : index) cout << i << " ";
   cout << endl;
   qsym_space_print(qrow,"qrow");
   qsym_space_print(qcol,"qcol");
   if(level >= 1){
      cout << "qblocks: nblocks=" << qblocks.size() << endl;
      int nnz = 0;
      for(const auto& p : qblocks){
         auto& t = p.first;
         auto& m = p.second;
         auto sym_row = get<0>(t);
         auto sym_col = get<1>(t);
         if(m.size() > 0){
            nnz++;
            cout << "idx=" << nnz 
		 << " block[" << sym_row << "," << sym_col << "]"
                 << " size=" << m.size() 
                 << endl; 
            if(level >= 2){
               m.print("mat");
            } // level=2
         }
      }
      cout << "total no. of nonzero blocks=" << nnz << endl;
   } // level=1
}

matrix qtensor2::to_matrix() const{
   int m = get_dim_row();
   int n = get_dim_col();
   matrix mat(m,n);
   int joff = 0;
   for(const auto& pj : qcol){
      const auto& qj = pj.first;
      const int mj = pj.second;
      int ioff = 0;
      for(const auto& pi : qrow){
	 const auto& qi = pi.first;
	 const int mi = pi.second;
	 const auto& blk = qblocks.at(make_pair(qi,qj));
	 if(blk.size() != 0){
	    // save
	    for(int j=0; j<mj; j++){
	       for(int i=0; i<mi; i++){
	          mat(ioff+i,joff+j) = blk(i,j);
	       } 
	    }
	 }
	 ioff += mi; 
      } // pi
      joff += mj;
   } // pj
   return mat;
}

qtensor2 qtensor2::T() const{
   qtensor2 qt2;
   qt2.sym = -sym;
   qt2.qrow = qcol;
   qt2.qcol = qrow;
   qt2.index = index;
   // (pq)^+ = q^+p^+
   reverse(qt2.index.begin(), qt2.index.end());
   for(const auto& pr : qt2.qrow){
      const auto& qr = pr.first;
      for(const auto& pc : qt2.qcol){
	 const auto& qc = pc.first;
	 auto key = make_pair(qr,qc);
         if(qr == qt2.sym + qc){
            auto tkey = make_pair(qc,qr);
	    qt2.qblocks[key] = qblocks.at(tkey).T();
	 }else{
	    qt2.qblocks[key] = matrix();
	 }
      }
   }
   return qt2;
}

// nonzero blocks scaled by fac*(-1)^p(c)
qtensor2 qtensor2::col_signed(const double fac) const{
   qtensor2 qt2;
   qt2.sym = sym;
   qt2.qrow = qrow;
   qt2.qcol = qcol;
   qt2.index = index;
   qt2.qblocks = qblocks;
   for(const auto& pr : qrow){
      const auto& qr = pr.first;
      for(const auto& pc : qcol){
	 const auto& qc = pc.first;
         if(qr == sym + qc){
	    double fac2 = qc.parity()*fac;
	    auto key = make_pair(qr,qc);
	    qt2.qblocks[key] *= fac2;
	 }
      }
   }
   return qt2;
}

qtensor2 qtensor2::operator -() const{
   qtensor2 qt2;
   qt2.sym = sym;
   qt2.qrow = qrow;
   qt2.qcol = qcol;
   qt2.index = index;
   qt2.qblocks = qblocks;
   for(const auto& pr : qrow){
      const auto& qr = pr.first;
      for(const auto& pc : qcol){
	 const auto& qc = pc.first;
         if(qr == sym + qc){
	    auto key = make_pair(qr,qc);
	    qt2.qblocks[key] *= -1;
	 }
      }
   }
   return qt2;
}

// algorithmic operations like matrix
qtensor2& qtensor2::operator +=(const qtensor2& qt){
   assert(sym == qt.sym); // symmetry blocking must be the same
   for(const auto& pr : qrow){
      const auto& qr = pr.first;
      for(const auto& pc : qcol){
	 const auto& qc = pc.first;
	 auto key = make_pair(qr,qc);
	 auto& blk = qblocks[key];
	 assert(blk.size() == qt.qblocks.at(key).size());
	 if(blk.size() > 0) blk += qt.qblocks.at(key);
      }
   }
   return *this;
}

qtensor2& qtensor2::operator -=(const qtensor2& qt){
   assert(sym == qt.sym); // symmetry blocking must be the same
   for(const auto& pr : qrow){
      const auto& qr = pr.first;
      for(const auto& pc : qcol){
	 const auto& qc = pc.first;
	 auto key = make_pair(qr,qc);
	 auto& blk = qblocks[key];
	 assert(blk.size() == qt.qblocks.at(key).size());
	 if(blk.size() > 0) blk -= qt.qblocks.at(key);
      }
   }
   return *this;
}

qtensor2 tns::operator +(const qtensor2& qta, const qtensor2& qtb){
   qtensor2 qt2 = qta;
   qt2 += qtb;
   return qt2;
}

qtensor2 tns::operator -(const qtensor2& qta, const qtensor2& qtb){
   qtensor2 qt2 = qta;
   qt2 -= qtb;
   return qt2;
}

// fac*qt2
qtensor2& qtensor2::operator *=(const double fac){
   for(const auto& pr : qrow){
      const auto& qr = pr.first;
      for(const auto& pc : qcol){
	 const auto& qc = pc.first;
	 auto key = make_pair(qr,qc);
	 auto& blk = qblocks[key];
	 if(blk.size() > 0) blk *= fac;
      }
   }
   return *this;
}

qtensor2 tns::operator *(const double fac, const qtensor2& qt){
   qtensor2 qt2 = qt; // use default assignment constructor;
   qt2 *= fac;
   return qt2;
}

qtensor2 tns::operator *(const qtensor2& qt, const double fac){
   return fac*qt;
}

double qtensor2::normF() const{
   double sum = 0.0;
   for(const auto& pr : qrow){
      const auto& qr = pr.first;
      for(const auto& pc : qcol){
	 const auto& qc = pc.first;	
	 auto key = make_pair(qr,qc);
	 const auto& blk = qblocks.at(key);
	 if(blk.size() > 0){
	    sum += pow(linalg::normF(blk),2);
	 }
      }
   }
   return sqrt(sum);
}
