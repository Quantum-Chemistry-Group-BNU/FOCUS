#include "../core/linalg.h"
#include "tns_qtensor.h"
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
		   const vector<bool> dir1){
   dir = dir1;
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
	 // symmetry rule for <r|O|c>
         if(ifconserve(qr,qc)){
	    qblocks[key] = matrix(rdim,cdim);
	 }else{
	    qblocks[key] = matrix();
	 }
      }
   }
}

void qtensor2::print(const string msg, const int level) const{
   cout << "qtensor2: " << msg << " sym=" << sym;
   cout << " dir=";
   for(auto b : dir) cout << b << " ";
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
                 << " rows,cols=(" << m.rows() << "," << m.cols() << ")"
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
	 if(blk.size() > 0){
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
   for(const auto& p : qblocks){
      const auto& key = p.first;
      const auto& blk = qblocks.at(key);
      auto tkey = make_pair(key.second,key.first);
      if(blk.size() > 0){
         qt2.qblocks[tkey] = blk.T();
      }else{
         qt2.qblocks[tkey] = matrix();
      } 
   }
   return qt2;
}

// nonzero blocks scaled by fac*(-1)^p(c)
qtensor2 qtensor2::col_signed(const double fac) const{
   qtensor2 qt2;
   qt2.dir = dir;
   qt2.sym = sym;
   qt2.qrow = qrow;
   qt2.qcol = qcol;
   qt2.qblocks = qblocks;
   for(auto& p : qt2.qblocks){
      auto& key = p.first;
      auto& blk = p.second;
      if(blk.size() > 0){
         auto qc = key.second;
         double fac2 = qc.parity()==0? fac : -fac;
	 blk *= fac2;
      }
   }
   return qt2;
}

qtensor2 qtensor2::operator -() const{
   qtensor2 qt2;
   qt2.dir = dir;
   qt2.sym = sym;
   qt2.qrow = qrow;
   qt2.qcol = qcol;
   qt2.qblocks = qblocks;
   for(auto& p : qt2.qblocks){
      auto& key = p.first;
      auto& blk = p.second;
      if(blk.size() > 0) blk *= -1;
   }
   return qt2;
}

// algorithmic operations like matrix
qtensor2& qtensor2::operator +=(const qtensor2& qt){
   assert(dir == qt.dir);
   assert(sym == qt.sym); // symmetry blocking must be the same
   for(auto& p : qblocks){
      auto& key = p.first;
      auto& blk = p.second;
      assert(blk.size() == qt.qblocks.at(key).size());
      if(blk.size() > 0) blk += qt.qblocks.at(key);
   }
   return *this;
}

qtensor2& qtensor2::operator -=(const qtensor2& qt){
   assert(dir == qt.dir);
   assert(sym == qt.sym); // symmetry blocking must be the same
   for(auto& p : qblocks){
      auto& key = p.first;
      auto& blk = p.second;
      assert(blk.size() == qt.qblocks.at(key).size());
      if(blk.size() > 0) blk -= qt.qblocks.at(key);
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
   for(auto& p : qblocks){
      auto& key = p.first;
      auto& blk = p.second;
      if(blk.size() > 0) blk *= fac;
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
   for(const auto& p : qblocks){
      const auto& blk = p.second;
      if(blk.size() > 0) sum += pow(linalg::normF(blk),2);
   }
   return sqrt(sum);
}

// check whether <l|o|r> is a faithful rep for o=I
double qtensor2::check_identity(const double thresh_ortho,
			        const bool debug) const{
   if(debug) cout << "qtensor2::check_identity thresh_ortho=" 
	  	  << thresh_ortho << endl;
   double mdiff = -1.0;
   for(const auto& pr : qrow){
      const auto& qr = pr.first;
      for(const auto& pc : qcol){
	 const auto& qc = pc.first;	
	 auto key = make_pair(qr,qc);
	 const auto& blk = qblocks.at(key);
	 if(blk.size() > 0){
	    if(qr != qc){
	       cout << "error: not a block-diagonal matrix!";
	       exit(1);
	    }
            int ndim = pr.second;
            double diff = linalg::normF(blk - identity_matrix(ndim));
	    mdiff = max(diff,mdiff);
	    if(debug){
               cout << "qsym=" << qr << " ndim=" << ndim 
     	 	    << " |Sr-Id|_F=" << diff << endl;
	    }
            if(diff > thresh_ortho){
	       cout << "error: not an identity matrix at qsym=" << qr << endl;
	       blk.print("block sym"+sym.to_string());
	    }
	 }
      }
   }
   return mdiff;
}

void qtensor2::random(){
   for(auto& p : qblocks){
      auto& blk = p.second;
      if(blk.size()>0){
	 int rdim = blk.rows();
	 int cdim = blk.cols();
	 blk = random_matrix(rdim,cdim);
      }
   }
}

int qtensor2::get_dim() const{
   int dim = 0;
   for(const auto& p : qblocks){
      auto& blk = p.second;
      dim += blk.size();
   }
   return dim;
}

// decimation: -<-*->- split into qt3
qtensor3 qtensor2::split_lc(const qsym_space& qlx,
			    const qsym_space& qcx,
			    const qsym_dpt& dpt) const{
   return split_qt3_qt2_lc(*this, qlx, qcx, dpt);
}

qtensor3 qtensor2::split_cr(const qsym_space& qcx,
			    const qsym_space& qrx,
		    	    const qsym_dpt& dpt) const{	    
   return split_qt3_qt2_cr(*this, qcx, qrx, dpt);
}

qtensor3 qtensor2::split_lr(const qsym_space& qlx,
			    const qsym_space& qrx,
			    const qsym_dpt& dpt) const{
   return split_qt3_qt2_lr(*this, qlx, qrx, dpt);
}

// rdm from wf: rdm[r1,r2] += wf[r1,c]*wf[r2,c]^* = M.M^d
qtensor2 qtensor2::get_rdm_row() const{
   qtensor2 rdm(qsym(0,0), qrow, qrow);
   for(const auto& pr : qrow){
      auto& qr = pr.first;
      auto& key = make_pair(qr,qr);
      for(const auto& pc : qcol){
	 auto& qc = pc.first;
	 const auto& blk = qblocks.at(make_pair(qr,qc));
	 if(blk.size() == 0) continue;
	 rdm.qblocks[key] += dgemm("N","N",blk,blk.T());
      }
   }
}

// rdm from wf: rdm[c1,c2] += wf[r,c1]*wf[r,c2]^* = M^t.M^*
qtensor2 qtensor2::get_rdm_col() const{
   qtensor2 rdm(qsym(0,0), qcol, qcol);
   for(const auto& pc : qcol){
      auto& qc = pc.first;
      for(const auto& pr : qrow){
         auto& qr = pr.first;
         auto& key = make_pair(qc,qc);
	 const auto& blk = qblocks.at(make_pair(qr,qc));
	 if(blk.size() == 0) continue;
	 rdm.qblocks[key] += dgemm("N","N",blk.T(),blk);
      }
   }
}
