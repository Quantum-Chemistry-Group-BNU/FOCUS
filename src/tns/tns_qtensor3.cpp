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
	       vector<matrix<double>> blk(mdim,matrix<double>(rdim,cdim));
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
         double fac2 = qm.parity()==0? fac : -fac;
	 for(int m=0; m<blk.size(); m++){
	    blk[m] *= fac2;
	 }
      }
   }
   return qt3;
}

qtensor3 qtensor3::row_signed(const double fac) const{
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
         auto qr = get<1>(key);
         double fac2 = qr.parity()==0? fac : -fac;
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
         double fac2 = qc.parity()==0? fac : -fac;
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
   // qblocks
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
         if(level >= 1){
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
	 } // level>=1
      }
   }
   cout << "total no. of nonzero blocks=" << nnz << endl;
}

// simple operations
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

qtensor3 qtensor3::perm_signed() const{
   qtensor3 qt3;
   qt3 = *this;
   for(auto& p : qt3.qblocks){
      auto& key = p.first;
      auto& blk = p.second;
      if(blk.size() > 0){
	 auto qm = get<0>(key);
	 auto qc = get<2>(key);
	 if(qm.parity()*qc.parity() == 1){
	    for(int m=0; m<blk.size(); m++){	
	       blk[m] = -blk[m];
	    }
	 }
      }
   }
   return qt3;
}

// for Davidson algorithm
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

// decimation
pair<qsym_space,qsym_dpt> qtensor3::dpt_lc() const{
   return qsym_space_dpt(qrow,qmid);
}

pair<qsym_space,qsym_dpt> qtensor3::dpt_cr() const{
   return qsym_space_dpt(qmid,qcol);
}

pair<qsym_space,qsym_dpt> qtensor3::dpt_lr() const{
   return qsym_space_dpt(qrow,qcol);
}

qtensor2 qtensor3::merge_lc() const{
   auto dp = dpt_lc();
   return merge_qt3_qt2_lc(*this,dp.first,dp.second);
}

qtensor2 qtensor3::merge_cr() const{
   auto dp = dpt_cr();
   return merge_qt3_qt2_cr(*this,dp.first,dp.second);
}

qtensor2 qtensor3::merge_lr() const{
   auto dp = dpt_lr();
   return merge_qt3_qt2_lr(*this,dp.first,dp.second);
}

qtensor4 qtensor3::split_lc1(const qsym_space& qlx,
			     const qsym_space& qc1,
			     const qsym_dpt& dpt) const{
   return split_qt4_qt3_lc1(*this, qlx, qc1, dpt);
}

qtensor4 qtensor3::split_c2r(const qsym_space& qc2,
			     const qsym_space& qrx,
			     const qsym_dpt& dpt) const{
   return split_qt4_qt3_c2r(*this, qc2, qrx, dpt);
}

// for random sampling
qtensor2 qtensor3::fix_qphys(const qsym& sym_p) const{
   assert(dir[0] == true); // out
   assert(qmid.at(sym_p) == 1); // 1d
   qsym sym1 = sym-sym_p; // merged
   vector<bool> dir1 = {dir[1], dir[2]};
   qtensor2 qt2(sym1, qrow, qcol, dir1);
   for(const auto& pr : qrow){
      const auto& qr = pr.first;
      for(const auto& pc : qcol){
         const auto& qc = pc.first;
	 auto& blk0 = qblocks.at(make_tuple(sym_p,qr,qc));
	 if(blk0.size() > 0) qt2.qblocks[make_pair(qr,qc)] = blk0[0];
      }
   }
   return qt2;
}
