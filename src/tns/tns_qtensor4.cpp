#include "../core/linalg.h"
#include "tns_qtensor.h"
#include <iostream>
#include <algorithm>

using namespace std;
using namespace linalg;
using namespace tns;

// --- rank-4 tensor ---
qtensor4::qtensor4(const qsym& sym1,
		   const qsym_space& qmid1,
		   const qsym_space& qver1,
		   const qsym_space& qrow1, 
		   const qsym_space& qcol1){
   sym = sym1;
   qmid = qmid1;
   qver = qver1;
   qrow = qrow1;
   qcol = qcol1;
   for(const auto& pm : qmid){
      const auto& qm = pm.first;
      int mdim = pm.second; 
      for(const auto& pv : qver){
         const auto& qv = pv.first;
	 int vdim = pv.second;
         for(const auto& pr : qrow){
            const auto& qr = pr.first;
            int rdim = pr.second;
            for(const auto& pc : qcol){
               const auto& qc = pc.first;
               int cdim = pc.second;
	       // initialization
	       auto key = make_tuple(qm,qv,qr,qc);
               if(sym == qm+qv+qr+qc){
	          vector<matrix<double>> blk(mdim*vdim,matrix<double>(rdim,cdim));
	          qblocks[key] = blk;
	       }else{
	          qblocks[key] = empty_block;
	       }
	    }
	 }
      }
   }
}

void qtensor4::print(const string msg, const int level) const{
   cout << "qtensor4: " << msg << " sym=" << sym << endl;
   qsym_space_print(qmid,"qmid");
   qsym_space_print(qver,"qver");
   qsym_space_print(qrow,"qrow");
   qsym_space_print(qcol,"qcol");
   // qblocks
   cout << "qblocks: nblocks=" << qblocks.size() << endl;
   int nnz = 0;
   for(const auto& p : qblocks){
      auto& t = p.first;
      auto& m = p.second;
      auto sym_mid = get<0>(t);
      auto sym_ver = get<1>(t);
      auto sym_row = get<2>(t);
      auto sym_col = get<3>(t);
      if(m.size() > 0){
         nnz++;
         if(level >= 1){
            cout << "idx=" << nnz 
     	         << " block[" << sym_mid << "," << sym_ver << "," 
     	    	 << sym_row << "," << sym_col << "]"
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
qtensor4& qtensor4::operator +=(const qtensor4& qt){
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

qtensor4& qtensor4::operator -=(const qtensor4& qt){
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

qtensor4& qtensor4::operator *=(const double fac){
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

qtensor4 tns::operator *(const double fac, const qtensor4& qt){
   qtensor4 qt4 = qt; // use default assignment constructor;
   qt4 *= fac;
   return qt4;
}

qtensor4 tns::operator *(const qtensor4& qt, const double fac){
   return fac*qt;
}

qtensor4 qtensor4::perm_signed() const{
   qtensor4 qt4;
   qt4 = *this;
   for(auto& p : qt4.qblocks){
      auto& key = p.first;
      auto& blk = p.second;
      if(blk.size() > 0){
	 auto qm = get<0>(key);
	 auto qv = get<1>(key);
	 auto qc = get<3>(key);
	 if(((qm.parity()+qv.parity())*qc.parity())%2 == 1){
	    for(int m=0; m<blk.size(); m++){
	       blk[m] = -blk[m];
	    }
	 }
      }
   }
   return qt4;
}

// for Davidson algorithm
double qtensor4::normF() const{
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

void qtensor4::random(){
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

int qtensor4::get_dim() const{
   int dim = 0;
   for(const auto& p : qblocks){
      auto& blk = p.second;
      if(blk.size() > 0){
	 dim += blk.size()*blk[0].size();
      }
   }
   return dim;
}

void qtensor4::from_array(const double* array){
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

void qtensor4::to_array(double* array) const{
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
pair<qsym_space,qsym_dpt> qtensor4::dpt_lc1() const{
   return qsym_space_dpt(qrow,qmid);
}

pair<qsym_space,qsym_dpt> qtensor4::dpt_c2r() const{
   return qsym_space_dpt(qver,qcol);
}

pair<qsym_space,qsym_dpt> qtensor4::dpt_lr() const{
   return qsym_space_dpt(qrow,qcol);
}

pair<qsym_space,qsym_dpt> qtensor4::dpt_c1c2() const{
   return qsym_space_dpt(qmid,qver);
}

qtensor3 qtensor4::merge_lc1() const{
   auto dp = dpt_lc1();
   return merge_qt4_qt3_lc1(*this,dp.first,dp.second);
}

qtensor3 qtensor4::merge_c2r() const{
   auto dp = dpt_c2r();
   return merge_qt4_qt3_c2r(*this,dp.first,dp.second);
}

qtensor2 qtensor4::merge_lr_c1c2() const{
   auto dp1 = dpt_lr();
   auto dp2 = dpt_c1c2();
   return merge_qt4_qt2_lr_c1c2(*this,dp1.first,dp1.second,dp2.first,dp2.second); 
}
