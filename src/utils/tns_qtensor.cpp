#include "tns_qtensor.h"
#include "../core/linalg.h"
#include <iostream>
#include <algorithm>

using namespace std;
using namespace linalg;
using namespace tns;

// --- rank-2 tensor ---
qtensor2::qtensor2(const qsym& msym1,
		   const qsym_space& qrow1, 
		   const qsym_space& qcol1,
		   const int nindex){
   if(nindex>0) index.resize(nindex); 
   msym = msym1;
   qrow = qrow1;
   qcol = qcol1;
   for(const auto& pr : qrow){
      const auto& qr = pr.first;
      int rdim = pr.second;
      for(const auto& pc : qcol){
	 const auto& qc = pc.first;
	 int cdim = pc.second;
	 auto key = make_pair(qr,qc);
         if(qr == msym + qc){
	    qblocks[key] = matrix(rdim,cdim);
	 }else{
	    qblocks[key] = matrix();
	 }
      }
   }
}

void qtensor2::print(const string msg, const int level) const{
   cout << "qtensor2: " << msg << " msym=" << msym << " index=";
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
      const auto& qsymj = pj.first;
      const int mj = pj.second;
      int ioff = 0;
      for(const auto& pi : qrow){
	 const auto& qsymi = pi.first;
	 const int mi = pi.second;
	 const auto& blk = qblocks.at(make_pair(qsymi,qsymj));
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

qtensor2 qtensor2::transpose() const{
   qtensor2 qt2;
   qt2.msym = -msym;
   qt2.qrow = qcol;
   qt2.qcol = qrow;
   qt2.index = index;
   // (pq)^+ = q^+p^+
   reverse(qt2.index.begin(), qt2.index.end());
   for(const auto& pr : qt2.qrow){
      const auto& qr = pr.first;
      int rdim = pr.second;
      for(const auto& pc : qt2.qcol){
	 const auto& qc = pc.first;
	 int cdim = pc.second;
	 auto key = make_pair(qr,qc);
         if(qr == qt2.msym + qc){
            auto tkey = make_pair(qc,qr);
	    qt2.qblocks[key] = qblocks.at(tkey).transpose();
	 }else{
	    qt2.qblocks[key] = matrix();
	 }
      }
   }
   return qt2;
}

qtensor2 qtensor2::col_signed(const double fac) const{
   qtensor2 qt2;
   qt2.msym = msym;
   qt2.qrow = qrow;
   qt2.qcol = qcol;
   qt2.index = index;
   qt2.qblocks = qblocks;
   for(const auto& pr : qrow){
      const auto& qr = pr.first;
      for(const auto& pc : qcol){
	 const auto& qc = pc.first;
	 auto key = make_pair(qr,qc);
	 // nonzero blocks scaled by fac*(-1)^p(c)
	 double fac2 = qc.parity()*fac;
         if(qr == msym + qc){
	    qt2.qblocks[key] *= fac2;
	 }
      }
   }
   return qt2;
}

qtensor2 qtensor2::operator -() const{
   qtensor2 qt2;
   qt2.msym = msym;
   qt2.qrow = qrow;
   qt2.qcol = qcol;
   qt2.index = index;
   qt2.qblocks = qblocks;
   for(const auto& pr : qrow){
      const auto& qr = pr.first;
      for(const auto& pc : qcol){
	 const auto& qc = pc.first;
	 auto key = make_pair(qr,qc);
         if(qr == msym + qc){
	    qt2.qblocks[key] *= -1;
	 }
      }
   }
   return qt2;
}

// algorithmic operations like matrix
qtensor2& qtensor2::operator +=(const qtensor2& qt){
   assert(msym == qt.msym); // symmetry blocking must be the same
   for(const auto& pr : qrow){
      const auto& qr = pr.first;
      for(const auto& pc : qcol){
	 const auto& qc = pc.first;
	 auto key = make_pair(qr,qc);
	 auto& blk = qblocks[key];
	 if(blk.size() > 0){
	    assert(blk.size() == qt.qblocks.at(key).size());
	    blk += qt.qblocks.at(key);
	 }
      }
   }
   return *this;
}

qtensor2& qtensor2::operator -=(const qtensor2& qt){
   assert(msym == qt.msym); // symmetry blocking must be the same
   for(const auto& pr : qrow){
      const auto& qr = pr.first;
      for(const auto& pc : qcol){
	 const auto& qc = pc.first;
	 auto key = make_pair(qr,qc);
	 auto& blk = qblocks[key];
	 if(blk.size() > 0){
	    assert(blk.size() == qt.qblocks.at(key).size());
	    blk -= qt.qblocks.at(key);
	 }
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
	 if(blk.size() > 0){
	    blk *= fac;
	 }
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

// --- rank-3 tensor ---
qtensor3::qtensor3(const qsym_space& qmid1,
		   const qsym_space& qrow1, 
		   const qsym_space& qcol1){
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
            if(qr == qm + qc){
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
         int rdim = pr.second;
         for(const auto& pc : qcol){
            const auto& qc = pc.first;
            int cdim = pc.second;
	    // initialization
	    auto key = make_tuple(qm,qr,qc);
            if(qr == qm + qc){
	       auto& blk = qt3.qblocks[key]; 
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
   qt3.qmid = qmid;
   qt3.qrow = qrow;
   qt3.qcol = qcol;
   qt3.qblocks = qblocks;
   for(const auto& pm : qmid){
      const auto& qm = pm.first;
      int mdim = pm.second; 
      for(const auto& pr : qrow){
         const auto& qr = pr.first;
         int rdim = pr.second;
         for(const auto& pc : qcol){
            const auto& qc = pc.first;
            int cdim = pc.second;
            double fac2 = qc.parity()*fac;
	    // initialization
	    auto key = make_tuple(qm,qr,qc);
            if(qr == qm + qc){
	       auto& blk = qt3.qblocks[key]; 
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
   cout << "qtensor3: " << msg << endl;
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

// --- tensor linear algebra : contractions ---

//          r---*--\ qt3a
// q(r,c) =     |m |x     = <r|c> = \sum_n An^C*Bn^T
//          c---*--/ qt3b
qtensor2 tns::contract_qt3_qt3_cr(const qtensor3& qt3a, 
				  const qtensor3& qt3b,
				  const qsym& msym){
   qtensor2 qt2(msym,qt3a.qrow,qt3b.qrow);
   // loop over external indices
   for(const auto& pr : qt2.qrow){
      const qsym& rsym = pr.first; 
      int rdim = pr.second;
      for(const auto& pc : qt2.qcol){
	 const qsym& csym = pc.first;
	 int cdim = pc.second;
	 auto key = make_pair(rsym,csym);
	 auto& blk = qt2.qblocks[key];
	 // impose symmetry selection for the final blocks
	 if(blk.size() == 0) continue;
	 
	 // loop over contracted indices
         for(const auto& pm : qt3a.qmid){
            const qsym& msym = pm.first;
	    int mdim = pm.second;
	    for(const auto& px : qt3a.qcol){
	       const qsym& xsym = px.first;
	       // contract blocks
	       auto keya = make_tuple(msym,rsym,xsym);
	       auto keyb = make_tuple(msym,csym,xsym);
	       auto& blka = qt3a.qblocks.at(keya);
	       auto& blkb = qt3b.qblocks.at(keyb);
	       if(blka.size() == 0 || blkb.size() == 0) continue;
               for(int m=0; m<mdim; m++){
	          blk += dgemm("N","T",blka[m],blkb[m]); 
	       } // m
	    } // qx
	 } // qm

      } // qc
   } // qr
   return qt2;
}

//     m  \r2
//     |  *   = [m](r,r2) = A[m](r,x)*R(r2,x)^T
//  r--*--/x 
qtensor3 tns::contract_qt3_qt2_r(const qtensor3& qt3a, 
				 const qtensor2& qt2b){
   qtensor3 qt3(qt3a.qmid,qt3a.qrow,qt2b.qrow);
   // loop over external indices
   for(const auto& pm : qt3.qmid){
      const qsym& msym = pm.first;
      int mdim = pm.second;
      for(const auto& pr : qt3.qrow){
         const qsym& rsym = pr.first; 
         int rdim = pr.second;
         for(const auto& pc : qt3.qcol){
            const qsym& csym = pc.first;
            int cdim = pc.second;
	    auto key = make_tuple(msym,rsym,csym);
	    auto& blk = qt3.qblocks[key];
	    if(blk.size() == 0) continue;

	    // loop over contracted indices
	    for(const auto& px : qt3a.qcol){
	       const qsym& xsym = px.first;
	       // contract blocks
	       auto keya = make_tuple(msym,rsym,xsym);
	       auto keyb = make_pair(csym,xsym);
	       auto& blka = qt3a.qblocks.at(keya);
	       auto& blkb = qt2b.qblocks.at(keyb);
	       if(blka.size() == 0 || blkb.size() == 0) continue;
	       for(int m=0; m<mdim; m++){
	          blk[m] += dgemm("N","T",blka[m],blkb);
	       } // m
	    } // qx
	    
	 } // qc
      } // qr
   } // qm
   return qt3;
}

//     |m/r
//     *	 
//     |x/c  = [m](r,c) = B(m,x)* A[x](r,c)
//  r--*--c
qtensor3 tns::contract_qt3_qt2_c(const qtensor3& qt3a, 
			 	 const qtensor2& qt2b){
   qtensor3 qt3(qt2b.qrow,qt3a.qrow,qt3a.qcol);
   // loop over external indices
   for(const auto& pm : qt3.qmid){
      const qsym& msym = pm.first;
      int mdim = pm.second;
      for(const auto& pr : qt3.qrow){
         const qsym& rsym = pr.first; 
         int rdim = pr.second;
         for(const auto& pc : qt3.qcol){
            const qsym& csym = pc.first;
            int cdim = pc.second;
	    auto key = make_tuple(msym,rsym,csym);
	    auto& blk = qt3.qblocks[key];
	    if(blk.size() == 0) continue;

	    // loop over contracted indices
	    for(const auto& px : qt3a.qmid){
	       const qsym& xsym = px.first;
	       int xdim = px.second;
	       // contract blocks
	       auto keya = make_tuple(xsym,rsym,csym);
	       auto keyb = make_pair(msym,xsym);
	       auto& blka = qt3a.qblocks.at(keya);
	       auto& blkb = qt2b.qblocks.at(keyb);
	       if(blka.size() == 0 || blkb.size() == 0) continue;
	       for(int x=0; x<xdim; x++){
	          for(int m=0; m<mdim; m++){
	             blk[m] += blkb(m,x)*blka[x];
	          } // m
	       } // x 
	    } // qx
	    
	 } // qc
      } // qr
   } // qm
   return qt3;
}

//          /--*--r qt3a
// q(r,c) = |x |m  	  = <r|c> = \sum_n An^H*Bn
//          \--*--c qt3b
qtensor2 tns::contract_qt3_qt3_lc(const qtensor3& qt3a, 
				  const qtensor3& qt3b,
				  const qsym& msym){
   qtensor2 qt2(msym,qt3a.qcol,qt3b.qcol);
   // loop over external indices
   for(const auto& pr : qt2.qrow){
      const qsym& rsym = pr.first; 
      int rdim = pr.second;
      for(const auto& pc : qt2.qcol){
	 const qsym& csym = pc.first;
	 int cdim = pc.second;
	 auto key = make_pair(rsym,csym);
	 auto& blk = qt2.qblocks[key];
	 if(blk.size() == 0) continue;

	 // loop over contracted indices
         for(const auto& pm : qt3a.qmid){
            const qsym& msym = pm.first;
	    int mdim = pm.second;
	    for(const auto& px : qt3a.qrow){
	       const qsym& xsym = px.first;
	       // contract blocks
	       auto keya = make_tuple(msym,xsym,rsym);
	       auto keyb = make_tuple(msym,xsym,csym);
	       auto& blka = qt3a.qblocks.at(keya);
	       auto& blkb = qt3b.qblocks.at(keyb);
	       if(blka.size() == 0 || blkb.size() == 0) continue;
               for(int m=0; m<mdim; m++){
	          blk += dgemm("T","N",blka[m],blkb[m]); 
	       } // m
	    } // qx
	 } // qm
	 
      } // qc
   } // qr
   return qt2;
}

//  r/	m 
//   *  |     = [m](r,c) = L(r,x)*A[m](x,c)
//  x\--*--c
qtensor3 tns::contract_qt3_qt2_l(const qtensor3& qt3a, 
				 const qtensor2& qt2b){
   qtensor3 qt3(qt3a.qmid,qt2b.qrow,qt3a.qcol);
   // loop over external indices
   for(const auto& pm : qt3.qmid){
      const qsym& msym = pm.first;
      int mdim = pm.second;
      for(const auto& pr : qt3.qrow){
         const qsym& rsym = pr.first; 
         int rdim = pr.second;
         for(const auto& pc : qt3.qcol){
            const qsym& csym = pc.first;
            int cdim = pc.second;
	    auto key = make_tuple(msym,rsym,csym);
	    auto& blk = qt3.qblocks[key];
	    if(blk.size() == 0) continue;

	    // loop over contracted indices
	    for(const auto& px : qt3a.qrow){
	       const qsym& xsym = px.first;
	       // contract blocks
	       auto keya = make_tuple(msym,xsym,csym);
	       auto keyb = make_pair(rsym,xsym);
	       auto& blka = qt3a.qblocks.at(keya);
	       auto& blkb = qt2b.qblocks.at(keyb);
	       if(blka.size() == 0 || blkb.size() == 0) continue;
	       for(int m=0; m<mdim; m++){
	          blk[m] += dgemm("N","N",blkb,blka[m]);
	       } // m
	    } // qx
	    
	 } // qc
      } // qr
   } // qm
   return qt3;
}
