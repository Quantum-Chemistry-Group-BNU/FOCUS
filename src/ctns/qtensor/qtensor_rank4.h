#ifndef QTENSOR_RANK4_H
#define QTENSOR_RANK4_H

namespace ctns{

// rank-4 tensor: only for holding two-dot wavefunction psi[l,c1,c2,r] 
template <typename Tm>
struct qtensor4{
   private:
      // serialize
      friend class boost::serialization::access;
      template<class Archive>
      void serialize(Archive & ar, const unsigned int version){
         ar & sym & qmid & qver & qrow & qcol
	    & _mids & _vers & _rows & _cols & _qblocks;
      }
      // conservation: dir={1,1,1,1} 
      inline bool _ifconserve(const int bm, const int bv, const int br, const int bc) const{
	 return sym == qmid.get_sym(bm) + qver.get_sym(bv) + qrow.get_sym(br) + qcol.get_sym(bc);
      }
      // address for storaging block data
      inline int _addr(const int bm, const int bv, const int br, const int bc) const{
	 return bm*_vers*_rows*_cols + bv*_rows*_cols + br*_cols + bc;
      }
      inline void _addr_unpack(const int idx, int& bm, int& bv, int& br, int& bc) const{
	 bc = idx%_cols;
	 int mvr = idx/_cols;
	 br = mvr%_rows;
	 int mv = mvr/_rows;
	 bm = mv/_vers;
	 bv = mv%_vers;
      }
   public:
      // constructor
      qtensor4(){};
      qtensor4(const qsym& sym1, const qbond& qmid1, const qbond& qver1, const qbond& qrow1, const qbond& qcol1);
      void init(const qsym& sym1, const qbond& qmid1, const qbond& qver1, const qbond& qrow1, const qbond& qcol1);
      // helpers
      inline int mids() const{ return _mids; }
      inline int vers() const{ return _vers; }
      inline int rows() const{ return _rows; }
      inline int cols() const{ return _cols; }
      // access: return a vector of matrices, vector size = dm*dv
      std::vector<linalg::matrix<Tm>>& operator ()(const int bm, const int bv, const int br, const int bc){
	 return _qblocks[_addr(bm,bv,br,bc)];
      }
      const std::vector<linalg::matrix<Tm>>& operator ()(const int bm, const int bv, const int br, const int bc) const{
	 return _qblocks[_addr(bm,bv,br,bc)];
      }
      // print
      size_t get_size() const;
      void print_size(const std::string name) const;
      void print(const std::string name, const int level=0) const;
      // deal with fermionic sign in fermionic direct product
      qtensor4<Tm> permCR_signed() const; // wf[lc1c2r]->wf[lc1c2r]*(-1)^{(p[c1]+p[c2])*p[r]}
      // ZL20210510: application of time-reversal operation
      qtensor4<Tm> K(const int nbar=0) const;
      // simple arithmetic operations
      qtensor4<Tm>& operator +=(const qtensor4<Tm>& qt);
      qtensor4<Tm>& operator -=(const qtensor4<Tm>& qt);
      qtensor4<Tm>& operator *=(const Tm fac);
      friend qtensor4<Tm> operator +(const qtensor4<Tm>& qta, const qtensor4<Tm>& qtb){
         qtensor4<Tm> qt4 = qta;
         qt4 += qtb;
         return qt4;
      }
      friend qtensor4<Tm> operator -(const qtensor4<Tm>& qta, const qtensor4<Tm>& qtb){
         qtensor4<Tm> qt4 = qta;
         qt4 -= qtb;
         return qt4;
      }
      friend qtensor4<Tm> operator *(const Tm fac, const qtensor4<Tm>& qt){
	 qtensor4<Tm> qt4 = qt;
	 qt4 *= fac;
	 return qt4;
      }
      friend qtensor4<Tm> operator *(const qtensor4<Tm>& qt, const Tm fac){
	 return fac*qt;
      }
      // for Davidson algorithm
      double normF() const;
      int get_dim() const;
      void from_array(const Tm* array);
      void to_array(Tm* array) const;
      void add_noise(const double noise);
      // for decimation
      inline qproduct dpt_lc1()  const{ return qmerge(qrow,qmid); };
      inline qproduct dpt_c2r()  const{ return qmerge(qver,qcol); };
      inline qproduct dpt_c1c2() const{ return qmerge(qmid,qver); };
      inline qproduct dpt_lr()   const{ return qmerge(qrow,qcol); };
      // reshape: merge
      qtensor3<Tm> merge_lc1() const{ // wf3[lc1,c2,r] 
	 auto qprod = dpt_lc1();
	 return merge_qt4_qt3_lc1(*this, qprod.first, qprod.second);
      }
      qtensor3<Tm> merge_c2r() const{ // wf3[l,c1,c2r]
	 auto qprod = dpt_c2r();
	 return merge_qt4_qt3_c2r(*this, qprod.first, qprod.second);
      } 
      qtensor3<Tm> merge_c1c2() const{ // wf3[l,c1c2,r]
	 auto qprod = dpt_c1c2();
	 return merge_qt4_qt3_c1c2(*this, qprod.first, qprod.second);
      }
      qtensor3<Tm> merge_lr() const{ // wf3[lr,c1,c2]
	 auto qprod = dpt_lr();
	 return merge_qt4_qt3_lr(*this, qprod.first, qprod.second);
      }
   public:
      qsym sym; 
      qbond qmid, qver, qrow, qcol;
   private:
      int _mids, _vers, _rows, _cols;   
      std::vector<std::vector<linalg::matrix<Tm>>> _qblocks;
};

template <typename Tm>
void qtensor4<Tm>::init(const qsym& sym1, const qbond& qmid1, const qbond& qver1,
		        const qbond& qrow1, const qbond& qcol1){
   sym = sym1;
   qmid = qmid1;
   qver = qver1;
   qrow = qrow1;
   qcol = qcol1;
   _mids = qmid.size();
   _vers = qver.size();
   _rows = qrow.size();
   _cols = qcol.size();
   _qblocks.resize(_mids*_vers*_rows*_cols);
   for(int bm=0; bm<_mids; bm++){
      for(int bv=0; bv<_vers; bv++){
         for(int br=0; br<_rows; br++){
            for(int bc=0; bc<_cols; bc++){
  	       if(not _ifconserve(bm,bv,br,bc)) continue;
	       int mdim = qmid.get_dim(bm);
	       int vdim = qver.get_dim(bv);
	       int rdim = qrow.get_dim(br);
	       int cdim = qcol.get_dim(bc);
	       int addr = _addr(bm,bv,br,bc);
	       _qblocks[addr].resize(mdim*vdim); 
	       for(int imv=0; imv<mdim*vdim; imv++){
	          _qblocks[addr][imv].resize(rdim,cdim);
	       }
	    } // bc
	 } // br
      } // bv
   } // bm
}

template <typename Tm>
qtensor4<Tm>::qtensor4(const qsym& sym1, const qbond& qmid1, const qbond& qver1,
		       const qbond& qrow1, const qbond& qcol1){
   this->init(sym1, qmid1, qver1, qrow1, qcol1);
}
 
template <typename Tm>
void qtensor4<Tm>::print(const std::string name, const int level) const{
   std::cout << "\nqtensor4: " << name << " sym=" << sym << std::endl;
   qmid.print("qmid");
   qver.print("qver");
   qrow.print("qrow");
   qcol.print("qcol");
   // qblocks
   std::cout << "qblocks: nblocks=" << _qblocks.size() << std::endl;
   int nnz = 0;
   for(int idx=0; idx<_qblocks.size(); idx++){
      const auto& blk = _qblocks[idx];
      if(blk.size() > 0){
         nnz++;
         if(level >= 1){
	    int bm,bv,br,bc;
	    _addr_unpack(idx,bm,bv,br,bc);
	    std::cout << "idx=" << nnz 
     	              << " block[" << qmid.get_sym(bm) << "," << qver.get_sym(bv) << ","
		      << qrow.get_sym(br) << "," << qcol.get_sym(bc) << "]"
                      << " size=" << blk.size() 
                      << " rows,cols=(" << blk[0].rows() << "," << blk[0].cols() << ")" 
                      << std::endl; 
            if(level >= 2){
               for(int imv=0; imv<blk.size(); imv++){		 
                  blk[imv].print("mat"+std::to_string(imv));
               }
            } // level=2
	 } // level>=1
      }
   }
   std::cout << "total no. of nonzero blocks=" << nnz << std::endl;
}

template <typename Tm>
size_t qtensor4<Tm>::get_size() const{
   size_t size = 0;
   for(int idx=0; idx<_qblocks.size(); idx++){
      const auto& blk = _qblocks[idx];
      for(int im=0; im<blk.size(); im++){
         size += blk[im].size();
      }
   } // idx
   return size;
}

template <typename Tm>
void qtensor4<Tm>::print_size(const std::string name) const{
   size_t size = this->get_size(); 
   std::cout << "qtensor4: " << name << " size=" << size
             << " sizeMB=" << tools::sizeMB<Tm>(size) 
             << std::endl;
}

// wf[lc1c2r]->wf[lc1c2r]*(-1)^{(p[c1]+p[c2])*p[r]}
template <typename Tm>
qtensor4<Tm> qtensor4<Tm>::permCR_signed() const{
   qtensor4<Tm> qt4 = *this;
   for(int idx=0; idx<qt4._qblocks.size(); idx++){
      auto& blk = qt4._qblocks[idx];
      if(blk.size() > 0){
         int bm,bv,br,bc;
         _addr_unpack(idx,bm,bv,br,bc);
	 if(((qmid.get_parity(bm)+qver.get_parity(bv))*qcol.get_parity(bc))%2 == 0){
            for(int im=0; im<blk.size(); im++){
               blk[im] = -blk[im];
            }
	 }
      }
   }
   return qt4;
}

// ZL20210510: application of time-reversal operation
template <typename Tm>
qtensor4<Tm> qtensor4<Tm>::K(const int nbar) const{
   const double fpo = (nbar%2==0)? 1.0 : -1.0;
   qtensor4<Tm> qt4(sym, qmid, qver, qrow, qcol); 
   for(int idx=0; idx<qt4._qblocks.size(); idx++){
      auto& blk = qt4._qblocks[idx];
      if(blk.size() == 0) continue;
      int bm,bv,br,bc;
      _addr_unpack(idx,bm,bv,br,bc);
      // qt4_new(c1c2)[l,r] = qt4(c1c2_bar)[l_bar,r_bar]^*
      const auto& blk1 = _qblocks[idx];
      int pm = qmid.get_parity(bm);
      int pv = qver.get_parity(bv);
      int pr = qrow.get_parity(br);
      int pc = qcol.get_parity(bc);
      int mdim = qmid.get_dim(bm);
      int vdim = qver.get_dim(bv);
      if(pm == 0 && pv == 0){
         for(int imv=0; imv<blk.size(); imv++){
            blk[imv] = fpo*kramers::time_reversal(blk1[imv], pr, pc);
         }
      }else if(pm == 0 && pv == 1){
	 assert(vdim%2 == 0);
	 int vdim2 = vdim/2;
	 for(int iv=0; iv<vdim2; iv++){
	    for(int im=0; im<mdim; im++){
               int imv  = iv*mdim + im;
	       int imv2 = (iv+vdim2)*mdim + im;
	       blk[imv] = fpo*kramers::time_reversal(blk1[imv2], pr, pc);
	    }
	 }
	 for(int iv=0; iv<vdim2; iv++){
	    for(int im=0; im<mdim; im++){
	       int imv  = (iv+vdim2)*mdim + im;
	       int imv2 = iv*mdim + im;
	       blk[imv] = -fpo*kramers::time_reversal(blk1[imv2], pr, pc);
	    }
	 }
      }else if(pm == 1 && pv == 0){
	 assert(mdim%2 == 0);
	 int mdim2 = mdim/2;
	 for(int iv=0; iv<vdim; iv++){
	    for(int im=0; im<mdim2; im++){
               int imv  = iv*mdim + im;
	       int imv2 = iv*mdim + (im+mdim2);
	       blk[imv] = fpo*kramers::time_reversal(blk1[imv2], pr, pc);
	    }
	    for(int im=0; im<mdim2; im++){
	       int imv  = iv*mdim + (im+mdim2);
	       int imv2 = iv*mdim + im;
	       blk[imv] = -fpo*kramers::time_reversal(blk1[imv2], pr, pc);
	    }
	 }
      }else if(pm == 1 && pv == 1){
	 assert(mdim%2 == 0 && vdim%2 == 0);
	 int mdim2 = mdim/2;
	 int vdim2 = vdim/2;
	 for(int iv=0; iv<vdim2; iv++){
	    for(int im=0; im<mdim2; im++){
               int imv  = iv*mdim + im;
	       int imv2 = (iv+vdim2)*mdim + (im+mdim2);
	       blk[imv] = fpo*kramers::time_reversal(blk1[imv2], pr, pc);
	    }
	    for(int im=0; im<mdim2; im++){
	       int imv  = iv*mdim + (im+mdim2);
	       int imv2 = (iv+vdim2)*mdim + im;
	       blk[imv] = -fpo*kramers::time_reversal(blk1[imv2], pr, pc);
	    }
	 }
	 for(int iv=0; iv<vdim2; iv++){
	    for(int im=0; im<mdim2; im++){
	       int imv  = (iv+vdim2)*mdim + im;
               int imv2 = iv*mdim + (im+mdim2);
	       blk[imv] = -fpo*kramers::time_reversal(blk1[imv2], pr, pc);
	    }
	    for(int im=0; im<mdim2; im++){
	       int imv  = (iv+vdim2)*mdim + (im+mdim2);
	       int imv2 = iv*mdim + im;
	       blk[imv] = fpo*kramers::time_reversal(blk1[imv2], pr, pc);
	    }
	 }
      } // (pm,pv)
   } // idx
   return qt4;
}

//--------------------------------------------------------------
// The following functions are the same as those for qtensor3 !
//--------------------------------------------------------------
     
// simple arithmetic operations
template <typename Tm>
qtensor4<Tm>& qtensor4<Tm>::operator +=(const qtensor4<Tm>& qt){
   assert(sym == qt.sym); // symmetry blocking must be the same
   for(int i=0; i<_qblocks.size(); i++){
      auto& blk = _qblocks[i];
      assert(blk.size() == qt._qblocks[i].size());
      if(blk.size() > 0){
         for(int m=0; m<blk.size(); m++){
            blk[m] += qt._qblocks[i][m];
         } // m
      }
   } // i
   return *this;
}

template <typename Tm>
qtensor4<Tm>& qtensor4<Tm>::operator -=(const qtensor4<Tm>& qt){
   assert(sym == qt.sym); // symmetry blocking must be the same
   for(int i=0; i<_qblocks.size(); i++){
      auto& blk = _qblocks[i];
      assert(blk.size() == qt._qblocks[i].size());
      if(blk.size() > 0){
         for(int m=0; m<blk.size(); m++){
  	    blk[m] -= qt._qblocks[i][m];
         } // m
      }
   } // i
   return *this;
}

template <typename Tm>
qtensor4<Tm>& qtensor4<Tm>::operator *=(const Tm fac){
   for(auto& blk : _qblocks){
      if(blk.size() > 0){ 
	 for(int m=0; m<blk.size(); m++){
	    blk[m] *= fac;
	 } // m
      }
   } // blk
   return *this;
}

// for Davidson algorithm
template <typename Tm>
double qtensor4<Tm>::normF() const{
   double sum = 0.0;
   for(const auto& blk : _qblocks){
      if(blk.size() > 0){
         for(int m=0; m<blk.size(); m++){
            sum += std::pow(linalg::normF(blk[m]),2);
         }
      }
   }
   return std::sqrt(sum);
}

template <typename Tm>
int qtensor4<Tm>::get_dim() const{
   int dim = 0;
   for(const auto& blk : _qblocks){
      if(blk.size() > 0) dim += blk.size()*blk[0].size(); // A[l,c1,c2,r] = A[c1*c2](l,r)
   }
   return dim;
}

template <typename Tm>
void qtensor4<Tm>::from_array(const Tm* array){
   int ioff = 0;
   for(int idx=0; idx<_qblocks.size(); idx++){
      auto& blk = _qblocks[idx];
      if(blk.size() == 0) continue;
      int size = blk[0].size();
      for(int im=0; im<blk.size(); im++){
         auto psta = array+ioff+im*size;
	 // copy from array to blk
	 std::copy(psta, psta+size, blk[im].data());
      }
      ioff += blk.size()*size;
   }
}

template <typename Tm>
void qtensor4<Tm>::to_array(Tm* array) const{
   int ioff = 0;
   for(int idx=0; idx<_qblocks.size(); idx++){
      auto& blk = _qblocks[idx];
      if(blk.size() == 0) continue;
      int size = blk[0].size();
      for(int im=0; im<blk.size(); im++){
         auto psta = array+ioff+im*size;
	 // copy from blk to array
	 std::copy(blk[im].data(), blk[im].data()+size, psta);
      }
      ioff += blk.size()*size;
   }
}

template <typename Tm>
void qtensor4<Tm>::add_noise(const double noise){
   for(auto& blk : _qblocks){
      if(blk.size() > 0){
         int rdim = blk[0].rows();
         int cdim = blk[0].cols();
         for(int im=0; im<blk.size(); im++){
            blk[im] += noise*linalg::random_matrix<Tm>(rdim,cdim);
         } // im
      }
   }
}

} // ctns

#endif
