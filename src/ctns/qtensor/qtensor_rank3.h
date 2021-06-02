#ifndef QTENSOR_RANK3_H
#define QTENSOR_RANK3_H

namespace ctns{

template <typename Tm>
struct qtensor2;
template <typename Tm>
struct qtensor4;

// rank-3 tensor: (mid,row,col)
template <typename Tm>
struct qtensor3{
   private:
      // serialize
      friend class boost::serialization::access;
      template<class Archive>
      void serialize(Archive & ar, const unsigned int version){
	 ar & dir & sym & qmid & qrow & qcol 
	    & _mids & _rows & _cols & _qblocks;     
      }
      // conservation pattern determined by dir
      inline bool _ifconserve(const int bm, const int br, const int bc) const{
	 auto qsum = -sym; // default in
	 qsum += dir[0] ? qmid.get_sym(bm) : -qmid.get_sym(bm);
	 qsum += dir[1] ? qrow.get_sym(br) : -qrow.get_sym(br);
	 qsum += dir[2] ? qcol.get_sym(bc) : -qcol.get_sym(bc);
	 return qsum == qsym();
      }
      // address for storaging block data 
      inline int _addr(const int bm, const int br, const int bc) const{
         return bm*_rows*_cols + br*_cols + bc;
      }
      inline void _addr_unpack(const int idx, int& bm, int& br, int& bc) const{
 	 bc = idx%_cols;
	 int mr = idx/_cols;
	 bm = mr/_rows;	    
         br = mr%_rows;
      }
   public:
      // constructor
      qtensor3(){}
      qtensor3(const qsym& sym1, const qbond& qmid1, const qbond& qrow1, const qbond& qcol1, 
	       const std::vector<bool> dir1={1,0,1});
      void init(const qsym& sym1, const qbond& qmid1, const qbond& qrow1, const qbond& qcol1, 
		const std::vector<bool> dir1={1,0,1});
      // helpers
      inline int mids() const{ return _mids; }
      inline int rows() const{ return _rows; }
      inline int cols() const{ return _cols; }
      // access
      std::vector<linalg::matrix<Tm>>& operator ()(const int bm, const int br, const int bc){
         return _qblocks[_addr(bm,br,bc)];
      }
      const std::vector<linalg::matrix<Tm>>& operator ()(const int bm, const int br, const int bc) const{
         return _qblocks[_addr(bm,br,bc)];
      }
      // print
      size_t get_size() const;
      void print_size(const std::string name) const;
      void print(const std::string name, const int level=0) const;
      // fix middle index (bm,im) - bm-th block, im-idx - composite index!
      qtensor2<Tm> fix_mid(const std::pair<int,int> mdx) const;
      // deal with fermionic sign in fermionic direct product
      qtensor3<Tm> mid_signed(const double fac=1.0) const; // wf[lcr](-1)^{p(c)}
      qtensor3<Tm> row_signed(const double fac=1.0) const; // wf[lcr](-1)^{p(l)}
      qtensor3<Tm> permCR_signed() const; // wf[lcr]->wf[lcr]*(-1)^{p[c]*p[r]}
      // ZL20210413: application of time-reversal operation
      qtensor3<Tm> K(const int nbar=0) const;
      // simple arithmetic operations
      qtensor3<Tm> operator -() const; 
      qtensor3<Tm>& operator +=(const qtensor3<Tm>& qt);
      qtensor3<Tm>& operator -=(const qtensor3<Tm>& qt);
      qtensor3<Tm>& operator *=(const Tm fac);
      friend qtensor3<Tm> operator +(const qtensor3<Tm>& qta, const qtensor3<Tm>& qtb){
         qtensor3<Tm> qt3 = qta;
         qt3 += qtb;
         return qt3;
      }
      friend qtensor3<Tm> operator -(const qtensor3<Tm>& qta, const qtensor3<Tm>& qtb){
         qtensor3<Tm> qt3 = qta;
         qt3 -= qtb;
         return qt3;
      }
      friend qtensor3<Tm> operator *(const Tm fac, const qtensor3<Tm>& qt){
         qtensor3<Tm> qt3 = qt; 
         qt3 *= fac;
         return qt3;
      }
      friend qtensor3<Tm> operator *(const qtensor3<Tm>& qt, const Tm fac){
         return fac*qt;
      }
      // for Davidson algorithm
      double normF() const;
      int get_dim() const;
      void from_array(const Tm* array);
      void to_array(Tm* array) const;
      void add_noise(const double noise);
      // for decimation
      inline qproduct dpt_lc() const{ return qmerge(qrow,qmid); }
      inline qproduct dpt_cr() const{ return qmerge(qmid,qcol); }
      inline qproduct dpt_lr() const{ return qmerge(qrow,qcol); }
      // reshape: merge
      qtensor2<Tm> merge_lc() const{ 
	 auto qprod = dpt_lc();
	 return merge_qt3_qt2_lc(*this, qprod.first, qprod.second);
      }
      qtensor2<Tm> merge_cr() const{
	 auto qprod = dpt_cr(); 
	 return merge_qt3_qt2_cr(*this, qprod.first, qprod.second);
      }
      qtensor2<Tm> merge_lr() const{
	 auto qprod = dpt_lr();  
	 return merge_qt3_qt2_lr(*this, qprod.first, qprod.second);
      }
      // superblock rdm
      inline qtensor2<Tm> get_rdm(const std::string& superblock) const{
	 qtensor2<Tm> rdm;
         if(superblock == "lc"){
            rdm = (this->merge_lc()).get_rdm_row();
         }else if(superblock == "cr"){
            rdm = (this->merge_cr()).get_rdm_col();
         }else if(superblock == "lr"){
            // Need to first bring two dimensions adjacent to each other before merge!
   	    rdm = (this->permCR_signed()).merge_lr().get_rdm_row();
         }
	 return rdm;
      }
      // reshape: split
      qtensor4<Tm> split_lc1(const qbond& qlx, const qbond& qc1, const qdpt& dpt) const{
	 return split_qt4_qt3_lc1(*this, qlx, qc1, dpt);
      }
      qtensor4<Tm> split_c2r(const qbond& qc2, const qbond& qrx, const qdpt& dpt) const{
	 return split_qt4_qt3_c2r(*this, qc2, qrx, dpt); 
      }
      qtensor4<Tm> split_c1c2(const qbond& qc1, const qbond& qc2, const qdpt& dpt) const{
	 return split_qt4_qt3_c1c2(*this, qc1, qc2, dpt);
      }
      qtensor4<Tm> split_lr(const qbond& qlx, const qbond& qrx, const qdpt& dpt) const{
	 return split_qt4_qt3_lr(*this, qlx, qrx, dpt);
      }
   public:
      std::vector<bool> dir = {1,0,1}; // =0,in; =1,out; {mid,row,col}
      				       // {1,0,1} - RCF (default)
      				       // {1,1,0} - LCF
				       // {0,1,1} - CCF (for internal upward node)
				       // {1,1,1} - WF
      qsym sym; // in
      qbond qmid, qrow, qcol; 
   private:  
      int _mids, _rows, _cols; 
      std::vector<std::vector<linalg::matrix<Tm>>> _qblocks;
};

template <typename Tm>
void qtensor3<Tm>::init(const qsym& sym1, const qbond& qmid1, const qbond& qrow1, const qbond& qcol1, 
			const std::vector<bool> dir1){ 
   sym = sym1;
   qmid = qmid1;
   qrow = qrow1;
   qcol = qcol1;
   dir = dir1;
   _mids = qmid.size();
   _rows = qrow.size();
   _cols = qcol.size();
   _qblocks.resize(_mids*_rows*_cols); 
   for(int bm=0; bm<_mids; bm++){
      for(int br=0; br<_rows; br++){
         for(int bc=0; bc<_cols; bc++){
  	    if(not _ifconserve(bm,br,bc)) continue;
  	    int mdim = qmid.get_dim(bm);
  	    int rdim = qrow.get_dim(br);
  	    int cdim = qcol.get_dim(bc);
  	    int addr = _addr(bm,br,bc);
  	    _qblocks[addr].resize(mdim);
  	    for(int im=0; im<mdim; im++){
  	       _qblocks[addr][im].resize(rdim,cdim);
  	    }
         } // bc
      } // br
   } // bm
}

template <typename Tm>
qtensor3<Tm>::qtensor3(const qsym& sym1, const qbond& qmid1, const qbond& qrow1, const qbond& qcol1, 
		       const std::vector<bool> dir1){
   this->init(sym1, qmid1, qrow1, qcol1, dir1);
}

template <typename Tm>
void qtensor3<Tm>::print(const std::string name, const int level) const{
   std::cout << "\nqtensor3: " << name << " sym=" << sym;
   std::cout << " dir=";
   for(auto b : dir) std::cout << b << " ";
   std::cout << std::endl;
   qmid.print("qmid");
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
            int bm,br,bc;
            _addr_unpack(idx,bm,br,bc);
            std::cout << "idx=" << idx 
      	              << " block[" << qmid.get_sym(bm) << "," 
                      << qrow.get_sym(br) << "," << qcol.get_sym(bc) << "]" 
                      << " size=" << blk.size() 
                      << " rows,cols=(" << blk[0].rows() << "," << blk[0].cols() << ")" 
                      << std::endl; 
            if(level >= 2){
               for(int im=0; im<blk.size(); im++){
                  blk[im].print("mat"+std::to_string(im));
               }
            } // level=2
         } // level>=1
      }
   } // idx
   std::cout << "total no. of nonzero blocks=" << nnz << std::endl;
}

template <typename Tm>
size_t qtensor3<Tm>::get_size() const{
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
void qtensor3<Tm>::print_size(const std::string name) const{
   size_t size = this->get_size(); 
   std::cout << "qtensor3: " << name << " size=" << size
             << " sizeMB=" << tools::sizeMB<Tm>(size) 
             << std::endl;
}

// fix middle index (bm,im) - bm-th block, im-idx - composite index!
template <typename Tm>
qtensor2<Tm> qtensor3<Tm>::fix_mid(const std::pair<int,int> mdx) const{
   int bm = mdx.first, im = mdx.second;   
   auto symIn = dir[0] ? sym-qmid.get_sym(bm) : sym+qmid.get_sym(bm);
   qtensor2<Tm> qt2(symIn, qrow, qcol, {dir[1],dir[2]});
   for(int br=0; br<_rows; br++){
      for(int bc=0; bc<_cols; bc++){
         if(not _ifconserve(bm,br,bc)) continue;
         int addr = _addr(bm,br,bc);
         int mdim = qmid.get_dim(bm);
         assert(im < mdim);
         qt2(br,bc) = _qblocks[addr][im];
      } // bc
   } // br
   return qt2;
}

// deal with fermionic sign in fermionic direct product
template <typename Tm>
qtensor3<Tm> qtensor3<Tm>::mid_signed(const double fac) const{
   qtensor3<Tm> qt3 = *this;
   for(int idx=0; idx<qt3._qblocks.size(); idx++){
      auto& blk = qt3._qblocks[idx];
      if(blk.size() > 0){
         int bm,br,bc;
         _addr_unpack(idx,bm,br,bc);
         double fac2 = (qmid.get_parity(bm)==0)? fac : -fac;
         for(int im=0; im<blk.size(); im++){
            blk[im] *= fac2;
         }
      }
   }
   return qt3;
}

template <typename Tm>
qtensor3<Tm> qtensor3<Tm>::row_signed(const double fac) const{
   qtensor3<Tm> qt3 = *this;
   for(int idx=0; idx<qt3._qblocks.size(); idx++){
      auto& blk = qt3._qblocks[idx];
      if(blk.size() > 0){
         int bm,br,bc;
         _addr_unpack(idx,bm,br,bc);
         double fac2 = (qrow.get_parity(br)==0)? fac : -fac;
         for(int im=0; im<blk.size(); im++){
            blk[im] *= fac2;
         }
      }
   }
   return qt3;
}

template <typename Tm>
qtensor3<Tm> qtensor3<Tm>::permCR_signed() const{
   qtensor3<Tm> qt3 = *this;
   for(int idx=0; idx<qt3._qblocks.size(); idx++){
      auto& blk = qt3._qblocks[idx];
      if(blk.size() > 0){
         int bm,br,bc;
         _addr_unpack(idx,bm,br,bc);
	 if(qmid.get_parity(bm)*qcol.get_parity(bc) == 1){
            for(int im=0; im<blk.size(); im++){
               blk[im] = -blk[im];
            }
	 }
      }
   }
   return qt3;
}

// ZL20210413: application of time-reversal operation
template <typename Tm>
qtensor3<Tm> qtensor3<Tm>::K(const int nbar) const{
   const double fpo = (nbar%2==0)? 1.0 : -1.0;
   qtensor3<Tm> qt3(sym, qmid, qrow, qcol, dir); // assuming it only works for (N), no flip of symmetry is necessary
   for(int idx=0; idx<qt3._qblocks.size(); idx++){
      auto& blk = qt3._qblocks[idx];
      if(blk.size() == 0) continue;
      int bm,br,bc;
      _addr_unpack(idx,bm,br,bc);
      // qt3[c](l,r) = blk[bar{c}](bar{l},bar{r})^*
      const auto& blk1 = _qblocks[idx];
      int pm = qmid.get_parity(bm);
      int pr = qrow.get_parity(br);
      int pc = qcol.get_parity(bc);
      if(pm == 0){
         // c[e]
         for(int im=0; im<blk.size(); im++){
            blk[im] = fpo*time_reversal(blk1[im], pr, pc);
         }
      }else{
         assert(blk.size()%2 == 0);
         int dm2 = blk.size()/2;
         // c[o],c[\bar{o}]
         for(int im=0; im<dm2; im++){
            blk[im] = fpo*time_reversal(blk1[im+dm2], pr, pc);
         }
         for(int im=0; im<dm2; im++){
            blk[im+dm2] = -fpo*time_reversal(blk1[im], pr, pc);
         }
      } // pm
   } // idx
   return qt3;
}

// simple arithmetic operations
template <typename Tm>
qtensor3<Tm> qtensor3<Tm>::operator -() const{
   qtensor3<Tm> qt3 = *this;
   for(auto& blk : qt3._qblocks){
      if(blk.size() > 0){
         for(int im=0; im<blk.size(); im++){
            blk[im] *= -1;
         } // im
      }
   }
   return qt3;
}

template <typename Tm>
qtensor3<Tm>& qtensor3<Tm>::operator +=(const qtensor3<Tm>& qt){
   assert(dir == qt.dir); // direction must be the same
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
qtensor3<Tm>& qtensor3<Tm>::operator -=(const qtensor3<Tm>& qt){
   assert(dir == qt.dir); // direction must be the same
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
qtensor3<Tm>& qtensor3<Tm>::operator *=(const Tm fac){
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
double qtensor3<Tm>::normF() const{
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
int qtensor3<Tm>::get_dim() const{
   int dim = 0;
   for(const auto& blk : _qblocks){
      if(blk.size() > 0) dim += blk.size()*blk[0].size(); // A[l,c,r] = A[c](l,r)
   }
   return dim;
}

template <typename Tm>
void qtensor3<Tm>::from_array(const Tm* array){
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
void qtensor3<Tm>::to_array(Tm* array) const{
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
void qtensor3<Tm>::add_noise(const double noise){
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
