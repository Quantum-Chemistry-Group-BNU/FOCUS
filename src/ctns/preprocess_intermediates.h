#ifndef PREPROCESS_INTERMEDIATES_H
#define PREPROCESS_INTERMEDIATES_H

namespace ctns{

template <typename Tm>
struct intermediates{
public:
   // constructors
   intermediates(){}
   ~intermediates(){ 
      _imap.clear();
      delete[] _data; 
   }
   // form intermediates
   void init(const oper_dictmap<Tm>& qops_dict,
	     const symbolic_task<Tm>& H_formulae,
	     const bool debug=false);
   // helpers
   int count() const{ return _count; };
   int size() const{ return _size; };
public:
   std::map<std::pair<int,int>,int> _imap;
   size_t _count = 0, _size = 0;
   Tm* _data = nullptr;
};

template <typename Tm>
void intermediates<Tm>::init(const oper_dictmap<Tm>& qops_dict,
	     		     const symbolic_task<Tm>& H_formulae,
	     		     const bool debug){
   auto t0 = tools::get_time();
   if(debug){
      std::cout << "intermeidates<Tm>::init"
	        << " size(formulae)=" << H_formulae.size()
	        << std::endl;
   }
   // count the size of intermediates
   int hsize = H_formulae.size();
   std::vector<Tm> coeffs(hsize,1.0);
   for(int it=0; it<hsize; it++){
      const auto& HTerm = H_formulae.tasks[it];
      if(debug) std::cout << "it=" << it << " formulae=" << HTerm << std::endl;
      for(int idx=HTerm.size()-1; idx>=0; idx--){
         const auto& sop = HTerm.terms[idx];
         int len = sop.size();
         if(len == 1){
            coeffs[it] *= sop.sums[0].first; 
         }else{
            // define intermediate operators
	    _count += 1;
            _imap[std::make_pair(it,idx)] = _size;
            const auto& sop0 = sop.sums[0].second;
            const auto& block = sop0.block;
            const auto& label  = sop0.label;
            const auto& qops = qops_dict.at(block);
            _size += qops(label).at(sop0.index).size();
         }
      }
   } // it
   if(debug){
      std::cout << "no. of intermediate operators = " << _count << std::endl;
      std::cout << "size of intermediate operators = " << _size << std::endl;
   }
   // allocate memory
   _data = new Tm[_size];
   memset(_data, 0, _size*sizeof(Tm));
   // form intermediates via AXPY
   std::vector<std::pair<int,int>> _index(_count);
   int idx = 0;
   for(const auto& pr : _imap){
      _index[idx] = pr.first;
      idx++;
   }
/*
#ifdef _OPENMP
   #pragma omp parallel for schedule(dynamic)
#endif
*/
   for(int idx=0; idx<_count; idx++){
      const auto& item = _index[idx];
      int i = item.first;
      int j = item.second;
      const auto& sop = H_formulae.tasks[i].terms[j];
      std::cout << "idx=" << idx 
	        << " i,j=" << i << "," << j 
		<< " sop=" << sop << std::endl;
   } // idx 
 
/*
         const auto& index0 = sop0.index;
         const auto& parity = sop0.parity;
         const auto& dagger = sop0.dagger;
         // form operator
         auto optmp = symbolic_sum_oper(qops, sop, label, dagger, workspace);



stensor2<Tm> symbolic_sum_oper(const oper_dict<Tm>& qops,
			       const symbolic_sum<Tm>& sop,
	          	       const char& label,
			       const bool& dagger,
			       Tm* workspace){
   int len = sop.size();
   // we assume the rest of terms have the same label/dagger
   auto wt0 = sop.sums[0].first;
   const auto& sop0 = sop.sums[0].second;
   int index0 = sop0.index;
   int nbar0  = sop0.nbar;
   // form opsum = wt0*op0 + wt1*op1 + ...
   const auto& op0 = qops(label).at(index0);
   if(dagger) wt0 = tools::conjugate(wt0);
   stensor2<Tm> optmp;
   optmp.init(op0.info,false);
   optmp.setup_data(workspace);
   optmp.clear();
   if(nbar0 == 0){
      linalg::xaxpy(op0.size(), wt0, op0.data(), optmp.data());
   }else{
      auto op0k = op0.K(nbar0);
      linalg::xaxpy(op0.size(), wt0, op0k.data(), optmp.data());
   }
   for(int k=1; k<len; k++){
      auto wtk = sop.sums[k].first;
      const auto& sopk = sop.sums[k].second;
      int indexk = sopk.index;
      int nbark  = sopk.nbar;
      const auto& opk = qops(label).at(indexk);
      if(dagger) wtk = tools::conjugate(wtk);
      if(nbark == 0){
         linalg::xaxpy(opk.size(), wtk, opk.data(), optmp.data());
      }else{
         auto opkk = opk.K(nbark);
         linalg::xaxpy(opk.size(), wtk, opkk.data(), optmp.data());
      }
   } // k
   return optmp;
}
*/
   if(debug){
      auto t1 = tools::get_time();
      tools::timing("intermeidates<Tm>::init", t0, t1);
   }
}


} // ctns

#endif
