#include "../core/linalg.h"
#include "../core/tools.h"
#include "tns_decimation.h"

using namespace std;
using namespace tns;
using namespace linalg;

// wf[L,R] = U[L,l]*sl*Vt[l,R]
qtensor2 tns::decimation_row(const qtensor2& wf,
			     const int Dcut){
   bool debug = true;
   cout << "\ntns::decimation_row Dcut=" << Dcut << endl;
   const auto& qrow = wf.qrow;
   const auto& qcol = wf.qcol;
   // 1. compute reduced basis
   int dr = wf.get_dim_row();
   map<int,qsym> idx2qsym; 
   vector<double> sig2all(dr);
   map<qsym,matrix> rbasis;
   int idx = 0, ioff = 0;
   for(const auto& pr : qrow){
      auto qr = pr.first;
      int rdim = pr.second;
      matrix rdm(rdim,rdim);
      for(const auto& pc : qcol){
	 auto qc = pc.first;
	 const auto& blk = wf.qblocks.at(make_pair(qr,qc)); 
	 if(blk.size() == 0) continue;
	 rdm += dgemm("N","N",blk,blk.T()); 
      }
      // compute renormalized basis
      vector<double> sig2(rdim);
      eigen_solver(rdm, sig2, 1);
      // save
      for(int i=0; i<rdim; i++){
         idx2qsym[ioff+i] = qr;
      }
      copy(sig2.begin(), sig2.end(), &sig2all[ioff]);
      rbasis[qr] = rdm;
      ioff += rdim;
      if(debug){
	 cout << "idx=" << idx << " qr=" << qr << " rdim=" << rdim << endl;
	 cout << "    sig2=";
	 for(auto s : sig2) cout << s << " ";
	 cout << endl;
	 idx++;
      }
   }
   // 2. select important ones
   auto index = tools::sort_index(sig2all, 1);
   int nres = min(Dcut,dr); 
   qsym_space qres;
   double sum = 0.0;
   for(int i=0; i<nres; i++){
      int idx = index[i];
      auto q = idx2qsym[idx];
      auto it = qres.find(q);
      if(it == qres.end()){
         qres[q] = 1;
      }else{
	 qres[q] += 1;
      }
      sum += sig2all[idx];
      if(debug){
	if(i==0) cout << "sorted sig2: nres=" << nres << endl;     
	cout << "i=" << i << " q=" << q << " idx=" << qres[q]-1 
             << " sig2=" << sig2all[idx] 
	     << " accum=" << sum << endl;
      }
   }
   double dwt = 1.0-sum;
   if(debug) cout << "discarded weight=" << dwt << endl;
   // 3. form qt2
   idx = 0;
   qtensor2 qt2(qsym(0,0), qrow, qres);
   for(auto& p : qt2.qblocks){
      auto q = p.first;
      auto& blk = p.second;
      if(blk.size() > 0){
         assert(q.first == q.second);
	 auto qd = q.first;
	 auto& rbas = rbasis[qd];
	 copy(rbas.data(), rbas.data()+blk.size(), blk.data());
	 if(debug){
	    if(idx == 0) cout << "reduced basis:" << endl;
	    cout << "idx=" << idx << " qr=" << qd << " shape=(" 
		 << blk.rows() << "," << blk.cols() << ")"
		 << endl;
	    idx++; 
	 }
      }
   }
   return qt2;
}

qtensor2 tns::decimation_col(const qtensor2& wf,
			     const int Dcut){
   cout << "tns::decimation_col Dcut=" << Dcut << endl;
   auto wfT = wf.T();
   auto Ubas = decimation_row(wfT, Dcut);
   return Ubas.T();
}
