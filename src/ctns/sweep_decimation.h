#ifndef SWEEP_DECIMATION_H
#define SWEEP_DECIMATION_H

#include "sweep_prdm.h"

namespace ctns{

const double thresh_noise = 1.e-10;
extern const double thresh_noise;

const double thresh_sig2 = 1.e-20;
extern const double thresh_sig2;

// wf[L,R] = U[L,l]*sl*Vh[l,R]
template <typename Tm>
qtensor2<Tm> decimation_row_nkr(const qtensor2<Tm>& rdm,
			        const int dcut,
			        double& dwt,
			        int& deff){
   const bool debug = false;
   if(debug) std::cout << "ctns::decimation_row_nkr dcut=" << dcut << std::endl;
   // 0. normalize before diagonalization
   Tm rfac = 1.0/rdm.trace();
   // 1. compute reduced basis
   std::map<int,int> idx2sector; 
   std::vector<double> sig2all;
   std::map<int,linalg::matrix<Tm>> rbasis;
   auto offset = rdm.qrow.get_offset();
   for(int br=0; br<rdm.rows(); br++){
      const auto& blk = rdm(br,br);
      if(blk.size() == 0) continue;
      // compute renormalized basis
      int rdim = rdm.qrow.get_dim(br);
      std::vector<double> sig2(rdim);
      linalg::matrix<Tm> rbas(rdim,rdim);
      auto rblk = rfac*blk;
      linalg::eig_solver(rblk, sig2, rbas, 1);
      // save
      std::copy(sig2.begin(), sig2.end(), std::back_inserter(sig2all));
      rbasis[br] = rbas;
      int ioff = offset[br];
      for(int i=0; i<rdim; i++){
	 idx2sector[ioff+i] = br;
      }
      if(debug){
	 if(br == 0) std::cout << "diagonalization of rdm for each symmetry sector:" << std::endl;
	 std::cout << " br=" << br << " qr=" << rdm.qrow.get_sym(br) << " rdim=" << rdim << " sig2=";
	 for(auto s : sig2) std::cout << s << " ";
	 std::cout << std::endl;
      }
   }
   // 2. select important sig2
   auto index = tools::sort_index(sig2all, 1);
   std::map<int,std::pair<int,double>> kept; // br->(dim,wt)
   deff = 0;
   double sum = 0.0;
   double SvN = 0.0;
   for(int i=0; i<sig2all.size(); i++){
      if(i >= dcut) break; // discard rest
      int idx = index[i];
      if(sig2all[idx] < thresh_sig2) continue; // discard negative weights
      int br = idx2sector[idx];
      auto it = kept.find(br);
      if(it == kept.end()){
         kept[br].first = 1;
	 kept[br].second = sig2all[idx];
      }else{
	 kept[br].first += 1;
	 kept[br].second += sig2all[idx];
      }
      deff += 1;
      sum += sig2all[idx];
      SvN += -sig2all[idx]*std::log2(sig2all[idx]);
      if(debug){
	if(i == 0) std::cout << "sorted sig2:" << std::endl;     
	std::cout << " i=" << i << " (br,ith)=" << br << "," << kept[br].first-1 
             	  << " sig2=" << sig2all[idx] 
	          << " accum=" << sum << std::endl;
      }
   }
   dwt = 1.0-sum;
   std::cout << "decimation summary: reduce from " << sig2all.size() << " to " << deff
     	     << "  dwt=" << dwt << "  SvN=" << SvN << std::endl;
   // 3. construct qbond and qt2 by assembling blocks
   std::vector<int> br_matched;
   std::vector<std::pair<qsym,int>> dims;
   for(const auto& p : kept){
      const auto& br = p.first;
      const auto& dim = p.second.first;
      const auto& wt = p.second.second;
      br_matched.push_back(br);
      dims.push_back(std::make_pair(rdm.qrow.get_sym(br),dim));
      if(debug){
         std::cout << " br=" << p.first << " dim=" << dim 
           	   << " wt=" << wt << std::endl;
      }
   }
   qbond qkept(dims);
   qtensor2<Tm> qt2(qsym(), rdm.qrow, qkept);
   for(int bc=0; bc<qkept.size(); bc++){
      int br = br_matched[bc];
      auto& blk = qt2(br,bc); 
      auto& rbas = rbasis[br];
      std::copy(rbas.data(), rbas.data()+blk.size(), blk.data());
      if(debug){
         assert(rdm.qrow.get_sym(br) == qt2.qcol.get_sym(bc));
         if(bc == 0) std::cout << "reduced basis:" << std::endl;
         std::cout << " (br,bc)=" << br << "," << bc 
		   << " qsym=" << qt2.qcol.get_sym(bc)
		   << " shape=(" << blk.rows() << "," << blk.cols() << ")"
     	           << std::endl;
      }
   } // bc
   return qt2;
}

template <typename Tm>
qtensor2<Tm> decimation_row_kr(const qtensor2<Tm>& rdm,
			       const int dcut,
			       double& dwt,
			       int& deff,
 			       const qbond& qs1,
 			       const qbond& qs2,
 			       const qdpt& dpt){
   const bool debug = false;
   if(debug) std::cout << "ctns::decimation_row_kr dcut=" << dcut << std::endl;
   // 0. normalize before diagonalization
   Tm rfac = 1.0/rdm.trace();
   // 1. compute reduced basis
   std::map<int,int> idx2sector; 
   std::vector<double> sig2all;
   std::map<int,linalg::matrix<Tm>> rbasis;
   auto offset = rdm.qrow.get_offset();
   for(int br=0; br<rdm.rows(); br++){
      const auto& blk = rdm(br,br);
      if(blk.size() == 0) continue;
      // compute renormalized basis
      int rdim = rdm.qrow.get_dim(br);
      std::vector<double> sig2(rdim);
      linalg::matrix<Tm> rbas(rdim,rdim);
      auto rblk = rfac*blk;
      linalg::eig_solver(rblk, sig2, rbas, 1);

/*
      //
      // KRS-adapted decimation
      //
      auto qr = rdm.qrow.get_sym(br);

      std::cout << "\nqr=" << qr << std::endl;
      const auto& comb = dpt.at(qr);
      for(int i=0; i<comb.size(); i++){
         int b1 = std::get<0>(comb[i]);
         int b2 = std::get<1>(comb[i]);
         int ioff = std::get<2>(comb[i]);
         std::cout << "i=" << i << " " 
             << qs1.get_sym(b1) << " " << qs1.get_dim(b1) << " "
             << qs2.get_sym(b2) << " " << qs2.get_dim(b2) 
             << std::endl; 
      }

      std::vector<double> eigs(rdim); 
      linalg::matrix<std::complex<double>> U(rdim,rdim);

      if(qr.parity() == 1){

	 // o = {|le,ro>,|lo,re>}
	 std::vector<int> idx_up, idx_dw;
         std::cout << "qr=" << qr << std::endl;
	 int ioff = 0;
         const auto& comb = dpt.at(qr);
         for(int i=0; i<comb.size(); i++){
            int b1 = std::get<0>(comb[i]);
            int b2 = std::get<1>(comb[i]);
            int ioff = std::get<2>(comb[i]);
            std::cout << "i=" << i << " " 
                << qs1.get_sym(b1) << " " << qs1.get_dim(b1) << " "
                << qs2.get_sym(b2) << " " << qs2.get_dim(b2) 
                << std::endl;
	    auto q1 = qs1.get_sym(b1);
	    auto q2 = qs2.get_sym(b2);
	    int  d1 = qs1.get_dim(b1);
	    int  d2 = qs2.get_dim(b2);

	    // |le,ro> 
	    if(q1.parity() == 0 && q2.parity() == 1){

	       assert(d2%2 == 0);
	       for(int i2=0; i2<d2/2; i2++){
	          for(int i1=0; i1<d1; i1++){
		     int idxA = ioff + i2*d1 + i1; // |le,ro>
		     idx_up.push_back(idxA);
		  }
	       }
	       for(int i2=0; i2<d2/2; i2++){
	          for(int i1=0; i1<d1; i1++){
		     int idxB = ioff + (i2+d2/2)*d1 + i1; // |le,ro_bar>
		     idx_dw.push_back(idxB);
		  }
	       }

	    // |lo,re>   
	    }else if(q1.parity() == 1 && q2.parity() == 0){
	       
	       assert(d1%2 == 0);
	       for(int i2=0; i2<d2; i2++){
	          for(int i1=0; i1<d1/2; i1++){
	 	     int idxA = ioff + i2*d1 + i1; 
		     idx_up.push_back(idxA);
		  }
		  for(int i1=0; i1<d1/2; i1++){
	   	     int idxB = ioff + i2*d1 + i1 + d1/2;
		     idx_dw.push_back(idxB);
		  }
	       }

	    }else{
	       std::cout << "error: no such combination of parities!" << std::endl;
	       std::cout << "q1p,q2p=" << q1.parity() << "," << q2.parity() << std::endl;
	       exit(1);
	    } 
	       
	    ioff += d1*d2;
         }

	 std::cout << "idx_up: ";
	 for(auto p : idx_up) std::cout << p << " ";
	 std::cout << std::endl;
	 std::cout << "idx_dw: ";
	 for(auto p : idx_dw) std::cout << p << " ";
	 std::cout << std::endl;

	 std::vector<int> idx_all(idx_up);
	 idx_all.insert(idx_all.end(), idx_dw.begin(), idx_dw.end());

	 rblk.print("rblk");

	 auto rhor = rblk.reorder_rowcol(idx_all,idx_all);
 	 rhor.print("rhor");

	 int dim1 = idx_up.size();
 	 std::vector<int> partition = {dim1,dim1};
         blockMatrix<std::complex<double>> rmat(partition,partition);
	 rmat = rhor;
	 // Kramers projection
	 auto A = 0.5*(rmat(0,0) + rmat(1,1).conj());
	 auto B = 0.5*(rmat(0,1) - rmat(1,0).conj()); 
	 rmat(0,0) = A;
	 rmat(0,1) = B;
	 rmat(1,0) = -B.conj();
	 rmat(1,1) = A.conj();
	 rhor = rmat.to_matrix();
	 rhor.print("rhor_projected");
	 // TRS-preserving diagonalization (only half eigs are output) 
	 zquatev(rhor,eigs,U,1);
	 std::copy(eigs.begin(), eigs.begin()+dim1, eigs.begin()+dim1); // duplicate eigs!
	 
	 std::cout << "eigs: ";
	 for(auto p : eigs) std::cout << p << " ";
	 std::cout << std::endl;

	 U = U.reorder_row(idx_all,1);

      }else{

	 // o = {|le,re>,|lo,ro>}
	 std::vector<int> idx_up, idx_dw, idx_ee;
	 std::vector<double> phases;
         std::cout << "qr=" << qr << std::endl;
	 int ioff = 0;
         const auto& comb = dpt.at(qr);
         for(int i=0; i<comb.size(); i++){
            int b1 = std::get<0>(comb[i]);
            int b2 = std::get<1>(comb[i]);
            int ioff = std::get<2>(comb[i]);
            std::cout << "i=" << i << " " 
                << qs1.get_sym(b1) << " " << qs1.get_dim(b1) << " "
                << qs2.get_sym(b2) << " " << qs2.get_dim(b2) 
                << std::endl;
	    auto q1 = qs1.get_sym(b1);
	    auto q2 = qs2.get_sym(b2);
	    int  d1 = qs1.get_dim(b1);
	    int  d2 = qs2.get_dim(b2);
	    // |le,re> 
	    if(q1.parity() == 0 && q2.parity() == 0){
	       
	       for(int i2=0; i2<d2; i2++){
	          for(int i1=0; i1<d1; i1++){
		     int idx = ioff + i2*d1 + i1;
		     idx_ee.push_back(idx);
		  }
	       }

	    // |lo,ro> = {|lo,ro>,|lo_bar,ro>} + {|lo_bar,ro_bar>,|lo,ro_bar>}
	    }else if(q1.parity() == 1 && q2.parity() == 1){
	       
	       assert(d1%2 == 0 & d2%2 == 0);
	       for(int i2=0; i2<d2/2; i2++){
	          for(int i1=0; i1<d1; i1++){
		     if(i1<d1/2){		  
		        int idxA = ioff + i2*d1 + i1; // |lo,ro> 
		        idx_up.push_back(idxA);
		        int idxB = ioff + (i2+d2/2)*d1 + (i1+d1/2); // |lo_bar,ro_bar>
	   	        idx_dw.push_back(idxB);
		        phases.push_back(1.0);
		     }else{
		        int idxA = ioff + i2*d1 + i1; // |lo_bar,ro> 
		        idx_up.push_back(idxA);
		        int idxB = ioff + (i2+d2/2)*d1 + (i1-d1/2); // |lo,ro_bar>
	   	        idx_dw.push_back(idxB);
		        phases.push_back(-1.0);
		     }
		  }
	       }

	    }else{
	       std::cout << "error: no such combination of parities!" << std::endl;
	       std::cout << "q1p,q2p=" << q1.parity() << "," << q2.parity() << std::endl;
	       exit(1);
	    }

	    ioff += d1*d2;
         }

	 std::cout << "idx_up: ";
	 for(auto p : idx_up) std::cout << p << " ";
	 std::cout << std::endl;
	 std::cout << "idx_dw: ";
	 for(auto p : idx_dw) std::cout << p << " ";
	 std::cout << std::endl;
	 std::cout << "idx_ee: ";
	 for(auto p : idx_ee) std::cout << p << " ";
	 std::cout << std::endl;

	 std::vector<int> idx_all(idx_up);
	 idx_all.insert(idx_all.end(), idx_dw.begin(), idx_dw.end());
	 idx_all.insert(idx_all.end(), idx_ee.begin(), idx_ee.end());

	 rblk.print("rblk");

	 auto rhor = rblk.reorder_rowcol(idx_all,idx_all);
 	 rhor.print("rhor");

	 int dim1 = idx_up.size();
	 int dim0 = idx_ee.size();
	 int dim = 2*dim1+dim0;

	 std::vector<int> partition = {dim1,dim1,dim0};
	 blockMatrix<std::complex<double>> rmat(partition,partition);
	 rmat = rhor;
	 // col-1 & row-1
	 rmat(0,1).colscale(phases);
	 rmat(1,1).colscale(phases);
	 rmat(2,1).colscale(phases);
	 rmat(1,0).rowscale(phases);
         rmat(1,1).rowscale(phases);
         rmat(1,2).rowscale(phases);
	 // Kramers projection
	 auto A = 0.5*(rmat(0,0) + rmat(1,1).conj());
	 auto B = 0.5*(rmat(0,1) + rmat(1,0).conj());
	 auto C = 0.5*(rmat(0,2) + rmat(1,2).conj());
	 auto E = 0.5*(rmat(2,2) + rmat(2,2).conj());
	 // real matrix representation in {|->,|+>,|0>}
	 //  [   (a-b)r   (a+b)i   sqrt2*ci ]
	 //  [  -(a-b)i   (a+b)r   sqrt2*cr ] 
	 //  [ sqrt2*ciT sqrt2*crT     e    ]
	 auto ApB = A+B;
	 auto AmB = A-B;
	 double sqrt2 = sqrt(2.0), invsqrt2 = 1.0/sqrt2;
	 auto Cr = sqrt2*C.real();
	 auto Ci = sqrt2*C.imag();
	 blockMatrix<double> matr(partition,partition);
	 matr(0,0) = AmB.real();
	 matr(1,0) = -AmB.imag();
	 matr(2,0) = Ci.T();
	 matr(0,1) = ApB.imag();
	 matr(1,1) = ApB.real();
	 matr(2,1) = Cr.T();
	 matr(0,2) = Ci;
	 matr(1,2) = Cr;
	 matr(2,2) = E.real();
	 // diagonalization
	 linalg::matrix<double> rho = matr.to_matrix();
	 linalg::matrix<double> Ur;
	 linalg::eig_solver(rho,eigs,Ur,1);
	 // back to determinant basis {|D>,|Df>,|D0>} from {|->,|+>,|0>}
	 // [   i     1    0  ]       [ u[-] ]   [    u[+]+i*u[-]  ]
	 // [-s*i   s*1    0  ]/sqrt2 [ u[+] ] = [ s*(u[+]-i*u[-]) ]/sqrt2
	 // [   0     0  sqrt2]       [ u[0] ]   [   sqrt2*u[0]    ]
	 // where the sign comes from |Dbar>=|Df>*s
	 blockMatrix<double> matu(partition,{dim});
         matu = Ur;
	 blockMatrix<std::complex<double>> umat(partition,{dim});
	 const std::complex<double> iunit(0.0,1.0);
	 umat(0,0) = (matu(1,0) + iunit*matu(0,0))*invsqrt2;
	 umat(1,0) = umat(0,0).conj();
	 umat(1,0).rowscale(phases);
	 umat(2,0) = matu(2,0).as_complex();
	 U = umat.to_matrix();
	 
	 U = U.reorder_row(idx_all,1);
      }

      sig2 = eigs;     
      rbas = U;
*/
      
      // save
      std::copy(sig2.begin(), sig2.end(), std::back_inserter(sig2all));
      rbasis[br] = rbas;
      int ioff = offset[br];
      for(int i=0; i<rdim; i++){
	 idx2sector[ioff+i] = br;
      }
      if(debug){
	 if(br == 0) std::cout << "diagonalization of rdm for each symmetry sector:" << std::endl;
	 std::cout << " br=" << br << " qr=" << rdm.qrow.get_sym(br) << " rdim=" << rdim << " sig2=";
	 for(auto s : sig2) std::cout << s << " ";
	 std::cout << std::endl;
      }
   }
   // 2. select important sig2
   auto index = tools::sort_index(sig2all, 1);
   std::map<int,std::pair<int,double>> kept; // br->(dim,wt)
   deff = 0;
   double sum = 0.0;
   double SvN = 0.0;
   for(int i=0; i<sig2all.size(); i++){
      if(i >= dcut) break; // discard rest
      int idx = index[i];
      if(sig2all[idx] < thresh_sig2) continue; // discard negative weights
      int br = idx2sector[idx];
      auto it = kept.find(br);
      if(it == kept.end()){
         kept[br].first = 1;
	 kept[br].second = sig2all[idx];
      }else{
	 kept[br].first += 1;
	 kept[br].second += sig2all[idx];
      }
      deff += 1;
      sum += sig2all[idx];
      SvN += -sig2all[idx]*std::log2(sig2all[idx]);
      if(debug){
	if(i == 0) std::cout << "sorted sig2:" << std::endl;     
	std::cout << " i=" << i << " (br,ith)=" << br << "," << kept[br].first-1 
             	  << " sig2=" << sig2all[idx] 
	          << " accum=" << sum << std::endl;
      }
   }
   dwt = 1.0-sum;
   std::cout << "decimation summary: reduce from " << sig2all.size() << " to " << deff
     	     << "  dwt=" << dwt << "  SvN=" << SvN << std::endl;
   // 3. construct qbond and qt2 by assembling blocks
   std::vector<int> br_matched;
   std::vector<std::pair<qsym,int>> dims;
   for(const auto& p : kept){
      const auto& br = p.first;
      const auto& dim = p.second.first;
      const auto& wt = p.second.second;
      br_matched.push_back(br);
      dims.push_back(std::make_pair(rdm.qrow.get_sym(br),dim));
      if(debug){
         std::cout << " br=" << p.first << " dim=" << dim 
           	   << " wt=" << wt << std::endl;
      }
   }
   qbond qkept(dims);
   qtensor2<Tm> qt2(qsym(), rdm.qrow, qkept);
   for(int bc=0; bc<qkept.size(); bc++){
      int br = br_matched[bc];
      auto& blk = qt2(br,bc); 
      auto& rbas = rbasis[br];
      std::copy(rbas.data(), rbas.data()+blk.size(), blk.data());
      if(debug){
         assert(rdm.qrow.get_sym(br) == qt2.qcol.get_sym(bc));
         if(bc == 0) std::cout << "reduced basis:" << std::endl;
         std::cout << " (br,bc)=" << br << "," << bc 
		   << " qsym=" << qt2.qcol.get_sym(bc)
		   << " shape=(" << blk.rows() << "," << blk.cols() << ")"
     	           << std::endl;
      }
   } // bc
   return qt2;
}

template <typename Tm>
qtensor2<Tm> decimation_row(const qtensor2<Tm>& rdm,
			    const int dcut,
			    double& dwt,
			    int& deff,
			    const bool ifkr,
			    const qbond& qs1,
			    const qbond& qs2,
			    const qdpt& dpt){
   qtensor2<Tm> qt2;
   if(!ifkr){
      qt2 = decimation_row_nkr(rdm, dcut, dwt, deff);
   }else{
      qt2 = decimation_row_kr(rdm, dcut, dwt, deff, qs1, qs2, dpt);
   }
   return qt2;
}

template <typename Km>
void decimation_onedot_lc(sweep_data& sweeps,
		          const int isweep,
		          const int ibond, 
		          comb<Km>& icomb,
		          const linalg::matrix<typename Km::dtype>& vsol,
		          qtensor3<typename Km::dtype>& wf,
		          oper_dict<typename Km::dtype>& lqops,
		          oper_dict<typename Km::dtype>& cqops,
	                  const integral::two_body<typename Km::dtype>& int2e, 
	                  const integral::one_body<typename Km::dtype>& int1e,
	                  const std::string scratch){
   const auto& dcut  = sweeps.ctrls[isweep].dcut;
   const auto& noise = sweeps.ctrls[isweep].noise; 
   std::cout << " renormalize |lc> : (dcut,noise)=" << dcut << "," 
	     << std::scientific << std::setprecision(1) << noise << std::endl;
   auto& timing = sweeps.opt_timing[isweep][ibond];
   auto& result = sweeps.opt_result[isweep][ibond];
   auto qprod = qmerge(wf.qrow, wf.qmid);
   auto qlc = qprod.first;
   auto dpt = qprod.second;
   qtensor2<typename Km::dtype> rdm(qsym(), qlc, qlc);
   // 1. build pRDM 
   for(int i=0; i<vsol.cols(); i++){
      wf.from_array(vsol.col(i));
      rdm += wf.merge_lc().get_rdm_row();
      if(noise > thresh_noise) get_prdm_lc(wf, lqops, cqops, noise, rdm);
   }
   // 2. decimation
   const bool ifkr = kind::is_kramers<Km>();
   auto qt2 = decimation_row(rdm, dcut, result.dwt, result.deff,
		   	     ifkr, wf.qrow, wf.qmid, dpt);
   // 3. update site tensor
   const auto& p = sweeps.seq[ibond].p;
   icomb.lsites[p] = qt2.split_lc(wf.qrow, wf.qmid, dpt);
   //-------------------------------------------------------------------	 
   assert((qt2-icomb.lsites[p].merge_lc()).normF() < 1.e-10);
   auto ovlp = contract_qt3_qt3_lc(icomb.lsites[p],icomb.lsites[p]);
   assert(ovlp.check_identityMatrix(1.e-10,false)<1.e-10);
   //-------------------------------------------------------------------
   // 4. initial guess for next site within the bond
   if(sweeps.guess){
      const auto& p1 = sweeps.seq[ibond].p1;	   
      icomb.psi.clear();
      for(int i=0; i<vsol.cols(); i++){
         wf.from_array(vsol.col(i));
         auto cwf = qt2.H().dot(wf.merge_lc()); // <-W[alpha,r]->
         auto psi = contract_qt3_qt2_l(icomb.rsites[p1],cwf);
         icomb.psi.push_back(psi);
      }
   }
   timing.td = tools::get_time();
   // 5. renorm operators	 
   oper_dict<typename Km::dtype> qops;
   oper_renorm_opAll("lc", icomb, p, int2e, int1e, lqops, cqops, qops);
   auto fname = oper_fname(scratch, p, "lop");
   oper_save(fname, qops);
}

template <typename Km>
void decimation_onedot_lr(sweep_data& sweeps,
		          const int isweep,
		          const int ibond, 
		          comb<Km>& icomb,
		          const linalg::matrix<typename Km::dtype>& vsol,
		          qtensor3<typename Km::dtype>& wf,
		          oper_dict<typename Km::dtype>& lqops,
		          oper_dict<typename Km::dtype>& rqops,
	                  const integral::two_body<typename Km::dtype>& int2e, 
	                  const integral::one_body<typename Km::dtype>& int1e,
	                  const std::string scratch){
   const auto& dcut  = sweeps.ctrls[isweep].dcut;
   const auto& noise = sweeps.ctrls[isweep].noise; 
   std::cout << " renormalize |lr> (comb) : (dcut,noise)=" << dcut << "," 
	     << std::scientific << std::setprecision(1) << noise << std::endl;
   auto& timing = sweeps.opt_timing[isweep][ibond];
   auto& result = sweeps.opt_result[isweep][ibond];
   auto qprod = qmerge(wf.qrow, wf.qcol);
   auto qlr = qprod.first;
   auto dpt = qprod.second;
   qtensor2<typename Km::dtype> rdm(qsym(), qlr, qlr);
   // 1. build pRDM 
   for(int i=0; i<vsol.cols(); i++){
      wf.from_array(vsol.col(i));
      // Note: need to first bring two dimensions adjacent to each other before merge!
      rdm += wf.permCR_signed().merge_lr().get_rdm_row();
      if(noise > thresh_noise) get_prdm_lr(wf, lqops, rqops, noise, rdm);
   }
   // 2. decimation
   const bool ifkr = kind::is_kramers<Km>();
   auto qt2 = decimation_row(rdm, dcut, result.dwt, result.deff,
		             ifkr, wf.qrow, wf.qcol, dpt);
   // 3. update site tensor
   const auto& p = sweeps.seq[ibond].p;
   icomb.lsites[p]= qt2.split_lr(wf.qrow, wf.qcol, dpt);
   //-------------------------------------------------------------------	 
   assert((qt2-icomb.lsites[p].merge_lr()).normF() < 1.e-10);
   auto ovlp = contract_qt3_qt3_lr(icomb.lsites[p],icomb.lsites[p]);
   assert(ovlp.check_identityMatrix(1.e-10,false)<1.e-10);
   //-------------------------------------------------------------------	
   // 4. initial guess for next site within the bond
   if(sweeps.guess){	
      const auto& p1 = sweeps.seq[ibond].p1;	   
      icomb.psi.clear();
      for(int i=0; i<vsol.cols(); i++){
         wf.from_array(vsol.col(i));
         auto cwf = qt2.H().dot(wf.permCR_signed().merge_lr()); // <-W[alpha,r]->
         auto psi = contract_qt3_qt2_l(icomb.rsites[p1],cwf);
         icomb.psi.push_back(psi);
      }
   }
   timing.td = tools::get_time();
   // 5. renorm operators	 
   oper_dict<typename Km::dtype> qops;
   oper_renorm_opAll("lr", icomb, p, int2e, int1e, lqops, rqops, qops);
   auto fname = oper_fname(scratch, p, "lop");
   oper_save(fname, qops);
}

template <typename Km>
void decimation_onedot_cr(sweep_data& sweeps,
		          const int isweep,
		          const int ibond, 
		          comb<Km>& icomb,
		          const linalg::matrix<typename Km::dtype>& vsol,
		          qtensor3<typename Km::dtype>& wf,
		          oper_dict<typename Km::dtype>& cqops,
		          oper_dict<typename Km::dtype>& rqops,
	                  const integral::two_body<typename Km::dtype>& int2e, 
	                  const integral::one_body<typename Km::dtype>& int1e,
	                  const std::string scratch){
   const auto& dcut  = sweeps.ctrls[isweep].dcut;
   const auto& noise = sweeps.ctrls[isweep].noise; 
   std::cout << " renormalize |cr> : (dcut,noise)=" << dcut << "," 
	     << std::scientific << std::setprecision(1) << noise << std::endl;
   auto& timing = sweeps.opt_timing[isweep][ibond];
   auto& result = sweeps.opt_result[isweep][ibond];
   auto qprod = qmerge(wf.qmid, wf.qcol);
   auto qcr = qprod.first;
   auto dpt = qprod.second;
   qtensor2<typename Km::dtype> rdm(qsym(), qcr, qcr);
   // 1. build pRDM 
   for(int i=0; i<vsol.cols(); i++){
      wf.from_array(vsol.col(i));
      rdm += wf.merge_cr().get_rdm_col();
      if(noise > thresh_noise) get_prdm_cr(wf, cqops, rqops, noise, rdm);
   }
   // 2. decimation
   const bool ifkr = kind::is_kramers<Km>();
   auto qt2 = decimation_row(rdm, dcut, result.dwt, result.deff,
		             ifkr, wf.qmid, wf.qcol, dpt).T(); // permute two lines for RCF
   // 3. update site tensor
   const auto& p = sweeps.seq[ibond].p;
   icomb.rsites[p] = qt2.split_cr(wf.qmid, wf.qcol, dpt);
   //-------------------------------------------------------------------	
   assert((qt2-icomb.rsites[p].merge_cr()).normF() < 1.e-10);	 
   auto ovlp = contract_qt3_qt3_cr(icomb.rsites[p],icomb.rsites[p]);
   assert(ovlp.check_identityMatrix(1.e-10,false)<1.e-10);
   //-------------------------------------------------------------------	
   // 4. initial guess for next site within the bond
   if(sweeps.guess){
      const auto& p0 = sweeps.seq[ibond].p0;	  
      const auto& cturn = sweeps.seq[ibond].cturn; 
      icomb.psi.clear();
      for(int i=0; i<vsol.cols(); i++){
         wf.from_array(vsol.col(i));
         auto cwf = wf.merge_cr().dot(qt2.H()); // <-W[l,alpha]->
	 qtensor3<typename Km::dtype> psi;
         if(!cturn){
            psi = contract_qt3_qt2_r(icomb.lsites[p0],cwf.T());
         }else{
            // special treatment of the propagation downside to backbone
            psi = contract_qt3_qt2_c(icomb.lsites[p0],cwf.T());
            psi = psi.permCR_signed(); // |(lr)c> back to |lcr> order on backbone
         }
         icomb.psi.push_back(psi);
      }
   }
   timing.td = tools::get_time();
   // 5. renorm operators	 
   oper_dict<typename Km::dtype> qops;
   oper_renorm_opAll("cr", icomb, p, int2e, int1e, cqops, rqops, qops);
   auto fname = oper_fname(scratch, p, "rop");
   oper_save(fname, qops);
}

template <typename Km>
void decimation_onedot(sweep_data& sweeps,
		       const int isweep,
		       const int ibond, 
		       comb<Km>& icomb,
		       const linalg::matrix<typename Km::dtype>& vsol,
		       qtensor3<typename Km::dtype>& wf,
		       oper_dict<typename Km::dtype>& cqops,
		       oper_dict<typename Km::dtype>& lqops,
		       oper_dict<typename Km::dtype>& rqops,
	               const integral::two_body<typename Km::dtype>& int2e, 
	               const integral::one_body<typename Km::dtype>& int1e,
	               const std::string scratch){
   const auto& dbond = sweeps.seq[ibond];
   std::cout << "ctns::decimation_onedot (forward,cturn)=" 
	     << dbond.forward << "," << dbond.cturn;
   // build reduced density matrix & perform renormalization
   if(dbond.forward){
      // update lsites & ql
      if(!dbond.cturn){
	 decimation_onedot_lc(sweeps, isweep, ibond, icomb, vsol, wf, 
			      lqops, cqops, int2e, int1e, scratch);
      }else{
	 // special for comb
         decimation_onedot_lr(sweeps, isweep, ibond, icomb, vsol, wf, 
			      lqops, rqops, int2e, int1e, scratch);
      } // cturn
   }else{
      // update rsites & qr
      decimation_onedot_cr(sweeps, isweep, ibond, icomb, vsol, wf, 
		           cqops, rqops, int2e, int1e, scratch); 
   }
}

/*
void tns::decimation_twodot(comb& icomb, 
		            const directed_bond& dbond,
		            const int dcut, 
			    const matrix& vsol,
			    qtensor4& wf,
		            double& dwt,
			    int& deff){
   auto p0 = get<0>(dbond);
   auto p1 = get<1>(dbond);
   auto forward = get<2>(dbond);
   auto p = forward? p0 : p1;
   bool cturn = (icomb.type[p0] == 3 && p1.second == 1);
   cout << "tns::decimation_twodot (fw,ct,dcut)=(" 
	<< forward << "," << cturn << "," << dcut << ") "; 
   qtensor2 rdm;
   if(forward){
      // update lsites & ql
      if(!cturn){
         cout << "renormalize |lc1>" << endl;
	 for(int i=0; i<vsol.cols(); i++){
	    wf.from_array(vsol.col(i));
            auto wf3 = wf.merge_c2r();
	    if(i == 0){
	       rdm  = wf3.merge_lc().get_rdm_row();
	    }else{
	       rdm += wf3.merge_lc().get_rdm_row();
	    }
	 }
	 auto qt2 = decimation_row(rdm, dcut, dwt, deff);
         icomb.lsites[p] = qt2.split_lc(wf.qrow, wf.qmid, wf.dpt_lc1().second);
 	 //-------------------------------------------------------------------	 
	 assert((qt2-icomb.lsites[p].merge_lc()).normF() < 1.e-10);
	 auto ovlp = contract_qt3_qt3_lc(icomb.lsites[p],icomb.lsites[p]);
	 assert(ovlp.check_identityMatrix(1.e-10,false)<1.e-10);
 	 //-------------------------------------------------------------------	
	 // initial guess for next site within the bond
	 icomb.psi.clear();
	 for(int i=0; i<vsol.cols(); i++){
	    wf.from_array(vsol.col(i));
	    auto wf3 = wf.merge_c2r();
	    auto cwf = qt2.T().dot(wf3.merge_lc()); // <-W[alpha,r]->
	    auto psi = cwf.split_cr(wf.qver, wf.qcol, wf.dpt_c2r().second);
	    icomb.psi.push_back(psi);
	 }
      }else{
         cout << "renormalize |lr> (comb)" << endl;
	 for(int i=0; i<vsol.cols(); i++){
            wf.from_array(vsol.col(i));
	    if(i == 0){
	       rdm  = wf.permCR_signed().merge_lr_c1c2().get_rdm_row();
	    }else{
	       rdm += wf.permCR_signed().merge_lr_c1c2().get_rdm_row();
	    }
	 }
	 auto qt2 = decimation_row(rdm, dcut, dwt, deff);
	 icomb.lsites[p]= qt2.split_lr(wf.qrow, wf.qcol, wf.dpt_lr().second);
 	 //-------------------------------------------------------------------	 
	 assert((qt2-icomb.lsites[p].merge_lr()).normF() < 1.e-10);
	 auto ovlp = contract_qt3_qt3_lr(icomb.lsites[p],icomb.lsites[p]);
	 assert(ovlp.check_identityMatrix(1.e-10,false)<1.e-10);
 	 //-------------------------------------------------------------------	
	 // initial guess for next site within the bond
	 icomb.psi.clear();
	 for(int i=0; i<vsol.cols(); i++){
	    wf.from_array(vsol.col(i));
	    auto cwf = qt2.T().dot(wf.permCR_signed().merge_lr_c1c2());
	    auto psi = cwf.split_cr(wf.qmid, wf.qver, wf.dpt_c1c2().second);
	    icomb.psi.push_back(psi); // psi on branch
	 }
      }
   }else{
      // update rsites & qr
      if(!cturn){
         cout << "renormalize |c2r>" << endl;
	 for(int i=0; i<vsol.cols(); i++){
	    wf.from_array(vsol.col(i));
	    auto wf3 = wf.merge_lc1();
            if(i == 0){
               rdm  = wf3.merge_cr().get_rdm_col();
            }else{
               rdm += wf3.merge_cr().get_rdm_col();
            }
         }
         auto qt2 = decimation_row(rdm, dcut, dwt, deff, true);
   
	 if(permute) qt2 = qt2.P();
	 
	 
	 
	 icomb.rsites[p] = qt2.split_cr(wf.qver, wf.qcol, wf.dpt_c2r().second);
         //-------------------------------------------------------------------	
         assert((qt2-icomb.rsites[p].merge_cr()).normF() < 1.e-10);	 
         auto ovlp = contract_qt3_qt3_cr(icomb.rsites[p],icomb.rsites[p]);
         assert(ovlp.check_identityMatrix(1.e-10,false)<1.e-10);
         //-------------------------------------------------------------------	
         // initial guess for next site within the bond
	 icomb.psi.clear();
	 for(int i=0; i<vsol.cols(); i++){
	    wf.from_array(vsol.col(i));
	    auto wf3 = wf.merge_lc1();
	    auto cwf = wf3.merge_cr().dot(qt2.T()); // <-W[l,alpha]->
	    auto psi = cwf.split_lc(wf.qrow, wf.qmid, wf.dpt_lc1().second);
            icomb.psi.push_back(psi);
	 }
      }else{
         cout << "renormalize |c1c2> (comb)" << endl;
	 for(int i=0; i<vsol.cols(); i++){
            wf.from_array(vsol.col(i));
	    if(i == 0){
	       rdm  = wf.permCR_signed().merge_lr_c1c2().get_rdm_col();
	    }else{
	       rdm += wf.permCR_signed().merge_lr_c1c2().get_rdm_col();
	    }
	 }
	 auto qt2 = decimation_row(rdm, dcut, dwt, deff, true);
	 icomb.rsites[p]= qt2.split_cr(wf.qmid, wf.qver, wf.dpt_c1c2().second);
         //-------------------------------------------------------------------	
         assert((qt2-icomb.rsites[p].merge_cr()).normF() < 1.e-10);	 
         auto ovlp = contract_qt3_qt3_cr(icomb.rsites[p],icomb.rsites[p]);
         assert(ovlp.check_identityMatrix(1.e-10,false)<1.e-10);
         //-------------------------------------------------------------------	
         // initial guess for next site within the bond
	 icomb.psi.clear();
	 for(int i=0; i<vsol.cols(); i++){
	    wf.from_array(vsol.col(i));
	    auto cwf = wf.permCR_signed().merge_lr_c1c2().dot(qt2.T()); 
	    auto psi = cwf.split_lr(wf.qrow, wf.qcol, wf.dpt_lr().second);
	    psi = psi.permCR_signed();
            icomb.psi.push_back(psi); // psi on backbone
	 }
      }
   }
}
*/

/*
template <typename Km>
void renorm_twodot(const directed_bond& dbond,
		   const comb<Km>& icomb, 
	           oper_dict<typename Km::dtype>& c1qops,
	           oper_dict<typename Km::dtype>& c2qops,
	           oper_dict<typename Km::dtype>& lqops,
	           oper_dict<typename Km::dtype>& rqops,	
	           const integral::two_body<typename Km::dtype>& int2e, 
	           const integral::one_body<typename Km::dtype>& int1e,
	           const std::string scratch){
   std::cout << "ctns::renorm_twodot superblock=";
   const auto& p = dbond.p;
   const auto& forward = dbond.forward;
   const auto& cturn = dbond.cturn;
   oper_dict<typename Km::dtype> qops;
   if(forward){
      if(!cturn){
	 std::cout << "lc1" << std::endl;
	 oper_renorm_opAll("lc", icomb, p, int2e, int1e, lqops, c1qops, qops);
      }else{
	 std::cout << "lr" << std::endl;
	 oper_renorm_opAll("lr", icomb, p, int2e, int1e, lqops, rqops, qops);
      }
      std::string fname = oper_fname(scratch, p, "lop");
      oper_save(fname, qops);
   }else{
      if(!cturn){
         std::cout << "c2r" << std::endl;
         oper_renorm_opAll("cr", icomb, p, int2e, int1e, c2qops, rqops, qops);
      }else{
	 //     
	 //        c2      
	 //        |
	 //   c1---p
	 //        |
	 //    l---*---r
	 //
         std::cout << "c1c2" << std::endl;
         oper_renorm_opAll("cr", icomb, p, int2e, int1e, c1qops, c2qops, qops);
      }
      std::string fname = oper_fname(scratch, p, "rop");
      oper_save(fname, qops);
   }
}
*/

} // ctns

#endif
