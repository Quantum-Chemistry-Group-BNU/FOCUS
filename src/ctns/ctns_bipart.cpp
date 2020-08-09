#include "../core/linalg.h"
#include "ctns_bipart.h"
#include "ctns_kramers.h"

using namespace std;
using namespace fock;
using namespace linalg;
using namespace ctns;

renorm_basis<double> ctns::right_projection(const onspace& space,
		                            const vector<vector<double>>& vs,
		                            const int bpos, 
		                            const double thresh,
					    const bool debug){
   auto t0 = tools::get_time();
   const bool debug_basis = false;
   if(debug){
      cout << "\nctns::right_projection<real> thresh=" 
           << scientific << setprecision(4) << thresh << endl;
   }
   renorm_basis<double> rbasis;
   // 1. bipartition form of psi
   bipart_qspace lspace, rspace;
   for(int i=0; i<space.size(); i++){
      auto lstate = space[i].get_before(bpos).make_standard();
      auto itl = lspace.uset.find(lstate);
      if(itl == lspace.uset.end()){ // not found - new basis
	 lspace.uset.insert(lstate);
	 auto ql = qsym(lstate.nelec(),lstate.twoms());
	 lspace.basis[ql].push_back(lstate);
	 if(lstate.norb_single() != 0){
	    ql = ql.flip();
	    lspace.basis[ql].push_back(lstate.flip());
	 }
      }
      auto rstate = space[i].get_after(bpos).make_standard();
      auto itr = rspace.uset.find(rstate);
      if(itr == rspace.uset.end()){
	 rspace.uset.insert(rstate);
	 auto qr = qsym(rstate.nelec(),rstate.twoms());
	 rspace.basis[qr].push_back(rstate);
	 auto fstate = rstate.flip();
	 if(rstate.norb_single() != 0){
	    qr = qr.flip();		 
	    rspace.basis[qr].push_back(rstate.flip());
	 }
      }
   }
   // reorder rspace.basis to form Kramers-paired structure (for M=0)
   for(const auto& pr : rspace.basis){
      const auto& qr = pr.first;
      const auto& qr_space = pr.second;
      int dim = qr_space.size();
      // even electron case {|D>,|Df>,|D0>} 
      if(qr.tm() == 0){
	 onspace rdets0, rdets1, rdets;
	 for(int i=0; i<dim; i++){
	    auto& state = qr_space[i];
	    if(state.norb_single() == 0){
	       rdets0.push_back(state);
	    }else{
	       if(state.is_standard()){
	          rdets1.push_back(state);
	       }
	    }
	 }
	 rspace.dim0[qr] = rdets0.size();
         assert(2*rdets1.size()+rdets0.size() == dim);
	 rdets = rdets1;
	 transform(rdets1.begin(),rdets1.end(),rdets1.begin(),
	           [](const onstate& state){ return state.flip(); });		 
	 copy(rdets1.begin(),rdets1.end(),back_inserter(rdets));
	 copy(rdets0.begin(),rdets0.end(),back_inserter(rdets));
         rspace.basis[qr] = rdets;
      }
   }
   // update space info
   lspace.update();
   rspace.update();
   // construct wfs
   int nroot = vs.size(); 
   bipart_ciwfs<double> wfs;
   for(int i=0; i<space.size(); i++){
      auto lstate = space[i].get_before(bpos);
      auto rstate = space[i].get_after(bpos);
      auto ql = qsym(lstate.nelec(),lstate.twoms());
      auto qr = qsym(rstate.nelec(),rstate.twoms());
      int nl = lspace.dims[ql];
      int nr = rspace.dims[qr];
      int il = lspace.index[ql][lstate];
      int ir = rspace.index[qr][rstate];
      auto key = make_pair(ql,qr);
      if(wfs.qblocks[key].size() == 0){
	 wfs.qblocks[key].resize(nroot);
	 for(int iroot=0; iroot<nroot; iroot++){
	    matrix<double> mat(nl,nr);		
	    mat(il,ir) = vs[iroot][i]; 
	    wfs.qblocks[key][iroot] = mat;
	 }
      }else{
	 for(int iroot=0; iroot<nroot; iroot++){
	    wfs.qblocks[key][iroot](il,ir) = vs[iroot][i]; 
	 }
      }
   }
   // 2. form dm for each qr
   int dimBc = 0;
   double sumBc = 0.0, SvN = 0.0;
   for(const auto& qr : rspace.syms){
      if(qr.tm() < 0) continue;
      int dim = rspace.dims[qr];
      int dim0 = (qr.tm() != 0)? 0 : rspace.dim0[qr];
      int dim1 = (qr.tm() != 0)? dim : (dim-dim0)/2;
      int dgen = (qr.tm() != 0)? 2 : 1;
      // phase for determinant with open-shell electrons
      vector<double> phases(dim1);
      for(int i=0; i<dim1; i++){
         phases[i] = rspace.basis[qr][i].parity_flip();
      }
      if(debug_basis){
         cout << "qr=" << qr << " (dim,dim1,dim0)=" 
	      << dim << "," << dim1 << "," << dim0 
	      << " dgen=" << dgen << endl;
         for(const auto& state : rspace.basis[qr]){
            cout << " state=" << state << " index=" << rspace.index[qr][state] << endl;
         }
      }
      // rhor[r,r'] = psi[l,r]^T*psi[l,r']^*     
      matrix<double> rhor(dim,dim);
      for(const auto& ql : lspace.syms){
         auto key = make_pair(ql,qr);
         auto& blk = wfs.qblocks[key];
         if(blk.size() == 0) continue;
         for(int iroot=0; iroot<nroot; iroot++){
            rhor += xgemm("T","N",blk[iroot],blk[iroot].conj());
         }
      }
      rhor *= 1.0/nroot;     
      // diagonalize rhor properly to yield rbasis (depending on qr)
      vector<double> eigs(dim); 
      matrix<double> U(dim,dim);
      if(qr.tm() != 0){
	 // average over q(N,M+) and q(N,M-) sectors
         matrix<double> rhor1(dim,dim);
         for(const auto& ql : lspace.syms){
            auto key = make_pair(ql,qr.flip());
            auto& blk = wfs.qblocks[key];
            if(blk.size() == 0) continue;
            for(int iroot=0; iroot<nroot; iroot++){
               rhor1 += xgemm("T","N",blk[iroot],blk[iroot].conj());
            }
         }
         rhor1 *= 1.0/nroot;    
	 // TR basis |Dbar> = |Df>*s
	 rhor1.colscale(phases);
	 rhor1.rowscale(phases);
	 // Kramers projection
	 rhor1 = 0.5*(rhor + rhor1);
	 // diagonalization
	 eig_solver(rhor1,eigs,U,1);
      }else{
	 // from det basis {|D>,|Df>,|D0>} to TR basis {|->,|+>,|0>}
	 // |-> = (|D> - |Dbar>)/sqrt2 = (|D> - s*|Df>)/sqrt2
	 // |+> = (|D> + |Dbar>)/sqrt2 = (|D> + s*|Df>)/sqrt2
	 if(dim1 == 0){
	    // only {|0>}
	    assert(dim == dim0);
	    eig_solver(rhor,eigs,U,1);
	 }else{
            vector<int> partition = {dim1,dim1,dim0};
	    blockMatrix<double> rmat(partition,partition);
	    rmat = rhor;
 	    // col-1 & row-1
	    rmat(0,1).colscale(phases);
	    rmat(1,1).colscale(phases);
	    rmat(2,1).colscale(phases);
	    rmat(1,0).rowscale(phases);
            rmat(1,1).rowscale(phases);
            rmat(1,2).rowscale(phases);
	    // Kramers projection
	    auto A = 0.5*(rmat(0,0) + rmat(1,1));
	    auto B = 0.5*(rmat(0,1) + rmat(1,0));
	    auto C = 0.5*(rmat(0,2) + rmat(1,2));
	    // real matrix rep: block diagonal as no imaginary part 
	    //  [ (a-b)r    0          0    ]
	    //  [   0     (a+b)r   sqrt2*cr ] 
	    //  [   0    sqrt2*crT     e    ]
	    // 1. real matrix in basis {|->}
	    matrix<double> rhom = A-B;
	    matrix<double> Um;
	    vector<double> em(dim1);
	    eig_solver(rhom,em,Um,1);
	    // 2. real matrix in basis {|+>,|0>}
	    double sqrt2 = sqrt(2.0), invsqrt2 = 1.0/sqrt2;
	    blockMatrix<double> matr({dim1,dim0},{dim1,dim0});
	    matr(0,0) = A+B;
	    matr(0,1) = sqrt2*C; 
	    matr(1,0) = sqrt2*C.T();
	    matr(1,1) = rmat(2,2);
	    // diagonalization
	    matrix<double> rhop = matr.to_matrix();
	    matrix<double> Up;
	    vector<double> ep(dim1+dim0);
	    eig_solver(rhop,ep,Up,1);
	    // back to determinant basis {|D>,|Df>,|D0>} from {|->,|+>,|0>}
	    // [  1    1    0  ]       [u[-]][  0 ]   [   u[-]][    u[+]  ]
	    // [-s1   s1    0  ]/sqrt2 [ 0  ][u[+]] = [-s*u[-]][  s*u[+]  ]/sqrt2
	    // [  0    0  sqrt2]       [ 0  ][u[0]]   [   0   ][sqrt2*u[0]]
	    // where the sign comes from |Dbar>=|Df>*s
	    blockMatrix<double> matu({dim1,dim0},{dim1+dim0});
            matu = Up;
	    blockMatrix<double> umat(partition,{dim1,dim1+dim0});
	    // X_A
	    auto xa = Um*invsqrt2;
	    umat(0,0) = xa;
	    umat(1,0) = -xa;
	    umat(1,0).rowscale(phases);
	    // X_S,X_0
	    auto xs = matu(0,0)*invsqrt2;
	    umat(0,1) = xs;
	    umat(1,1) = xs;
	    umat(1,1).rowscale(phases);
	    umat(2,1) = matu(1,0);
	    U = umat.to_matrix();
	    copy(em.begin(),em.end(),eigs.begin());
	    copy(ep.begin(),ep.end(),eigs.begin()+dim1);
	 } // dim1
      }
      // 4. select important renormalized states from (eigs,U) 
      double sumBi = 0.0;
      vector<int> kept;
      for(int i=0; i<dim; i++){
	 if(eigs[i] > thresh){
	    kept.push_back(i);
	    sumBi += eigs[i];
            SvN += -dgen*eigs[i]*log2(eigs[i]); // compute entanglement entropy
	 }
      }
      int dimBi = kept.size();
      dimBc += dgen*dimBi;
      sumBc += dgen*sumBi;
      if(debug){
         cout << " qr=" << qr << " dimB,dimBi=" << dim << "," << dimBi 
              << " sumBi=" << sumBi << " sumBc=" << sumBc << endl;
      }
      // save sites
      if(dimBi > 0){
         renorm_sector<double> rsec;
         rsec.sym = qr;
         rsec.space = rspace.basis[qr];
	 rsec.coeff.resize(dim,dimBi);
	 int i = 0;
         for(const int idx : kept){
	    if(debug_basis) cout << " i=" << i << " eig=" << scientific << eigs[idx] << endl;
	    copy(U.col(idx), U.col(idx)+dim, rsec.coeff.col(i));
	    i += 1;
	 }
         rbasis.push_back(rsec);
	 // check orthogonality
	 if(debug_basis){
	    auto ova = xgemm("N","N",rsec.coeff.H(),rsec.coeff);
	    double diff = normF(ova - identity_matrix<double>(dimBi));
	    cout << " orthonormality=" << diff << endl;
	    if(diff > 1.e-10){
	       cout << " error: basis is not orthonormal!" << endl;
	       exit(1);
	    }
	 }
	 // sector |M-> = K|M+> = |Dbar>*C = |Df>*(s*C) 
	 if(qr.tm() != 0){
	    rsec.sym = qr.flip();
	    rsec.space = rspace.basis[qr.flip()];
	    rsec.coeff.rowscale(phases);
	    rbasis.push_back(rsec);
	 }
      } // dimBi>0
   } // qr
   if(debug){
      cout << "dim(space,lspace,rspace)=" << space.size() << "," 
           << lspace.get_dim() << "," << rspace.get_dim() 
           << " dimBc=" << dimBc << " sumBc=" << sumBc << " SvN=" << SvN << endl;
      auto t1 = tools::get_time();
      cout << "timing for ctns::right_projection<real> : " << setprecision(2) 
           << tools::get_duration(t1-t0) << " s" << endl;
   }
   return rbasis;
}

renorm_basis<complex<double>> ctns::right_projection(const onspace& space,
		                      		     const vector<vector<complex<double>>>& vs,
		                      		     const int bpos, 
		                      		     const double thresh,
						     const bool debug){
   auto t0 = tools::get_time();
   const bool debug_basis = false;
   if(debug){
      cout << "ctns::right_projection<cmplx> thresh=" 
   	   << scientific << setprecision(4) << thresh << endl;
   }
   renorm_basis<std::complex<double>> rbasis;
   // 1. bipartition form of psi
   bipart_qspace lspace, rspace;
   for(int i=0; i<space.size(); i++){
      auto lstate = space[i].get_before(bpos).make_standard();
      auto itl = lspace.uset.find(lstate);
      if(itl == lspace.uset.end()){ // not found - new basis
	 lspace.uset.insert(lstate);
	 auto ql = qsym(lstate.nelec(),0);
	 lspace.basis[ql].push_back(lstate);
	 if(lstate.norb_single() != 0) lspace.basis[ql].push_back(lstate.flip());
      }
      auto rstate = space[i].get_after(bpos).make_standard();
      auto itr = rspace.uset.find(rstate);
      if(itr == rspace.uset.end()){
	 rspace.uset.insert(rstate);
	 auto qr = qsym(rstate.nelec(),0);
	 rspace.basis[qr].push_back(rstate);
	 auto fstate = rstate.flip();
	 if(rstate.norb_single() != 0) rspace.basis[qr].push_back(rstate.flip());
      }
   }
   // reorder rspace.basis to form Kramers-paired structure
   for(const auto& pr : rspace.basis){
      const auto& qr = pr.first;
      const auto& qr_space = pr.second;
      int dim = qr_space.size();
      // odd electron case {|D>,|Df>}
      if(qr.ne()%2 == 1){
         assert(dim%2 == 0);
         onspace rdets(dim);
         for(int i=0; i<dim/2; i++){
            rdets[i] = qr_space[2*i];
            rdets[i+dim/2] = qr_space[2*i+1];
         }
         rspace.basis[qr] = rdets;
      // even electron case {|D>,|Df>,|D0>} 
      }else{
	 onspace rdets0, rdets1, rdets;
	 for(int i=0; i<dim; i++){
	    auto& state = qr_space[i];
	    if(state.norb_single() == 0){
	       rdets0.push_back(state);
	    }else{
	       if(state.is_standard()){
	          rdets1.push_back(state);
	       }
	    }
	 }
	 rspace.dim0[qr] = rdets0.size();
         assert(2*rdets1.size()+rdets0.size() == dim);
	 rdets = rdets1;
	 transform(rdets1.begin(),rdets1.end(),rdets1.begin(),
	           [](const onstate& state){ return state.flip(); });		 
	 copy(rdets1.begin(),rdets1.end(),back_inserter(rdets));
	 copy(rdets0.begin(),rdets0.end(),back_inserter(rdets));
         rspace.basis[qr] = rdets;
      }
   }
   // update space info
   lspace.update();
   rspace.update();
   // construct wfs
   int nroot = vs.size(); 
   bipart_ciwfs<complex<double>> wfs;
   for(int i=0; i<space.size(); i++){
      auto lstate = space[i].get_before(bpos);
      auto rstate = space[i].get_after(bpos);
      auto ql = qsym(lstate.nelec(),0);
      auto qr = qsym(rstate.nelec(),0);
      int nl = lspace.dims[ql];
      int nr = rspace.dims[qr];
      int il = lspace.index[ql][lstate];
      int ir = rspace.index[qr][rstate];
      auto key = make_pair(ql,qr);
      if(wfs.qblocks[key].size() == 0){
	 wfs.qblocks[key].resize(nroot);
	 for(int iroot=0; iroot<nroot; iroot++){
	    matrix<complex<double>> mat(nl,nr);		
	    mat(il,ir) = vs[iroot][i]; 
	    wfs.qblocks[key][iroot] = mat;
	 }
      }else{
	 for(int iroot=0; iroot<nroot; iroot++){
	    wfs.qblocks[key][iroot](il,ir) = vs[iroot][i]; 
	 }
      }
   }
   // 2. form dm for each qr
   int dimBc = 0;
   double sumBc = 0.0, SvN = 0.0;
   for(const auto& qr : rspace.syms){
      int dim = rspace.dims[qr];
      int dim0 = (qr.ne()%2 == 1)? 0 : rspace.dim0[qr];
      int dim1 = (dim-dim0)/2;
      assert(dim0 + dim1*2 == dim);
      // phase for determinant with open-shell electrons
      vector<double> phases(dim1);
      for(int i=0; i<dim1; i++){
         phases[i] = rspace.basis[qr][i].parity_flip();
      }
      if(debug_basis){
         cout << "qr=" << qr << " (dim,dim1,dim0)=" 
	      << dim << "," << dim1 << "," << dim0 << endl;
         for(const auto& state : rspace.basis[qr]){
            cout << " state=" << state << " index=" << rspace.index[qr][state] << endl;
         }
      }
      // rhor[r,r'] = psi[l,r]^T*psi[l,r']^*     
      matrix<complex<double>> rhor(dim,dim);
      for(const auto& ql : lspace.syms){
	 auto key = make_pair(ql,qr);
	 auto& blk = wfs.qblocks[key];
	 if(blk.size() == 0) continue;
	 for(int iroot=0; iroot<nroot; iroot++){
	    rhor += xgemm("T","N",blk[iroot],blk[iroot].conj());
	 }
      }
      rhor *= 1.0/nroot;      
      // 3. diagonalized properly to yield rbasis (depending on qr)
      vector<double> eigs(dim); 
      matrix<complex<double>> U(dim,dim);
      if(qr.ne()%2 == 1){
	 // from det basis {|D>,|Df>} to TR basis {|D>,|Dbar>}
	 vector<int> partition = {dim1,dim1};
	 blockMatrix<complex<double>> rmat(partition,partition);
	 rmat = rhor;
	 rmat(0,1).colscale(phases);
	 rmat(1,1).colscale(phases);
	 rmat(1,0).rowscale(phases);
         rmat(1,1).rowscale(phases);
	 // Kramers projection
	 auto A = 0.5*(rmat(0,0) + rmat(1,1).conj());
	 auto B = 0.5*(rmat(0,1) - rmat(1,0).conj()); 
	 rmat(0,0) = A;
	 rmat(0,1) = B;
	 rmat(1,0) = -B.conj();
	 rmat(1,1) = A.conj();
	 rhor = rmat.to_matrix();
	 // TRS-preserving diagonalization (only half eigs are output) 
	 zquatev(rhor,eigs,U,1);
	 copy(eigs.begin(), eigs.begin()+dim1, eigs.begin()+dim1);
	 // back to determinant basis {|D>,|Df>}
	 blockMatrix<complex<double>> umat(partition,{dim});
	 umat = U;
	 umat(1,0).rowscale(phases);
	 U = umat.to_matrix();
      }else{
	 // from det basis {|D>,|Df>,|D0>} to {|D>,|Dbar>,|D0>} to TR basis {|->,|+>,|0>}
	 // |-> = i(|D> - |Dbar>)/sqrt2 = i(|D> - s|Df>)/sqrt2
	 // |+> =  (|D> + |Dbar>)/sqrt2 =  (|D> + s|Df>)/sqrt2
	 vector<int> partition = {dim1,dim1,dim0};
	 blockMatrix<complex<double>> rmat(partition,partition);
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
	 matrix<double> rho = matr.to_matrix();
	 matrix<double> Ur;
	 eig_solver(rho,eigs,Ur,1);
	 // back to determinant basis {|D>,|Df>,|D0>} from {|->,|+>,|0>}
	 // [  I    1    0  ]       [ u[-] ]   [    u[+]+I*u[-]  ]
	 // [-sI   s1    0  ]/sqrt2 [ u[+] ] = [ s*(u[+]-I*u[-]) ]/sqrt2
	 // [  0    0  sqrt2]       [ u[0] ]   [   sqrt2*u[0]    ]
	 // where the sign comes from |Dbar>=|Df>*s
	 blockMatrix<double> matu(partition,{dim});
         matu = Ur;
	 blockMatrix<complex<double>> umat(partition,{dim});
	 const complex<double> iunit(0.0,1.0);
	 umat(0,0) = (matu(1,0) + iunit*matu(0,0))*invsqrt2;
	 umat(1,0) = umat(0,0).conj();
	 umat(1,0).rowscale(phases);
	 umat(2,0) = matu(2,0).as_complex();
	 U = umat.to_matrix();
      }
      // 4. select important renormalized states from (eigs,U) 
      double sumBi = 0.0;
      vector<int> kept;
      for(int i=0; i<dim; i++){
	 if(eigs[i] > thresh){
	    kept.push_back(i);
	    sumBi += eigs[i];
            SvN += -eigs[i]*log2(eigs[i]); // compute entanglement entropy
	 }
      }
      int dimBi = kept.size();
      dimBc += dimBi;
      sumBc += sumBi;
      if(debug){
         cout << " qr=" << qr << " dimB,dimBi=" << dim << "," << dimBi 
              << " sumBi=" << sumBi << " sumBc=" << sumBc << endl;
      }
      if(qr.ne()%2 == 1) assert(dimBi%2 == 0);
      // save sites
      if(dimBi > 0){
         renorm_sector<complex<double>> rsec;
         rsec.sym = qr;
         rsec.space = rspace.basis[qr];
	 rsec.coeff.resize(dim,dimBi);
	 int i = 0;
         for(const int idx : kept){
	    if(debug_basis) cout << " i=" << i << " eig=" << scientific << eigs[idx] << endl;
	    copy(U.col(idx), U.col(idx)+dim, rsec.coeff.col(i));
	    i += 1;
	 }
         rbasis.push_back(rsec);
	 // check orthogonality
	 if(debug_basis){
	    auto ova = xgemm("N","N",rsec.coeff.H(),rsec.coeff);
	    double diff = normF(ova - identity_matrix<complex<double>>(dimBi));
	    cout << " orthonormality=" << diff << endl;
	    if(diff > 1.e-10){
	       cout << " error: basis is not orthonormal!" << endl;
	       exit(1);
	    }
	 }
      } // dimBi>0
   } // qr
   if(debug){
      cout << "dim(space,lspace,rspace)=" << space.size() << "," 
           << lspace.get_dim() << "," << rspace.get_dim() 
           << " dimBc=" << dimBc << " sumBc=" << sumBc << " SvN=" << SvN << endl; 
      auto t1 = tools::get_time();
      cout << "timing for ctns::right_projection<cmplx> : " << setprecision(2) 
           << tools::get_duration(t1-t0) << " s" << endl;
   }
   return rbasis;
}
