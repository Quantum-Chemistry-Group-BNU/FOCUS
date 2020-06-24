#include "../core/linalg.h"
#include "../core/analysis.h"
#include "tns_comb.h" 
#include "tns_qtensor.h"
#include "tns_oper.h"

using namespace std;
using namespace linalg;
using namespace tns;
using namespace fock;

// <det|Comb[n]> by contracting the Comb
vector<double> comb::rcanon_CIcoeff(const onstate& state){
   int n = rsites[make_pair(0,0)].get_dim_row();
   vector<double> coeff(n,0.0);
   // compute fermionic sign changes
   auto sgn = state.permute_sgn(image2);
   // compute <n'|Comb> by contracting all sites
   qsym sym_p, sym_l, sym_r, sym_u, sym_d;
   matrix<double> mat_r, mat_u;
   // loop over sites on backbone
   sym_r = qsym(0,0);
   for(int i=nbackbone-1; i>=0; i--){
      auto p = make_pair(i,0);
      int tp = type[p];
      if(tp == 0 || tp == 1){
         // site on backbone with physical index
         int k = topo[i][0];
	 int na = state[2*k], nb = state[2*k+1];
	 qsym sym_p(na+nb,na);
         sym_l = sym_p + sym_r;
	 auto key = make_tuple(sym_p,sym_l,sym_r);
	 auto& blk = rsites[p].qblocks[key];
	 if(blk.size() == 0) return coeff; // in case comb does not encode this det
	 matrix<double>& mat = blk[0]; // as physical dimension for each fixed sym_p is 1
	 if(i==nbackbone-1){
	    mat_r = mat;
         }else{
	    // (out,x)*(x,in)->(out,in)
	    mat_r = xgemm("N","N",mat,mat_r); 
	 }
	 sym_r = sym_l; // update sym_r (in)
      }else if(tp == 3){
	 // propogate symmetry from leaves down to backbone
	 sym_u = qsym(0,0);
         for(int j=topo[i].size()-1; j>=1; j--){
	    int k = topo[i][j];
	    int na = state[2*k], nb = state[2*k+1];
	    qsym sym_p(na+nb,na);
	    sym_d = sym_p + sym_u;
	    auto key = make_tuple(sym_p,sym_d,sym_u);
	    auto& blk = rsites[make_pair(i,j)].qblocks[key];
	    if(blk.size() == 0) return coeff;
	    matrix<double>& mat = blk[0];
	    if(j==topo[i].size()-1){
	       mat_u = mat;
	    }else{
	       mat_u = xgemm("N","N",mat,mat_u);
	    }
	    sym_u = sym_d; // update sym_u (in)
         } // j
	 // deal with internal site without physical index
	 sym_l = sym_u + sym_r;
	 auto key = make_tuple(sym_u,sym_l,sym_r);
	 auto& blk = rsites[p].qblocks[key];
	 if(blk.size() == 0) return coeff;
	 int dim_l = blk[0].rows(); 
	 int dim_r = blk[0].cols();
	 matrix<double> mat(dim_l,dim_r);
	 // contract upper sites
	 int dim_u = rsites[p].qmid[sym_u];
	 for(int k=0; k<dim_u; k++){
	    // (c,in)*(c,l,r)->(in=1,l,r)
	    mat += mat_u(k,0)*blk[k]; 
	 }
	 // contract right matrix
	 mat_r = xgemm("N","N",mat,mat_r);
	 sym_r = sym_l;
      } // tp
   } // j
   assert(mat_r.cols() == 1 && mat_r.rows() == n);
   for(int j=0; j<n; j++){
      coeff[j] = sgn*mat_r(j,0);
   }
   return coeff;
}

// ovlp[m,n] = <SCI[m]|Comb[n]>
matrix<double> comb::rcanon_CIovlp(const onspace& space,
	                   const vector<vector<double>>& vs){
   cout << "\ncomb::rcanon_CIovlp" << endl;
   int n = rsites[make_pair(0,0)].get_dim_row();
   int dim = space.size();
   // cmat(n,d) = <d|Comb[n]>
   matrix<double> cmat(n,dim);
   for(int i=0; i<dim; i++){
      auto coeff = rcanon_CIcoeff(space[i]);
      copy(coeff.begin(),coeff.end(),cmat.col(i));
   };
   // ovlp(m,n) = vs(d,m)*cmat(n,d)
   int m = vs.size();
   matrix<double> vmat(dim,m);
   for(int im=0; im<m; im++){
      copy(vs[im].begin(),vs[im].end(),vmat.col(im));
   }
   auto ovlp = xgemm("T","T",vmat,cmat);
   return ovlp;
}

// test subroutines for building operators: Smat & Hmat
// use tensor contraction to compute Smat
matrix<double> tns::get_Smat(const comb& bra, 
  		     const comb& ket){
   cout << "\ntns::get_Smat" << endl;
   if(bra.nbackbone != ket.nbackbone){
      cout << "error: bra/ket nbackbone=" << bra.nbackbone 
	   << "," << ket.nbackbone << endl; 
      exit(1); 
   }
   int nbackbone = bra.nbackbone;
   qtensor2 qt2_r, qt2_u;
   // loop over sites on backbone
   for(int i=nbackbone-1; i>=0; i--){
      auto p = make_pair(i,0);
      int tp = bra.type.at(p);
      if(tp == 0 || tp == 1){
	 if(i==nbackbone-1){
	    qt2_r = contract_qt3_qt3_cr(bra.rsites.at(p),ket.rsites.at(p));
	 }else{
	    auto qtmp = contract_qt3_qt2_r(ket.rsites.at(p),qt2_r);
	    qt2_r = contract_qt3_qt3_cr(bra.rsites.at(p),qtmp);
	 }
      }else if(tp == 3){
         for(int j=bra.topo[i].size()-1; j>=1; j--){
	    auto pj = make_pair(i,j);
            if(j==bra.topo[i].size()-1){
	       qt2_u = contract_qt3_qt3_cr(bra.rsites.at(pj),ket.rsites.at(pj));	   
	    }else{
	       auto qtmp = contract_qt3_qt2_r(ket.rsites.at(pj),qt2_u);
	       qt2_u = contract_qt3_qt3_cr(bra.rsites.at(pj),qtmp);
	    }
	 } // j
	 // internal site without physical index
	 auto qtmp = contract_qt3_qt2_r(ket.rsites.at(p),qt2_r);
	 qtmp = contract_qt3_qt2_c(qtmp,qt2_u); // upper branch
	 qt2_r = contract_qt3_qt3_cr(bra.rsites.at(p),qtmp);
      }
   } // i
   // final: convert qt2_r to normal matrix
   auto Smat = qt2_r.to_matrix();
   return Smat;
}

matrix<double> tns::get_Hmat(const comb& bra, 
		     const comb& ket,
		     const integral::two_body<double>& int2e,
		     const integral::one_body<double>& int1e,
		     const double ecore,
		     const string scratch){
   cout << "\ntns::get_Hmat" << endl;
   // environement
   oper_env_right(bra, ket, int2e, int1e, scratch);
   // load
   oper_dict qops;
   auto p = make_pair(0,0); 
   string fname = oper_fname(scratch, p, "rop");
   oper_load(fname, qops);
   auto Hmat = qops['H'][0].to_matrix();
   Hmat += ecore*identity_matrix<double>(Hmat.rows());
   return Hmat;
}
