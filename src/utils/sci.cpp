#include <functional>
#include <algorithm>
#include "../core/dvdson.h"
#include "../core/hamiltonian.h"
#include "../settings/global.h"
#include "sci.h"

using namespace std;
using namespace fock;
using namespace linalg;
using namespace sci;

// constructor
product_space::product_space(const onspace& space){
   int dim = space.size();
   int udxA = 0, udxB = 0;
   // construct {U(A), D(A), B(A)}, {U(B), D(B), A(B)}
   for(int i=0; i<dim; i++){
      // even 
      auto strA = space[i].get_even();
      auto itA = umapA.find(strA);
      if(itA == umapA.end()){
         auto pr = umapA.insert({strA,udxA});
	 itA = pr.first;
	 udxA += 1;
	 usetA.push_back(strA);
	 dsetA.resize(udxA); // make space
	 bsetA.resize(udxA);
      };
      // odd 
      auto strB = space[i].get_odd();
      auto itB = umapB.find(strB);
      if(itB == umapB.end()){
         auto pr = umapB.insert({strB,udxB});
	 itB = pr.first;
	 udxB += 1;
	 usetB.push_back(strB);
	 dsetB.resize(udxB);
	 asetB.resize(udxB);
      }
      dsetA[itA->second].push_back(i);
      bsetA[itA->second].push_back(itB->second);
      dsetB[itB->second].push_back(i);
      asetB[itB->second].push_back(itA->second);
   }
   // nnzA,nnzB
   dimA = udxA;
   nnzA.resize(dimA);
   for(int ia=0; ia<dimA; ia++) nnzA[ia] = dsetA[ia].size();
   dimB = udxB;
   nnzB.resize(dimB);
   for(int ib=0; ib<dimB; ib++) nnzB[ib] = dsetB[ib].size();	  
   // dpt
   dpt.resize(dimA); 
   for(int ia=0; ia<dimA; ia++){
      dpt[ia].resize(dimB,-1);
      for(int ib=0; ib<nnzA[ia]; ib++){
	 int b = bsetA[ia][ib]; // nonzero column
         int d = dsetA[ia][ib]; // index of det in space 
	 dpt[ia][b] = d;
      }
   }
}

// constructor
coupling_table::coupling_table(const onspace& uset){
   dim = uset.size();
   C11.resize(dim);
   C22.resize(dim);
   for(int i=0; i<dim; i++){
      for(int j=0; j<dim; j++){
	 auto pr = uset[i].diff_type(uset[j]);
	 if(pr == make_pair(1,1)){
	    C11[i].push_back(j); 
	 }else if(pr == make_pair(2,2)){
	    C22[i].push_back(j);
	 }
      }
   }
}

void sparse_hamiltonian::get_diagonal(const onspace& space, 
				      const integral::two_body& int2e, 
				      const integral::one_body& int1e, 
				      const double ecore){
   for(int i=0; i<dim; i++){
      double Hii = fock::get_Hii(space[i], int2e, int1e) + ecore; 
      connect[i].push_back(i);
      value[i].push_back(Hii);
   }
}

// local term: (C11+C22)_A*Id_B
void sparse_hamiltonian::get_localA(const onspace& space,
				    const product_space& pspace,
				    const coupling_table& ctabA,
				    const integral::two_body& int2e,
				    const integral::one_body& int1e){
   // <I_A,I_B|H_A|J_A,J_B> = <I_A|H_A|J_A><I_B|J_B>	
   for(int ia=0; ia<pspace.dimA; ia++){
      auto I_A = pspace.usetA[ia];
      // <I_A|H1|J_A>	   
      for(int ja : ctabA.C11[ia]){
	 auto J_A = pspace.usetA[ja];
	 double Hij = fock::get_HijS(I_A, J_A, int2e, int1e, 0); 
	 // loop over I_B=J_B
	 for(int ib : pspace.bsetA[ia]){
	    int j = pspace.dpt[ja][ib];
	    if(j>=0){
	       int i = pspace.dpt[ia][ib];
	       connect[i].push_back(j);
	       value[i].push_back(Hij);
	    }
	 }
      }
      // <I_A|H2|J_A>	   
      for(int ja : ctabA.C22[ia]){
	 auto J_A = pspace.usetA[ja];
	 double Hij = fock::get_HijD(I_A, J_A, int2e, int1e, 0); 
	 // loop over I_B=J_B
	 for(int ib : pspace.bsetA[ia]){
	    int j = pspace.dpt[ja][ib];
	    if(j>=0){
	       int i = pspace.dpt[ia][ib];
	       connect[i].push_back(j);
	       value[i].push_back(Hij);
	    }
	 }
      }
   } // ia
}

// local term: Id_A*(C11+C22)_B
void sparse_hamiltonian::get_localB(const onspace& space,
				    const product_space& pspace,
				    const coupling_table& ctabB,
				    const integral::two_body& int2e,
				    const integral::one_body& int1e){
   // <I_A,I_B|H_B|J_A,J_B> = <I_A|J_A><I_B|H_B|J_B>	
   for(int ib=0; ib<pspace.dimB; ib++){
      auto I_B = pspace.usetB[ib];
      // <I_B|H1|J_B>	   
      for(int jb : ctabB.C11[ib]){
	 auto J_B = pspace.usetB[jb];
	 double Hij = fock::get_HijS(I_B, J_B, int2e, int1e, 1); 
	 // loop over I_A=J_A
	 for(int ia : pspace.asetB[ib]){
	    int j = pspace.dpt[ia][jb];
	    if(j>=0){
	       int i = pspace.dpt[ia][ib];
	       connect[i].push_back(j);
	       value[i].push_back(Hij);
	    }
	 }
      }
      // <I_B|H2|J_B>	   
      for(int jb : ctabB.C22[ib]){
	 auto J_B = pspace.usetB[jb];
	 double Hij = fock::get_HijD(I_B, J_B, int2e, int1e, 1); 
	 // loop over I_A=J_A
	 for(int ia : pspace.asetB[ib]){
	    int j = pspace.dpt[ia][jb];
	    if(j>=0){
	       int i = pspace.dpt[ia][ib];
	       connect[i].push_back(j);
	       value[i].push_back(Hij);
	    }
	 }
      }
   } // ib
}

// interaction term: C11_A*C11_B
void sparse_hamiltonian::get_int_11_11(const onspace& space,
				       const product_space& pspace,
				       const coupling_table& ctabA,
				       const coupling_table& ctabB,
				       const integral::two_body& int2e,
				       const integral::one_body& int1e){
   // <I_A,I_B|HAB|J_A,J_B> 
   for(int ia=0; ia<pspace.dimA; ia++){
      for(int ja : ctabA.C11[ia]){

         auto I_A = pspace.usetA[ia];
         auto J_A = pspace.usetA[ja];
         vector<int> cre_A,ann_A;
         I_A.diff_orb(J_A,cre_A,ann_A);
         int q = cre_A[0], qq = 2*q;
         int s = ann_A[0], ss = 2*s;
	 auto sgn_A = I_A.parity(q)*J_A.parity(s);

         for(int ib : pspace.bsetA[ia]){
            int i = pspace.dpt[ia][ib];	      
   	    for(int jb : ctabB.C11[ib]){
   	       int j = pspace.dpt[ja][jb];
   	       if(j>=0){

	          double Hij = fock::get_HijD(space[i],space[j],int2e,int1e,2);
		  
		  auto I_B = pspace.usetB[ib];
		  auto J_B = pspace.usetB[jb];
		  vector<int> cre_B,ann_B;
                  I_B.diff_orb(J_B,cre_B,ann_B);
                  int p = cre_B[0], pp = 2*p+1;
                  int r = ann_B[0], rr = 2*r+1;
	          auto sgn_B = I_B.parity(p)*J_B.parity(r);

	          // <pBqA||rBsA> = [pr|qs]	  
                  double Hij2;
		  Hij2 = sgn_A*sgn_B*int2e.get(pp,rr,qq,ss);
//		  cout << "Hij=" << Hij << " " << Hij2 << endl;

		  connect[i].push_back(j);
	          value[i].push_back(Hij2);
	       }
	    }
	 } // ja
      } // ib
   } // ia
}

// compute sparse H
sparse_hamiltonian::sparse_hamiltonian(const onspace& space,
				       const product_space& pspace,
				       const coupling_table& ctabA,
				       const coupling_table& ctabB,
				       const integral::two_body& int2e,
				       const integral::one_body& int1e,
				       const double ecore){
   dim = space.size();
   connect.resize(dim);
   value.resize(dim);
   nnz.resize(dim);
   bool debug = true;
  
   // diagonal term:
   auto t0 = global::get_time();
   this->get_diagonal(space, int2e, int1e, ecore);
   auto ta = global::get_time();
   if(debug) cout << "timing for get_diagonal : " << setprecision(2) 
		  << global::get_duration(ta-t0) << " s" << endl;

   // local term: (C11+C22)_A*Id_B
   this->get_localA(space, pspace, ctabA, int2e, int1e);
   auto tb = global::get_time();
   if(debug) cout << "timing for get_localA : " << setprecision(2) 
		  << global::get_duration(tb-ta) << " s" << endl;
   
   // local term: Id_A*(C11+C22)_B
   this->get_localB(space,pspace, ctabB, int2e, int1e);
   auto tc = global::get_time();
   if(debug) cout << "timing for get_localB : " << setprecision(2) 
		  << global::get_duration(tc-tb) << " s" << endl;

   // interaction term: C11_A*C11_B
   this->get_int_11_11(space, pspace, ctabA, ctabB, int2e, int1e);
   auto td = global::get_time();
   if(debug) cout << "timing for get_int_11_11: " << setprecision(2) 
   		  << global::get_duration(td-tc) << " s" << endl;
   
   // compute nnz
   for(int i=0; i<dim; i++){
      nnz[i] = connect[i].size();
   }
}

// matrix-vector product using stored H
void sci::get_Hx(double* y,
	         const double* x,
	         const sparse_hamiltonian& sparseH){
   // y[i] = sum_j H[i,j]*x[j] 
   for(int i=0; i<sparseH.dim; i++){
      y[i] = 0.0;
      for(int jdx=0; jdx<sparseH.nnz[i]; jdx++){
         int j = sparseH.connect[i][jdx];
	 double Hij = sparseH.value[i][jdx];
	 y[i] += Hij*x[j];
      }
   }
}

// solve eigenvalue problem in this space
void sci::ci_solver(vector<double>& es,
	       	    matrix& vs,	
		    const onspace& space,
	       	    const integral::two_body& int2e,
	       	    const integral::one_body& int1e,
	       	    const double ecore){
   cout << "\nsci::ci_solver" << endl; 
   bool debug = true;
  
   // setup product_space
   auto t0 = global::get_time();
   product_space pspace(space);
   auto ta = global::get_time();
   if(debug) cout << "timing for pspace : " << setprecision(2) 
		  << global::get_duration(ta-t0) << " s" << endl;
  
   // setupt coupling_table
   coupling_table ctabA(pspace.usetA);
   auto tb = global::get_time();
   coupling_table ctabB(pspace.usetB);
   auto tc = global::get_time();
   if(debug) cout << "timing for ctabA/B : " << setprecision(2) 
		  << global::get_duration(tb-ta) << " s" << " "
		  << global::get_duration(tc-tb) << " s" << endl;
   
   // compute sparse_hamiltonian
   sparse_hamiltonian sparseH(space, pspace, ctabA, ctabB,
		   	      int2e, int1e, ecore);
   auto td = global::get_time();
   if(debug) cout << "timing for sparseH : " << setprecision(2) 
		  << global::get_duration(td-tc) << " s" << endl;

   // Davidson solver 
   dvdsonSolver solver;
   solver.iprt = 2;
   solver.ndim = space.size();
   solver.neig = es.size();
   auto Diag = fock::get_Hdiag(space, int2e, int1e, ecore);
   solver.Diag = Diag.data(); 
   using std::placeholders::_1;
   using std::placeholders::_2;
   solver.HVec = bind(sci::get_Hx, _1, _2, cref(sparseH));
   // solve
   solver.solve_iter(es.data(), vs.data());
   //solver.solve_diag(es.data(), vs.data());
   
   auto t1 = global::get_time();
   cout << "timing for sci::ci_solver : " << setprecision(2) 
	<< global::get_duration(t1-t0) << " s" << endl;
   exit(1);
}
