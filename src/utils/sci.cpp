#include <functional>
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
   // construct {U(A), D(A), B(A)}, {U(B), D(B), A(B)}
   int dim = space.size();
   int udxA = 0, udxB = 0;
   for(size_t i=0; i<dim; i++){
      // even 
      auto strA = space[i].get_even();
      auto itA = UsetA.find(strA);
      if(itA == UsetA.end()){
         auto pr = UsetA.insert({strA,udxA});
	 itA = pr.first;
	 udxA += 1;
	 DsetA.resize(udxA); // make space
	 BsetA.resize(udxA);
      };
      // odd 
      auto strB = space[i].get_odd();
      auto itB = UsetB.find(strB);
      if(itB == UsetB.end()){
         auto pr = UsetB.insert({strB,udxB});
	 itB = pr.first;
	 udxB += 1;
	 DsetB.resize(udxB);
	 AsetB.resize(udxB);
      }
      DsetA[itA->second].push_back(i);
      BsetA[itA->second].push_back(itB->second);
      DsetB[itB->second].push_back(i);
      AsetB[itB->second].push_back(itA->second);
   }
}

// debug
void product_space::print(){
   // debug for UsetA 
   cout << "\nUsetA size=" << UsetA.size() << endl; 
   for(const auto& p : UsetA){
      cout << p.first << " " << p.second << endl;
   }
   // sequential from 0 to dim for FCI space
   for(int ia=0; ia<DsetA.size(); ia++){
      cout << "ia=" << ia << endl;
      for(auto d : DsetA[ia])
         cout << d << " ";
      cout << endl;
   }
   // all equal to 0 1 2 3 4 5 6 7 8 9 10 11 12 13 ... for FCI space
   for(int ia=0; ia<BsetA.size(); ia++){
      cout << "ia=" << ia << endl;
      for(auto d : BsetA[ia])
         cout << d << " ";
      cout << endl;
   }
   // debug for UsetB
   cout << "\nUsetB size=" << UsetB.size() << endl; 
   for(const auto& p : UsetB){
      cout << p.first << " " << p.second << endl;
   }
   for(int ib=0; ib<DsetB.size(); ib++){
      cout << "ib=" << ib << endl;
      for(auto d : DsetB[ib])
         cout << d << " ";
      cout << endl;
   }
   for(int ib=0; ib<AsetB.size(); ib++){
      cout << "ib=" << ib << endl;
      for(auto d : AsetB[ib])
         cout << d << " ";
      cout << endl;
   }
}

// constructor
coupling_table::coupling_table(const std::map<onstate,int>& Uset){
   dim = Uset.size();
   C11.resize(dim);
   C22.resize(dim);
   for(const auto& p : Uset){
      for(const auto& q : Uset){
	 auto pr = p.first.diff_type(q.first);
	 if(pr == make_pair(1,1)){
	    C11[p.second].push_back(q.second); 
	 }else if(pr == make_pair(2,2)){
	    C22[p.second].push_back(q.second);
	 }
      }
   }
}

// compute sparse H
sparse_hamiltonian::sparse_hamiltonian(const onspace& space,
				       const product_space& pspace,
				       const coupling_table& ctableA,
				       const coupling_table& ctableB,
				       const integral::two_body& int2e,
				       const integral::one_body& int1e,
				       const double ecore){
   dim = space.size();
   connect.resize(dim);
   value.resize(dim);
   nnz.resize(dim);
   // C00,C00 diagonal term
   for(int i=0; i<dim; i++){
      double Hii = fock::get_Hii(space[i], int2e, int1e) + ecore; 
      connect[i].push_back(i);
      value[i].push_back(Hii);
   }
 

   // compute nnz
   for(int i=0; i<dim; i++){
      nnz[i] = connect[i].size();
   }

   // debug
   /* 
   for(int i=0; i<dim; i++){
      cout << "i=" << i << endl;
      for(auto j : connect[i])
         cout << j << " ";
      cout << endl;
   }
   */
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
   if(debug) cout << "\ntiming for pspace : " << setprecision(2) 
		  << global::get_duration(ta-t0) << " s" << endl;
  
   // setupt coupling_table
   coupling_table ctableA(pspace.UsetA);
   auto tb = global::get_time();
   coupling_table ctableB(pspace.UsetB);
   auto tc = global::get_time();
   if(debug) cout << "\ntiming for ctableA/B : " << setprecision(2) 
		  << global::get_duration(tb-ta) << " s" << " "
		  << global::get_duration(tc-tb) << " s" << endl;
   
   // compute sparse_hamiltonian
   sparse_hamiltonian sparseH(space, pspace, ctableA, ctableB,
		   	      int2e, int1e, ecore);
   auto td = global::get_time();
   if(debug) cout << "\ntiming for sparseH : " << setprecision(2) 
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
   auto t1 = global::get_time();
   cout << "\ntiming for sci::ci_solver : " << setprecision(2) 
	<< global::get_duration(t1-t0) << " s" << endl;
   exit(1);
}
