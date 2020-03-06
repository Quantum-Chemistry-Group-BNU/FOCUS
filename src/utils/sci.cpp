#include "sci.h"

using namespace std;
using namespace fock;
using namespace linalg;

// solve eigenvalue problem in this space
void sci::ci_solver(vector<double>& es,
	       	    matrix& vs,	
		    const onspace& space,
	       	    const integral::two_body& int2e,
	       	    const integral::one_body& int1e,
	       	    const double ecore){

};
/*
   cout << "\nsci::ci_solver" << endl; 
   auto t0 = global::get_time();
   // Davidson solver 
   dvdsonSolver solver;
   solver.ndim = space.size();
   solver.neig = es.size();
   // Hdiag
   auto Diag = get_Hdiag(space, int2e, int1e, ecore);
   solver.Diag = Diag.data(); 
   // y=H*x, see https://en.cppreference.com/w/cpp/utility/functional/ref
   using std::placeholders::_1;
   using std::placeholders::_2;
   solver.HVec = bind(fock::get_Hx, _1, _2, cref(space), cref(int2e), cref(int1e), ecore);
   // solve
   solver.solve_iter(es.data(), vs.data());
   //solver.full_diag(es.data(), vs.data());
   auto t1 = global::get_time();
   cout << "\ntiming for fock::ci_solver : " << setprecision(2) 
	<< global::get_duration(t1-t0) << " s" << endl;
}
*/

