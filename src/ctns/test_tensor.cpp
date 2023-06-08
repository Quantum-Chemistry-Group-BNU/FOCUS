#include <iostream>
#include <iomanip>
#include <string>
#include "../qtensor/qtensor.h"
#include "tests_ctns.h"
//#include "init_phys.h"

using namespace std;
using namespace ctns;

int tests::test_tensor(){
   cout << endl;	
   cout << tools::line_separator << endl;	
   cout << "tests::test_tensor" << endl;
   cout << tools::line_separator << endl;	

   using Tm = double;
   std::vector<int> syms({0,1,2});

/*
   for(int isym : syms){
      auto qphys = get_qbond_phys(isym);

      // c[0] = kA^+
      // [[0. 0. 0. 0.]
      //  [0. 0. 0. 1.]
      //  [1. 0. 0. 0.]
      //  [0. 0. 0. 0.]]
      linalg::matrix<Tm> mat(4,4);
      mat(1,3) = 1;
      mat(2,0) = 1;
      auto sym_op = get_qsym_opC(isym,0);

      qtensor2<Tm> qt2(sym_op, qphys, qphys);
      qt2.from_matrix(mat);
      qt2.print("qt2",1);
      qt2.to_matrix().print("c0+_qt2");

      stensor2<Tm> st2(sym_op, qphys, qphys);
      st2.from_matrix(mat);
      st2.print("st2",1);
      st2.to_matrix().print("c0+_st2");
   }
*/
   return 0;
}
