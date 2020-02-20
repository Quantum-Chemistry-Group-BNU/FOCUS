#include <iostream>
#include "../utils/onstate.h"

using namespace std;

int tmp(const fock::onstate& y){
   return y[0];
}

int test_onstate(){
   cout << "\ntest_onstate" << endl;
 
   // test constructor and getocc/setocc
   fock::onstate state1(65);
   state1[0] = 1;
   state1[4] = 1;
   state1[9] = 1;
   state1[64] = 1;
   cout << "len=" << state1.len() << endl;
   cout << "size=" << state1.size() << endl;
   for(int i=0; i<state1.size(); i++){
      cout << "i=" << i << " occ=" << state1[i] << endl;
   };
   cout << "state1=" << state1 << endl;

   // test constructor from string
   fock::onstate state2("0000101");
   cout << "state2=" << state2 << endl;

   // test assignment
   fock::onstate state3(state1.size());
   state3 = state1;
   cout << "state3=" << state3 << endl; 

   // test count
   cout << "Ne1=" << state1.Ne() << endl;
   cout << "Ne2=" << state2.Ne() << endl;
   cout << "Ne3=" << state3.Ne() << endl;

   return 0;   
}
