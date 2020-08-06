#include <tuple>
#include <bitset>
#include <cassert>
#include "../core/onstate.h"
#include "../core/tools.h"
#include "tests.h"

using namespace std;

int tests::test_onstate(){
   cout << endl;
   cout << tools::line_separator << endl;	
   cout << "test::test_onstate" << endl;
   cout << tools::line_separator << endl;	
 
   // test constructor and getocc/setocc
   fock::onstate state1(66);
   state1[0] = 1;
   state1[4] = 1;
   state1[9] = 1;
   state1[64] = 1;
   state1[65] = 1;
   cout << "len=" << state1.len() << endl;
   cout << "size=" << state1.size() << endl;
   for(int i=0; i<state1.size(); i++){
      cout << "i=" << i << " occ=" << state1[i] << endl;
   };
   cout << "state1=" << state1 << endl;
   cout << "state1=" << state1.to_string2() << endl; 

   // test constructor from string
   fock::onstate state2("00000101");
   cout << "state2=" << state2 << endl;
   cout << "state2=" << state2.to_string2() << endl; 

   // test assignment
   fock::onstate state3(state1.size());
   state3 = state1;
   state3[1] = 1;
   cout << "state3=" << state3 << endl; 
   cout << "state3=" << state3.to_string2() << endl; 

   // test count
   cout << "Ne1=" << " " << state1.nelec() 
   		  << " " << state1.nelec_a() 
   		  << " " << state1.nelec_b() << endl;
   cout << "Ne2=" << " " << state2.nelec() 
   		  << " " << state2.nelec_a() 
   		  << " " << state2.nelec_b() << endl;
   cout << "Ne3=" << " " << state3.nelec() 
   		  << " " << state3.nelec_a() 
   		  << " " << state3.nelec_b() << endl;

   // test copy
   fock::onstate state4(state3);
   cout << "state3=" << state3.to_string2() << endl; 
   cout << state4.diff_num(state3) << endl;
   cout << state4.diff_num(state1) << endl;

   // test non
   fock::onstate non;
   cout << "non=" << non << endl;
   cout << "non=" << non.to_string() << endl;
   cout << "non=" << non.to_string2() << endl;

   // test cre/ann
   cout << "input\n" << state4 << endl;
   cout << "\ncre 1" << endl; 
   auto x = state4.cre(1);   // copy constructor
   cout << "x.f=" << x.first << endl;
   cout << "x.s=" << x.second << endl;
   
   cout << "\nann 1" << endl;
   auto y = state4.ann(1);   // copy+move constructor 
   cout << "y.f=" << y.first << endl;
   cout << "y.s=" << y.second << endl;
   
   cout << "\nann 3" << endl;
   auto z = state4.ann(3);   // copy+move constructor 
   cout << "z.f=" << z.first << endl;
   cout << "z.s=" << z.second << endl;

   cout << "\ncre 6" << endl;
   cout << "y.second = " << y.second << endl;
   auto m = y.second.cre(6);
   cout << "m.f=" << m.first << endl;
   cout << "m.s=" << m.second << endl;

   cout << "\nann 64" << endl;
   cout << "y.second = " << y.second << endl;
   // move constructor 
   fock::onstate tmp = y.second.ann(65).second;
   // move assignment 
   fock::onstate sta;
   sta = y.second.ann(65).second; // move =
   cout << "sta=" << sta << endl;
   int fac;
   std::tie(fac,sta) = y.second.ann(65);
   cout << "fac=" << fac << endl;
   cout << "sta=" << sta << endl;
   /*
   for(int i=0; i<64; i++){
      long tmp = fock::allones(i);
      cout << bitset<64>(tmp) << endl;
      cout << bitset<64>(~tmp) << endl;
   }
   */
   cout << "parity(0)=" << sta.parity(0) << endl;
   cout << "parity(1)=" << sta.parity(1) << endl;
   cout << "parity(0,3)=" << sta.parity(0,3) << endl;
   cout << "parity(0,4)=" << sta.parity(0,4) << endl;
   cout << "parity(0,5)=" << sta.parity(0,5) << endl;
   cout << "parity(0,6)=" << sta.parity(0,6) << endl;
   cout << "parity(3,10)=" << sta.parity(3,10) << endl;
   cout << "parity(0,64)=" << sta.parity(0,64) << endl;
   assert(sta.parity(0)==1);
   assert(sta.parity(1)==-1);
   assert(sta.parity(0,3)==-1);
   assert(sta.parity(0,4)==-1);
   assert(sta.parity(0,5)==1);
   assert(sta.parity(0,6)==1);
   assert(sta.parity(3,10)==1);
   assert(sta.parity(0,64)==-1);

   // test kramers
   cout << "input\n" << state4 << endl;
   cout << state4.to_string2() << endl;
   cout << state4.nelec_a() << "," << state4.nelec_b() << endl;
   cout << "is_standard=" << state4.is_standard() << endl;
   fock::onstate state5 = state4.flip();
   cout << "flip=" << state5 << endl;
   cout << "flip=" << state5.to_string2() << endl;
   cout << (state4 == state5) << endl;
   cout << (state4 < state5) << endl;
  
   // test for make_standard() 
   cout << "--- initial ---" << endl;
   cout << state5 << endl;
   cout << state4 << endl; 
   cout << "--- state5 ---" << endl;
   cout << state5.is_standard() << endl;
   state5 = state5.make_standard();
   cout << state5 << endl;
   state5[2] = 1;
   cout << state5 << endl;
   cout << "--- state4 ---" << endl;
   cout << state4.is_standard() << endl;
   state4 = state4.make_standard();
   cout << state4 << endl;
   state4[3] = 1;
   cout << state4 << endl;
   cout << "--- final ---" << endl;
   cout << state5 << endl;
   cout << state4 << endl; 

   cout << state4.to_string2() << endl;
   cout << "norb_single=" << state4.norb_single() << endl;
   cout << "norb_double=" << state4.norb_double() << endl;
   cout << "norb_vacant=" << state4.norb_vacant() << endl;
   cout << state4.norb_single()
	   +state4.norb_double()
	   +state4.norb_vacant() << endl;
   cout << state4.to_string2() << endl;
   cout << state4.parity_flip() << endl;
   cout << state4.flip().to_string2() << endl;
   cout << state4.flip().parity_flip() << endl;

   vector<int> olst,vlst;
   state4.get_olst(olst);
   state4.get_vlst(vlst);
   cout << "olst=";
   for(auto i : olst)
      cout << " " << i;
   cout << endl;
   cout << "vlst="; 
   for(auto i : vlst)
      cout << " " << i;
   cout << endl;
   
   olst.clear();
   vlst.clear();
   state5.get_olst(olst);
   state5.get_vlst(vlst);
   cout << "olst=";
   for(auto i : olst)
      cout << " " << i;
   cout << endl;
   cout << "vlst="; 
   for(auto i : vlst)
      cout << " " << i;
   cout << endl;

   // test for diff_orb 
   fock::onstate state_v1("0000");
   fock::onstate state_v2("1011");
   fock::onstate state_v3("0101");
   vector<int> cre, ann;

   cre.clear();
   ann.clear();
   state_v1.diff_orb(state_v2, cre, ann);
   cout << "\ndiff_orb: 1,2 : " << state_v1 << " " << state_v2 << endl;
   cout << "diff_num " << state_v1.diff_num(state_v2) << endl;
   cout << "cre " << cre.size() << " : ";
   for(int x : cre) cout << x;
   cout << endl;
   cout << "ann " << ann.size() << " : ";
   for(int x : ann) cout << x;
   cout << endl;
   auto p = state_v1.diff_type(state_v2);
   cout << "type " << p.first << "," << p.second << endl;

   cre.clear();
   ann.clear();
   state_v1.diff_orb(state_v3, cre, ann);
   cout << "\ndiff_orb: 1,3 : " << state_v1 << " " << state_v3 << endl;
   cout << "diff_num " << state_v1.diff_num(state_v3) << endl;
   cout << "cre " << cre.size() << " : ";
   for(int x : cre) cout << x;
   cout << endl;
   cout << "ann " << ann.size() << " : ";
   for(int x : ann) cout << x;
   cout << endl;
   p = state_v1.diff_type(state_v3);
   cout << "type " << p.first << "," << p.second << endl;

   cre.clear();
   ann.clear();
   state_v2.diff_orb(state_v3, cre, ann);
   cout << "\ndiff_orb: 2,3 : " << state_v2 << " " << state_v3 << endl;
   cout << "diff_num " << state_v2.diff_num(state_v3) << endl;
   cout << "cre " << cre.size() << " : ";
   for(int x : cre) cout << x;
   cout << endl;
   cout << "ann " << ann.size() << " : ";
   for(int x : ann) cout << x;
   cout << endl;
   p = state_v2.diff_type(state_v3);
   cout << "type " << p.first << "," << p.second << endl;
   
   cre.clear();
   ann.clear();
   state_v3.diff_orb(state_v2, cre, ann);
   cout << "\ndiff_orb: 3,2 : " << state_v3 << " " << state_v2 << endl;
   cout << "diff_num " << state_v3.diff_num(state_v2) << endl;
   cout << "cre " << cre.size() << " : ";
   for(int x : cre) cout << x;
   cout << endl;
   cout << "ann " << ann.size() << " : ";
   for(int x : ann) cout << x;
   cout << endl;
   p = state_v3.diff_type(state_v2);
   cout << "type " << p.first << "," << p.second << endl;

   cout << "\neven and odd:" << endl;
   cout << state_v1 << " " << state_v1.get_even() << " " << state_v1.get_odd() << endl;
   cout << state_v2 << " " << state_v2.get_even() << " " << state_v2.get_odd() << endl;
   cout << state_v3 << " " << state_v3.get_even() << " " << state_v3.get_odd() << endl;

   return 0;   
}
