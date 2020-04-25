#include "tns_comb.h"
#include "../core/serialization.h"

using namespace std;
using namespace tns;

void comb::rcanon_save(const string fname){
   cout << "\ncomb::rcanon_save" << endl;
   ofstream ofs(fname, std::ios::binary);
   boost::archive::binary_oarchive save(ofs);
   save << rsites;
}

void comb::rcanon_load(const string fname){
   cout << "\ncomb:rcanon_load" << endl;
   ifstream ifs(fname, std::ios::binary);
   boost::archive::binary_iarchive load(ifs);
   load >> rsites;
}
