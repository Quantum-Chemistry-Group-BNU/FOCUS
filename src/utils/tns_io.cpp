#include "tns_comb.h"
#include <fstream>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/map.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/binary_object.hpp>

using namespace std;
using namespace tns;

void comb::save_rsites(const string fname){
   cout << "\ncomb::save_rsites" << endl;
   ofstream ofs(fname, std::ios::binary);
   boost::archive::binary_oarchive save(ofs);
   save << rsites;
}

void comb::load_rsites(const string fname){
   cout << "\ncomb:load_rsites" << endl;
   ifstream ifs(fname, std::ios::binary);
   boost::archive::binary_iarchive load(ifs);
   load >> rsites;
}
