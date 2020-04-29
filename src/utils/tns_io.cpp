#include "../core/serialization.h"
#include "tns_comb.h"
#include "tns_oper.h"

using namespace std;
using namespace tns;

void comb::rcanon_save(const string fname){
   cout << "\ncomb::rcanon_save fname=" << fname << endl;
   ofstream ofs(fname, std::ios::binary);
   boost::archive::binary_oarchive save(ofs);
   save << rsites;
}

void comb::rcanon_load(const string fname){
   cout << "\ncomb:rcanon_load fname=" << fname << endl;
   ifstream ifs(fname, std::ios::binary);
   boost::archive::binary_iarchive load(ifs);
   load >> rsites;
}

void tns::oper_save(const string fname, const qopers& qops){
   cout << "\ntns::oper_save fname=" << fname << endl;
   ofstream ofs(fname, std::ios::binary);
   boost::archive::binary_oarchive save(ofs);
   save << qops;
}

void tns::oper_load(const string fname, qopers& qops){
   cout << "\ntns::oper_load fname=" << fname << endl;
   ifstream ifs(fname, std::ios::binary);
   boost::archive::binary_iarchive load(ifs);
   load >> qops;
}
