#include "../core/serialization.h"
#include "tns_comb.h"
#include "tns_oper.h"

using namespace std;
using namespace tns;

// for comb
void comb::rcanon_save(const string fname){
   cout << "comb::rcanon_save fname=" << fname << endl;
   ofstream ofs(fname, std::ios::binary);
   boost::archive::binary_oarchive save(ofs);
   save << rsites;
}

void comb::rcanon_load(const string fname){
   cout << "comb:rcanon_load fname=" << fname << endl;
   ifstream ifs(fname, std::ios::binary);
   boost::archive::binary_iarchive load(ifs);
   load >> rsites;
}

// for qopers
string tns::oper_fname(const string scratch, 
  	 	       const comb_coord& p,
		       const string optype){
   string fname = scratch+"/oper_("
	         +to_string(p.first)+","
	         +to_string(p.second)+")_"
		 +optype;
   return fname;
}
 
void tns::oper_save(const string fname, const qopers& qops){
   cout << "tns::oper_save fname=" << fname 
	<< " size=" << qops.size() << endl;
   ofstream ofs(fname, std::ios::binary);
   boost::archive::binary_oarchive save(ofs);
   save << qops;
}

void tns::oper_load(const string fname, qopers& qops){
   cout << "tns::oper_load fname=" << fname << endl;
   ifstream ifs(fname, std::ios::binary);
   boost::archive::binary_iarchive load(ifs);
   load >> qops;
}
