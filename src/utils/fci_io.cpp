#include "fci.h"
#include "../core/serialization.h"

using namespace std;
using namespace fock;

void fci::ci_save(const onspace& space,
	          const vector<vector<double>>& vs,
		  const string fname){
   cout << "\nfci::ci_save" << endl;
   ofstream ofs(fname, std::ios::binary);
   boost::archive::binary_oarchive save(ofs);
   save << space << vs;
}

void fci::ci_load(onspace& space,
	          vector<vector<double>>& vs,
		  const string fname){
   cout << "\nfci::ci_load" << endl;
   ifstream ifs(fname, std::ios::binary);
   boost::archive::binary_iarchive load(ifs);
   load >> space >> vs;
}
