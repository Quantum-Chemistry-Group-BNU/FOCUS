#include <cmath> // pow
#include "tools.h"

using namespace std;
using namespace tools;

double tools::mem_size(size_t sz, const int dfac){
   return sz*dfac/pow(1024.0,2); // in MB
}

// timing
chrono::high_resolution_clock::time_point tools::get_time(){
   return chrono::high_resolution_clock::now();
}

//std::random_device linalg::rd; // non-deterministic hardware gen
std::seed_seq tools::seeds{0}; //linalg::rd()};
std::default_random_engine tools::generator(tools::seeds);

ostream& tools::operator <<(ostream& os, const perm& pm){
   for(int i=0; i<pm.size; i++){
      os << pm.image[i] << " ";
   }
   return os;
}
