#include "tools.h"

using namespace std;
using namespace tools;

//std::random_device linalg::rd; // non-deterministic hardware gen
std::seed_seq tools::seeds{0}; //linalg::rd()};
std::default_random_engine tools::generator(tools::seeds);

ostream& tools::operator <<(ostream& os, const perm& pm){
   for(int i=0; i<pm.size; i++){
      os << pm.image[i] << " ";
   }
   return os;
}
