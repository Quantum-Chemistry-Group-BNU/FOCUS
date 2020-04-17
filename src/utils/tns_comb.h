#ifndef TNS_COMB_H
#define TNS_COMB_H

#include <vector>
#include <string>
#include <tuple>
#include <map>

namespace tns{

class comb{
   public:
      void read_topology(std::string topology); 
      void print();
      void init();
   public:
      int nbackbone, nphysical, ninternal, ntotal;
      std::vector<std::vector<int>> topo;
      using coord = std::pair<int,int>;
      std::map<coord,std::vector<int>> rsupport;
      std::vector<std::pair<coord,coord>> sweep_seq;
};

} // tns

#endif
