#ifndef CTNS_TOPO_H
#define CTNS_TOPO_H

#include <iostream>
#include <tuple>
#include <vector>
#include <string>
#include <map>

namespace ctns{

// coordinates (i,j) for sites of ctns
using comb_coord = std::pair<int,int>;
std::ostream& operator <<(std::ostream& os, const comb_coord& coord);
const comb_coord coord_phy = std::make_pair(-1,-1);  
const comb_coord coord_vac = std::make_pair(-2,-2); 
extern const comb_coord coord_phy, coord_vac;

// node information for sites of ctns
struct node{
   public:
      friend std::ostream& operator <<(std::ostream& os, const node& nd);
   public:
      int pindex; // physical index
      int type;	  // type of node: 0 [boundary], 1 [backbone], 2 [branch], 3 [internal]
      comb_coord middle; // m-neighbor
      comb_coord left;   // l-neighbor
      comb_coord right;  // r-neighbor
};

// sweep sequence for optimization of ctns 
using directed_bond = std::tuple<comb_coord,comb_coord,bool>;

// topology information of ctns
struct topology{
   public:
      topology(const std::string& topology_file); 
      void print() const;
      // helper for support 
      std::vector<int> support_rest(const std::vector<int>& rsupp) const;
      // sweep sequence 
      std::vector<directed_bond> get_sweeps(const bool debug=true) const;
   public:
      int nbackbone, nphysical;
      std::vector<std::vector<node>> nodes; // nodes on comb
      std::vector<comb_coord> rcoord; // coordinate of each node in rvisit order
      				      // used in constructing right environment
      //--- support ---
      int iswitch; // for i<=iswitch on backbone, size(lsupp)<size(rsupp)
      std::map<comb_coord,std::vector<int>> rsupport;
      std::map<comb_coord,std::vector<int>> lsupport;
      std::vector<int> image2; // 1D ordering of CTNS for |n_p...> 
};

} // ctns

#endif
