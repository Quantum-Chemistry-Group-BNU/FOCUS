#ifndef COMB_TOPO_H
#define COMB_TOPO_H

#include <iostream>
#include <tuple>
#include <vector>
#include <string>

namespace comb{

using comb_coord = std::pair<int,int>;
std::ostream& operator <<(std::ostream& os, const comb_coord& coord);

const comb_coord coord_phy = std::make_pair(-1,-1);
const comb_coord coord_vac = std::make_pair(-2,-2); 
extern const comb_coord coord_phy, coord_vac;

struct node{
   public:
      friend std::ostream& operator <<(std::ostream& os, const node& nd);
   public:
     int pindex;    // physical index
     int type;	    // type of nodes 0 [boundary], 1 [backbone], 2 [branch], 3 [internal]
     comb_coord middle; // m-neighbor
     comb_coord left;   // l-neighbor
     comb_coord right;  // r-neighbor
};

struct topology{
   public:
      void read(std::string topology_file); 
      void print() const;
   public:
      int nbackbone, nphysical;
      std::vector<std::vector<node>> nodes; // nodes on comb
      std::vector<comb_coord> rcoord; // coordinate of each node in rvisit order
      				     //  used in constructing right environment

//      std::map<comb_coord,std::vector<int>> rsupport;
//      std::map<comb_coord,std::vector<int>> lsupport;
//      // --- 1D ordering ---
//      std::vector<int> image2; // mapping of physical indices
//      std::vector<int> orbord; // map orbital to 1D position
};
      
//std::vector<int> support_rest(const std::vector<int>& rsupp);

// --- sweep sequence ---
using directed_bond = std::tuple<comb_coord,comb_coord,bool>;
std::vector<directed_bond> get_sweeps(const topology& topo, const bool debug=false);

} // tns

#endif
