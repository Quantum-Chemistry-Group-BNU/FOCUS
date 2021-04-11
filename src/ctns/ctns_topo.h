#ifndef CTNS_TOPO_H
#define CTNS_TOPO_H

#include <tuple>
#include <vector>

namespace ctns{

// the position of each site in a CTNS is specified by a 2D coordinate (i,j)
using comb_coord = std::pair<int,int>;
std::ostream& operator <<(std::ostream& os, const comb_coord& coord);

// special coord 
const comb_coord coord_vac = std::make_pair(-2,-2);
extern const comb_coord coord_vac;
const comb_coord coord_phys = std::make_pair(-1,-1);  
extern const comb_coord coord_phys;

// node information for sites of ctns in right 
//		            r
//      c		    |
//      |               c---*
//  l---*---r               |
//		            l
struct node{
   public:
      friend std::ostream& operator <<(std::ostream& os, const node& nd);
   public:
      int pindex; // physical index: p-th spatial orbital; =-1 for internal sites 
      int type;	  // type of node: 0 [boundary], 1 [backbone], 2 [branch], 3 [internal]
      comb_coord center; // c-neighbor
      comb_coord left;   // l-neighbor
      comb_coord right;  // r-neighbor
      // the bipartite bond is chosen as l-*
      std::vector<int> rsupport;  // orbitals in the right part 
      std::vector<int> lsupport;  // orbitals in the left part
};

// directed_bond used in sweep sequence for optimization of ctns 
using directed_bond = std::tuple<comb_coord,comb_coord,bool>;

// topology information of ctns
struct topology{
   public:
      topology(const std::string& topology_file); 
      void print() const;
      // node
      const node& get_node(const comb_coord& p) const{ return nodes[p.first][p.second]; }
      // type
      int node_type(const comb_coord& p) const{ return nodes[p.first][p.second].type; }
      // helper for support 
      std::vector<int> support_rest(const std::vector<int>& rsupp) const;
      // sweep sequence 
      std::vector<directed_bond> get_sweeps(const bool debug=true) const;
      // cturn: bond that at the turning points to branches
      //
      //           |
      //        ---* (i,1)
      //       |   |   |
      //    ---*---*---*---
      //             (i,0)
      inline bool is_cturn(const comb_coord& p0, const comb_coord& p1) const{
	 return p0.second == 0 && p1.second == 1;
      }
   public:
      int nbackbone, nphysical;
      std::vector<std::vector<node>> nodes; // nodes on comb
      std::vector<comb_coord> rcoord; // coordinate of each node in rvisit order ("sliced from right")
      				      // used in constructing right environment
      std::vector<int> image2; // 1D ordering of CTNS for mapping |n_p...> 
};

} // ctns

#endif
