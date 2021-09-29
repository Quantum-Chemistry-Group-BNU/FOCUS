#ifndef CTNS_TOPO_H
#define CTNS_TOPO_H

#include <tuple>
#include <vector>
#include "../core/serialization.h"
#include "../core/tools.h"

namespace ctns{

// the position of each site in a CTNS is specified by a 2D coordinate (i,j)
using comb_coord = std::pair<int,int>;
std::ostream& operator <<(std::ostream& os, const comb_coord& coord);

// special coord 
const comb_coord coord_vac = std::make_pair(-2,-2);
extern const comb_coord coord_vac;
const comb_coord coord_phys = std::make_pair(-1,-1);  
extern const comb_coord coord_phys;

//
// node information for sites of ctns in right 
//		            r
//      c		    |
//      |               c---*
//  l---*---r               |
//		            l
struct node{
   private:
      // serialize
      friend class boost::serialization::access;
      template<class Archive>
      void serialize(Archive & ar, const unsigned int version){
	 ar & pindex & type & center & left & right
	    & rsupport & lsupport;
      }
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

// directed_bond used in sweep sequence for optimization of ctns: (p0,p1,forward,p,cturn)
struct directed_bond{
   public:
      directed_bond(const comb_coord _p0, const comb_coord _p1, const bool _forward, 
		    const comb_coord _p, const bool _cturn):
	p0(_p0), p1(_p1), forward(_forward), p(_p), cturn(_cturn) {}
   public:
      comb_coord p0, p1, p;
      bool forward, cturn;
};

// topology information of ctns
struct topology{
   private:
      // serialize
      friend class boost::serialization::access;
      template<class Archive>
      void serialize(Archive & ar, const unsigned int version){
	 ar & ntotal & nbackbone & nphysical 
            & nodes & rcoord & rindex & image2;
      }
   public:
      topology(){};
      void read(const std::string& topology_file); 
      void print() const;
      // helpers
      const node& get_node(const comb_coord& p) const{ return nodes[p.first][p.second]; }
      int get_type(const comb_coord& p) const{ return nodes[p.first][p.second].type; }
      //				  |
      //    MPS-like:	    Additional: --pc
      //     \|/			 \|/
      //    --p--			--p--
      //
      std::vector<int> get_suppc(const comb_coord& p) const{
         auto pc = get_node(p).center;
         bool physical = (pc == coord_phys);
         auto suppc = physical? std::vector<int>({get_node(p).pindex}) : get_node(pc).rsupport;
         return suppc;
      }
      // 			        |
      //    MPS-like:     Additional: --p 
      //      |      |                 /|\
      //    --pl-->--p--     	      --pl--
      //
      std::vector<int> get_suppl(const comb_coord& p) const{
         auto suppl = get_node(p).lsupport;
         return suppl;
      }
      //
      // MPS-like:
      //    |     |
      //  --p--<--pr-- : qrow of rsites[pr]
      //
      std::vector<int> get_suppr(const comb_coord& p) const{
         auto pr = get_node(p).right;
         auto suppr = get_node(pr).rsupport;
         return suppr;
      }
      //
      // cturn: bond that at the turning points to branches
      //              |
      //           ---*(i,1)
      //       |      I      |
      //    ---*------*------*---
      //               (i,0)
      //
      bool is_cturn(const comb_coord& p0, const comb_coord& p1) const{
	 return p0.second == 0 && p1.second == 1;
      }
      // helper for support 
      std::vector<int> support_rest(const std::vector<int>& rsupp) const;
      // sweep sequence 
      std::vector<directed_bond> get_sweeps(const bool debug=true) const;
   public:
      int ntotal, nbackbone, nphysical;
      // nodes on comb
      std::vector<std::vector<node>> nodes; 
      // coordinate of each node in rvisit order ("sliced from right")
      // used in constructing right environment
      std::vector<comb_coord> rcoord;  // idx->(i,j) 
      std::map<comb_coord,int> rindex; // (i,j)->idx
      // 1D ordering of CTNS for mapping |n_p...> in ctns_alg 
      std::vector<int> image2; 
};

} // ctns

#endif
