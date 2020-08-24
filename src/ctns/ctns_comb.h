#ifndef CTNS_COMB_H
#define CTNS_COMB_H

#include <iostream>
#include <tuple>
#include <vector>
#include <string>
#include <map>
#include "ctns_rbasis.h"
#include "ctns_qtensor.h"

namespace ctns{

// coordinates (i,j) for sites of ctns
using comb_coord = std::pair<int,int>;
std::ostream& operator <<(std::ostream& os, const comb_coord& coord);
const comb_coord coord_vac = std::make_pair(-2,-2); 
const comb_coord coord_phys = std::make_pair(-1,-1);  
extern const comb_coord coord_vac, coord_phys;

// node information for sites of ctns
struct node{
   public:
      friend std::ostream& operator <<(std::ostream& os, const node& nd);
   public:
      int pindex; // physical index
      int type;	  // type of node: 0 [boundary], 1 [backbone], 2 [branch], 3 [internal]
      comb_coord center; // c-neighbor
      comb_coord left;   // l-neighbor
      comb_coord right;  // r-neighbor
      std::vector<int> rsupport;
      std::vector<int> lsupport;
};

// sweep sequence for optimization of ctns 
using directed_bond = std::tuple<comb_coord,comb_coord,bool>;

// topology information of ctns
struct topology{
   public:
      topology(const std::string& topology_file); 
      void print() const;
      // node
      const node& get_node(const comb_coord& p) const{ return nodes[p.first][p.second]; }
      // helper for support 
      std::vector<int> support_rest(const std::vector<int>& rsupp) const;
      // sweep sequence 
      std::vector<directed_bond> get_sweeps(const bool debug=true) const;
   public:
      int nbackbone, nphysical;
      std::vector<std::vector<node>> nodes; // nodes on comb
      std::vector<comb_coord> rcoord; // coordinate of each node in rvisit order
      				      // used in constructing right environment
      int iswitch; // for i<=iswitch on backbone, size(lsupp)<size(rsupp)
      std::vector<int> image2; // 1D ordering of CTNS for |n_p...> 
};

// comb tensor network states 
template <typename Tm>	
class comb{
   public:
      comb(const topology& topo1): topo(topo1) {}
      // helpers
      int get_nphysical() const{ return topo.nphysical; }
      int get_nstate() const{
	 assert(rwfuns.rows() == 1); // only one symmetry sector
	 return rwfuns.qrow.get_dim(0);
      }
      qsym get_sym_state() const{
	 assert(rwfuns.rows() == 1); // only one symmetry sector
         return rwfuns.qrow.get_sym(0);
      }
      qtensor2<Tm> get_state(const int istate) const{
         assert(rwfuns.rows() == 1);
	 qsym_space qrow({{rwfuns.qrow.get_sym(0),1}});
	 qtensor2<Tm> rwfun(rwfuns.sym, qrow, rwfuns.qcol, rwfuns.dir);
	 const auto& blk0 = rwfuns(0,0);
	 auto& blk = rwfun(0,0);
	 for(int ic=0; ic<rwfuns.qcol.get_dim(0); ic++){
	    blk(0,ic) = blk0(istate,ic);
	 }
	 return rwfun;
      }

//      // --- neightbor ---
//      int get_kp(const comb_coord& p) const{ return topo[p.first][p.second]; }
//      comb_coord get_c(const comb_coord& p) const{ return std::get<0>(neighbor.at(p)); }
//      comb_coord get_l(const comb_coord& p) const{ return std::get<1>(neighbor.at(p)); }
//      comb_coord get_r(const comb_coord& p) const{ return std::get<2>(neighbor.at(p)); }
//      bool ifbuild_c(const comb_coord& p) const{ return get_c(p) == std::make_pair(-1,-1); }
//      bool ifbuild_l(const comb_coord& p) const{ return type.at(get_l(p)) == 0; }
//      bool ifbuild_r(const comb_coord& p) const{ return type.at(get_r(p)) == 0; }
//      // --- environmental quantum numbers --- 
//      qsym_space get_qc(const comb_coord& p) const{
//         auto pc = get_c(p);
//	 bool physical = (pc == std::make_pair(-1,-1));
//         return physical? phys_qsym_space : rsites.at(pc).qrow; 
//      }
//      qsym_space get_ql(const comb_coord& p) const{
//         auto pl = get_l(p);
//         bool cturn = (type.at(pl) == 3 and p.second == 1);
//	 return cturn? lsites.at(pl).qmid : lsites.at(pl).qcol;
//      }
//      qsym_space get_qr(const comb_coord& p) const{
//         auto pr = get_r(p);
//         return rsites.at(pr).qrow;
//      }

   public:
      topology topo;
      std::map<comb_coord,renorm_basis<Tm>> rbases; // renormalized basis from SCI
      std::map<comb_coord,qtensor3<Tm>> rsites; // right canonical form 
      qtensor2<Tm> rwfuns; // wavefunction at the left boundary -*-
      std::map<comb_coord,qtensor3<Tm>> lsites; // left canonical form 
      //std::vector<qtensor3<Tm>> psi; // propagation of initial guess 
};

} // ctns

#endif
