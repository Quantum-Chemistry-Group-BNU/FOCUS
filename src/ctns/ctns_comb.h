#ifndef CTNS_COMB_H
#define CTNS_COMB_H

#include <iostream>
#include <tuple>
#include <vector>
#include <string>
#include <map>
#include "ctns_rbasis.h"
#include "ctns_qtensor.h"
//#include "../core/integral.h"
//#include "../core/onspace.h"
//#include "../core/matrix.h"

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
      comb(const topology topo1): topo(topo1) {}

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

//      // --- boundary site ---
//      qtensor3 get_lbsite() const; 
//      qtensor3 get_rbsite() const; 
//      // --- from SCI wavefunctions ---
//      // compute renormalized bases {|r>} from SCI wf 
//      void get_rbases(const fock::onspace& space,
//		      const std::vector<std::vector<double>>& vs,
//		      const double thresh_proj=1.e-14);
//      // compute wave function at the start for right canonical form 
//      qtensor3 get_rwavefuns(const fock::onspace& space,
//		      	     const std::vector<std::vector<double>>& vs,
//			     const std::vector<int>& order,
//			     const renorm_basis& rbasis);

//      // --- right canonical form (RCF) ---
//      void rcanon_init(const fock::onspace& space,
//		       const std::vector<std::vector<double>>& vs,
//		       const double thresh_proj){
//         ::rcanon_init(rsites, topo, space, vs, thresh_proj)
//      }
//
//      void rcanon_check(const double thresh_ortho, // =1.e-10
//		        const bool ifortho=false); // check last site

//      // io for rsites
//      void rcanon_save(const std::string fname="rcanon.info");
//      void rcanon_load(const std::string fname="rcanon.info");

//      // --- overlap with SCI wavefunctions --- 
//      // <det|Comb[n]> by contracting the Comb
//      std::vector<double> rcanon_CIcoeff(const fock::onstate& state);
//      // ovlp[m,n] = <SCI[m]|Comb[n]>
//      linalg::matrix rcanon_CIovlp(const fock::onspace& space,
//		                   const std::vector<std::vector<double>>& vs);
//      // sampling of Comb state to get {|det>,p(det)=|<det|Psi[i]>|^2}
//      std::pair<fock::onstate,double> rcanon_sampling(const int istate);
//      // sampling approach for estimating Sd
//      double rcanon_sampling_Sd(const int nsample, const int istate, const int nprt=0);
//      // check by explict list all dets in the FCI space
//      void rcanon_sampling_check(const int istate);

   public:
      topology topo;
      std::map<comb_coord,renorm_basis<Tm>> rbases; // renormalized basis from SCI
      std::map<comb_coord,qtensor3<Tm>> rsites; // right canonical form 
      qtensor2<Tm> rwfuns; // wavefunction at the left boundary -*-
      //std::map<comb_coord,qtensor3<Tm>> lsites; // left canonical form 
      //std::vector<qtensor3<Tm>> psi; // propagation of initial guess 
};

//linalg::matrix get_Smat(const comb& bra, 
//  		        const comb& ket);
//
//linalg::matrix get_Hmat(const comb& bra, 
//		        const comb& ket,
//		        const integral::two_body& int2e,
//		        const integral::one_body& int1e,
//		        const double ecore,
//		        const std::string scratch);

} // ctns

#endif
