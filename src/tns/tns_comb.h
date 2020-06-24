#ifndef TNS_COMB_H
#define TNS_COMB_H

#include "../core/integral.h"
#include "../core/onspace.h"
#include "../core/matrix.h"
#include "tns_pspace.h"
#include "tns_qtensor.h"
#include <tuple>
#include <vector>
#include <string>

namespace tns{

// --- comb tensor networks ---

using comb_coord = std::pair<int,int>;

using directed_bond = std::tuple<comb_coord,comb_coord,bool>;

using triple_coord = std::tuple<comb_coord,comb_coord,comb_coord>;

class comb{
   public:
      // --- topology of comb ---
      std::vector<int> support_rest(const std::vector<int>& rsupp);
      void topo_read(std::string topology_file); 
      void topo_init();
      void topo_print() const;
      // --- neightbor ---
      int get_kp(const comb_coord& p) const{ return topo[p.first][p.second]; }
      comb_coord get_c(const comb_coord& p) const{ return std::get<0>(neighbor.at(p)); }
      comb_coord get_l(const comb_coord& p) const{ return std::get<1>(neighbor.at(p)); }
      comb_coord get_r(const comb_coord& p) const{ return std::get<2>(neighbor.at(p)); }
      bool ifbuild_c(const comb_coord& p) const{ return get_c(p) == std::make_pair(-1,-1); }
      bool ifbuild_l(const comb_coord& p) const{ return type.at(get_l(p)) == 0; }
      bool ifbuild_r(const comb_coord& p) const{ return type.at(get_r(p)) == 0; }
      // --- environmental quantum numbers --- 
      qsym_space get_qc(const comb_coord& p) const{
         auto pc = get_c(p);
	 bool physical = (pc == std::make_pair(-1,-1));
         return physical? phys_qsym_space : rsites.at(pc).qrow; 
      }
      qsym_space get_ql(const comb_coord& p) const{
         auto pl = get_l(p);
         bool cturn = (type.at(pl) == 3 and p.second == 1);
	 return cturn? lsites.at(pl).qmid : lsites.at(pl).qcol;
      }
      qsym_space get_qr(const comb_coord& p) const{
         auto pr = get_r(p);
         return rsites.at(pr).qrow;
      }
      // --- boundary site ---
      qtensor3 get_lbsite() const; 
      qtensor3 get_rbsite() const; 
      // --- sweep sequence ---
      std::vector<directed_bond> get_sweeps();
      // --- from SCI wavefunctions ---
      // compute renormalized bases {|r>} from SCI wf 
      void get_rbases(const fock::onspace& space,
		      const std::vector<std::vector<double>>& vs,
		      const double thresh_proj=1.e-14);
      // compute wave function at the start for right canonical form 
      qtensor3 get_rwavefuns(const fock::onspace& space,
		      	     const std::vector<std::vector<double>>& vs,
			     const std::vector<int>& order,
			     const renorm_basis& rbasis);
      // --- right canonical form ---
      // build site tensor from {|r>} bases
      void rcanon_init(const fock::onspace& space,
		       const std::vector<std::vector<double>>& vs,
		       const double thresh_proj); // =1.e-14
      void rcanon_check(const double thresh_ortho, // =1.e-10
		        const bool ifortho=false); // check last site
      // io for rsites
      void rcanon_save(const std::string fname="rcanon.info");
      void rcanon_load(const std::string fname="rcanon.info");
      // --- overlap with SCI wavefunctions --- 
      // <det|Comb[n]> by contracting the Comb
      std::vector<double> rcanon_CIcoeff(const fock::onstate& state);
      // ovlp[m,n] = <SCI[m]|Comb[n]>
      linalg::matrix<double> rcanon_CIovlp(const fock::onspace& space,
		                   const std::vector<std::vector<double>>& vs);
      // sampling of Comb state to get {|det>,p(det)=|<det|Psi[i]>|^2}
      std::pair<fock::onstate,double> rcanon_sampling(const int istate);
      // sampling approach for estimating Sd
      double rcanon_sampling_Sd(const int nsample, const int istate, const int nprt=0);
      // check by explict list all dets in the FCI space
      void rcanon_sampling_check(const int istate);
   public:
      int nbackbone, nphysical, ninternal, nboundary, ntotal;
      int iswitch = -1; // i<=iswitch size_lsupp<size_rsupp;
      std::vector<std::vector<int>> topo; // save site index
      std::map<comb_coord,int> type; // type of nodes 0,1,2
      std::map<comb_coord,triple_coord> neighbor; // internal nodes
      std::vector<comb_coord> rcoord; // coordinate of each node in rvisit order
      std::map<comb_coord,std::vector<int>> rsupport;
      // --- 1D ordering ---
      std::vector<int> image2; // mapping of physical indices
      std::vector<int> orbord; // map orbital to 1D position
      // --- rbases ---
      std::map<comb_coord,renorm_basis> rbases;
      // --- right canonical form ---
      std::map<comb_coord,qtensor3> rsites;
      // --- left canonical form ---
      std::map<comb_coord,std::vector<int>> lsupport;
      std::map<comb_coord,qtensor3> lsites;
      // --- propagation of initial guess ---
      std::vector<qtensor3> psi;
};

linalg::matrix<double> get_Smat(const comb& bra, 
  		        const comb& ket);

linalg::matrix<double> get_Hmat(const comb& bra, 
		        const comb& ket,
		        const integral::two_body<double>& int2e,
		        const integral::one_body<double>& int1e,
		        const double ecore,
		        const std::string scratch);

} // tns

#endif
