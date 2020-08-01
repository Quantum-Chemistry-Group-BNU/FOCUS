#ifndef CTNS_COMB_H
#define CTNS_COMB_H

#include "ctns_topo.h"
//#include "../core/integral.h"
//#include "../core/onspace.h"
//#include "../core/matrix.h"
//#include "tns_pspace.h"
//#include "tns_qtensor.h"
//#include <tuple>
//#include <vector>
//#include <string>

namespace ctns{

// --- comb tensor network states ---

template <typename Tm>	
class comb{
   public:
      comb(const topology topo1, const int isym1): topo(topo1), isym(isym1) {}

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
//      // --- right canonical form ---
//      // build site tensor from {|r>} bases
//      void rcanon_init(const fock::onspace& space,
//		       const std::vector<std::vector<double>>& vs,
//		       const double thresh_proj); // =1.e-14
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
      short isym; // =0, (N,Na); =1, (N) [symmetry of comb]
//      std::map<comb_coord,renorm_basis> rbases;
//      std::map<comb_coord,qtensor3> rsites; // right canonical form 
//      std::map<comb_coord,qtensor3> lsites; // left canonical form 
//      std::vector<qtensor3> psi; // propagation of initial guess 
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
