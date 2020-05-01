#ifndef TNS_COMB_H
#define TNS_COMB_H

#include "../core/onspace.h"
#include "../core/matrix.h"
#include "tns_pspace.h"
#include "tns_qtensor.h"
#include <vector>
#include <string>

namespace tns{

// --- comb tensor networks ---
using comb_coord = std::pair<int,int>;
using comb_rbases = std::map<comb_coord,renorm_basis>;
class comb{
   public:
      // --- topology of comb ---
      std::vector<int> support_rest(const std::vector<int>& rsupp);
      void topo_read(std::string topology_file); 
      void topo_init();
      void topo_print() const;
      // --- from SCI wavefunctions ---
      // compute renormalized bases {|r>} from SCI wf 
      comb_rbases get_rbases(const fock::onspace& space,
		      	     const std::vector<std::vector<double>>& vs,
		      	     const double thresh_proj=1.e-15);
      // compute wave function at the start for right canonical form 
      qtensor3 get_rwfuns(const fock::onspace& space,
		      	  const std::vector<std::vector<double>>& vs,
			  const std::vector<int>& order,
			  const renorm_basis& rbasis);
      // --- right canonical form ---
      // build site tensor from {|r>} bases
      void rcanon_init(const fock::onspace& space,
		       const std::vector<std::vector<double>>& vs,
		       const double thresh_proj=1.e-15,
		       const double thresh_ortho=1.e-10);
      // <n|Comb[i]>
      std::vector<double> rcanon_coeff(const fock::onstate& state);
      // ovlp[n,m] = <Comb[n]|SCI[m]>
      linalg::matrix rcanon_ovlp(const fock::onspace& space,
		                 const std::vector<std::vector<double>>& vs);
      // io for rsites
      void rcanon_save(const std::string fname="rcanon.info");
      void rcanon_load(const std::string fname="rcanon.info");
   public:
      int nbackbone, nphysical, ninternal, ntotal;
      int iswitch=-1; // i<=iswitch size_lsupp<size_rsupp;
      std::vector<std::vector<int>> topo; // save site index
      std::map<comb_coord,int> type; // type of nodes 0,1,2
      std::vector<comb_coord> rcoord; // coordinate of each node in rvisit order
      std::map<comb_coord,std::vector<int>> rsupport;
      std::vector<int> image2; // mapping of physical indices
      // --- right canonical form ---
      std::map<comb_coord,qtensor3> rsites;
      // --- left canonical form ---
      std::map<comb_coord,std::vector<int>> lsupport;
      std::map<comb_coord,qtensor3> lsites;
      // --- sweep ---
      std::vector<std::pair<comb_coord,comb_coord>> sweep_seq;
};

} // tns

#endif
