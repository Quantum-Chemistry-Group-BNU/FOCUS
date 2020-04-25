#ifndef TNS_COMB_H
#define TNS_COMB_H

#include "../core/serialization.h"
#include "../core/onspace.h"
#include "../core/matrix.h"
#include "tns_pspace.h"
#include <vector>
#include <string>
#include <tuple>
#include <map>

namespace tns{

// --- qsym = (Ne,Na) ---
using qsym = std::pair<int,int>;
using qsym_space = std::map<qsym,int>;


// --- physical degree of freedoms  ---
// symmetry
const std::vector<qsym> phys_sym({{0,0}, {1,0}, {1,1}, {2,1}});
extern const std::vector<qsym> phys_sym;
// states
const fock::onstate phys_0("00"), phys_b("10"), phys_a("01"), phys_2("11");
const fock::onspace phys_space({phys_0, phys_b, phys_a, phys_2});
extern const fock::onspace phys_space;
// qsym_space
const qsym_space phys_qsym_space({{phys_sym[0],1},
			          {phys_sym[1],1},
			          {phys_sym[2],1},
			          {phys_sym[3],1}});
extern const qsym_space phys_qsym_space;
// rbasis for leaves
inline renorm_basis get_rbasis_phys(){
   renorm_basis rbasis(4);
   for(int i=0; i<4; i++){
      rbasis[i].sym = phys_sym[i]; 
      rbasis[i].space.push_back(phys_space[i]);
      rbasis[i].coeff = linalg::identity_matrix(1);
   }
   return rbasis;
}

// --- qspace ---
// total dimension
inline int qsym_space_dim(const qsym_space& qs){
   int dim = 0;
   for(const auto& p : qs) dim += p.second;
   return dim;
}
// print
inline void qsym_space_print(const qsym_space& qs, const std::string& name){ 
   std::cout << name 
	     << " nsym=" << qs.size() 
	     << " dim=" << qsym_space_dim(qs) 
	     << std::endl;
   // loop over symmetry sectors
   for(const auto& p : qs){
      auto sym = p.first;
      auto dim = p.second;
      std::cout << " (" << sym.first << "," << sym.second << "):" << dim;
   }
   std::cout << std::endl;
}

// --- rank-3 tensor ---
const std::vector<linalg::matrix> empty_block;
extern const std::vector<linalg::matrix> empty_block;
// <in0,in1|out> = [in0](in1,out)
struct site_tensor{
   private:
      friend class boost::serialization::access;
      template<class Archive>
      void serialize(Archive & ar, const unsigned int version){
         ar & qspace0;
	 ar & qspace1;
	 ar & qspace;
	 ar & qblocks;
      }
   public:
      inline int get_dim0() const{ return qsym_space_dim(qspace0); }
      inline int get_dim1() const{ return qsym_space_dim(qspace1); }
      inline int get_dim() const{ return qsym_space_dim(qspace); }
      void print(const std::string msg, const int level=0){
	 std::cout << "site_tensor: " << msg << std::endl;
	 qsym_space_print(qspace0,"qspace0");
	 qsym_space_print(qspace1,"qspace1");
	 qsym_space_print(qspace,"qspace");
	 if(level >= 1){
	    std::cout << "qblocks: nblocks=" << qblocks.size() << std::endl;
	    int nnz = 0;
	    for(const auto& p : qblocks){
	       auto& t = p.first;
	       auto& m = p.second;
	       auto sym0 = get<0>(t);
	       auto sym1 = get<1>(t);
	       auto sym = get<2>(t);
	       if(m.size() > 0){
		  nnz++;
		  std::cout << "idx=" << nnz << " block[" 
		       << "(" << sym0.first << "," << sym0.second << "),"
	               << "(" << sym1.first << "," << sym1.second << ")," 
	               << "(" << sym.first  << "," << sym.second  << ")]"
	               << " size=" << m.size() 
		       << " rows,cols=(" << m[0].rows() << "," << m[0].cols() << ")" 
		       << std::endl; 
	          if(level >= 2){
		     for(int i=0; i<m.size(); i++){		 
		        m[i].print("mat"+std::to_string(i));
		     }
		  } // level=2
	       }
	    }
	    std::cout << "total no. of nonzero blocks =" << nnz << std::endl;
	 } // level=1
      }
   public:
      qsym_space qspace0; // central [sym,dim]
      qsym_space qspace1; // in [sym,dim]
      qsym_space qspace;  // out [sym,dim]
      std::map<std::tuple<qsym,qsym,qsym>,std::vector<linalg::matrix>> qblocks;
};

// --- comb tensor networks ---
using comb_coord = std::pair<int,int>;
using comb_rbases = std::map<comb_coord,renorm_basis>;
class comb{
   public:
      void read_topology(std::string topology); 
      void init();
      void print();
      // compute renormalized bases {|r>} from SCI wf 
      comb_rbases get_rbases(const fock::onspace& space,
		      	     const std::vector<std::vector<double>>& vs,
		      	     const double thresh_proj=1.e-15);
      // compute wave function at the start for right canonical form 
      site_tensor get_rwfuns(const fock::onspace& space,
		      	     const std::vector<std::vector<double>>& vs,
			     const std::vector<int>& order,
			     const renorm_basis& rbasis);
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
      std::vector<std::vector<int>> topo; // save site index
      std::map<comb_coord,int> type; // type of nodes 0,1,2
      std::vector<comb_coord> rcoord; // coordinate of each node in rvisit order
      std::map<comb_coord,std::vector<int>> rsupport;
      std::vector<int> image2; // mapping of physical indices
      // --- right canonical form ---
      std::map<comb_coord,site_tensor> rsites;
      // --- left canonical form ---
      // std::map<comb_coord,std::vector<int>> lsupport;
      // std::map<comb_coord,site_tensor> lsites;
      // --- sweep ---
      std::vector<std::pair<comb_coord,comb_coord>> sweep_seq;
};

} // tns

#endif
