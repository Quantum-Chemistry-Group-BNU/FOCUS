#ifndef TNS_COMB_H
#define TNS_COMB_H

#include "../core/onspace.h"
#include "../core/matrix.h"
#include "tns_pspace.h"
#include <vector>
#include <string>
#include <tuple>
#include <map>

namespace tns{

using qsym = std::pair<int,int>;

// physical indieces
const std::vector<qsym> qphys({{0,0},{1,0},{1,1},{2,1}});
extern const std::vector<qsym> qphys;

const fock::onstate phys_0("00");
const fock::onstate phys_b("10");
const fock::onstate phys_a("01");
const fock::onstate phys_2("11");
const fock::onspace space_phys({phys_0,phys_b,phys_a,phys_2});
extern const fock::onspace space_phys;

inline renorm_basis get_rbasis_phys(){
   renorm_basis rbasis(4);
   for(int i=0; i<4; i++){
      rbasis[i].sym = qphys[i]; 
      rbasis[i].space.push_back(space_phys[i]);
      rbasis[i].coeff = linalg::identity_matrix(1);
   }
   return rbasis;
}

// <in0,in1|out> = [in0](in1,out)
struct renorm_tensor{
   public:
      inline int get_dim0() const{
	 return qspace0.size(); 
      }
      inline int get_dim1() const{
	 int dim = 0;
	 for(const auto& p : qspace1) dim += p.second;
	 return dim;
      }
      inline int get_dim() const{
	 int dim = 0;
	 for(const auto& p : qspace) dim += p.second;
	 return dim;
      }
      inline int get_size() const{
	 int size = 0;
	 for(const auto& p : qblocks) size += p.second.size();
	 return size;
      }
      void print(std::string msg, const int level=0){
         cout << msg << " level=" << level << endl;
	 // qspace0
	 cout << "qspace0: dim0=" << get_dim0() << endl;
	 for(int i=0; i<qspace0.size(); i++){
	    cout << " " << i << ":(" << qspace0[i].first << ","
		 << qspace0[i].second << ")";
	 } 
	 cout << endl;
 	 // qspace1
	 cout << "qspace1: nsym1=" << qspace1.size() << " dim1=" << get_dim1() << endl;
	 for(const auto& p : qspace1){
	    auto sym = p.first;
	    auto dim = p.second;
	    cout << " (" << sym.first << "," << sym.second << "):" << dim;
	 }
	 cout << endl;
	 // qspace
	 cout << "qspace: nsym=" << qspace.size() << " dim=" << get_dim() << endl;
	 for(const auto& p : qspace){
	    auto sym = p.first;
	    auto dim = p.second;
	    cout << " (" << sym.first << "," << sym.second << "):" << dim;
	 }
	 cout << endl;
	 // qblocks
	 if(level >= 1){
	    cout << "qblocks nblocks=" << qblocks.size() << endl;
	    int nnz = 0;
	    for(const auto& p : qblocks){
	       auto& t = p.first;
	       auto& m = p.second;
	       if(m.size() > 0){
		  nnz++;
	          cout << " block[" << get<0>(t) << "," 
	               << "(" << get<1>(t).first << "," << get<1>(t).second << ")," 
	               << "(" << get<2>(t).first << "," << get<2>(t).second << ")]"
	               << " rows,cols,size=(" << m.rows() << "," << m.cols() << "," 
	               << m.size() << ")" << endl;
	          if(level >= 2) m.print("mat");
	       }
	    }
	    cout << "total nnz=" << nnz << " size=" << get_size() << endl;
	 }
      }
   public:
      std::vector<qsym>  qspace0; // central
      std::map<qsym,int> qspace1; // in [sym,dim]
      std::map<qsym,int> qspace;  // out [sym,dim]
      std::map<std::tuple<int,qsym,qsym>,linalg::matrix> qblocks; 
};

class comb{
   public:
      void read_topology(std::string topology); 
      void init();
      void print();
      // compute renormalized bases {|r>} 
      void get_rbases(const fock::onspace& space,
		      const std::vector<std::vector<double>>& vs,
		      const double thresh_proj=1.e-15); 
      // build site tensor from {|r>} basis
      void get_rcanon(const double thresh_ortho=1.e-10);
   public:
      using coord = std::pair<int,int>;
      int nbackbone, nphysical, ninternal, ntotal;
      std::vector<std::vector<int>> topo; // save site index
      std::map<coord,int> type; // type of nodes 0,1,2
      // right canonical form
      std::vector<coord> rcoord; // coordinate of each node in visit order
      std::map<coord,std::vector<int>> rsupport;
      // right canonical form
      std::map<coord,renorm_basis> rbases;
      std::map<coord,renorm_tensor> rsites;
      // sweep
      std::vector<std::pair<coord,coord>> sweep_seq;
};

} // tns

#endif
