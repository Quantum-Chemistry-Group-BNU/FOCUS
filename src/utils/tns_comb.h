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
const std::vector<qsym> qphys({{0,0},{1,0},{1,1},{2,0}});
extern const std::vector<qsym> qphys;

const fock::onstate phys_0("00");
const fock::onstate phys_b("10");
const fock::onstate phys_a("01");
const fock::onstate phys_2("11");
const fock::onspace phys({phys_0,phys_b,phys_a,phys_2});
extern const fock::onspace phys;

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
         cout << msg << endl;
	 // qspace0
	 cout << "qspace0 dim0=" << get_dim0() << endl;
	 for(int i=0; i<qspace0.size(); i++){
	    cout << i << ":(" << qspace0[i].first << ","
		 << qspace0[i].second << ") ";
	 } 
	 cout << endl;
 	 // qspace1
	 cout << "qspace1 dim1=" << get_dim1() << endl;
	 for(const auto& p : qspace1){
	    auto sym = p.first;
	    auto dim = p.second;
	    cout << "(" << sym.first << "," << sym.second << "):" << dim << " ";
	 }
	 cout << endl;
	 // qspace
	 cout << "qspace dim=" << get_dim() << endl;
	 for(const auto& p : qspace){
	    auto sym = p.first;
	    auto dim = p.second;
	    cout << "(" << sym.first << "," << sym.second << "):" << dim << " ";
	 }
	 cout << endl;
	 // qblocks
	 if(level >= 1){
	    cout << "qblocks nnz=" << qblocks.size() << endl;
	    for(const auto& p : qblocks){
	       auto& t = p.first;
	       auto& m = p.second;
	       if(m.size() > 0){
	          cout << "block[" << get<0>(t) << "," 
	               << "(" << get<1>(t).first << "," << get<1>(t).second << ")," 
	               << "(" << get<2>(t).first << "," << get<2>(t).second << ")]"
	               << " rows=" << m.rows() << " cols=" << m.cols() 
	               << " size=" << m.size() << endl;
	          if(level >= 2) m.print("mat");
	       }
	    }
	    cout << "total size=" << get_size() << endl;
	 }
      }
   public:
      std::vector<qsym> qspace0;
      std::map<qsym,int> qspace1;
      std::map<qsym,int> qspace;
      std::map<std::tuple<int,qsym,qsym>,linalg::matrix> qblocks; 
};

class comb{
   public:
      void read_topology(std::string topology); 
      void print();
      void init();
      // right canonical form
      void get_rbases(const fock::onspace& space,
		      const std::vector<std::vector<double>>& vs,
		      const double thresh=1.e-6);
      void get_rcanon(const fock::onspace& space,
		      const std::vector<std::vector<double>>& vs,
		      const double thresh=1.e-6);
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
