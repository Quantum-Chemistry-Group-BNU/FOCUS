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
   //      |      c---*
   //  l---*---r      |
   //		            l
   struct node{
      private:
         // serialize
         friend class boost::serialization::access;
         template<class Archive>
            void serialize(Archive & ar, const unsigned int version){
               ar & lindex & porb & type & center & left & right
                  & rsupport & lsupport
                  & corbs & lorbs & rorbs;
            }
      public:
         friend std::ostream& operator <<(std::ostream& os, const node& nd);
      public:
         int lindex; // position in the occupation number vector
         int porb; // physical index: p-th spatial orbital; =-1 for internal sites 
         int type; // type of node: 0 [boundary], 1 [backbone], 2 [branch], 3 [internal]
         comb_coord center; // c-neighbor
         comb_coord left;   // l-neighbor
         comb_coord right;  // r-neighbor
                            //				    |
                            // the bipartite bond is chosen as -|-*-- [remove the bond leads to two parts]
                            //
         std::vector<int> lsupport;  // orbitals in the left part
         std::vector<int> rsupport;  // orbitals in the right part
                                     //              c
                                     //  	      |	
                                     // orbitals: l--*--r [remove the dot leads to three parts] (lorbs=lsupport)
                                     //
         std::vector<int> corbs;
         std::vector<int> lorbs;
         std::vector<int> rorbs;
   };

   // directed_bond used in sweep sequence for optimization of ctns: (p0,p1,forward)
   struct directed_bond{
      public:
         // constructor
         directed_bond(const comb_coord _p0, 
               const comb_coord _p1, 
               const bool _forward): 
            p0(_p0), p1(_p1), forward(_forward) {}
         // node to be updated
         comb_coord get_current() const{ return forward? p0 : p1; }
         comb_coord get_next() const{ return forward? p1 : p0; }
         //
         // cturn: a bond that at the turning points to branches
         //              |
         //           ---*(i,1)
         //       |      I      |
         //    ---*------*------*---
         //               (i,0)
         //
         bool is_cturn() const{ return p0.second == 0 && p1.second == 1; }
      public:
         friend std::ostream& operator <<(std::ostream& os, const directed_bond& dbond);
      public:
         comb_coord p0, p1;
         bool forward;
   };

   // topology information of ctns
   struct topology{
      private:
         // serialize
         friend class boost::serialization::access;
         template<class Archive>
            void serialize(Archive & ar, const unsigned int version){
               ar & ifmps & ntotal & nbackbone & nphysical
                  & nodes & rcoord & rindex & image2;
            }
      public:
         topology(){};
         void read(const std::string& topology_file, const bool debug=true); 
         void print() const;
         std::vector<int> get_image1() const;
         // helpers
         const node& get_node(const comb_coord& p) const{ return nodes[p.first][p.second]; }
         int get_type(const comb_coord& p) const{ return nodes[p.first][p.second].type; }
         // supports 
         std::vector<int> get_supp_rest(const std::vector<int>& rsupp) const;
         std::vector<int> get_suppc(const comb_coord& p) const{ return get_node(p).corbs; }
         std::vector<int> get_suppl(const comb_coord& p) const{ return get_node(p).lorbs; }
         std::vector<int> get_suppr(const comb_coord& p) const{ return get_node(p).rorbs; }
         // sweep related 
         std::vector<directed_bond> get_sweeps(const bool debug=false) const;
         std::vector<directed_bond> get_mps_fsweeps(const bool debug=false) const;
         std::vector<directed_bond> get_mps_bsweeps(const bool debug=false) const;
         std::vector<directed_bond> get_mps_sweeps(const bool debug=false) const;
         std::vector<int> check_partition(const int dots, const directed_bond& dbond, 
               const bool debug, const int verbose=0) const;
         // get qops around a dot p 
         std::string get_fqop(const comb_coord& p,
               const std::string kind,
               const std::string scratch) const;
         // get qops around a bond configuration in sweep 
         std::vector<std::string> get_fqops(const int dots,
               const directed_bond& dbond,
               const std::string scratch,
               const bool debug=false) const;
         // get a pair (frop,fdel) for a bond
         std::pair<std::string,std::string> get_fbond(const directed_bond& dbond,
               const std::string scratch,
               const bool debug=false) const;
      public:
         bool ifmps = true;
         int ntotal, nbackbone, nphysical;
         // nodes on comb
         std::vector<std::vector<node>> nodes; 
         // coordinate of each node in rvisit order ("sliced from right")
         // used in constructing right environment
         std::vector<comb_coord> rcoord;  // idx->(i,j) 
         std::map<comb_coord,int> rindex; // (i,j)->idx
         std::vector<int> image2; // 1D ordering of CTNS for mapping |n_p...> in ctns_alg 
   };

} // ctns

#endif
