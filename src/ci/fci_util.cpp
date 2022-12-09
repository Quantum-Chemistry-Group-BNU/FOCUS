#include "../core/tools.h"
#include "../core/hamiltonian.h"
#include "fci_util.h"

using namespace std;
using namespace fock;
using namespace linalg;
using namespace fci;

// constructor
void product_space::get_pspace(const onspace& space,
      const int istart){
   dimA0 = spaceA.size();
   dimB0 = spaceB.size();
   int udxA = dimA0, udxB = dimB0;
   // construct {U(A), D(A), B(A)}, {U(B), D(B), A(B)}
   int dim = space.size();
   for(int i=istart; i<dim; i++){
      // even bits 
      onstate strA = space[i].get_even();
      auto itA = umapA.find(strA);
      if(itA == umapA.end()){
         spaceA.push_back(strA);
         auto pr = umapA.insert({strA,udxA});
         itA = pr.first;
         udxA += 1;
         rowA.resize(udxA); // reserve additional space for the new row
      };
      // odd bits
      onstate strB = space[i].get_odd();
      auto itB = umapB.find(strB);
      if(itB == umapB.end()){
         spaceB.push_back(strB);
         auto pr = umapB.insert({strB,udxB});
         itB = pr.first;
         udxB += 1;
         colB.resize(udxB);
      }
      rowA[itA->second].emplace_back(itB->second,i);
      colB[itB->second].emplace_back(itA->second,i);
   }
   assert(udxA == spaceA.size());
   assert(udxB == spaceB.size());
   dimA = udxA;
   dimB = udxB;
}

// update Cmn 
void coupling_table::update_Cmn(const onspace& space,
      const int istart,
      const pair<int,int>& key,
      vector<set<int>>& Cmn){
   int dim = space.size(); 
   Cmn.resize(dim);
   for(int i=0; i<dim; i++){
      if(i<istart){
         for(int j=istart; j<dim; j++){
            auto pr = space[i].diff_type(space[j]);
            if(pr == key) Cmn[i].insert(j);
         }
      }else{
         for(int j=0; j<dim; j++){
            auto pr = space[i].diff_type(space[j]);
            if(pr == key) Cmn[i].insert(j);
         }
      }
   }
}

// coupling_table
void coupling_table::get_Cmn(const onspace& space,
      const bool Htype,
      const int istart){
   auto t0 = tools::get_time();
   // C11
   auto key11 = make_pair(1,1); 
   update_Cmn(space, istart, key11, C11);
   // relativistic Hamiltonian
   if(Htype){
      // C01, C10, C20, C02
      auto key01 = make_pair(0,1);
      update_Cmn(space, istart, key01, C01);
      auto key10 = make_pair(1,0);
      update_Cmn(space, istart, key10, C10);
      auto key20 = make_pair(2,0);
      update_Cmn(space, istart, key20, C20);
      auto key02 = make_pair(0,2);
      update_Cmn(space, istart, key02, C02);
   } // Htype
   auto t1 = tools::get_time();
   cout << "coupling_table::get_Cmn : dim=" << space.size()
      << " timing : " << tools::get_duration(t1-t0) << " s" << endl; 
}
