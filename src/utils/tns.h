#ifndef TNS_H
#define TNS_H

#include "../core/onspace.h"
#include "../core/tools.h"
#include <vector>
#include <tuple>

namespace tns{

// represent space of dets by direct product structure
struct product_space{
   public:
      void get_pspace(const fock::onspace& space, const int n);
      std::pair<int,double> projection(const std::vector<std::vector<double>>& vs,
		      		       const double thresh=1.e-4);
   public:
      // second int is used for indexing in constructing rowA, colB 
      std::map<fock::onstate,int> umapA, umapB;
      fock::onspace spaceA, spaceB; // ordered by appearance
      std::vector<std::vector<std::pair<int,int>>> rowA, colB;  
      int dim, dimA, dimB;
};

// compute SvN for permuted orbitals
void bipartite_entanglement(const fock::onspace& space,
 	                    const std::vector<std::vector<double>>& vs,
			    const std::vector<int>& order,
			    std::vector<int>& bdims,
			    double& SvN);

// brute-force
void ordering_bf(const fock::onspace& space,
	         const std::vector<std::vector<double>>& vs,
	         std::vector<int>& order,
		 double& Smin);

class GApop{
   public:
      // init by random permutations
      GApop(const int k, const int psize){
         tools::perm pm(k);
	 size = psize;
	 pop.resize(size);
	 fitness.resize(size);
         for(int i=0; i<size; i++){
            pop[i] = pm.image;
            pm.shuffle();
         } // i
      }
      // fitness defined as 1/SvN;
      void eval_fitness(const fock::onspace& space,
 	                const std::vector<std::vector<double>>& vs){
         for(int i=0; i<size; i++){
            vector<int> bdims;
            double SvN;
            bipartite_entanglement(space, vs, pop[i], bdims, SvN);
            fitness[i] = 1.0/SvN;
	 } // i
	 // sort by fitness
         auto index = tools::sort_index(fitness);
	 std::vector<std::vector<int>> pop2(size);
         std::vector<double> fitness2(size);
         for(int i=0; i<size; i++){
            pop2[i] = pop[index[i]];
            fitness2[i] = fitness[index[i]]; 
         }
         pop = pop2;
         fitness = fitness2;
      }
      // Partially Mapped Crossover
      std::vector<int> crossover(const int i1, const int i2){
	 auto a = pop[i1];
	 auto b = pop[i2];
         // see https://github.com/sanshar/Block/blob/master/genetic/CrossOver.C
         int nSize = a.size();
         map<int, int> aMap;
         for(int i = 0; i < nSize; i++) aMap.insert(make_pair(a[i], i));
         map<int, int> bMap;
         for(int i = 0; i < nSize; i++) bMap.insert(make_pair(b[i], i));
         vector<int> mask(nSize, 0);
	 uniform_real_distribution<double> rdist(0,1);
	 for(int i = 0; i < nSize; i++){
	    double r = rdist(tools::generator);
	    if(r < 0.5) mask[i] = 1;
	 }
	 vector<int> c(nSize, -1);
         vector<int> bNew(b);
         for(int i = 0; i < nSize; ++i)
         {
           if(mask[i] == 1)
           {
             c[i] = a[i];
             int k = aMap.find(b[i])->second;
             if(mask[k] == 0)
             {
               int j = bMap.find(a[i])->second;
               while(mask[j]) j = bMap.find(a[j])->second;
               swap(bNew[i], bNew[j]);
             }
           }
         }
         for(int i = 0; i < nSize; ++i) if(mask[i] == 0) c[i] = bNew[i];
	 return c;
      }
   public:
      int size;
      std::vector<std::vector<int>> pop;
      std::vector<double> fitness;
};

// compute ordering
void ordering_ga(const fock::onspace& space,
	         const std::vector<std::vector<double>>& vs,
	         std::vector<int>& order,
		 double& Smin);

} // tns

#endif
