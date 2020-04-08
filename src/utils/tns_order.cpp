#include "tns.h"
#include <numeric>
#include <algorithm> // swap

using namespace std;
using namespace fock;
using namespace linalg;
using namespace tns;

// compute SvN for permuted orbitals
void tns::bipartite_entanglement(const onspace& space,
 	                         const vector<vector<double>>& vs,
			         const vector<int>& order,
			         vector<int>& bdims,
			         double& SvN){
   // image2
   int k = order.size();
   vector<int> image2(2*k);
   for(int i=0; i<k; i++){
      image2[2*i] = 2*order[i];
      image2[2*i+1] = 2*order[i]+1;
   }
   // update basis vector and signs 
   onspace sci_space2;
   vector<int> sgns;
   for(const auto& state : space){
      sci_space2.push_back(state.permute(image2));
      sgns.push_back(state.permute_sgn(image2));
   }
   int dim = space.size();
   int nroot = vs.size();
   vector<vector<double>> vs2(nroot);
   for(int i=0; i<nroot; i++){
      vs2[i].resize(dim);
      transform(vs[i].begin(),vs[i].end(),sgns.begin(),vs2[i].begin(),
	        [](const double& x, const int& y){ return x*y; });
   }
   // (n-1) bipartitions of spatial lattice 
   SvN = 0.0;
   for(int pos=1; pos<k; pos++){
      tns::product_space pspace2;
      pspace2.get_pspace(sci_space2, 2*pos);
      auto pr = pspace2.projection(vs2);
      SvN += pr.second;
      bdims.push_back(pr.first);
   }
}

// brute-force  
void tns::ordering_bf(const onspace& space, 
		      const vector<vector<double>>& vs,
		      vector<int>& order,
		      double& Smin){
   cout << "\ntns::ordering_bf" << endl;
   int k = space[0].size();
   vector<int> sord(k/2);
   iota(sord.begin(),sord.end(),0);
   int idx = 0;
   Smin = 1.e10;
   do{
       // generate permutation of spatial sites
       vector<int> bdims;
       double SvN;
       bipartite_entanglement(space, vs, sord, bdims, SvN);
       // save data
       if(SvN < Smin){
	  Smin = SvN;
	  order = sord;
       }
       // check
       {
          cout << "idx=" << idx << " : ";	   
          for(int i : sord) cout << i << " ";
          cout << "SvN=" << defaultfloat << setprecision(12) << SvN 
               << " bdims=";
          for(int i : bdims) cout << i << " ";  
          cout << endl;
       }
       idx++;
   }while(next_permutation(sord.begin(), sord.end()));
}

// ordering computed by genetic algorithm
void tns::ordering_ga(const onspace& space, 
		      const vector<vector<double>>& vs,
		      vector<int>& order,
		      double& Smin){
   cout << "\ntns::ordering_ga" << endl;
   // compute exact solution by brute-force enumeration
   bool debug = false;
   if(debug){
      ordering_bf(space, vs, order, Smin);
      cout << "Smin=" << Smin << " : ";
      for(int i : order) cout << i << " ";
      cout << endl;
   }
   // parameters
   const int nelite = 1;
   int popsize = 100;
   int maxgen = 1000;
   double crxprob = 0.8;
   double mutprob = 0.2;
   // init population by random sequence
   int k = space[0].size()/2;
   GApop pop(k, popsize);
   pop.eval_fitness(space, vs);
   // Evolution
   uniform_real_distribution<double> rdist(0,1);
   for(int igen=0; igen<maxgen; igen++){
      discrete_distribution<int> dist(pop.fitness.begin(), pop.fitness.end());
      vector<vector<int>> newpop(popsize);
      // elitism
      for(int i=0; i<nelite; i++){
	 newpop[i] = pop.pop[i];
      }
      // start evolution
      for(int i=nelite; i<popsize; i++){
         double r = rdist(tools::generator);
         // crossoverPMX
	 vector<int> offspring;
         if(r < crxprob){
	    int i1 = dist(tools::generator);
	    int i2 = dist(tools::generator);
	    offspring = pop.crossover(i1,i2);
	 }else{
	    int i1 = dist(tools::generator);
	    offspring = pop.pop[i1];
	 }
         // muationSWAP
         r = rdist(tools::generator);
	 if(r < mutprob){
	    int i12 = k*(k-1)/2*rdist(tools::generator);
	    auto pr = tools::inverse_pair0(i12); 
	    int i1 = pr.first, i2 = pr.second;
	    swap(offspring[i1],offspring[i2]);
	 }
	 newpop[i] = offspring;
      }
      pop.pop = newpop;
      pop.eval_fitness(space, vs);
      // print
      cout << "igen=" << igen << endl;
      for(int i=0; i<min(popsize,5); i++){
         cout << " candidate " << i << defaultfloat << setprecision(12) 
	      << " SvN=" << 1.0/pop.fitness[i] << " pm=";
         for(int j : pop.pop[i]) cout << j << " ";
         cout << endl;
      }
   } // igen
}
