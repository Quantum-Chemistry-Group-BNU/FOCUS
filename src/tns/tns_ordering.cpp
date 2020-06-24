#include "../core/linalg.h"
#include "tns_pspace.h"
#include "tns_ordering.h"
#include <numeric>
#include <algorithm> // swap

using namespace std;
using namespace fock;
using namespace linalg;
using namespace tns;

// transform space and coefficient upon permutation
void tns::transform_coeff(const onspace& space,
 	                  const vector<vector<double>>& vs,
			  const vector<int>& order,
			  onspace& space2,
			  vector<vector<double>>& vs2){
   // image2
   int k = order.size();
   vector<int> image2(2*k);
   for(int i=0; i<k; i++){
      image2[2*i] = 2*order[i];
      image2[2*i+1] = 2*order[i]+1;
   }
   // update basis vector and signs 
   space2.clear();
   vector<int> sgns;
   for(const auto& state : space){
      space2.push_back(state.permute(image2));
      sgns.push_back(state.permute_sgn(image2));
   }
   int dim = space.size();
   int nroot = vs.size();
   vs2.resize(nroot);
   for(int i=0; i<nroot; i++){
      vs2[i].resize(dim);
      transform(vs[i].begin(),vs[i].end(),sgns.begin(),vs2[i].begin(),
	        [](const double& x, const int& y){ return x*y; });
   }
}

// compute SvN for permuted orbitals
void tns::bipartite_entanglement(const onspace& space,
 	                         const vector<vector<double>>& vs,
			         const vector<int>& order,
			         vector<int>& bdims,
			         double& SvN){
   onspace space2;
   vector<vector<double>> vs2;
   transform_coeff(space, vs, order, space2, vs2);
   // (n-1) bipartitions of spatial lattice 
   bdims.clear();
   SvN = 0.0;
   for(int pos=1; pos<order.size(); pos++){
      tns::product_space pspace2;
      pspace2.get_pspace(space2, 2*pos);
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
   bool debug = false;
   cout << "\ntns::ordering_ga" << endl;
   // parameters for truncation
   const int ndet = 500;
   // parameters fro genetic optimization
   const int nelite = 1;
   int popsize = 50;
   int maxgen = 1000;
   double crxprob = 0.8;
   double mutprob = 0.2;
   // compute exact solution by brute-force enumeration
   if(debug){
      ordering_bf(space, vs, order, Smin);
      cout << "Smin=" << Smin << " : ";
      for(int i : order) cout << i << " ";
      cout << endl;
   }
   // truncation
   int nroot = vs.size();
   int dim = space.size();
   cout << " ndet=" << ndet << " dim=" << dim << endl;
   onspace space2;
   vector<vector<double>> vs2(nroot);
   if(dim < ndet){
      space2 = space;
      vs2 = vs;
   }else{
      vector<double> p2(dim,0.0);
      for(int i=0; i<dim; i++){
         for(int j=0; j<nroot; j++){
            p2[i] += vs[j][i];
         }
      }
      auto index = tools::sort_index(p2, 1);
      space2.resize(ndet);
      for(int i=0; i<ndet; i++){
         space2[i] = space[index[i]];
      }
      for(int j=0; j<nroot; j++){
         vs2[j].resize(ndet);
         for(int i=0; i<ndet; i++){
            vs2[j][i] = vs[j][index[i]];
         }
      }
   }
   // init population by random sequence
   int k = space[0].size()/2;
   GApop pop(k, popsize);
   pop.eval_fitness(space2, vs2);
   // evolution
   uniform_real_distribution<double> rdist(0,1);
   for(int igen=0; igen<maxgen; igen++){
      // roulette wheel selection:
      //discrete_distribution<int> dist(pop.fitness.begin(), pop.fitness.end());
      // rank selection:
      vector<int> rank(popsize);
      iota(rank.begin(),rank.end(),0);
      discrete_distribution<int> dist(rank.rbegin(), rank.rend());
      // evolution 
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
      pop.eval_fitness(space2, vs2);
      // print
      cout << "igen=" << igen << endl;
      for(int i=0; i<min(popsize,5); i++){
         cout << " candidate " << i << defaultfloat << setprecision(12) 
	      << " SvN=" << 1.0/pop.fitness[i] << " pm=";
         for(int j : pop.pop[i]) cout << j << " ";
         cout << endl;
      }
      vector<int> bdims;
      double SvN;
      bipartite_entanglement(space2, vs2, pop.pop[0], bdims, SvN);
      cout << "bdims=";
      for(int i : bdims) cout << i << " ";
      cout << endl;
   } // igen
}

// fiedler ordering
void tns::ordering_fiedler(const vector<double>& data,
		           vector<int>& order){
   cout << "\ntns::ordering_fiedler" << endl;
   int k = tools::inverse_pair(data.size()).first;
   matrix<double> kij(k/2,k/2);
   for(int i=0; i<k; i+=2){
      for(int j=0; j<k; j+=2){
	 int ij = i>j? i*(i+1)/2+j : j*(j+1)/2+i;
	 kij(j/2,i/2) = data[ij];
      } 
   }
   auto lij = -kij;
   for(int i=0; i<k/2; i++){
      for(int j=0; j<k/2; j++){
         lij(i,i) += kij(j,i);
      }
   }
   vector<double> e(k/2);
   matrix<double> lap(lij);
   eig_solver(lap,e,lij);
   cout << "e0=" << e[0] << " e1=" << e[1] << endl;
   for(int i=0; i<k/2; i++) e[i] = lij(i,1);
   order = tools::sort_index(e, 1);
   if(order[0] > order[k/2-1]) reverse(order.begin(), order.end());
   cout << "fiedler ordering : ";
   for(int i : order) cout << i << " ";
   cout << endl;
}
