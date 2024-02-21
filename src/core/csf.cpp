#include "csf.h"
#include "spin.h"

using namespace fock;

csfspace fock::get_csf_space(const int k, const int n, const int ts){
   std::cout << "get_csf_space" << std::endl;
   assert(k <= 32); // no. of spatial orbitals should be smaller than 32
   csfspace space;
   csfstate vacuum(k);
   space.push_back(vacuum);
   // gradually construct FCI space
   for(int i=0; i<k; i++){
      std::cout << "\ni=" << i << std::endl;
      csfspace space_new;
      int kres = k-i-1;
      size_t idx = 0;
      for(const auto& state : space){
         std::cout << " idx=" << idx << " state=" << state.repr << std::endl;
         idx += 1;
         // produce new state
         for(int d=0; d<4; d++){
            auto state0 = state;
            state0.repr[2*i] = d%2;
            state0.repr[2*i+1] = d/2;
            int nelec = state0.nelec();
            int twos = state0.twos();
            // check whether state is acceptible
            if(twos >= 0 &&
               (nelec <= n && nelec+2*kres >= n)){
               int nres = n - nelec;
               int tsmin = (nres%2==0)? 0 : 1;
               int tsmax = (nres<=kres)? nres : 2*kres-nres; 
               
               std::cout << "  state0=" << state0.repr
                  << " twos,tsmin,tsmax,ts="
                  << twos << "," << tsmin << "," 
                  << tsmax << "," << ts 
                  << std::endl;
               
               // twos + {tsmin,...,tsmax} => ts
               for(int tsval=tsmin; tsval<=tsmax; tsval+=2){
                  std::cout << "    tsval=" << tsval
                     << " spin_triangle(twos,tsval,ts)=" << spin_triangle(twos,tsval,ts)
                     << std::endl;
                  if(spin_triangle(twos,tsval,ts)){ 
                     space_new.push_back(state0);
                     break;
                  }
               }
               //space_new.push_back(state0);
            }
         }
      }
      // apply selection?
      space = std::move(space_new);
      // count the number of dimensions?
   } 
   // check dimension of the CSF space
   assert(space.size() == dim_csf_space(k,n,ts)); 
   return space;
}
