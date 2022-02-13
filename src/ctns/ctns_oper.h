#ifndef CTNS_OPER_H
#define CTNS_OPER_H

#include "oper_env.h"

namespace ctns{

// Hij = <CTNS[i]|H|CTNS[j]>
template <typename Km>
linalg::matrix<typename Km::dtype> get_Hmat(const comb<Km>& icomb, 
		            		    const integral::two_body<typename Km::dtype>& int2e,
		            		    const integral::one_body<typename Km::dtype>& int1e,
		            		    const double ecore,
		            		    const std::string scratch,
					    const int algorithm=0){
   int size = 1, rank = 0;
#ifndef SERIAL
   size = icomb.world.size();
   rank = icomb.world.rank();
#endif   
   
   // build operators for environement
   oper_env_right(icomb, int2e, int1e, scratch, algorithm);

   // load operators from file
   using Tm = typename Km::dtype;
   oper_dict<Tm> qops;
   auto p = std::make_pair(0,0); 
   auto fname = oper_fname(scratch, p, "r");
   oper_load(fname, qops);
 
   //if(rank == 0) std::cout << "\nctns::get_Hmat" << std::endl;
   auto Hmat = qops('H')[0].to_matrix();
   if(rank == 0){ 
      Hmat += ecore*linalg::identity_matrix<Tm>(Hmat.rows()); // avoid repetition
   }
   // deal with rwfuns(istate,ibas): Hij = w*[i,a] H[a,b] w[j,b] = (w^* H w^T) 
   auto wfmat = icomb.rwfuns.to_matrix();
   auto tmp = linalg::xgemm("N","T",Hmat,wfmat);
   Hmat = linalg::xgemm("N","N",wfmat.conj(),tmp);
#ifndef SERIAL
   // reduction of partial H formed on each processor
   if(size > 1){
      linalg::matrix<Tm> Hmat2(Hmat.rows(),Hmat.cols());
      boost::mpi::reduce(icomb.world, Hmat, Hmat2, std::plus<linalg::matrix<Tm>>(), 0);
      Hmat = Hmat2;
   }
#endif 
   return Hmat;
}

} // ctns

#endif
