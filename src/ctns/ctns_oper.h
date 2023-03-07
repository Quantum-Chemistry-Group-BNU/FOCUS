#ifndef CTNS_OPER_H
#define CTNS_OPER_H

#include "oper_env.h"
#ifndef SERIAL
#include "mpi_wrapper.h"
#endif

namespace ctns{

// Hij = <CTNS[i]|H|CTNS[j]>
template <typename Km>
linalg::matrix<typename Km::dtype> get_Hmat(const comb<Km>& icomb, 
		            		    const integral::two_body<typename Km::dtype>& int2e,
		            		    const integral::one_body<typename Km::dtype>& int1e,
		            		    const double ecore,
					    const input::schedule& schd,
					    const std::string scratch){
   int size = 1, rank = 0;
#ifndef SERIAL
   size = icomb.world.size();
   rank = icomb.world.rank();
#endif   
   
   // build operators for environement
   oper_env_right(icomb, int2e, int1e, schd, scratch);

   // load operators from file
   using Tm = typename Km::dtype;
   oper_dict<Tm> qops;
   auto p = std::make_pair(0,0); 
   auto fname = oper_fname(scratch, p, "r");
   oper_load(schd.ctns.iomode, fname, qops, (rank==0));
   // communicate
   auto Hmat = qops('H')[0].to_matrix();
   if(rank == 0) Hmat += ecore*linalg::identity_matrix<Tm>(Hmat.rows()); // avoid repetition
#ifndef SERIAL
   if(!schd.ctns.ifdist1 and size > 1){
      // reduction of partial H formed on each processor if ifdist1 = false
      linalg::matrix<Tm> Hmat2(Hmat.rows(),Hmat.cols());
      mpi_wrapper::reduce(icomb.world, Hmat.data(), Hmat.size(), Hmat2.data(), std::plus<Tm>(), 0);
      Hmat = Hmat2;
   }
#endif 
   return Hmat;
}

} // ctns

#endif
