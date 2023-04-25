#ifndef CTNS_OPER_H
#define CTNS_OPER_H

#include "oper_env.h"

#ifndef SERIAL
#include "../core/mpi_wrapper.h"
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
   auto Hmat = qops('H')[0].to_matrix();
   if(rank == 0) Hmat += ecore*linalg::identity_matrix<Tm>(Hmat.rows()); // avoid repetition
                                                                         //
   // deal with rwfuns(istate,ibas): Hij = w*[i,a] H[a,b] w[j,b] = (w^* H w^T)
   auto wfmat = icomb.get_wf2().to_matrix();
   auto tmp = linalg::xgemm("N","T",Hmat,wfmat);
   Hmat = linalg::xgemm("N","N",wfmat.conj(),tmp);
#ifndef SERIAL
   // reduction of partial H formed on each processor if ifdist1 = false
   if(size > 1 and !schd.ctns.ifdist1){
      mpi_wrapper::reduce(icomb.world, Hmat.data(), Hmat.size(), 0);
   }
#endif 
   return Hmat;
}

} // ctns

#endif
