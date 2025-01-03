#ifndef CTNS_OPER_H
#define CTNS_OPER_H

#include "oper_env.h"

#ifndef SERIAL
#include "../core/mpi_wrapper.h"
#endif

namespace ctns{

   // Hij = <CTNS[i]|H|CTNS[j]> [full construction]
   template <typename Qm, typename Tm>
      linalg::matrix<Tm> get_Hmat(comb<Qm,Tm>& icomb, // icomb may be modified ifoutcore=true 
            const integral::two_body<Tm>& int2e,
            const integral::one_body<Tm>& int1e,
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
         // load operators at (0,0) from file
         qoper_dict<Qm::ifabelian,Tm> qops;
         auto p = std::make_pair(0,0); 
         auto fname = oper_fname(scratch, p, "r");
         oper_load(schd.ctns.iomode, fname, qops, (rank==0));
         // transform Hmat
         auto Hmat = qops('H').at(0).to_matrix();
         if(rank == 0) Hmat += ecore*linalg::identity_matrix<Tm>(Hmat.rows()); // avoid repetition    
         // deal with rwfuns(istate,ibas): Hij = w*[i,a] H[a,b] w[j,b] = (w^* H w^T)
         auto wfmat = icomb.get_wf2().to_matrix();
         auto tmp = linalg::xgemm("N","T",Hmat,wfmat);
         Hmat = linalg::xgemm("N","N",wfmat.conj(),tmp);
#ifndef SERIAL
         // reduction of partial H formed on each processor if ifdist1 = false
         if(size > 1 and !schd.ctns.ifdist1){
            mpi_wrapper::allreduce(icomb.world, Hmat.data(), Hmat.size());
         }
#endif 
         return Hmat;
      }

   template <typename Qm, typename Tm>
      linalg::matrix<Tm> oper_final(const comb<Qm,Tm>& icomb, 
            const integral::two_body<Tm>& int2e,
            const integral::one_body<Tm>& int1e,
            const double ecore,
            const input::schedule& schd,
            const std::string scratch,
            qoper_pool<Qm::ifabelian,Tm>& qops_pool,
            const int isweep){
         int size = 1, rank = 0;
#ifndef SERIAL
         size = icomb.world.size();
         rank = icomb.world.rank();
#endif   
         if(rank == 0){
            std::cout << "\nctns::oper_final - build Hmat for RCF at isweep=" << isweep << std::endl;
         }

         // c-R0-[R1]-... & c-[R0]-R1-...
         std::vector<comb_coord> plst = {std::make_pair(1,0),std::make_pair(0,0)};
         for(const auto& pcoord : plst){ 
            if(rank == 0) std::cout << "\ncoord=" << pcoord << std::endl;

            // a. get operators from memory / disk    
            std::vector<std::string> fneed(2);
            fneed[0] = icomb.topo.get_fqop(pcoord, "c", scratch);
            fneed[1] = icomb.topo.get_fqop(pcoord, "r", scratch);
            qops_pool.fetch_to_memory(fneed, schd.ctns.alg_renorm>10);
            const auto& cqops = qops_pool.at(fneed[0]);
            const auto& rqops = qops_pool.at(fneed[1]);
            if(rank==0 && schd.ctns.verbose>0){
               cqops.print("cqops");
               rqops.print("rqops");
            }

            // b. perform renormalization for superblock {|cr>}
            std::string frop = oper_fname(scratch, pcoord, "r");
            std::string superblock = "cr";
            std::string fname, fmmtask;
            dot_timing timing_local;

           //xiangchunyang 20241220
            icomb.world.barrier();

            oper_renorm(superblock, icomb, pcoord, int2e, int1e, schd,
                  cqops, rqops, qops_pool[frop], fname, timing_local, fmmtask);

           //xiangchunyang 20241220
            icomb.world.barrier();
            // c. save operators to disk
            qops_pool.join_and_erase(fneed);
            qops_pool.save_to_disk(frop, schd.ctns.async_save);
         }
         
         // transform Hmat
         auto fname = oper_fname(scratch, plst[1], "r");
         const auto& qops = qops_pool.at(fname);
         auto Hmat = qops('H').at(0).to_matrix();
         if(rank == 0) Hmat += ecore*linalg::identity_matrix<Tm>(Hmat.rows()); // avoid repetition    
         // deal with rwfuns(istate,ibas): Hij = w*[i,a] H[a,b] w[j,b] = (w^* H w^T)
         auto wfmat = icomb.get_wf2().to_matrix();
         auto tmp = linalg::xgemm("N","T",Hmat,wfmat);
         Hmat = linalg::xgemm("N","N",wfmat.conj(),tmp);
#ifndef SERIAL
         // reduction of partial H formed on each processor if ifdist1 = false
         if(size > 1 and !schd.ctns.ifdist1){
            mpi_wrapper::allreduce(icomb.world, Hmat.data(), Hmat.size());
         }
#endif 
         if(rank == 0){
            std::cout << std::endl;
            Hmat.print("Hmat_isweep"+std::to_string(isweep), schd.ctns.outprec);
            std::cout << std::endl;
         }
         return Hmat;
      }

} // ctns

#endif
