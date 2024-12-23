#ifndef OPER_RENORM_OPS_H
#define OPER_RENORM_OPS_H

namespace ctns{

#ifndef SERIAL
   // ZL@2024/12/22 opS
   template <typename Qm, typename Tm>
      void oper_renorm_opS_kernel(const std::string superblock,
            const comb<Qm,Tm>& icomb,
            const integral::two_body<Tm>& int2e,
            const input::schedule& schd,
            const stensor3su2<Tm>& site,
            const qoper_dict<Qm::ifabelian,Tm>& qops1,
            const qoper_dict<Qm::ifabelian,Tm>& qops2,
            qoper_dict<Qm::ifabelian,Tm>& qops,
            const int size,
            const int rank){
         std::cout << "error: no implementation of oper_renorm_opS_kernel for su2!" << std::endl;
         exit(1);
      }
   template <typename Qm, typename Tm>
      void oper_renorm_opS_kernel(const std::string superblock,
            const comb<Qm,Tm>& icomb,
            const integral::two_body<Tm>& int2e,
            const input::schedule& schd,
            const stensor3<Tm>& site,
            const qoper_dict<Qm::ifabelian,Tm>& qops1,
            const qoper_dict<Qm::ifabelian,Tm>& qops2,
            qoper_dict<Qm::ifabelian,Tm>& qops,
            const int size,
            const int rank){
         auto sindex = oper_index_opS(qops.krest, qops.ifkr);
         for(const auto& index : sindex){
            if(rank==0 and schd.ctns.verbose>1){
               std::cout << " opS: index=" << index << std::endl;
            }
            // compute opS[p](iproc) 
            auto opxwf = oper_compxwf_opS(superblock, site, qops1, qops2, int2e, index, size, rank, schd.ctns.ifdist1);
            auto op = contract_qt3_qt3(superblock, site, opxwf);
            // reduction of op
            int iproc = distribute1(qops.ifkr, size, index);
            mpi_wrapper::reduce(icomb.world, op.data(), op.size(), iproc);
            if(iproc == rank){
               auto& opS = qops('S')[index];
               linalg::xcopy(op.size(), op.data(), opS.data());
            }
         }
      }

   template <typename Qm, typename Tm>
      void oper_renorm_opS(const std::string superblock,
            const comb<Qm,Tm>& icomb,
            const integral::two_body<Tm>& int2e,
            const input::schedule& schd,
            const qtensor3<Qm::ifabelian,Tm>& site,
            const qoper_dict<Qm::ifabelian,Tm>& qops1,
            const qoper_dict<Qm::ifabelian,Tm>& qops2,
            qoper_dict<Qm::ifabelian,Tm>& qops){
         int size = icomb.world.size();
         int rank = icomb.world.rank();
         const bool ifdist1 = schd.ctns.ifdist1;
         const bool ifdistc = schd.ctns.ifdistc;
         const bool ifdists = schd.ctns.ifdists;
         const int alg_renorm = schd.ctns.alg_renorm;
         const bool ifab = Qm::ifabelian;
         const int isym = Qm::isym;
         const bool ifkr = Qm::ifkr;
         const bool debug = (rank == 0); 
         if(debug and schd.ctns.verbose>0){ 
            std::cout << "ctns::oper_renorm_opS"
               << " superblock=" << superblock 
               << " ifab=" << ifab
               << " isym=" << isym 
               << " ifkr=" << ifkr
               << " ifdist1=" << ifdist1
               << " ifdists=" << ifdists 
               << " alg_renorm=" << alg_renorm	
               << " mpisize=" << size
               << std::endl;
         }
         assert(ifdist1 and ifdists);
         auto t0 = tools::get_time();

         /*
         // loop over opS[p]
         for(const auto& index : sindex){
            auto opS = symbolic_compxwf_opS<Tm>(oplist1, oplist2, block1, block2, cindex1, cindex2,
                  int2e, index, isym, ifkr, size, rank, ifdist1, ifdistc);
            // opS can be empty for ifdist1=true
            if(opS.size() == 0) continue;
            formulae.append(std::make_tuple('S', index, opS));
            counter["S"] += opS.size();	   
            if(ifsave){
               std::cout << "idx=" << idx++;
               opS.display("opS["+std::to_string(index)+"]", print_level);
            }
         }
         */
         if(alg_renorm == 0){

            oper_renorm_opS_kernel(superblock, icomb, int2e, schd, site, qops1, qops2, qops, size, rank);

         }else if(alg_renorm == 1){
         
         }else if(alg_renorm == 2){

         }else if(alg_renorm == 4){

         }else if(alg_renorm == 6 || alg_renorm == 7 || alg_renorm == 8 || alg_renorm == 9){

#ifdef GPU
         }else if(alg_renorm == 16 || alg_renorm == 17 || alg_renorm == 18 || alg_renorm == 19){


#endif
         }else{
            std::cout << "error: no such option for alg_renorm=" << alg_renorm << std::endl;
            exit(1);
         } // alg_renorm

         auto t1 = tools::get_time();
         if(debug){
            double t_tot = tools::get_duration(t1-t0);
            std::cout << "----- TIMING FOR oper_renorm_opS : " << t_tot << " S"
               << " rank=" << rank << " -----"
               << std::endl;
         }
      }
#endif

} // ctns

#endif
