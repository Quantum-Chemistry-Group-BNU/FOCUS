#ifndef SWEEP_HVEC_H
#define SWEEP_HVEC_H

namespace ctns{

   template <typename Qm, typename Tm>
      struct HVec_wrapper{
         public:
            void init();
            void finalize();
         public:
            HVec_type<Tm> Hx;
         private:
            std::map<std::string,int> oploc;
            Tm* opaddr[5] = {nullptr,nullptr,nullptr,nullptr,nullptr};
            Hx_functors<Tm> Hx_funs; // hvec0
            symbolic_task<Tm> H_formulae; // hvec1,2
            std::map<qsym,QInfo> info_dict; // hvec2
            size_t opsize=0, wfsize=0, tmpsize=0, worktot=0;
            bipart_task<Tm> H_formulae2; // hvec3
            hintermediates<ifab,Tm> hinter; // hvec4,5,6
            Hxlist<Tm> Hxlst; // hvec4
            Hxlist2<Tm> Hxlst2; // hvec5
            HMMtask<Tm> Hmmtask; // hvec6-9,16-19
            HMMtasks<Tm> Hmmtasks; 
            Tm scale;
            size_t blksize=0, blksize0=0;
            double cost=0.0;
            Tm* workspace = nullptr;
#ifdef GPU
            Tm* dev_opaddr[5] = {nullptr,nullptr,nullptr,nullptr,nullptr};
            Tm* dev_workspace = nullptr;
            Tm* dev_red = nullptr;
#endif
            size_t batchsize=0, gpumem_dvdson=0, gpumem_batch=0;
      }

   template <typename Qm, typename Tm>
      void HVec_wrapper<Qm,Tm>::init(const int _dots, 
            const int alg_hvec,
            const int alg_hinter,   
            const int fname,
            const oper_dictmap<Tm>& qops_dict,
            const integral::two_body<Tm>& int2e,
            const double ecore,
            const input::schedule& schd,
            const int& size,
            const int& rank,
            dot_timing& timing){
         const bool debug_formulae = schd.ctns.verbose>0;
         // basic setup
         dots = _dots;
         alg_hvec = _alg_hvec;
         fname =_fname;
         opaddr[0] = qops_dict.at("l")._data;
         opaddr[1] = qops_dict.at("r")._data;
         if(dots == 1){
            opaddr[2] = qops_dict.at("c")._data;
            oploc = {{"l",0},{"r",1},{"c",2}};
         }else{
            opaddr[2] = qops_dict.at("c1")._data;
            opaddr[3] = qops_dict.at("c2")._data;
            oploc = {{"l",0},{"r",1},{"c1",2},{"c2",3}};
         }

         opsize = preprocess_opsize<ifab,Tm>(qops_dict);
         wfsize = preprocess_wfsize<ifab,Tm>(wf.info, info_dict);
         scale = qkind::is_qNK<Qm>()? 0.5*ecore : 1.0*ecore;
         using std::placeholders::_1;
         using std::placeholders::_2;

         // setup HVec 
         timing.tb1 = tools::get_time();
         if(alg_hvec == 0){

            // oldest version
            Hx_funs = twodot_Hx_functors(qops_dict, int2e, ecore, wf, size, rank, 
                  schd.ctns.ifdist1, debug_formulae);
            Hx = bind(&ctns::twodot_Hx<ifab,Tm>, _1, _2, std::ref(Hx_funs),
                  std::ref(wf), std::cref(size), std::cref(rank));

         }else if(alg_hvec == 1){

            // raw version: symbolic formulae + dynamic allocation of memory 
            H_formulae = symbolic_formulae_twodot(qops_dict, int2e, size, rank, fname,
                  schd.ctns.sort_formulae, schd.ctns.ifdist1, schd.ctns.ifdistc, debug_formulae); 
            Hx = bind(&ctns::symbolic_Hx<ifab,Tm,qtensor4<ifab,Tm>>, _1, _2, std::cref(H_formulae),
                  std::cref(qops_dict), std::cref(ecore),
                  std::ref(wf), std::cref(size), std::cref(rank));

         }else if(alg_hvec == 2){ 

            // symbolic formulae + preallocation of workspace 
            H_formulae = symbolic_formulae_twodot(qops_dict, int2e, size, rank, fname,
                  schd.ctns.sort_formulae, schd.ctns.ifdist1, schd.ctns.ifdistc, debug_formulae);
            tmpsize = opsize + 3*wfsize;
            worktot = maxthreads*tmpsize;
            if(debug && schd.ctns.verbose>0){
               std::cout << "preprocess for Hx: opsize=" << opsize << " wfsize=" << wfsize 
                  << " worktot=" << worktot << ":" << tools::sizeMB<Tm>(worktot) << "MB"
                  << ":" << tools::sizeGB<Tm>(worktot) << "GB" << std::endl; 
            }
            workspace = new Tm[worktot];
            Hx = bind(&ctns::symbolic_Hx2<ifab,Tm,qtensor4<ifab,Tm>,qinfo4type<ifab,Tm>>, _1, _2, 
                  std::cref(H_formulae), std::cref(qops_dict), std::cref(ecore), 
                  std::ref(wf), std::cref(size), std::cref(rank), std::cref(info_dict), 
                  std::cref(opsize), std::cref(wfsize), std::cref(tmpsize),
                  std::ref(workspace));

         }else if(alg_hvec == 3){

            // symbolic formulae (factorized) + preallocation of workspace 
            H_formulae2 = symbolic_formulae_twodot2(qops_dict, int2e, size, rank, fname,
                  schd.ctns.sort_formulae, schd.ctns.ifdist1, schd.ctns.ifdistc, debug_formulae); 
            tmpsize = opsize + 4*wfsize;
            worktot = maxthreads*tmpsize;
            if(debug && schd.ctns.verbose>0){
               std::cout << "preprocess for Hx: opsize=" << opsize << " wfsize=" << wfsize 
                  << " worktot=" << worktot << ":" << tools::sizeMB<Tm>(worktot) << "MB"
                  << ":" << tools::sizeGB<Tm>(worktot) << "GB" << std::endl; 
            }
            workspace = new Tm[worktot];
            HVec = bind(&ctns::symbolic_Hx3<ifab,Tm,qtensor4<ifab,Tm>,qinfo4type<ifab,Tm>>, _1, _2, 
                  std::cref(H_formulae2), std::cref(qops_dict), std::cref(ecore), 
                  std::ref(wf), std::cref(size), std::cref(rank), std::cref(info_dict), 
                  std::cref(opsize), std::cref(wfsize), std::cref(tmpsize),
                  std::ref(workspace));

         }else if(alg_hvec == 4){

            // OpenMP + Single Hxlst: symbolic formulae + hintermediates + preallocation of workspace
            H_formulae = symbolic_formulae_twodot(qops_dict, int2e, size, rank, fname,
                  schd.ctns.sort_formulae, schd.ctns.ifdist1, schd.ctns.ifdistc, debug_formulae);

            const bool ifDirect = false;
            const int batchgemv = 1;
            hinter.init(ifDirect, schd.ctns.alg_hinter, batchgemv, qops_dict, oploc, opaddr, H_formulae, debug);

            preprocess_formulae_Hxlist(ifDirect, schd.ctns.alg_hcoper, 
                  qops_dict, oploc, opaddr, H_formulae, wf, hinter,
                  Hxlst, blksize, blksize0, cost, rank==0 && schd.ctns.verbose>0);

            get_MMlist2(Hxlst);

            worktot = maxthreads*(blksize*2+ndim);
            if(debug && schd.ctns.verbose>0){
               std::cout << "preprocess for Hx: ndim=" << ndim << " blksize=" << blksize 
                  << " worktot=" << worktot << ":" << tools::sizeMB<Tm>(worktot) << "MB"
                  << ":" << tools::sizeGB<Tm>(worktot) << "GB" << std::endl; 
            }

            HVec = bind(&ctns::preprocess_Hx<Tm>, _1, _2,
                  std::cref(scale), std::cref(size), std::cref(rank),
                  std::cref(ndim), std::cref(blksize), 
                  std::ref(Hxlst), std::ref(opaddr));

         }else if(alg_hvec == 5){

            // OpenMP + Hxlist2: symbolic formulae + hintermediates + preallocation of workspace
            H_formulae = symbolic_formulae_twodot(qops_dict, int2e, size, rank, fname,
                  schd.ctns.sort_formulae, schd.ctns.ifdist1, schd.ctns.ifdistc, debug_formulae); 

            const bool ifDirect = false;
            const int batchgemv = 1;
            hinter.init(ifDirect, schd.ctns.alg_hinter, batchgemv, qops_dict, oploc, opaddr, H_formulae, debug);

            preprocess_formulae_Hxlist2(ifDirect, schd.ctns.alg_hcoper, 
                  qops_dict, oploc, opaddr, H_formulae, wf, hinter,
                  Hxlst2, blksize, blksize0, cost, rank==0 && schd.ctns.verbose>0);

            get_MMlist2(Hxlst2);

            worktot = maxthreads*blksize*3;
            if(debug && schd.ctns.verbose>0){
               std::cout << "preprocess for Hx: ndim=" << ndim << " blksize=" << blksize 
                  << " worktot=" << worktot << ":" << tools::sizeMB<Tm>(worktot) << "MB"
                  << ":" << tools::sizeGB<Tm>(worktot) << "GB" << std::endl; 
            }

            HVec = bind(&ctns::preprocess_Hx2<Tm>, _1, _2,
                  std::cref(scale), std::cref(size), std::cref(rank),
                  std::cref(ndim), std::cref(blksize), 
                  std::ref(Hxlst2), std::ref(opaddr));

         }else if(alg_hvec == 6 || alg_hvec == 7 || alg_hvec == 8 || alg_hvec == 9){

            // BatchGEMM: symbolic formulae + hintermediates + preallocation of workspace
            H_formulae = symbolic_formulae_twodot(qops_dict, int2e, size, rank, fname,
                  schd.ctns.sort_formulae, schd.ctns.ifdist1, schd.ctns.ifdistc, debug_formulae);

            const bool ifSingle = alg_hvec > 7;
            const bool ifDirect = alg_hvec % 2 == 1;
            const int batchgemv = 1;
            hinter.init(ifDirect, schd.ctns.alg_hinter, batchgemv, qops_dict, oploc, opaddr, H_formulae, debug);

            size_t maxbatch = 0;
            if(!ifSingle){
               preprocess_formulae_Hxlist2(ifDirect, schd.ctns.alg_hcoper, 
                     qops_dict, oploc, opaddr, H_formulae, wf, hinter,
                     Hxlst2, blksize, blksize0, cost, rank==0 && schd.ctns.verbose>0);
               for(int i=0; i<Hxlst2.size(); i++){
                  maxbatch = std::max(maxbatch, Hxlst2[i].size());
               } // i
            }else{
               preprocess_formulae_Hxlist(ifDirect, schd.ctns.alg_hcoper, 
                     qops_dict, oploc, opaddr, H_formulae, wf, hinter,
                     Hxlst, blksize, blksize0, cost, rank==0 && schd.ctns.verbose>0);
               maxbatch = Hxlst.size();
            }
            if(!ifDirect) assert(blksize0 == 0); 

            if(blksize > 0){
               // determine batchsize dynamically
               size_t blocksize = 2*blksize+blksize0;
               preprocess_cpu_batchsize<Tm>(schd.ctns.batchmem, blocksize, maxbatch, 
                     batchsize, worktot);
               if(debug && schd.ctns.verbose>0){
                  std::cout << "preprocess for Hx: ndim=" << ndim << " blksize=" << blksize 
                     << " blksize0=" << blksize0 << " batchsize=" << batchsize
                     << " worktot=" << worktot << ":" << tools::sizeMB<Tm>(worktot) << "MB"
                     << ":" << tools::sizeGB<Tm>(worktot) << "GB" << std::endl; 
               }
               workspace = new Tm[worktot];

               // generate Hmmtasks
               const int batchblas = schd.ctns.alg_hinter; // use the same keyword for GEMM_batch
               auto batchhvec = std::make_tuple(batchblas,batchblas,batchblas);
               if(!ifSingle){
                  Hmmtasks.resize(Hxlst2.size());
                  for(int i=0; i<Hmmtasks.size(); i++){
                     Hmmtasks[i].init(Hxlst2[i], schd.ctns.alg_hcoper, batchblas, batchhvec, batchsize, blksize*2, blksize0);
                     if(debug && schd.ctns.verbose>1 && Hxlst2[i].size()>0){
                        std::cout << " rank=" << rank << " iblk=" << i 
                           << " size=" << Hxlst2[i][0].size 
                           << " Hmmtasks.totsize=" << Hmmtasks[i].totsize
                           << " batchsize=" << Hmmtasks[i].batchsize 
                           << " nbatch=" << Hmmtasks[i].nbatch 
                           << std::endl;
                     }
                  } // i
                  if(fmmtask.size()>0) save_mmtask(Hmmtasks, fmmtask);
               }else{
                  Hmmtask.init(Hxlst, schd.ctns.alg_hcoper, batchblas, batchhvec, batchsize, blksize*2, blksize0);
                  if(debug && schd.ctns.verbose>1){
                     std::cout << " rank=" << rank 
                        << " Hxlst.size=" << Hxlst.size()
                        << " Hmmtask.totsize=" << Hmmtask.totsize
                        << " batchsize=" << Hmmtask.batchsize 
                        << " nbatch=" << Hmmtask.nbatch 
                        << std::endl;
                  }
                  if(fmmtask.size()>0) save_mmtask(Hmmtask, fmmtask);
               }
            } // blksize>0

            if(!ifSingle){
               if(!ifDirect){
                  HVec = bind(&ctns::preprocess_Hx_batch<Tm>, _1, _2,
                        std::cref(scale), std::cref(size), std::cref(rank),
                        std::cref(ndim), std::ref(Hmmtasks), std::ref(opaddr), std::ref(workspace));
               }else{
                  opaddr[4] = workspace + batchsize*blksize*2; // memory layout [workspace|inter]
                  HVec = bind(&ctns::preprocess_Hx_batchDirect<Tm>, _1, _2,
                        std::cref(scale), std::cref(size), std::cref(rank),
                        std::cref(ndim), std::ref(Hmmtasks), std::ref(opaddr), std::ref(workspace),
                        std::ref(hinter._data));
               }
            }else{
               if(!ifDirect){
                  HVec = bind(&ctns::preprocess_Hx_batchSingle<Tm>, _1, _2,
                        std::cref(scale), std::cref(size), std::cref(rank),
                        std::cref(ndim), std::ref(Hmmtask), std::ref(opaddr), std::ref(workspace));
               }else{
                  opaddr[4] = workspace + batchsize*blksize*2; // memory layout [workspace|inter]
                  HVec = bind(&ctns::preprocess_Hx_batchDirectSingle<Tm>, _1, _2,
                        std::cref(scale), std::cref(size), std::cref(rank),
                        std::cref(ndim), std::ref(Hmmtask), std::ref(opaddr), std::ref(workspace),
                        std::ref(hinter._data));
               }
            }

#ifdef GPU
         }else if(alg_hvec == 16 || alg_hvec == 17 || alg_hvec == 18 || alg_hvec == 19){

            // BatchGEMM on GPU: symbolic formulae + hintermediates + preallocation of workspace

            // allocate memery on GPU & copy qops
            for(int i=0; i<4; i++){
               const auto& tqops = qops_pool.at(fneed[i]);
               assert(tqops.avail_gpu());
               dev_opaddr[i] = tqops._dev_data;
            }
            size_t gpumem_oper = sizeof(Tm)*opertot;
            if(debug && schd.ctns.verbose>0){
               std::cout << "rank=" << rank
                  << " GPUmem(GB): used=" << GPUmem.used()/std::pow(1024.0,3)
                  << " (oper)=" << gpumem_oper/std::pow(1024.0,3) 
                  << std::endl;
            }
            timing.tb2 = tools::get_time();

            H_formulae = symbolic_formulae_twodot(qops_dict, int2e, size, rank, fname,
                  schd.ctns.sort_formulae, schd.ctns.ifdist1, schd.ctns.ifdistc, debug_formulae);

            timing.tb3 = tools::get_time();

            // compute hintermediates on GPU directly
            const bool ifSingle = alg_hvec > 17;
            const bool ifDirect = alg_hvec % 2 == 1;
            const int batchgemv = std::get<0>(schd.ctns.batchhvec); 
            hinter.init(ifDirect, schd.ctns.alg_hinter, batchgemv, qops_dict, oploc, dev_opaddr, H_formulae, debug);
            size_t gpumem_hinter = sizeof(Tm)*hinter.size();
            if(debug && schd.ctns.verbose>0){
               std::cout << "rank=" << rank
                  << " GPUmem(GB): used=" << GPUmem.used()/std::pow(1024.0,3)
                  << " (oper,hinter)=" << gpumem_oper/std::pow(1024.0,3) 
                  << "," << gpumem_hinter/std::pow(1024.0,3) 
                  << std::endl;
            }
            timing.tb4 = tools::get_time();
            timing.tb5 = tools::get_time();

            // GEMM list and GEMV list
            size_t maxbatch = 0;
            if(!ifSingle){
               preprocess_formulae_Hxlist2(ifDirect, schd.ctns.alg_hcoper, 
                     qops_dict, oploc, opaddr, H_formulae, wf, hinter, 
                     Hxlst2, blksize, blksize0, cost, rank==0 && schd.ctns.verbose>0);
               for(int i=0; i<Hxlst2.size(); i++){
                  maxbatch = std::max(maxbatch, Hxlst2[i].size());
               } // i
            }else{
               preprocess_formulae_Hxlist(ifDirect, schd.ctns.alg_hcoper, 
                     qops_dict, oploc, opaddr, H_formulae, wf, hinter, 
                     Hxlst, blksize, blksize0, cost, rank==0 && schd.ctns.verbose>0);
               maxbatch = Hxlst.size();
            }
            if(!ifDirect) assert(blksize0 == 0); 
            timing.tb6 = tools::get_time();

            // Determine batchsize dynamically
            gpumem_dvdson = sizeof(Tm)*2*ndim;
            if(blksize > 0){
               size_t blocksize = 2*blksize+blksize0+1;
               preprocess_gpu_batchsize<Tm>(schd.ctns.batchmem, blocksize, maxbatch, gpumem_dvdson, rank,
                     batchsize, gpumem_batch);
            }
            dev_workspace = (Tm*)GPUmem.allocate(gpumem_dvdson+gpumem_batch);
            if(debug && schd.ctns.verbose>0){
               std::cout << "rank=" << rank
                  << " GPUmem(GB): used=" << GPUmem.used()/std::pow(1024.0,3)
                  << " (oper,hinter,dvdson,batch)=" << gpumem_oper/std::pow(1024.0,3) 
                  << "," << gpumem_hinter/std::pow(1024.0,3) 
                  << "," << gpumem_dvdson/std::pow(1024.0,3)
                  << "," << gpumem_batch/std::pow(1024.0,3)
                  << " blksize=" << blksize
                  << " blksize0=" << blksize0
                  << " batchsize=" << batchsize 
                  << std::endl;
            }

            // generate Hmmtasks given batchsize
            const int batchblas = 2; // GPU
            if(!ifSingle){
               Hmmtasks.resize(Hxlst2.size());
               for(int i=0; i<Hmmtasks.size(); i++){
                  Hmmtasks[i].init(Hxlst2[i], schd.ctns.alg_hcoper, batchblas, schd.ctns.batchhvec, batchsize, blksize*2, blksize0);
                  if(debug && schd.ctns.verbose>1 && Hxlst2[i].size()>0){
                     std::cout << " rank=" << rank << " iblk=" << i 
                        << " size=" << Hxlst2[i][0].size 
                        << " Hmmtasks.totsize=" << Hmmtasks[i].totsize
                        << " batchsize=" << Hmmtasks[i].batchsize 
                        << " nbatch=" << Hmmtasks[i].nbatch 
                        << std::endl;
                  }
               } // i
               if(fmmtask.size()>0) save_mmtask(Hmmtasks, fmmtask);
            }else{
               Hmmtask.init(Hxlst, schd.ctns.alg_hcoper, batchblas, schd.ctns.batchhvec, batchsize, blksize*2, blksize0);
               if(debug && schd.ctns.verbose>1){
                  std::cout << " rank=" << rank
                     << " Hxlst.size=" << Hxlst.size()
                     << " Hmmtasks.totsize=" << Hmmtask.totsize
                     << " batchsize=" << Hmmtask.batchsize 
                     << " nbatch=" << Hmmtask.nbatch 
                     << std::endl;
                  if(schd.ctns.verbose>2){
                     for(int k=0; k<Hmmtask.nbatch; k++){
                        for(int i=0; i<Hmmtask.mmbatch2[k].size(); i++){
                           if(Hmmtask.mmbatch2[k][i].size==0) continue;
                           std::cout << " Hmmbatch2: k/nbatch=" << k << "/" << Hmmtask.nbatch
                              << " i=" << i << " size=" << Hmmtask.mmbatch2[k][i].size
                              << " group=" << Hmmtask.mmbatch2[k][i].gsta.size()-1
                              << " average=" << Hmmtask.mmbatch2[k][i].size/(Hmmtask.mmbatch2[k][i].gsta.size()-1)
                              << std::endl;
                        }
                     }
                  }
               }
               if(fmmtask.size()>0) save_mmtask(Hmmtask, fmmtask);
            }
            timing.tb7 = tools::get_time();

            // GPU version of Hx
            dev_red = dev_workspace + 2*ndim + batchsize*(blksize*2+blksize0); 
            if(!ifSingle){
               if(!ifDirect){
                  HVec = bind(&ctns::preprocess_Hx_batchGPU<Tm>, _1, _2,
                        std::cref(scale), std::cref(size), std::cref(rank),
                        std::cref(ndim), std::ref(Hmmtasks), std::ref(dev_opaddr), std::ref(dev_workspace),
                        std::ref(dev_red), std::cref(schd.ctns.ifnccl));
               }else{
                  dev_opaddr[4] = dev_workspace + 2*ndim + batchsize*blksize*2; // memory layout [workspace|inter]
                  HVec = bind(&ctns::preprocess_Hx_batchDirectGPU<Tm>, _1, _2,
                        std::cref(scale), std::cref(size), std::cref(rank),
                        std::cref(ndim), std::ref(Hmmtasks), std::ref(dev_opaddr), std::ref(dev_workspace),
                        std::ref(hinter._dev_data), std::ref(dev_red), std::cref(schd.ctns.ifnccl));
               }
            }else{
               if(!ifDirect){
                  HVec = bind(&ctns::preprocess_Hx_batchGPUSingle<Tm>, _1, _2,
                        std::cref(scale), std::cref(size), std::cref(rank),
                        std::cref(ndim), std::ref(Hmmtask), std::ref(dev_opaddr), std::ref(dev_workspace),
                        std::ref(dev_red), std::cref(schd.ctns.ifnccl));
               }else{
                  dev_opaddr[4] = dev_workspace + 2*ndim + batchsize*blksize*2; // memory layout [workspace|inter]
                  HVec = bind(&ctns::preprocess_Hx_batchDirectGPUSingle<Tm>, _1, _2,
                        std::cref(scale), std::cref(size), std::cref(rank),
                        std::cref(ndim), std::ref(Hmmtask), std::ref(dev_opaddr), std::ref(dev_workspace),
                        std::ref(hinter._dev_data), std::ref(dev_red), std::cref(schd.ctns.ifnccl));
               }
            }

#endif // GPU

         }else{
            std::cout << "error: no such option for alg_hvec=" << alg_hvec << std::endl;
            exit(1);
         } // alg_hvec
      }

   template <typename Qm, typename Tm>
      void HVec_wrapper<Qm,Tm>::finalize(){
         if(alg_hvec==2 || alg_hvec==3 || 
               alg_hvec==6 || alg_hvec==7 ||
               alg_hvec==8 || alg_hvec==9){
            delete[] workspace;
         }
#ifdef GPU
         if(alg_hvec>10) GPUmem.deallocate(dev_workspace, gpumem_dvdson+gpumem_batch);
#endif
      }

} // ctns

#endif
