#ifndef PREPROCESS_INTER_H
#define PREPROCESS_INTER_H

namespace ctns{

   template <typename Tm>
      struct intermediates{
         public:
            // constructors
            intermediates(){}
            ~intermediates(){ 
               _offset.clear();
               delete[] _data; 
            }
            // initialization
            void init(const int alg_inter,
                  const oper_dictmap<Tm>& qops_dict,
                  const symbolic_task<Tm>& H_formulae,
                  const bool debug=false){
            }
            // form intermediates
            void init_omp(const oper_dictmap<Tm>& qops_dict,
                  const symbolic_task<Tm>& H_formulae,
                  const bool debug);
            void init_batch_cpu(const oper_dictmap<Tm>& qops_dict,
                  const symbolic_task<Tm>& H_formulae,
                  const bool debug);
            // helpers
            size_t count() const{ return _count; };
            size_t size() const{ return _size; };
         public:
            std::map<std::pair<int,int>,size_t> _offset; // map from (it,idx) in H_formulae to offset
            size_t _count = 0, _size = 0;
            Tm* _data = nullptr;
      };

   template <typename Tm>
      void intermediates<Tm>::init(const int alg_inter,
            const oper_dictmap<Tm>& qops_dict,
            const symbolic_task<Tm>& H_formulae,
            const bool debug){
         if(alg_inter == 0){
            this->init_omp(qops_dic, H_formulae, debug);
         }else if(alg_inter == 1){
            this->init_batch_cpu(qops_dic, H_formulae, debug);
         }else if(alg_inter == 2){
            //this->kernel_batch_gpu(qops_dic, H_formulae, debug);
            std::cout << "NOT IMPLEMENTED YET!" << std::endl;
            exit(1);
         }
      }

   // openmp version with symbolic_sum_oper
   template <typename Tm>
      void intermediates<Tm>::init_omp(const oper_dictmap<Tm>& qops_dict,
            const symbolic_task<Tm>& H_formulae,
            const bool debug){
         auto t0 = tools::get_time();
#ifdef _OPENMP
         int maxthreads = omp_get_max_threads();
#else
         int maxthreads = 1;
#endif
         if(debug){
            std::cout << "intermediates<Tm>::init_omp maxthreads=" << maxthreads << std::endl;
            std::cout << " no. of formulae=" << H_formulae.size() << std::endl;
         }

         // count the size of intermediates
         for(int it=0; it<H_formulae.size(); it++){
            const auto& HTerm = H_formulae.tasks[it];
            for(int idx=HTerm.size()-1; idx>=0; idx--){
               const auto& sop = HTerm.terms[idx];
               // define intermediate operators
               if(sop.size() > 1){
                  _count += 1;
                  _offset[std::make_pair(it,idx)] = _size; 
                  const auto& sop0 = sop.sums[0].second;
                  const auto& block = sop0.block;
                  const auto& label  = sop0.label;
                  const auto& qops = qops_dict.at(block);
                  _size += qops(label).at(sop0.index).size();
               }
            }
         } // it
         if(debug){
            std::cout << " no. of intermediate operators=" << _count << std::endl;
            std::cout << " size of intermediate operators=" << _size 
               << ":" << tools::sizeMB<Tm>(_size) << "MB"
               << ":" << tools::sizeGB<Tm>(_size) << "GB"
               << std::endl;
         }

         // 2. allocate memory on CPU
         _data = new Tm[_size];
         memset(_data, 0, _size*sizeof(Tm));

         // 3. form intermediates via AXPY
         std::vector<std::pair<int,int>> _index(_count);
         size_t idx = 0;
         for(const auto& pr : _offset){
            _index[idx] = pr.first;
            idx++;
         }
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
         for(size_t idx=0; idx<_count; idx++){
            const auto& item = _index[idx];
            int i = item.first;
            int j = item.second;
            const auto& sop = H_formulae.tasks[i].terms[j];
            Tm* workspace = _data+_offset.at(item);
            symbolic_sum_oper(qops_dict, sop, workspace);
         } // idx 

         if(debug){
            auto t1 = tools::get_time();
            tools::timing("intermediates<Tm>::init_omp", t0, t1);
         }
      }

   // This subroutine does not work for cNK. Besides, we make the
   // assumption that the C operators are stored contegously.
   template <typename Tm>
      void intermediates<Tm>::init_batch_cpu(const oper_dictmap<Tm>& qops_dict,
            const symbolic_task<Tm>& H_formulae,
            const bool debug){
         auto t0 = tools::get_time();
#ifdef _OPENMP
         int maxthreads = omp_get_max_threads();
#else
         int maxthreads = 1;
#endif
         if(debug){
            std::cout << "intermediates<Tm>::init_batch_cpu maxthreads=" << maxthreads << std::endl;
            std::cout << " no. of formulae=" << H_formulae.size() << std::endl;
         }

         // count the size of intermediates
         size_t alpha_size = 0;
         for(int it=0; it<H_formulae.size(); it++){
            const auto& HTerm = H_formulae.tasks[it];
            for(int idx=HTerm.size()-1; idx>=0; idx--){
               const auto& sop = HTerm.terms[idx];
               int len = sop.size();
               // define intermediate operators
               if(sop.size() > 1){
                  _count += 1;
                  _offset[std::make_pair(it,idx)] = _size; 
                  const auto& sop0 = sop.sums[0].second;
                  const auto& block = sop0.block;
                  const auto& label  = sop0.label;
                  const auto& qops = qops_dict.at(block);
                  _size += qops(label).at(sop0.index).size();
                  alpha_size += len; 
               }
            }
         } // it
         if(debug){
            std::cout << " no. of intermediate operators=" << _count << std::endl;
            std::cout << " no. of coefficients=" << alpha_size << std::endl;
            std::cout << " size of intermediate operators=" << _size 
               << ":" << tools::sizeMB<Tm>(_size) << "MB"
               << ":" << tools::sizeGB<Tm>(_size) << "GB"
               << std::endl;
         }
 
         // allocate memory on CPU
         _data = new Tm[_size];
         Tm* alpha_vec = new Tm[alpha_size];

         // setup GEMV_BATCH
         MVlst<Tm> mvlst(_count);
         size_t idx = 0, adx = 0;
         for(const auto& pr : _offset){
            const auto& item = pr.first;
            int i = item.first;
            int j = item.second;
            const auto& sop = H_formulae.tasks[i].terms[j];
            // form mvinfo for sop = wt0*op0 + wt1*op1 + ...,
            // where we assume all the terms have the same label/dagger !!!
            const auto& sop0 = sop.sums[0].second;
            const auto& index0 = sop0.index;
            const auto& block = sop0.block;
            const auto& label  = sop0.label;
            const auto& dagger= sop0.dagger;
            const auto& qops = qops_dict.at(block);
            const auto& op0 = qops(label).at(index0);
            size_t ndim = op0.size();
            int len = sop.size();
            MVinfo<Tm> mv;
            mv.transA = 'N';
            mv.M = op0.size(); 
            mv.N = len;
            mv.LDA = op0.sym.isym()==2? 2*ndim : ndim; // spin symmetry
            mv.locA = oploc.at(block); mv.offA = qops._offset.at(std::make_pair(label,index0)); // qops
            mv.locx = 4; mv.offx = adx;
            mv.locy = 5; mv.offy = _offset.at(item); // intermediates
            mvlst[idx] = mv;
            for(int k=0; k<len; k++){
               auto wtk = sop.sums[k].first;
               alpha_vec[adx+k] = dagger? tools::conjugate(wtk) : wtk; 
            } 
            adx += len;
            idx += 1;
         }
         assert(idx == _count && adx == alpha_size);

         // perform GEMV_BATCH
         MVbatch<Tm> mvbatch;
         mvbatch.init(mvlst);
         batchgemv = 0;
         Tm* ptrs[6];
         ptrs[0] = opaddr[0];
         ptrs[1] = opaddr[1];
         ptrs[2] = opaddr[2];
         ptrs[3] = opaddr[3];
         ptrs[4] = alpha_vec.data(); 
         ptrs[5] = inter._data;
         mvbatch.kernel(batchgemv, ptrs);

         if(debug){
            auto t1 = tools::get_time();
            tools::timing("intermediates<Tm>::init_batch_cpu", t0, t1);
         }
      }

} // ctns

#endif
