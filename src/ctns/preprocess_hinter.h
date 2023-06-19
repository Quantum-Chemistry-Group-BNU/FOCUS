#ifndef PREPROCESS_HINTER_H
#define PREPROCESS_HINTER_H

#include "time.h"
#include "sys/time.h"
#include "preprocess_header.h"
#include "preprocess_mvbatch.h"

namespace ctns{

   template <typename Tm>
      struct hintermediates{
         public:
            ~hintermediates(){
               delete[] _data;
#ifdef GPU
               GPUmem.deallocate(_dev_data, _size*sizeof(Tm));
#endif
            }
#ifdef GPU
            void allocate_gpu(){
               _dev_data = (Tm*)GPUmem.allocate(_size*sizeof(Tm));
            }
#endif
            // initialization
            void init(const bool ifDirect,
                  const int alg_hinter,
                  const int batchgemv,
                  const oper_dictmap<Tm>& qops_dict,
                  const std::map<std::string,int>& oploc,
                  Tm** opaddr,
                  const symbolic_task<Tm>& H_formulae,
                  const bool debug=false){
               if(!ifDirect){
                  if(alg_hinter == 0){
                     this->init_omp(qops_dict, H_formulae, debug);
                     opaddr[locInter] = _data;
                  }else if(alg_hinter == 1){
                     this->init_batch_cpu(qops_dict, oploc, opaddr, H_formulae, batchgemv, debug);
                     opaddr[locInter] = _data;
#ifdef GPU
                  }else if(alg_hinter == 2){
                     this->init_batch_gpu(qops_dict, oploc, opaddr, H_formulae, batchgemv, debug);
                     opaddr[locInter] = _dev_data;
#endif
                  }else{
                     std::cout << "error: no such option in Intermediates::init alg_hinter=" 
                        << alg_hinter << std::endl;
                     exit(1);
                  }
               }else{
                  if(alg_hinter == 1){
                     this->initDirect_batch_cpu(H_formulae, debug);
#ifdef GPU
                  }else if(alg_hinter == 2){
                     this->initDirect_batch_gpu(H_formulae, debug);
#endif
                  }else{
                     std::cout << "error: no such option in Intermediates::initDirect alg_hinter=" 
                        << alg_hinter << std::endl;
                     exit(1);
                  }
               }
            }
            // form hintermediates
            void init_omp(const oper_dictmap<Tm>& qops_dict,
                  const symbolic_task<Tm>& H_formulae,
                  const bool debug);
            void init_batch_cpu(const oper_dictmap<Tm>& qops_dict,
                  const std::map<std::string,int>& oploc,
                  Tm** opaddr,
                  const symbolic_task<Tm>& H_formulae,
                  const int batchgemv,
                  const bool debug);
            void initDirect_batch_cpu(const symbolic_task<Tm>& H_formulae,
                  const bool debug);
#ifdef GPU
            void init_batch_gpu(const oper_dictmap<Tm>& qops_dict,
                  const std::map<std::string,int>& oploc,
                  Tm** opaddr,
                  const symbolic_task<Tm>& H_formulae,
                  const int batchgemv,
                  const bool debug);
            void initDirect_batch_gpu(const symbolic_task<Tm>& H_formulae,
                  const bool debug);
#endif
            // helpers
            size_t count() const{ return _count; };
            size_t size() const{ return _size; };
         public:
            std::map<std::pair<int,int>,size_t> _offset; // map from (it,idx) in H_formulae to offset
            size_t _count = 0, _size = 0;
            Tm* _data = nullptr;
            Tm* _dev_data = nullptr;
      };

   // openmp version with symbolic_sum_oper
   template <typename Tm>
      void hintermediates<Tm>::init_omp(const oper_dictmap<Tm>& qops_dict,
            const symbolic_task<Tm>& H_formulae,
            const bool debug){
         auto t0 = tools::get_time();
#ifdef _OPENMP
         int maxthreads = omp_get_max_threads();
#else
         int maxthreads = 1;
#endif
         if(debug){
            std::cout << "hintermediates<Tm>::init_omp maxthreads=" << maxthreads << std::endl;
            std::cout << " no. of formulae=" << H_formulae.size() << std::endl;
         }

         // count the size of hintermediates
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
                  size_t opsize = qops(label).at(sop0.index).size();
                  _size += opsize;
               }
            }
         } // it
         if(debug){
            std::cout << " no. of hintermediate operators=" << _count << std::endl;
            std::cout << " size of hintermediate operators=" << _size 
               << ":" << tools::sizeMB<Tm>(_size) << "MB"
               << ":" << tools::sizeGB<Tm>(_size) << "GB"
               << std::endl;
         }
         if(_size == 0) return;

         // 2. allocate memory on CPU
         _data = new Tm[_size];
         memset(_data, 0, _size*sizeof(Tm));

         // 3. form hintermediates via AXPY
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
            Tm* workspace = _data + _offset.at(item);
            symbolic_sum_oper(qops_dict, sop, workspace);
         } // idx 

         if(debug){
            auto t1 = tools::get_time();
            tools::timing("hintermediates<Tm>::init_omp", t0, t1);
         }
      }

   // This subroutine does not work for cNK. Besides, we make the
   // assumption that the C operators are stored contegously.
   template <typename Tm>
      void hintermediates<Tm>::init_batch_cpu(const oper_dictmap<Tm>& qops_dict,
            const std::map<std::string,int>& oploc,
            Tm** opaddr,
            const symbolic_task<Tm>& H_formulae,
            const int batchgemv,
            const bool debug){
         auto t0 = tools::get_time();
#ifdef _OPENMP
         int maxthreads = omp_get_max_threads();
#else
         int maxthreads = 1;
#endif
         if(debug){
            std::cout << "hintermediates<Tm>::init_batch_cpu maxthreads=" << maxthreads << std::endl;
            std::cout << " no. of formulae=" << H_formulae.size() << std::endl;
         }

         // count the size of hintermediates
         size_t alpha_size = 0;
         for(int it=0; it<H_formulae.size(); it++){
            const auto& HTerm = H_formulae.tasks[it];
            for(int idx=HTerm.size()-1; idx>=0; idx--){
               const auto& sop = HTerm.terms[idx];
               int len = sop.size();
               // define intermediate operators
               if(len > 1){
                  _count += 1;
                  _offset[std::make_pair(it,idx)] = _size; 
                  const auto& sop0 = sop.sums[0].second;
                  const auto& block = sop0.block;
                  const auto& label  = sop0.label;
                  const auto& qops = qops_dict.at(block);
                  size_t opsize = qops(label).at(sop0.index).size();
                  _size += opsize;
                  alpha_size += len; 
               }
            }
         } // it
         if(debug){
            std::cout << " no. of hintermediate operators=" << _count << std::endl;
            std::cout << " no. of coefficients=" << alpha_size << std::endl;
            std::cout << " size of hintermediate operators=" << _size 
               << ":" << tools::sizeMB<Tm>(_size) << "MB"
               << ":" << tools::sizeGB<Tm>(_size) << "GB"
               << std::endl;
         }
         if(_size == 0) return;

         // allocate memory on CPU
         _data = new Tm[_size];
         std::vector<Tm> alpha_vec(alpha_size);

         // setup GEMV_BATCH
         MVlist<Tm> mvlst(_count);
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
            const auto& sop1 = sop.sums[1].second; // used for determine LDA
            const auto& index1 = sop1.index;
            const auto& op1 = qops(label).at(index1);
            MVinfo<Tm> mv;
            mv.transA = 'N';
            mv.M = ndim;
            mv.N = len;
            //mv.LDA = std::distance(op0._data, op1._data); // Ca & Cb can be of different dimes for isym=2
            mv.LDA = qops._offset.at(std::make_pair(label,index1)) - qops._offset.at(std::make_pair(label,index0));
            mv.locA = oploc.at(block); 
            mv.offA = qops._offset.at(std::make_pair(label,index0)); // qops
            mv.locx = 4; 
            mv.offx = adx;
            mv.locy = 5; 
            mv.offy = _offset.at(item); // hintermediates
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
         // sort
         std::stable_sort(mvlst.begin(), mvlst.end(),
               [](const MVinfo<Tm>& mv1, const MVinfo<Tm>& mv2){
               return mv1 > mv2;
               });
         mvbatch.init(mvlst);
         Tm* ptrs[6];
         ptrs[0] = opaddr[0]; // l
         ptrs[1] = opaddr[1]; // r
         ptrs[2] = opaddr[2]; // c1
         ptrs[3] = opaddr[3]; // c2
         ptrs[4] = alpha_vec.data(); 
         ptrs[5] = _data;
         struct timeval t0gemv, t1gemv;
         gettimeofday(&t0gemv, NULL);
         mvbatch.kernel(batchgemv, ptrs);
         gettimeofday(&t1gemv, NULL);
         double dt = ((double)(t1gemv.tv_sec - t0gemv.tv_sec) 
               + (double)(t1gemv.tv_usec - t0gemv.tv_usec)/1000000.0);
         if(debug){
            std::cout << "cost_hinter=" << mvbatch.cost
               << " time=" << dt << " flops=" << mvbatch.cost/dt
               << std::endl;
            auto t1 = tools::get_time();
            tools::timing("hintermediates<Tm>::init_batch_cpu", t0, t1);
         }
      }

#ifdef GPU
   // This subroutine does not work for cNK. Besides, we make the
   // assumption that the C operators are stored contegously.
   template <typename Tm>
      void hintermediates<Tm>::init_batch_gpu(const oper_dictmap<Tm>& qops_dict,
            const std::map<std::string,int>& oploc,
            Tm** opaddr,
            const symbolic_task<Tm>& H_formulae,
            const int batchgemv,
            const bool debug){
         auto t0 = tools::get_time();
#ifdef _OPENMP
         int maxthreads = omp_get_max_threads();
#else
         int maxthreads = 1;
#endif
         if(debug){
            std::cout << "hintermediates<Tm>::init_batch_gpu maxthreads=" << maxthreads << std::endl;
            std::cout << " no. of formulae=" << H_formulae.size() << std::endl;
         }

         // count the size of hintermediates
         size_t alpha_size = 0;
         for(int it=0; it<H_formulae.size(); it++){
            const auto& HTerm = H_formulae.tasks[it];
            for(int idx=HTerm.size()-1; idx>=0; idx--){
               const auto& sop = HTerm.terms[idx];
               int len = sop.size();
               // define intermediate operators
               if(len > 1){
                  _count += 1;
                  _offset[std::make_pair(it,idx)] = _size; 
                  const auto& sop0 = sop.sums[0].second;
                  const auto& block = sop0.block;
                  const auto& label  = sop0.label;
                  const auto& qops = qops_dict.at(block);
                  size_t opsize = qops(label).at(sop0.index).size();
                  _size += opsize;
                  alpha_size += len; 
               }
            }
         } // it
         if(debug){
            std::cout << " no. of hintermediate operators=" << _count << std::endl;
            std::cout << " no. of coefficients=" << alpha_size << std::endl;
            std::cout << " size of hintermediate operators=" << _size 
               << ":" << tools::sizeMB<Tm>(_size) << "MB"
               << ":" << tools::sizeGB<Tm>(_size) << "GB"
               << std::endl;
         }
         if(_size == 0) return;

         // setup GEMV_BATCH
         std::vector<Tm> alpha_vec(alpha_size);
         MVlist<Tm> mvlst(_count);
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
            const auto& sop1 = sop.sums[1].second; // used for determine LDA
            const auto& index1 = sop1.index;
            const auto& op1 = qops(label).at(index1);
            MVinfo<Tm> mv;
            mv.transA = 'N';
            mv.M = ndim;
            mv.N = len;
            //mv.LDA = std::distance(op0._data, op1._data); // Ca & Cb can be of different dimes for isym=2
            mv.LDA = qops._offset.at(std::make_pair(label,index1)) - qops._offset.at(std::make_pair(label,index0));
            mv.locA = oploc.at(block); 
            mv.offA = qops._offset.at(std::make_pair(label,index0)); // qops
            mv.locx = 4; 
            mv.offx = adx;
            mv.locy = 5; 
            mv.offy = _offset.at(item); // hintermediates
            mvlst[idx] = mv;
            for(int k=0; k<len; k++){
               auto wtk = sop.sums[k].first;
               alpha_vec[adx+k] = dagger? tools::conjugate(wtk) : wtk; 
            } 
            adx += len;
            idx += 1;
         }
         assert(idx == _count && adx == alpha_size);

         this->allocate_gpu();
         size_t gpumem_alpha = alpha_size*sizeof(Tm);
         Tm* dev_alpha_vec = (Tm*)GPUmem.allocate(gpumem_alpha);
         GPUmem.to_gpu(dev_alpha_vec, alpha_vec.data(), gpumem_alpha);

         // perform GEMV_BATCH
         MVbatch<Tm> mvbatch;
         // sort
         std::stable_sort(mvlst.begin(), mvlst.end(),
               [](const MVinfo<Tm>& mv1, const MVinfo<Tm>& mv2){
               return mv1 > mv2;
               });
         mvbatch.init(mvlst);
         Tm* ptrs[6];
         ptrs[0] = opaddr[0]; // l
         ptrs[1] = opaddr[1]; // r
         ptrs[2] = opaddr[2]; // c1
         ptrs[3] = opaddr[3]; // c2
         ptrs[4] = dev_alpha_vec;
         ptrs[5] = _dev_data;
         struct timeval t0gemv, t1gemv;
         gettimeofday(&t0gemv, NULL);
         mvbatch.kernel(batchgemv, ptrs);
         gettimeofday(&t1gemv, NULL);
         double dt = ((double)(t1gemv.tv_sec - t0gemv.tv_sec) 
               + (double)(t1gemv.tv_usec - t0gemv.tv_usec)/1000000.0);
         if(debug){
            std::cout << "cost_hinter=" << mvbatch.cost
               << " time=" << dt << " flops=" << mvbatch.cost/dt
               << std::endl;
         }

         GPUmem.deallocate(dev_alpha_vec, gpumem_alpha);

         if(debug){
            auto t1 = tools::get_time();
            tools::timing("hintermediates<Tm>::init_batch_gpu", t0, t1);
         }
      }
#endif

   // This subroutine does not work for cNK. Besides, we make the
   // assumption that the C operators are stored contegously.
   template <typename Tm>
      void hintermediates<Tm>::initDirect_batch_cpu(const symbolic_task<Tm>& H_formulae,
            const bool debug){
         auto t0 = tools::get_time();
#ifdef _OPENMP
         int maxthreads = omp_get_max_threads();
#else
         int maxthreads = 1;
#endif
         if(debug){
            std::cout << "hintermediates<Tm>::initDirect_batch_cpu maxthreads=" << maxthreads << std::endl;
            std::cout << " no. of formulae=" << H_formulae.size() << std::endl;
         }

         // count the size of hintermediates
         for(int it=0; it<H_formulae.size(); it++){
            const auto& HTerm = H_formulae.tasks[it];
            for(int idx=HTerm.size()-1; idx>=0; idx--){
               const auto& sop = HTerm.terms[idx];
               int len = sop.size();
               // define intermediate operators
               if(len > 1){
                  _count += 1;
                  _offset[std::make_pair(it,idx)] = _size; 
                  _size += len; 
               }
            }
         } // it
         if(debug){
            std::cout << " no. of hintermediate operators=" << _count << std::endl;
            std::cout << " no. of coefficients=" << _size
               << ":" << tools::sizeMB<Tm>(_size) << "MB"
               << ":" << tools::sizeGB<Tm>(_size) << "GB"
               << std::endl;
         }
         if(_size == 0) return;

         _data = new Tm[_size];
         size_t adx = 0;
         for(const auto& pr : _offset){
            const auto& item = pr.first;
            int i = item.first;
            int j = item.second;
            const auto& sop = H_formulae.tasks[i].terms[j];
            const auto& sop0 = sop.sums[0].second;
            const auto& dagger= sop0.dagger;
            int len = sop.size();
            for(int k=0; k<len; k++){
               auto wtk = sop.sums[k].first;
               _data[adx+k] = dagger? tools::conjugate(wtk) : wtk; 
            } 
            adx += len;
         }
         assert(adx == _size);

         if(debug){
            auto t1 = tools::get_time();
            tools::timing("hintermediates<Tm>::initDirect_batch_cpu", t0, t1);
         }
      }

#ifdef GPU
   // This subroutine does not work for cNK. Besides, we make the
   // assumption that the C operators are stored contegously.
   template <typename Tm>
      void hintermediates<Tm>::initDirect_batch_gpu(const symbolic_task<Tm>& H_formulae,
            const bool debug){
         auto t0 = tools::get_time();
#ifdef _OPENMP
         int maxthreads = omp_get_max_threads();
#else
         int maxthreads = 1;
#endif
         if(debug){
            std::cout << "hintermediates<Tm>::initDirect_batch_gpu maxthreads=" << maxthreads << std::endl;
            std::cout << " no. of formulae=" << H_formulae.size() << std::endl;
         }

         // count the size of hintermediates
         for(int it=0; it<H_formulae.size(); it++){
            const auto& HTerm = H_formulae.tasks[it];
            for(int idx=HTerm.size()-1; idx>=0; idx--){
               const auto& sop = HTerm.terms[idx];
               int len = sop.size();
               // define intermediate operators
               if(len > 1){
                  _count += 1;
                  _offset[std::make_pair(it,idx)] = _size; 
                  _size += len; 
               }
            }
         } // it
         if(debug){
            std::cout << " no. of hintermediate operators=" << _count << std::endl;
            std::cout << " no. of coefficients=" << _size 
               << ":" << tools::sizeMB<Tm>(_size) << "MB"
               << ":" << tools::sizeGB<Tm>(_size) << "GB"
               << std::endl;
         }
         if(_size == 0) return;

         std::vector<Tm> alpha_vec(_size);
         size_t adx = 0;
         for(const auto& pr : _offset){
            const auto& item = pr.first;
            int i = item.first;
            int j = item.second;
            const auto& sop = H_formulae.tasks[i].terms[j];
            const auto& sop0 = sop.sums[0].second;
            const auto& dagger= sop0.dagger;
            int len = sop.size();
            for(int k=0; k<len; k++){
               auto wtk = sop.sums[k].first;
               alpha_vec[adx+k] = dagger? tools::conjugate(wtk) : wtk; 
            } 
            adx += len;
         }
         assert(adx == _size);

         this->allocate_gpu();
         GPUmem.to_gpu(_dev_data, alpha_vec.data(), _size*sizeof(Tm));

         if(debug){
            auto t1 = tools::get_time();
            tools::timing("hintermediates<Tm>::initDirect_batch_gpu", t0, t1);
         }
      }
#endif

} // ctns

#endif
