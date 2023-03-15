#ifndef PREPROCESS_HINTER_H
#define PREPROCESS_HINTER_H

#include "time.h"
#include "sys/time.h"
#include "preprocess_mvbatch.h"

namespace ctns{

   template <typename Tm>
      struct hintermediates{
         public:
            // constructors
            hintermediates(){}
            ~hintermediates(){ 
               _offset.clear();
               delete[] _data; 
            }
            // initialization
            void init(const int alg_inter,
                  const oper_dictmap<Tm>& qops_dict,
                  const std::map<std::string,int>& oploc,
                  Tm** opaddr,
                  const symbolic_task<Tm>& H_formulae,
                  const bool debug=false){
               if(alg_inter == 0){
                  this->init_omp(qops_dict, H_formulae, debug);
               }else if(alg_inter == 1){
                  this->init_batch_cpu(qops_dict, oploc, opaddr, H_formulae, debug);
#ifdef GPU
               }else if(alg_inter == 2){
                  this->kernel_batch_gpu(qops_dic, oploc, opaddr, H_formulae, debug);
#endif
               }else{
                  std::cout << "error: no such option in Intermediates::init alg_inter=" 
                     << alg_inter << std::endl;
                  exit(1);
               }
               opaddr[4] = _data;
            }
            // form hintermediates
            void init_omp(const oper_dictmap<Tm>& qops_dict,
                  const symbolic_task<Tm>& H_formulae,
                  const bool debug);
            void init_batch_cpu(const oper_dictmap<Tm>& qops_dict,
                  const std::map<std::string,int>& oploc,
                  Tm** opaddr,
                  const symbolic_task<Tm>& H_formulae,
                  const bool debug);
#ifdef GPU
            void init_batch_gpu(const oper_dictmap<Tm>& qops_dict,
                  const std::map<std::string,int>& oploc,
                  Tm** opaddr,
                  const symbolic_task<Tm>& H_formulae,
                  const bool debug);
#endif
            // helpers
            size_t count() const{ return _count; };
            size_t size() const{ return _size; };
         public:
            std::map<std::pair<int,int>,size_t> _offset; // map from (it,idx) in H_formulae to offset
            size_t _count = 0, _size = 0;
            Tm* _data = nullptr;
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
                  _size += qops(label).at(sop0.index).size();
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
            Tm* workspace = _data+_offset.at(item);
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
            std::cout << " no. of hintermediate operators=" << _count << std::endl;
            std::cout << " no. of coefficients=" << alpha_size << std::endl;
            std::cout << " size of hintermediate operators=" << _size 
               << ":" << tools::sizeMB<Tm>(_size) << "MB"
               << ":" << tools::sizeGB<Tm>(_size) << "GB"
               << std::endl;
         }

         // allocate memory on CPU
         _data = new Tm[_size];
         std::vector<Tm> alpha_vec(alpha_size);

         Tm* _data2 = new Tm[_size];

         const auto& qops = qops_dict.at("r");
         for(const auto& qop : qops('C')){
            const auto& idx = qop.first;
            const auto& op = qop.second;
            std::cout << "idx=" << idx << std::endl;
            op.print("op");
         }

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
            mv.M = op0.size(); 
            mv.N = len;
            mv.LDA = std::distance(op0._data, op1._data); // Ca & Cb can be of different dimes for isym=2
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

            Tm* workspace = _data2+_offset.at(item);
            symbolic_sum_oper(qops_dict, sop, workspace);
         }
         assert(idx == _count && adx == alpha_size);

         // perform GEMV_BATCH
         MVbatch<Tm> mvbatch;
         mvbatch.init(mvlst);
         Tm* ptrs[6];
         ptrs[0] = opaddr[0];
         ptrs[1] = opaddr[1];
         ptrs[2] = opaddr[2];
         ptrs[3] = opaddr[3];
         ptrs[4] = alpha_vec.data(); 
         ptrs[5] = _data;
         struct timeval t0gemv, t1gemv;
         int batchgemv = 1;
         gettimeofday(&t0gemv, NULL);
         mvbatch.kernel(batchgemv, ptrs);
         gettimeofday(&t1gemv, NULL);
         double dt = ((double)(t1gemv.tv_sec - t0gemv.tv_sec) 
               + (double)(t1gemv.tv_usec - t0gemv.tv_usec)/1000000.0);
         std::cout << "--- cost_inter=" << mvbatch.cost
            << " time=" << dt
            << " flops=" << mvbatch.cost/dt
            << std::endl;

         double diff = 0.0;
         for(size_t i=0; i<_size; i++){
            diff += std::abs(_data[i]-_data2[i]);
         }
         std::cout << "difference hinter=" << diff << std::endl;
         delete[] _data2;

         if(debug){
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
            std::cout << " no. of hintermediate operators=" << _count << std::endl;
            std::cout << " no. of coefficients=" << alpha_size << std::endl;
            std::cout << " size of hintermediate operators=" << _size 
               << ":" << tools::sizeMB<Tm>(_size) << "MB"
               << ":" << tools::sizeGB<Tm>(_size) << "GB"
               << std::endl;
         }

         // allocate memory on CPU
         _data = new Tm[_size];
         std::vector<Tm> alpha_vec(alpha_size);

         Tm* _data2 = new Tm[_size];

         const auto& qops = qops_dict.at("r");
         for(const auto& qop : qops('C')){
            const auto& idx = qop.first;
            const auto& op = qop.second;
            std::cout << "idx=" << idx << std::endl;
            op.print("op");
         }

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
            mv.M = op0.size(); 
            mv.N = len;
            mv.LDA = std::distance(op0._data, op1._data); // Ca & Cb can be of different dimes for isym=2
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

            Tm* workspace = _data2+_offset.at(item);
            symbolic_sum_oper(qops_dict, sop, workspace);
         }
         assert(idx == _count && adx == alpha_size);

         // perform GEMV_BATCH
         MVbatch<Tm> mvbatch;
         mvbatch.init(mvlst);
         Tm* ptrs[6];
         ptrs[0] = opaddr[0];
         ptrs[1] = opaddr[1];
         ptrs[2] = opaddr[2];
         ptrs[3] = opaddr[3];
         ptrs[4] = alpha_vec.data(); 
         ptrs[5] = _data;
         struct timeval t0gemv, t1gemv;
         int batchgemv = 2;
         gettimeofday(&t0gemv, NULL);
         mvbatch.kernel(batchgemv, ptrs);
         gettimeofday(&t1gemv, NULL);
         double dt = ((double)(t1gemv.tv_sec - t0gemv.tv_sec) 
               + (double)(t1gemv.tv_usec - t0gemv.tv_usec)/1000000.0);
         std::cout << "--- cost_inter=" << mvbatch.cost
            << " time=" << dt
            << " flops=" << mvbatch.cost/dt
            << std::endl;

         double diff = 0.0;
         for(size_t i=0; i<_size; i++){
            diff += std::abs(_data[i]-_data2[i]);
         }
         std::cout << "difference=" << diff << std::endl;
         delete[] _data;
         delete[] _data2;
         exit(1);

         if(debug){
            auto t1 = tools::get_time();
            tools::timing("hintermediates<Tm>::init_batch_gpu", t0, t1);
         }
      }
#endif

} // ctns

#endif
