#ifndef QNUM_QBOND_H
#define QNUM_QBOND_H
#include "qnum_qsym.h"

namespace ctns{

   // qbond: std::vector<std::pair<qsym,int>> dims;
   class qbond{
      private:
         friend class boost::serialization::access;	   
         template<class Archive>
            void serialize(Archive & ar, const unsigned int version){
               ar & dims;
            }
      public:
         // constructor
         qbond(){};
         qbond(const std::vector<std::pair<qsym,int>>& ds): dims(ds) {}
         // order
         qbond sort_by_sym() const{
            qbond qs;
            qs.dims = dims;
            std::stable_sort(qs.dims.begin(), qs.dims.end(),
                  [](const std::pair<qsym,int>& t1,
                     const std::pair<qsym,int>& t2){
                  return t2.first < t1.first;
                  });
            return qs;
         }
         qbond sort_by_dim() const{
            qbond qs;
            qs.dims = dims; 
            std::stable_sort(qs.dims.begin(), qs.dims.end(),
                  [](const std::pair<qsym,int>& t1,
                     const std::pair<qsym,int>& t2){
                  return t2.second < t1.second;
                  });
            return qs;
         }
         // helpers
         int size() const{ return dims.size(); }
         qsym get_sym(const int i) const{ return dims[i].first; } 
         int get_dim(const int i) const{ return dims[i].second; }
         int get_parity(const int i) const{ return dims[i].first.parity(); }
         // total dimension
         int get_dimAll() const{
            int dim = 0;
            for(const auto& p : dims) dim += p.second;
            return dim;
         }
         // total dimension U1
         int get_dimAllU1() const{
            int dim = 0;
            for(const auto& p : dims){
               const auto& sym = p.first;
               assert(sym.isym() == 3);
               dim += p.second*(sym.ts()+1); // *(2S+1)
            }
            return dim;
         }
         // offset 
         std::vector<int> get_offset() const{
            std::vector<int> offset;
            int ioff = 0;
            for(int i=0; i<dims.size(); i++){
               offset.push_back(ioff);
               ioff += dims[i].second; 
            }
            return offset;
         }
         // comparison
         bool operator ==(const qbond& qs) const{
            bool ifeq = (dims.size() == qs.size());
            if(not ifeq) return false;
            for(int i=0; i<dims.size(); i++){
               ifeq = ifeq && dims[i].first == qs.dims[i].first &&
                  dims[i].second == qs.dims[i].second;
               if(not ifeq) return false;
            }
            return true;
         }
         void print(const std::string name, const bool debug=true) const{
            std::cout << " qbond: " << name 
               << " nsym=" << dims.size()
               << " dimAll=" << get_dimAll() 
               << std::endl;
            // loop over symmetry sectors
            if(debug){
               for(int i=0; i<dims.size(); i++){
                  auto sym = dims[i].first;
                  auto dim = dims[i].second;
                  std::cout << " " << sym << ":" << dim;
               }
               std::cout << std::endl;
            }
         }
         // ZL@20221207 dump (symmetry label is not stored)
         void dump(std::ofstream& ofs) const{
            int sz = dims.size();
            std::vector<int> ds(1+3*sz,0);
            ds[0] = sz;
            for(int i=0; i<sz; i++){
               ds[3*i+1] = dims[i].first.ne();
               ds[3*i+2] = dims[i].first.tm();
               ds[3*i+3] = dims[i].second;
            }
            ofs.write((char*)(ds.data()), sizeof(ds[0])*(3*sz+1));
         }
         // look for sym
         int existQ(const qsym& sym) const{
            int idx = -1;
            for(int i=0; i<dims.size(); i++){
               if(sym == dims[i].first){
                  idx = i;
                  break;
               }
            }
            return idx;
         }
      public:
         std::vector<std::pair<qsym,int>> dims;
   };

} // ctns

#endif
