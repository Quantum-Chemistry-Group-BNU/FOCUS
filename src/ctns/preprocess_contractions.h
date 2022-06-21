#ifndef PREPROCESS_CONTRACTIONS_H
#define PREPROCESS_CONTRACTIONS_H

namespace ctns{

template <typename Tm>
struct Hmu_ptr{
public:
   template <typename QTm>
   void gen_Hxblocks(const QTm& wf);
public:
   qinfo2<Tm>* info[4] = {nullptr,nullptr,nullptr,nullptr};
   Tm* data[4] = {nullptr,nullptr,nullptr,nullptr};
   bool parity[4] = {false,false,false,false};
   bool dagger[4] = {false,false,false,false};
   Tm coeff = 1.0; 
};

template <typename Tm>
void Hmu_ptr<Tm>::gen_Hxblocks(const stensor4<Tm>& wf){

}

} // ctns

#endif
