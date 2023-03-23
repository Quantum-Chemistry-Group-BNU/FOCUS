#ifndef PREPROCESS_HEADER_H
#define PREPROCESS_HEADER_H

namespace ctns{

   const int locInter = 4; // intermediates
   const int locIn    = 5; // x
   const int locOut   = 6; // y
   extern const int locInter;
   extern const int locIn;
   extern const int locOut;

   // intermediates
   const int ipack = 10;
   extern const int ipack;
   inline size_t inter_pack(const size_t it, const size_t idx){
      return idx + it*ipack;
   }
   inline std::pair<size_t,size_t> oper_unpack(const size_t ij){
      return std::make_pair(ij/ipack,ij%ipack);
   }

} // ctns

#endif
