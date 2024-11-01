#ifndef RDM_DISTRIBUTE_H
#define RDM_DISTRIBUTE_H

namespace ctns{

   // rule for distributed assembling of RDMs
   template <bool ifab, typename Tm>   
      void setup_evalmap(const std::string num_string,
            const bool ifkr,
            const int sorb,
            const qoper_map<ifab,Tm>& lops, 
            const qoper_map<ifab,Tm>& rops, 
            std::vector<int>& leval, 
            std::vector<int>& reval,
            const int size, 
            const int rank){
         // select cases
         if(num_string == "020" || num_string == "200" ||
               num_string == "031" || num_string == "121" || num_string == "301" ||
               num_string == "310" || num_string == "400" || 
               num_string == "011" || num_string == "040" || num_string == "013" ||
               num_string == "004" ||
               // 1p0h & 0p1h
               num_string == "010" || num_string == "100" || num_string == "001" ||
               // 2p1h & 1p2h
               num_string == "030" || num_string == "021" || num_string == "300" ||
               num_string == "012" || num_string == "003" || 
               // 3p2h 
               num_string == "041" || num_string == "311" || num_string == "320" ||
               num_string == "410" || num_string == "401"){
            for(const auto& rpr : rops){
               const auto& rdx = rpr.first;
               if(rdx % size == rank) reval.push_back(rdx);
            }
            for(const auto& lpr : lops){
               const auto& ldx = lpr.first;
               leval.push_back(ldx);
            }
         // parallelize over left one-indexed ops
         }else if(num_string == "110" || num_string == "101" || num_string == "112" ||
               num_string == "130" || num_string == "103" ||
               // by putting 002 and 022 here, they will only be evaluated on rank-0
               // which will avoid repeated calculations in different ranks for ifhermi=true 
               num_string == "002" || num_string == "022" ||
               // 2p1h & 1p2h
               num_string == "120" || num_string == "111" || num_string == "102" ||
               // 3p2h
               num_string == "140" || num_string == "131" || num_string == "113" ||
               num_string == "023" || num_string == "104" || num_string == "014"){
            for(const auto& rpr : rops){
               const auto& rdx = rpr.first;
               reval.push_back(rdx);
            }
            for(const auto& lpr : lops){
               const auto& ldx = lpr.first;
               if(ldx % size == rank) leval.push_back(ldx);
            }
         // parallelize over left two-indexed ops
         }else if(num_string == "220" || num_string == "211" || num_string == "202" ||
               // 2p1h & 1p2h
               num_string == "201" || num_string == "210" ||
               // 3p2h
               num_string == "230" || num_string == "221" || num_string == "203"){
            for(const auto& rpr : rops){
               const auto& rdx = rpr.first;
               reval.push_back(rdx); 
            }
            char key;
            for(const auto& lpr : lops){
               const auto& ldx = lpr.first;
               if(distribute2(key, ifkr, size, ldx, sorb) == rank) leval.push_back(ldx);
            }
         // parallelize over right two-indexed ops
         }else if(num_string == "032" || num_string == "122" || num_string == "302"){
            char key;
            for(const auto& rpr : rops){
               const auto& rdx = rpr.first;
               if(distribute2(key, ifkr, size, rdx, sorb) == rank) reval.push_back(rdx); 
            }
            for(const auto& lpr : lops){
               const auto& ldx = lpr.first;
               leval.push_back(ldx);
            }
         // needs spectical treatment later
         }else if(num_string == "212"){
            for(const auto& rpr : rops){
               const auto& rdx = rpr.first;
               reval.push_back(rdx); 
            }
            char key;
            for(const auto& lpr : lops){
               const auto& ldx = lpr.first;
               if(distribute2(key, ifkr, size, ldx, sorb) == rank) leval.push_back(ldx);
            }
         }else{
            tools::exit("error: no such option for num_string="+num_string);
         }
      }

} // ctns

#endif
