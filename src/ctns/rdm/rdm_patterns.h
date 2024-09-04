#ifndef RDM_PATTERNS_H
#define RDM_PATTERNS_H

namespace ctns{

   // N-RDM: order=N
   
   // Number Patterns: follows J. Chem. Theory Comput. 2016, 12, 1583−1591
   inline std::vector<std::tuple<int,int,int>> number_patterns(const int order){
      std::vector<std::tuple<int,int,int>> patterns;
      for(int ns=0; ns<=order; ns++){
         for(int ne=0; ne<=order-1; ne++){
            for(int nd=1; nd<=4; nd++){
               if(ns + nd + ne == 2*order){
                  patterns.push_back(std::make_tuple(ns,nd,ne));
               }
            }
         }
      }
      return patterns;
   }

   inline std::vector<std::tuple<int,int,int>> first_number_patterns(const int order){
      std::vector<std::tuple<int,int,int>> patterns;
      for(int ns=order+1; ns<=std::min(4,2*order); ns++){
         for(int nd=0; nd<=4; nd++){
            int ne = 2*order - ns - nd;
            if(ne >= 0) patterns.push_back(std::make_tuple(ns,nd,ne));
         }
      }
      return patterns;
   }

   inline std::vector<std::tuple<int,int,int>> last_number_patterns(const int order){
      std::vector<std::tuple<int,int,int>> patterns;
      for(int ne=order; ne<=std::min(4,2*order); ne++){
         for(int nd=0; nd<=4; nd++){
            int ns = 2*order - ne - nd;
            if(ns >= 0) patterns.push_back(std::make_tuple(ns,nd,ne));
         }
      }
      return patterns;
   }

   // From Number Pattern to Type Pattern
   std::string integer2binaryString(const int n, const int k){
      std::string str(n,'-');
      int kt = k, idx = 0;
      while(kt > 0){
         str[n-1-idx] = (kt%2)? '+' : '-';
         kt /= 2;
         idx += 1;
      }
      return str; 
   }

   struct type_pattern{
      public:
         type_pattern(const std::tuple<int,int,int>& num_pattern,
               const int ts, const int td, const int te){
            left   = integer2binaryString(std::get<0>(num_pattern), ts);        
            center = integer2binaryString(std::get<1>(num_pattern), td);
            right  = integer2binaryString(std::get<2>(num_pattern), te);
         }
         std::string to_string() const{ 
            return std::to_string(left.size())
               +std::to_string(center.size())
               +std::to_string(right.size())+":"
               +left+"|"+center+"|"+right; 
         }
         bool valid(const int ncre, const int nann, const int dot=1) const{
            return (this->valid_dotop(dot)) and
               (this->get_ncre() == ncre) and 
               (this->get_nann() == nann);
         }
         bool valid_dotop(const int dot) const{
            //
            // because for RDM, operators must be of form p+q+rs
            // then assuming operators are ordered by spatial orbitals
            // single-dot operators must have "-" always comes later than "+".
            // valid local operators:
            //    "+":a,b, "-":a,b
            //    "++":aa,ab,bb, "+-":aa,ab,ba,bb, "--":aa,ab,bb
            //    "+++", "++-", "+--", "---": ...
            //
            std::string str;
            if(dot == 0){
              str = left;
            }else if(dot == 1){
              str = center;
            }else if(dot == 2){
              str = right;
            }
            bool valid = true;
            bool minusSeen = false; 
            for(char ch : str){
               if(ch == '-'){
                  minusSeen = true;
               }else if(ch == '+'){
                  if(minusSeen){
                     valid = false; 
                  }
               }
            }
            return valid;
         }
         int get_ncre() const{ 
            return this->get_ncre_left()+
               this->get_ncre_center()+
               this->get_ncre_right();
         }
         int get_ncre_left() const{ return std::count(left.begin(), left.end(), '+'); }
         int get_ncre_center() const{ return std::count(center.begin(), center.end(), '+'); }
         int get_ncre_right() const{ return std::count(right.begin(), right.end(), '+'); }
         int get_nann() const{ 
            return this->get_nann_left()+
               this->get_nann_center()+
               this->get_nann_right();
         }
         int get_nann_left() const{ return std::count(left.begin(), left.end(), '-'); }
         int get_nann_center() const{ return std::count(center.begin(), center.end(), '-'); }
         int get_nann_right() const{ return std::count(right.begin(), right.end(), '-'); }
      public:
         std::string left   = "";
         std::string center = "";
         std::string right  = "";
   };

   inline std::vector<type_pattern> gen_type_patterns(const std::tuple<int,int,int>& num_pattern,
         const int ncre,
         const int nann,
         const int dot=1){
      std::vector<type_pattern> tpatterns;
      for(int ts=0; ts<std::pow(2,std::get<0>(num_pattern)); ts++){
         for(int td=0; td<std::pow(2,std::get<1>(num_pattern)); td++){
            for(int te=0; te<std::pow(2,std::get<2>(num_pattern)); te++){
               type_pattern tpattern(num_pattern, ts, td, te);
               if(!tpattern.valid(ncre, nann, dot)) continue;
               tpatterns.push_back(tpattern);
            }
         }
      }
      return tpatterns;
   }

   inline std::vector<type_pattern> all_type_patterns(const int order){
      std::vector<type_pattern> tpatterns;
      auto patterns = number_patterns(order);
      for(const auto& pt : patterns){
         auto tps = gen_type_patterns(pt, order, order, 1);
         std::copy(tps.begin(), tps.end(), std::back_inserter(tpatterns));          
      }
      return tpatterns;
   }

   inline std::vector<type_pattern> all_first_type_patterns(const int order){
      std::vector<type_pattern> tpatterns;
      auto patterns = first_number_patterns(order);
      for(const auto& pt : patterns){
         auto tps = gen_type_patterns(pt, order, order, 0);
         std::copy(tps.begin(), tps.end(), std::back_inserter(tpatterns));          
      }
      return tpatterns;
   }

   inline std::vector<type_pattern> all_last_type_patterns(const int order){
      std::vector<type_pattern> tpatterns;
      auto patterns = last_number_patterns(order);
      for(const auto& pt : patterns){
         auto tps = gen_type_patterns(pt, order, order, 2);
         std::copy(tps.begin(), tps.end(), std::back_inserter(tpatterns));          
      }
      return tpatterns;
   }

   inline void display_patterns(const std::vector<type_pattern>& patterns){
      std::cout << "ctns::display_patterns size=" << patterns.size() << std::endl;
      for(int i=0; i<patterns.size(); i++){
         std::cout << " i=" << i << " " << patterns[i].to_string() << std::endl;
      }
   }

} // ctns

#endif
