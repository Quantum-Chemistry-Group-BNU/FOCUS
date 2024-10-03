#ifndef RDM_PATTERNS_H
#define RDM_PATTERNS_H

namespace ctns{

   // Number Patterns: generalization of J. Chem. Theory Comput. 2016, 12, 1583−1591 for npmh
   // example: (nsystem,ndot,nenvironment)
   // 1 - 0,1,0 
   // 2 - 1,1,0
   // 3 - 1,1,1
   // 4 - 2,1,1
   // 5 - 2,1,2
   // In this construction, we always have ns_max >= ne_max
   std::pair<int,int> get_nsnemax(const int ntot){
      int ns_max = ntot/2;
      int ne_max = (ntot-1)/2;
      return std::make_pair(ns_max,ne_max);
   }

   inline std::vector<std::tuple<int,int,int>> number_patterns(const int ntot){
      auto pr = get_nsnemax(ntot);
      int ns_max = pr.first, ne_max = pr.second;
      std::vector<std::tuple<int,int,int>> patterns;
      for(int ns=0; ns<=ns_max; ns++){
         for(int ne=0; ne<=ne_max; ne++){
            for(int nd=1; nd<=4; nd++){
               if(ns + nd + ne == ntot) patterns.push_back(std::make_tuple(ns,nd,ne));
            } // nd
         } // ne
      } // ns 
      return patterns;
   }

   inline std::vector<std::tuple<int,int,int>> first_number_patterns(const int ntot){
      auto pr = get_nsnemax(ntot);
      int ns_max = pr.first, ne_max = pr.second;
      std::vector<std::tuple<int,int,int>> patterns;
      for(int ns=ns_max+1; ns<=std::min(4,ntot); ns++){
         for(int nd=0; nd<=4; nd++){
            int ne = ntot - ns - nd;
            if(ne >= 0) patterns.push_back(std::make_tuple(ns,nd,ne));
         }
      }
      return patterns;
   }

   inline std::vector<std::tuple<int,int,int>> last_number_patterns(const int ntot){
      auto pr = get_nsnemax(ntot);
      int ns_max = pr.first, ne_max = pr.second;
      std::vector<std::tuple<int,int,int>> patterns;
      for(int ne=ne_max+1; ne<=std::min(4,ntot); ne++){
         for(int nd=0; nd<=4; nd++){
            int ns = ntot - ne - nd;
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
         std::string num_string() const{
            return std::to_string(left.size())
               +std::to_string(center.size())
               +std::to_string(right.size());
         }
         std::string to_string() const{ 
            return this->num_string()+":"+left+"|"+center+"|"+right; 
         }
         bool valid(const int ncre, const int nann, const std::string dots) const{
            bool valid = (this->get_ncre() == ncre) and (this->get_nann() == nann);
            for(const auto& dot : dots){
               valid = valid and (this->valid_dotop(dot));
            }
            return valid; 
         }
         bool valid_dotop(const char dot) const{
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
            if(dot == 'l'){
              str = left;
            }else if(dot == 'c'){
              str = center;
            }else if(dot == 'r'){
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
         type_pattern adjoint() const{
            type_pattern adj = *this;
            std::reverse(adj.left.begin(), adj.left.end());
            std::transform(adj.left.begin(), adj.left.end(), adj.left.begin(), [](char c){
               return (c == '+') ? '-' : '+';
            });
            std::reverse(adj.center.begin(), adj.center.end());
            std::transform(adj.center.begin(), adj.center.end(), adj.center.begin(), [](char c){
               return (c == '+') ? '-' : '+';
            });
            std::reverse(adj.right.begin(), adj.right.end());
            std::transform(adj.right.begin(), adj.right.end(), adj.right.begin(), [](char c){
               return (c == '+') ? '-' : '+';
            });
            return adj;
         }
         bool operator ==(const type_pattern& tp) const{
            return left == tp.left and center == tp.center and right == tp.right; 
         }
         // https://en.cppreference.com/w/cpp/language/ascii + is smaller than -
         bool operator <(const type_pattern& tp) const{
            return this->to_string() < tp.to_string();
         }
         // special pattern that the adjoint operation maps pattern to itself 
         bool hermi() const{
            return this->to_string() == this->adjoint().to_string();
         }
      public:
         std::string left   = "";
         std::string center = "";
         std::string right  = "";
   };
         
   inline std::vector<type_pattern> gen_type_patterns(const std::tuple<int,int,int>& num_pattern,
         const int ncre,
         const int nann,
         const std::string dots){
      std::vector<type_pattern> tpatterns;
      for(int ts=0; ts<std::pow(2,std::get<0>(num_pattern)); ts++){
         for(int td=0; td<std::pow(2,std::get<1>(num_pattern)); td++){
            for(int te=0; te<std::pow(2,std::get<2>(num_pattern)); te++){
               type_pattern tpattern(num_pattern, ts, td, te);
               // check whether the pattern is valid
               if(!tpattern.valid(ncre, nann, dots)) continue;
               tpatterns.push_back(tpattern);
            }
         }
      }
      return tpatterns;
   }

   inline void display_patterns(const std::vector<type_pattern>& patterns,
         const std::string name){
      std::cout << "ctns::display_patterns name=" << name << " size=" << patterns.size() << std::endl;
      for(int i=0; i<patterns.size(); i++){
         std::cout << " i=" << i << " " << patterns[i].to_string() 
            << " hermi=" << patterns[i].hermi()
            << std::endl;
      }
   }

   // remove patterns that can be related with other patterns by Hermitian operation
   // which can reduce computation roughly by half for icomb = icomb2
   inline std::vector<type_pattern> remove_hermitian(const std::vector<type_pattern>& tpatterns){
      std::vector<type_pattern> tpatterns_new;
      for(int i=0; i<tpatterns.size(); i++){
         /*
         std::cout << "i=" << i << " tp=" << tpatterns[i].to_string()
            << " tp.adj=" << tpatterns[i].adjoint().to_string()
            << " smaller=" << (tpatterns[i]<tpatterns[i].adjoint())
            << std::endl;
         */
         const auto& tp = tpatterns[i];
         auto tpadj = tp.adjoint();
         if(std::find(tpatterns_new.begin(), tpatterns_new.end(), tp) == tpatterns_new.end() and
               std::find(tpatterns_new.begin(), tpatterns_new.end(), tpadj) == tpatterns_new.end()){
            auto tpmin = std::min(tp,tpadj);
            auto tpstr = tpmin.to_string();
            // these two patterns are also not needed, because they will be formed from
            // "220:+-|+-" and "202:+-||+-". For instance,
            // pL+qL-rC+sC- (p<q,r<s) is equivalent to qL-pL+rC+sC- (q>p), and by hermitian
            // qL+pL-sC+rC- (q>p,r<s) and pL-qL+sC+rC- (p<q). Therefore, only pL+qL-rC+sC- (p<q,r<s)
            // needs to be computed, as with ifhermi=true the part qL+pL-sC+rC- (q>p,r<s) will 
            // emerge using rdm(i,j) = tools::conjugate(rdm(j,i)). 
            if(tpstr != "220:-+|+-|" and tpstr != "202:-+||+-"){
               tpatterns_new.push_back( std::min(tp,tpadj) );
            }
         }
      }
      return tpatterns_new;
   }

   // is_same = false: B is fully constructed, which is sufficient for all 2-TDMs
   // In the TDM case, 211:-+|+|- <l|p1p2+|l'> is the same as -<l|p2+p1|l'> (p1!=p2), thus
   // it will be produce the same TDM as 211:+-|+|-. Hence, this pattern can be skipped.
   inline std::vector<type_pattern> remove_minusplus(const std::vector<type_pattern>& tpatterns){
      std::vector<type_pattern> tpatterns_new;
      for(int i=0; i<tpatterns.size(); i++){
         const auto& tp = tpatterns[i];
         if(tp.left == "-+" || tp.center == "-+" || tp.right == "-+") continue;
         tpatterns_new.push_back( tp );
      }
      return tpatterns_new;
   }

   inline std::vector<type_pattern> all_type_patterns(const int ncre, const int nann,
         const bool is_same){
      std::vector<type_pattern> tpatterns;
      auto patterns = number_patterns(ncre+nann);
      for(const auto& pt : patterns){
         std::cout << std::get<0>(pt) << "|" << std::get<1>(pt) << "|" << std::get<2>(pt) << std::endl;
         auto tps = gen_type_patterns(pt, ncre, nann, "c");
         std::copy(tps.begin(), tps.end(), std::back_inserter(tpatterns));          
      }
      if(is_same and ncre==nann) tpatterns = remove_hermitian(tpatterns);
      if(!is_same) tpatterns = remove_minusplus(tpatterns); 
      return tpatterns;
   }

   inline std::vector<type_pattern> all_first_type_patterns(const int ncre, const int nann,
         const bool is_same){
      std::vector<type_pattern> tpatterns;
      auto patterns = first_number_patterns(ncre+nann);
      for(const auto& pt : patterns){
         auto tps = gen_type_patterns(pt, ncre, nann, "lc");
         std::copy(tps.begin(), tps.end(), std::back_inserter(tpatterns));          
      }
      if(is_same and ncre==nann) tpatterns = remove_hermitian(tpatterns); 
      if(!is_same) tpatterns = remove_minusplus(tpatterns); 
      return tpatterns;
   }

   inline std::vector<type_pattern> all_last_type_patterns(const int ncre, const int nann,
         const bool is_same){
      std::vector<type_pattern> tpatterns;
      auto patterns = last_number_patterns(ncre+nann);
      for(const auto& pt : patterns){
         auto tps = gen_type_patterns(pt, ncre, nann, "cr");
         std::copy(tps.begin(), tps.end(), std::back_inserter(tpatterns));          
      }
      if(is_same and ncre==nann) tpatterns = remove_hermitian(tpatterns); 
      if(!is_same) tpatterns = remove_minusplus(tpatterns); 
      return tpatterns;
   }

} // ctns

#endif
