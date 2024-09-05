#ifndef RDM_STRING_H
#define RDM_STRING_H

namespace ctns{

   // creation and annihilation operators
   using ca_type = std::pair<int,bool>;
   using ca_string = std::vector<ca_type>;

   inline ca_string get_calst(const char key, 
         const int idx, 
         const bool ifdagger){
      ca_string calst;
      if(key == 'I'){
         // pass 
      }else if(key == 'C'){
         calst.resize(1);
         calst[0] = std::make_pair(idx,(!ifdagger? 1 : 0));
      }else if(key == 'A'){
         calst.resize(2);
         auto pr = oper_unpack(idx);
         // (p1^+ p2^+)^+ (p1<p2) = p2 p1 (p2>p1)
         calst[0] = std::make_pair((!ifdagger? pr.first : pr.second), (!ifdagger? 1 : 0));
         calst[1] = std::make_pair((!ifdagger? pr.second : pr.first), (!ifdagger? 1 : 0));
      }else if(key == 'B'){
         calst.resize(2);
         auto pr = oper_unpack(idx);
         // (p1^+ p2)^+ = p2^+ p1
         calst[0] = std::make_pair((!ifdagger? pr.first : pr.second), 1);
         calst[1] = std::make_pair((!ifdagger? pr.second : pr.first), 0);
      }else{
         tools::exit("error: not implemented yet in get_calst!");
      }
      return calst;
   }

   struct rdmstring{
      public:
         rdmstring(const ca_string& lstr,
               const ca_string& cstr,
               const ca_string& rstr){
            std::copy(lstr.begin(), lstr.end(), std::back_inserter(calst));
            std::copy(cstr.begin(), cstr.end(), std::back_inserter(calst));
            std::copy(rstr.begin(), rstr.end(), std::back_inserter(calst));
         }
         bool ordered(const ca_type ca_op1,
               const ca_type ca_op2) const{
            if(ca_op1.second and !ca_op2.second){
               return true;
            }else if(!ca_op1.second and ca_op2.second){
               return false;
            }else if(ca_op1.second and ca_op2.second){
               assert(ca_op1.first != ca_op2.first);
               return ca_op1.first > ca_op2.first;
            }else if(!ca_op1.second and !ca_op2.second){
               assert(ca_op1.first != ca_op2.first);
               return ca_op1.first < ca_op2.first;
            }
         }
         // sort into p1+p2+q1q2 (p1>p2,q1<q2)
         int sort(){
            int sgn = 1;
            for(int i=1; i<calst.size(); i++){
               for(int j=calst.size()-1; j>=i; j--){
                  /*
                  std::cout << "i,j=" << i << "," << j << " " << calst[j-1].first << ":" << calst[j].first 
                     << " order=" << this->ordered(calst[j-1],calst[j]) 
                     << std::endl;
                  */
                  if(!this->ordered(calst[j-1],calst[j])){
                     std::swap(calst[j-1],calst[j]);
                     sgn = -sgn;
                  }
               }
            }
            return sgn;        
         }
         // rdmN(i,j)
         std::pair<int,int> get_ijdx() const{
            int idx, jdx;
            int order = calst.size();
            assert(order%2 == 0 and order<=4); // only for 1,2-RDMs
            order = order/2;
            if(order == 1){
               idx = calst[0].first;
               jdx = calst[1].first; 
            }else if(order == 2){
               idx = calst[0].first*(calst[0].first-1)/2 + calst[1].first;
               jdx = calst[2].first*(calst[3].first-1)/2 + calst[3].first; 
            }
            return std::make_pair(idx,jdx);
         }
         // print
         std::string to_string() const{
            std::string str="";
            for(int i=0; i<calst.size(); i++){
               str += std::to_string(calst[i].first)
                    + (calst[i].second? "+" : "-");
            }
            return str;
         }
      public:
         ca_string calst;
   };

} // ctns

#endif
