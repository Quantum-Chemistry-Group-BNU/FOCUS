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
      }else if(key == 'D'){
         calst.resize(1);
         calst[0] = std::make_pair(idx,(!ifdagger? 0 : 1));
      }else if(key == 'A'){
         calst.resize(2);
         auto pr = oper_unpack(idx);
         // (p1^+ p2^+)^+ (p1<p2) = p2 p1 (p2>p1)
         calst[0] = std::make_pair((!ifdagger? pr.first : pr.second), (!ifdagger? 1 : 0));
         calst[1] = std::make_pair((!ifdagger? pr.second : pr.first), (!ifdagger? 1 : 0));
      }else if(key == 'M'){
         calst.resize(2);
         auto pr = oper_unpack(idx);
         // (p1 p2) (p1<p2) [dagger does not apper]
         assert(!ifdagger);
         calst[0] = std::make_pair(pr.first, 0);
         calst[1] = std::make_pair(pr.second, 0);
      }else if(key == 'B'){
         calst.resize(2);
         auto pr = oper_unpack(idx);
         // (p1^+ p2)^+ = p2^+ p1
         calst[0] = std::make_pair((!ifdagger? pr.first : pr.second), 1);
         calst[1] = std::make_pair((!ifdagger? pr.second : pr.first), 0);
      // just for dot operators 
      }else if(key == 'T'){
         calst.resize(3);
         if(idx%2 == 0){
            // a+ba and (a+ba)^+=a+b+a
            calst[0] = std::make_pair(idx, 1);
            calst[1] = std::make_pair(idx+1, (!ifdagger? 0 : 1));
            calst[2] = std::make_pair(idx, 0);
         }else{
            // b+ba and (b+ba)^+=a+b+b=b+ba+
            calst[0] = std::make_pair(idx, 1);
            calst[1] = std::make_pair(idx, 0);
            calst[2] = std::make_pair(idx-1, (!ifdagger? 0 : 1));
         }
      }else if(key == 'F'){
         calst.resize(4);
         calst[0] = std::make_pair(idx, 1);
         calst[1] = std::make_pair(idx+1, 1);
         calst[2] = std::make_pair(idx+1, 0);
         calst[3] = std::make_pair(idx, 0);
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
         int get_ncre() const{
            int ncre = 0;
            for(int i=0; i<calst.size(); i++){
               ncre += calst[i].second; 
            }
            return ncre;
         }
         // rdmN(i,j)
         std::pair<size_t,size_t> get_ijdx() const{
            int ncre = this->get_ncre();
            int nann = calst.size() - ncre;
            size_t idx, jdx;
            assert(ncre < 4 and nann < 4);
            // (1,0)
            if(ncre == 1 and nann == 0){
               idx = calst[0].first;
               jdx = 0;
            // (0,1)
            }else if(ncre == 0 and nann == 1){
               idx = 0;
               jdx = calst[0].first;
            // (1,1)
            }else if(ncre == 1 and nann == 1){
               idx = calst[0].first;
               jdx = calst[1].first;
            // (2,0)
            }else if(ncre == 2 and nann == 0){
               assert(calst[0] > calst[1]);
               idx = calst[0].first*(calst[0].first-1)/2 + calst[1].first;
               jdx = 0;
            // (0,2)
            }else if(ncre == 0 and nann == 2){
               idx = 0;
               assert(calst[0] < calst[1]);
               jdx = calst[1].first*(calst[1].first-1)/2 + calst[0].first;
            // (2,1)
            }else if(ncre == 2 and nann == 1){
               assert(calst[0] > calst[1]);
               idx = calst[0].first*(calst[0].first-1)/2 + calst[1].first;
               jdx = calst[2].first;
            // (1,2)
            }else if(ncre == 1 and nann == 2){
               idx = calst[0].first;
               assert(calst[1] < calst[2]);
               jdx = calst[2].first*(calst[2].first-1)/2 + calst[1].first;
            // (3,0)
            }else if(ncre == 3 and nann == 0){
               assert(calst[0] > calst[1] and calst[1] > calst[2]);
               idx = calst[0].first*(calst[0].first-1)*(calst[0].first-2)/6
                   + calst[1].first*(calst[1].first-1)/2
                   + calst[2].first;
               jdx = 0;
            // (0,3)
            }else if(ncre == 0 and nann == 3){
               idx = 0;
               assert(calst[0] < calst[1] and calst[1] < calst[2]);
               jdx = calst[2].first*(calst[2].first-1)*(calst[2].first-2)/6
                   + calst[1].first*(calst[1].first-1)/2
                   + calst[0].first;
            // (2,2)
            }else if(ncre == 2 and nann == 2){
               // RDM2(p0p1,q0q1)=<p0+p1+q1q0> (p0>p1,q1<q0)
               assert(calst[0] > calst[1]);
               idx = calst[0].first*(calst[0].first-1)/2 + calst[1].first;
               assert(calst[2] < calst[3]);
               jdx = calst[3].first*(calst[3].first-1)/2 + calst[2].first;
            // (3,1)
            }else if(ncre == 3 and nann == 1){
               assert(calst[0] > calst[1] and calst[1] > calst[2]);
               idx = calst[0].first*(calst[0].first-1)*(calst[0].first-2)/6
                   + calst[1].first*(calst[1].first-1)/2
                   + calst[2].first;
               jdx = calst[3].first;
            // (1,3)
            }else if(ncre == 1 and nann == 3){
               idx = calst[0].first;
               assert(calst[1] < calst[2] and calst[2] < calst[3]);
               jdx = calst[3].first*(calst[3].first-1)*(calst[3].first-2)/6
                   + calst[2].first*(calst[2].first-1)/2
                   + calst[1].first;
            // (3,2)
            }else if(ncre == 3 and nann == 2){
               assert(calst[0] > calst[1] and calst[1] > calst[2]);
               idx = calst[0].first*(calst[0].first-1)*(calst[0].first-2)/6
                   + calst[1].first*(calst[1].first-1)/2
                   + calst[2].first;
               assert(calst[3] < calst[4]);
               jdx = calst[4].first*(calst[4].first-1)/2 + calst[3].first;
            // (2,3)
            }else if(ncre == 2 and nann == 3){
               assert(calst[0] > calst[1]);
               idx = calst[0].first*(calst[0].first-1)/2 + calst[1].first;
               assert(calst[2] < calst[3] and calst[3] < calst[4]);
               jdx = calst[4].first*(calst[4].first-1)*(calst[4].first-2)/6
                   + calst[3].first*(calst[3].first-1)/2
                   + calst[2].first;
            // other cases
            }else{
               std::cout << "error: (ncre,nann)=" << ncre << "," << nann 
                  << " is not supported in get_ijdx!"
                  << std::endl;
               exit(1); 
            }
            return std::make_pair(idx,jdx);
         }
         // print
         std::string to_string() const{
            std::string str="";
            for(int i=0; i<calst.size(); i++){
               str += std::to_string(calst[i].first/2)
                    + (calst[i].first%2==0? "A" : "B")
                    + (calst[i].second? "+" : "-");
            }
            return str;
         }
         std::string to_string1() const{
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
