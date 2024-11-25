#ifndef OPER_PARTITION_H
#define OPER_PARTITION_H

namespace ctns{

   inline bool ifexistQ(const std::string oplist, const char key){
      return oplist.find(key) != std::string::npos;
   }

   inline bool determine_NCorCN_Ham(const std::string oplist1,
         const std::string oplist2,
         const size_t csize1,
         const size_t csize2){
      bool ifAP = ifexistQ(oplist1,'A') and ifexistQ(oplist2,'P');
      bool ifPA = ifexistQ(oplist1,'P') and ifexistQ(oplist2,'A');
      bool ifBQ = ifexistQ(oplist1,'B') and ifexistQ(oplist2,'Q');
      bool ifQB = ifexistQ(oplist1,'Q') and ifexistQ(oplist2,'B');
      assert(ifAP or ifPA);
      assert(ifBQ or ifQB);
      assert(ifAP == ifBQ);
      assert(ifPA == ifQB);
      const bool ifNC = (ifAP==ifPA and csize1<=csize2) or
                        (ifAP!=ifPA and ifAP==true);
      return ifNC;
   }

   inline bool determine_NCorCN_BQ(const std::string oplist1,
         const std::string oplist2,
         const size_t csize1,
         const size_t csize2){
      bool ifBQ = ifexistQ(oplist1,'B') and ifexistQ(oplist2,'Q');
      bool ifQB = ifexistQ(oplist1,'Q') and ifexistQ(oplist2,'B');
      assert(ifBQ or ifQB);
      const bool ifNC = (ifBQ==ifQB and csize1<=csize2) or
                        (ifBQ!=ifQB and ifBQ==true);
      return ifNC;
   }
 
} // ctns

#endif
