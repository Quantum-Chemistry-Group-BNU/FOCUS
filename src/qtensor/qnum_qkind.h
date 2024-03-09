#ifndef QNUM_QKIND_H
#define QNUM_QKIND_H

#include <complex>
#include <string>

namespace ctns{

   // kind of CTNS
   namespace qkind{ 

      // isym = 0:
      struct qZ2{
         static const int isym = 0;
         static const bool ifkr = false;
         static const bool ifabelian = true;
      };
      // isym = 1:
      struct qN{
         static const int isym = 1;
         static const bool ifkr = false;
         static const bool ifabelian = true;
      };
      // isym = 2:
      struct qNSz{
         static const int isym = 2;
         static const bool ifkr = false;
         static const bool ifabelian = true;
      };
      // Kramers symmetry: relativistic H with SOC
      struct qNK{
         static const int isym = 1;
         static const bool ifkr = true;
         static const bool ifabelian = true;
      };
      // isym = 3;
      struct qNS{
         static const int isym = 3;
         static const bool ifkr = true; // algorithmically similar to qNK
         static const bool ifabelian = false;
      };

      // is_available
      template <typename Qm>
         inline bool is_available(){ return false; }
      template <> inline bool is_available<qZ2>(){ return true; } 
      template <> inline bool is_available<qN>(){ return true; } 
      template <> inline bool is_available<qNSz>(){ return true; }
      template <> inline bool is_available<qNK>(){ return true; }
      template <> inline bool is_available<qNS>(){ return true; }

      // get_name
      template <typename Qm>
         inline std::string get_name(){ return ""; }
      template <> inline std::string get_name<qZ2>(){ return "qZ2"; } 
      template <> inline std::string get_name<qN>(){ return "qN"; } 
      template <> inline std::string get_name<qNSz>(){ return "qNSz"; }
      template <> inline std::string get_name<qNK>(){ return "qNK"; }
      template <> inline std::string get_name<qNS>(){ return "qNS"; }

      template <typename Qm>
         inline bool is_qNSz(){ return false; }
      template <> inline bool is_qNSz<qNSz>(){ return true; }

   } // qkind

} // ctns

#endif
