#ifndef RDM_OPUTIL_H
#define RDM_OPUTIL_H

namespace ctns{

   // From type pattern to operators in qops
   // is_same = true
   const std::map<std::string,std::pair<char,bool>> str2optype_same = {
      {"",{'I',0}},
      {"+",{'C',0}},
      {"-",{'C',1}},
      {"++",{'A',0}},
      {"--",{'A',1}},
      {"+-",{'B',0}},
      {"-+",{'B',1}},
      // only needed for dot operators: cop, lop[fpattern], rop[lpattern]
      {"+--",{'T',0}},
      {"++-",{'T',1}},
      {"++--",{'F',0}}
   };
   extern const std::map<std::string,std::pair<char,bool>> str2optype_same;
   
   // is_same = false
   const std::map<std::string,std::pair<char,bool>> str2optype_diff = {
      {"",{'I',0}},
      {"+",{'C',0}},
      {"-",{'D',0}},
      {"++",{'A',0}},
      {"--",{'M',0}},
      {"+-",{'B',0}},
      // only needed for dot operators: cop, lop[fpattern], rop[lpattern]
      {"+--",{'T',0}},
      {"++-",{'T',1}},
      {"++--",{'F',0}}
   };
   extern const std::map<std::string,std::pair<char,bool>> str2optype_diff;

   // Parity of operators
   const std::map<char,int> op2parity = {
      {'I',0},
      {'C',1},{'D',1},
      {'A',0},{'B',0},{'M',0},
      {'T',1},{'F',0}
   };
   extern const std::map<char,int> op2parity;

} // ctns

#endif
