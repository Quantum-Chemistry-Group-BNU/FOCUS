#include "ctns/ctns_header.h"

using namespace std;
using namespace ctns;

int main(int argc, char** argv){

   if(argc != 2){
      std::cout << "error: input must be given!" << std::endl;
      exit(1);
   }
   
   string fop(argv[1]);
   string fname = "./scratch/sweep/"+fop;

   int iomode = 0;
   bool debug = true;
   using Tm = double;
   oper_dict<Tm> qops;
   
   // load
   oper_load(iomode, fname, qops, debug);

   // process
   string oplist = "C";
   for(const auto& key : oplist){
      const auto& qop = qops(key);
      for(const auto& pr : qop){
         const auto& index = pr.first;
         const auto& op = pr.second;
         std::cout << "\nop=" << key << " index=" << index << std::endl;
         op.print("op");
         for(int br=0; br<op.rows(); br++){
            int bc = op.info._br2bc[br];
            if(bc == -1) continue;
            auto blk2 = op(br,bc);
            auto mat2 = blk2.to_matrix();
            std::cout << "br,bc=" << br << "," << bc << std::endl;
            mat2.print("mat");
         } 
         exit(1);
      }
  }

  return 0;
}
