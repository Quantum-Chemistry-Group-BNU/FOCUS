#ifndef CTNS_FROMNOSYM_H
#define CTNS_FROMNOSYM_H

#include "../init_phys.h"

namespace ctns{

   // convert MPS without symmetry to qNSz via sweep projection
   template <typename Qm, typename Tm>
      void rcanon_fromnosym(comb<Qm,Tm>& icomb){
         std::cout << "\nctns::rcanon_fromnosym" << std::endl;
         std::cout << "error: not implemented yet!" << std::endl;
         exit(1);
      }
   template <typename Tm>
      void rcanon_fromnosym(comb<qkind::qNSz,Tm>& icomb,
                            const std::string prefix="rmps"){
         const bool debug = true;
         int nsite = icomb.get_nphysical();
         std::cout << "\nctns::rcanon_fromnosym nsite=" << nsite << std::endl;
         auto t0 = tools::get_time();

         // load nsectors
         std::vector<int> nsectors(nsite+2);
         {
            std::ifstream ifs2(prefix+".nsectors", std::ios::binary);
            ifs2.read((char*)(nsectors.data()), nsectors.size()*sizeof(int));
            ifs2.close();
         }
         if(debug) tools::print_vector(nsectors, "nsectors");
        
         // load qbonds 
         std::vector<qbond> qbonds(nsite+2);
         for(int i=0; i<nsite+2; i++){
            std::vector<int> data(nsectors[i]*3);
            std::ifstream ifs2(prefix+".qbond"+std::to_string(i), std::ios::binary);
            ifs2.read((char*)(data.data()), data.size()*sizeof(int));
            ifs2.close();
            // setup qbond
            qbonds[i].dims.resize(nsectors[i]);
            for(int k=0; k<nsectors[i]; k++){
               int ne  = data[3*k];
               int tm  = data[3*k+1];
               int dim = data[3*k+2];
               qsym sym(2,ne,tm);
               qbonds[i].dims[k] = std::make_pair(sym,dim);
            }
            if(debug) qbonds[i].print("qbond");
         }

         // load sites
         icomb.sites.resize(nsite);
         for(int i=0; i<nsite; i++){
            const auto& qrow = qbonds[i+1];
            const auto& qcol = qbonds[i];
            const auto& qmid = qbonds[1];
            icomb.sites[i].init(qsym(2,0,0),qrow,qcol,qmid);
            // load data
            std::ifstream ifs2(prefix+".site"+std::to_string(i), std::ios::binary);
            ifs2.read((char*)(icomb.sites[i].data()), icomb.sites[i].size()*sizeof(Tm));
            ifs2.close();
         }
         
         // load rwfuns
         qsym sym_state = qbonds[nsite+1].get_sym(0);
         int nroot = qbonds[nsite+1].get_dim(0);
         qbond qleft;
         qleft.dims.push_back(std::make_pair(sym_state,1)); 
         std::cout << "sym_state=" << sym_state << " nroot=" << nroot << std::endl;
         // load rwfuns
         std::ifstream ifs2(prefix+".rwfuns", std::ios::binary);
         icomb.rwfuns.resize(nroot);
         for(int i=0; i<nroot; i++){
            icomb.rwfuns[i].init(qsym(2,0,0),qleft,qbonds[nsite],dir_RWF);
            ifs2.read((char*)(icomb.rwfuns[i].data()), icomb.rwfuns[i].size()*sizeof(Tm));
         }
         ifs2.close();

         // check
         icomb.display_shape();

         auto t1 = tools::get_time();
         tools::timing("ctns::rcanon_fromnosym", t0, t1);
      }

} // ctns

#endif
