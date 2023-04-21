#ifndef SWEEP_DECIM_UTIL_H
#define SWEEP_DECIM_UTIL_H

#ifdef _OPENMP
#include <omp.h>
#endif

#include <numeric>

namespace ctns{

   const double thresh_sig2 = 1.e-14;
   extern const double thresh_sig2;

   const bool debug_decimation = false;
   extern const bool debug_decimation;

   // select important sigs
   inline void decimation_selection(const bool ifkr,
         const qbond& qrow,
         const std::vector<bool>& ifmatched,
         const std::vector<double>& sig2all,
         const std::map<int,int>& idx2sector,
         const int& dcut,
         double& dwt,
         int& deff,
         std::vector<int>& br_kept,
         std::vector<std::pair<qsym,int>>& dims,
         const std::string fname){

      std::streambuf *psbuf, *backup;
      std::ofstream file;
      bool ifsave = !fname.empty();
      if(ifsave){
         // http://www.cplusplus.com/reference/ios/ios/rdbuf/
         file.open(fname);
         backup = std::cout.rdbuf(); // back up cout's streambuf
         psbuf = file.rdbuf(); // get file's streambuf
         std::cout.rdbuf(psbuf); // assign streambuf to cout
      }

      auto index = tools::sort_index(sig2all, 1); // sort all sigs
      const int nqr = qrow.size();
      std::vector<int> kept_dim(nqr,0); // no. of states kept in each symmetry sector
      std::vector<double> kept_wts(nqr,0.0); // weights kept in each symmetry sector
      deff = 0; // bond dimension kept (including additional for symmetry)
      double accum = 0.0, SvN = 0.0;
      std::cout << "sorted renormalized states: total=" << sig2all.size()
         << " dcut=" << dcut << " thresh_sig2=" << thresh_sig2 
         << std::endl;
      for(int i=0; i<sig2all.size(); i++){
         if(dcut > -1 && deff >= dcut) break; // discard rest
         int idx = index[i];
         if(sig2all[idx] < thresh_sig2) continue; // discard negative weights
         int br = idx2sector.at(idx);
         auto qr = qrow.get_sym(br);
         int nfac = (ifkr && qr.parity()==1)? 2 : 1; // odd case: kept KR-pair
         deff += nfac;
         kept_dim[br] += nfac;
         kept_wts[br] += nfac*sig2all[idx];
         accum += nfac*sig2all[idx];
         SvN += -nfac*sig2all[idx]*std::log2(sig2all[idx]);
         std::cout << " i=" << i << " qr=" << qr 
            << " " << kept_dim[br]/nfac-1 << "-th"
            << " sig2=" << sig2all[idx] 
            << " accum=" << accum << std::endl;
      } // i
      dwt = 1.0-accum;
      // construct qbond & recompute deff including additional states 
      deff = 0;
      accum = 0.0;
      // order symmetry sectors by kept weights / dimensions
      std::vector<int> index2;
      if(sort_by_dim){
         index2 = tools::sort_index(kept_dim, 1);
      }else{
         index2 = tools::sort_index(kept_wts, 1); 
      }
      std::cout << "select renormalized states per symmetry sector: nqr=" << nqr << std::endl;
      for(int iqr=0; iqr<nqr; iqr++){
         int br = index2[iqr];
         const auto& qr = qrow.get_sym(br);
         const auto& dim0 = qrow.get_dim(br);
         const auto& dim = kept_dim[br];
         const auto& wts = kept_wts[br];
         if(dim != 0){
            br_kept.push_back(br);
            dims.emplace_back(qr,dim);
            accum += wts;    
            deff += dim;
            // save information
            std::cout << " iqr=" << iqr << " qr=" << qr
               << " dim[full,kept]=" << dim0 << "," << dim 
               << " wts=" << wts << " accum=" << accum << " deff=" << deff 
               << std::endl;
         }else{
            // additional: kept at least one state per sector
            // ZL@20220517 disable such choice, since it will create many sector with dim=1 
            /*
               if(!ifmatched[br]) continue;
               br_kept.push_back(br);
               int dmin = (ifkr && qr.parity()==1)? 2 : 1;
               dims.emplace_back(qr,dmin);
               deff += dmin;
            // save information
            std::cout << " iqr=" << iqr << " qr=" << qr
            << " dim[full,kept]=" << dim0 << "," << dmin 
            << " wts=" << wts << " accum=" << accum << " deff=" << deff
            << " (additional)" << std::endl;
            */
         }
      } // iqr

      if(ifsave){
         std::cout << "decimation summary: " << qrow.get_dimAll() << "->" << deff  
            << " dwt=" << std::showpos << std::scientific << std::setprecision(3) << dwt 
            << " SvN=" << std::noshowpos << SvN << std::endl;
         std::cout.rdbuf(backup); // restore cout's original streambuf
         file.close();
      }
      std::cout << "decimation summary: " << qrow.get_dimAll() << "->" << deff  
         << " dwt=" << std::showpos << std::scientific << std::setprecision(3) << dwt 
         << " SvN=" << std::noshowpos << SvN << std::endl;
   }

} // ctns

#endif
