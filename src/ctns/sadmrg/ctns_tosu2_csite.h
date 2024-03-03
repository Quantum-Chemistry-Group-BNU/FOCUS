#ifndef CTNS_TOSU2_CSITE_H
#define CTNS_TOSU2_CSITE_H

#include "ctns_tosu2_wmat.h"
#include "ctns_tosu2_msite.h"
#include "../../core/spin.h"

namespace ctns{

   using quintet = std::tuple<int,int,int,qsym,qsym>;
   using qdpt3 = std::map<qsym3,std::vector<quintet>>;
   using qproduct3 = std::pair<qbond3,qdpt3>;

   inline qproduct3 qmerge(const qbond3& qs1, const qbond3& qs2){
      const bool debug = false;
      qdpt3 dpt;
      // natural ordering qs1,qs2 based on the double loop
      for(int i1=0; i1<qs1.size(); i1++){
         auto q1 = qs1[i1].first;
         int ts1 = std::get<1>(q1);
         for(int i2=0; i2<qs2.size(); i2++){
            auto q2 = qs2[i2].first;
            int ts2 = std::get<1>(q2);
            // q1 * q2
            int n12 = std::get<0>(q1) + std::get<0>(q2);
            int tm12 = std::get<2>(q1) + std::get<2>(q2);
            for(int ts12=std::abs(ts1-ts2); ts12<=ts1+ts2; ts12+=2){
               // assemble those direct product blocks that can contribute to (N,S,M)
               if(std::abs(tm12) > ts12) continue;
               assert((ts12-tm12)%2==0);
               qsym3 q12({n12,ts12,tm12});
               qsym qns1(3,std::get<0>(q1),ts1);
               qsym qns2(3,std::get<0>(q2),ts2);
               dpt[q12].push_back(std::make_tuple(i1,i2,-1,qns1,qns2));
            }
         }
      }
      // sort combination: this will ensure direct product {(N,S1,M1)(N,S2,M1)}
      // with different M for the same N and S will present consecutively.
      for(auto& pr : dpt){
         auto& q12 = pr.first;
         auto& comp = pr.second;
         std::stable_sort(comp.begin(), comp.end(),
               [](const quintet& item1, const quintet& item2){
               return std::get<3>(item1) < std::get<3>(item2) or
               (std::get<3>(item1) == std::get<3>(item2) &&
                std::get<4>(item1) < std::get<4>(item2));
               });
      }
      // evaluate offset 
      qbond3 qs12(dpt.size());
      int idx = 0; 
      for(auto& pr : dpt){
         auto& q12 = pr.first;
         auto& comp = pr.second;
         if(debug){
            std::cout << "\nsym=(" << std::get<0>(q12) << ","
               << std::get<1>(q12) << "," 
               << std::get<2>(q12) << ")"
               << std::endl;
         }
         int i1 = std::get<0>(comp[0]);
         int i2 = std::get<1>(comp[0]);
         auto q1 = qs1[i1].first;
         auto q2 = qs2[i2].first;
         int d1 = qs1[i1].second;
         int d2 = qs2[i2].second;
         qsym sym1(3,std::get<0>(q1),std::get<1>(q1));
         qsym sym2(3,std::get<0>(q2),std::get<1>(q2));
         size_t off = 0; 
         for(int i=0; i<comp.size(); i++){
            int i1 = std::get<0>(comp[i]);
            int i2 = std::get<1>(comp[i]);
            auto q1 = qs1[i1].first;
            auto q2 = qs2[i2].first;
            qsym sym1new(3,std::get<0>(q1),std::get<1>(q1));
            qsym sym2new(3,std::get<0>(q2),std::get<1>(q2));
            // new combination is found
            if(sym1 != sym1new || sym2 != sym2new){
               off += d1*d2; // d1,d2 are old dimensions
               sym1 = sym1new;
               sym2 = sym2new;
            }
            // update d1 and d2
            std::get<2>(comp[i]) = off;
            d1 = qs1[i1].second;
            d2 = qs2[i2].second;
            if(debug){
               std::cout << " i=" << i << " i1,i2=" << i1 << "," << i2 
                  << " q1=(" << std::get<0>(q1) 
                  << "," << std::get<1>(q1) 
                  << "," << std::get<2>(q1) << ")" 
                  << " q2=(" << std::get<0>(q2) 
                  << "," << std::get<1>(q2)
                  << "," << std::get<2>(q2) << ")"
                  << " d1=" << d1 << " d2=" << d2
                  << " off=" << std::get<2>(comp[i])
                  << std::endl;
            }
         } // i
         off += d1*d2;
         qs12[idx] = std::make_pair(q12,off);
         idx += 1;
         if(debug) std::cout << "dim=" << off << std::endl;
      }
      //if(debug) display_qbond3(qs12,"qc");
      return std::make_pair(qs12,dpt);
   }

   // msite[l,c,r] => msite[l,cr] => msite[l,(cr)]
   template <typename Tm> 
      Wmatrix<Tm> formCoupledRSite(const MixedRSite<Tm>& msite,
            const qproduct3& qprod,
            const qbond3& qs1,
            const qbond3& qs2,
            const bool debug=true){
         if(debug) std::cout << "\nctns::formCoupledRSite" << std::endl; 
         // construct csite[l,(cr)] = A[l,c,r] by appropriate cgcoeff
         const auto& qs12 = qprod.first;
         const auto& dpt = qprod.second;
         const auto& qrow = msite.qrow;
         Wmatrix<Tm> csite(qrow,qs12);
         for(int j=0; j<qs12.size(); j++){
            const auto& q12 = qs12[j].first;
            const auto& comp = dpt.at(q12);
            int ts12 = std::get<1>(q12);
            int tm12 = std::get<2>(q12);
            for(int k=0; k<comp.size(); k++){
               int k1 = std::get<0>(comp[k]); // c
               int k2 = std::get<1>(comp[k]); // r
               auto q1 = qs1[k1].first;
               auto q2 = qs2[k2].first;
               int ts1 = std::get<1>(q1);
               int ts2 = std::get<1>(q2);
               int tm1 = std::get<2>(q1);
               int tm2 = std::get<2>(q2);
               assert(tm12 == tm1+tm2);
               Tm cg = fock::cgcoeff(ts1,ts2,ts12,tm1,tm2,tm12);
               int d1 = qs1[k1].second;
               int d2 = qs2[k2].second;
               // offcr need to multiply dimr 
               size_t offcr = std::get<2>(comp[k]); 
               for(int i=0; i<qrow.size(); i++){
                  size_t off = csite._offset[std::make_pair(i,j)];
                  size_t off0 = msite._offset.at(std::make_tuple(i,k2,k1));
                  if(off == 0 || off0 == 0) continue;
                  // qt3[l,r,c] -> qt2[l,cr], storage c[slow]r,
                  // which is different from merge_cr, in order to 
                  // be local in space
                  int dl = qrow.get_dim(i);
                  int dc = d1;
                  int dr = d2;
                  const Tm* xptr = msite._data.data() + off0-1;
                  Tm* yptr = csite._data.data() + off-1 + offcr*dl;
                  size_t N = dl*dr*dc;
                  linalg::xaxpy(N, cg, xptr, yptr);
                  /*
                     const Tm* xptr = msite._data.data() + off0-1;
                     Tm* yptr = csite._data.data() + off-1 + offcr;
                     int rdim = qrow.get_dim(i);
                     int mdim = d1;
                     int cdim = d2;
                     for(int ic=0; ic<cdim; ic++){
                     for(int im=0; im<mdim; im++){
                     for(int ir=0; ir<rdim; ir++){
                  // qt3[l,r,c] -> qt2[l,cr], storage c[fast]r
                   *(yptr + (ic*mdim+im)*rdim + ir) = (*(xptr + (im*cdim+ic)*rdim + ir)) * cg; 
                   } // ir
                   } // ic
                   } // im
                   */
               } // i
            } // k
         } // j
         return csite;
      }

} // ctns

#endif
