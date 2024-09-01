#ifndef CTNS_LOADCONFS_H
#define CTNS_LOADCONFS_H

#include "ctns_det2mps.h"
#include "sadmrg/ctns_csf2samps.h"

namespace ctns{

   template <typename Tm>
      double parse_confs(const std::string fname,
            std::vector<std::string>& confs,
            std::vector<Tm>& coeffs){
         std::cout << "\nctns::parse_confs";
         size_t idx=0, size=-1;
         std::string line; 
         std::ifstream file(fname+".txt");
         double psum = 0.0, psum0 = 0.0;
         while(!file.eof()){
            line.clear();
            std::getline(file,line);
            if(idx == size) continue;
            if(line.substr(0,5)=="size="){
               std::string tmp;
               std::istringstream is(line.substr(5));
               is >> size >> tmp >> psum0;
               confs.resize(size);
               coeffs.resize(size);
               std::cout << " size=" << size << " psum0=" << psum0;
            }else{
               std::istringstream is(line);   
               std::string conf;
               Tm coeff;
               is >> conf >> coeff;
               confs[idx] = conf;
               coeffs[idx] = coeff;
               psum += std::norm(coeff);
               idx++;
            }
         }
         file.close();
         std::cout << " psum=" << psum << " diff=" << psum-psum0 << std::endl;
         if(std::abs(psum-psum0) > 1.e-10){
            tools::exit("error: inconsistent psum!");
         }
         return psum;
      }

   // Addition of single-state MPS:
   template <typename Qm, typename Tm, std::enable_if_t<Qm::ifabelian,int> = 0>
      void rmps_addition_rsite0(const comb<Qm,Tm>& mps1,
            const comb<Qm,Tm>& mps2,
            comb<Qm,Tm>& mps12,
            const bool debug){
         if(debug) std::cout << "ctns::rmps_addition_rsite0" << std::endl;
         const auto& site1 = mps1.sites[0];
         const auto& site2 = mps2.sites[0];
         const auto& qrow1 = site1.info.qrow;
         const auto& qrow2 = site2.info.qrow;
         const auto& qcol1 = site1.info.qcol;
         const auto& qcol2 = site2.info.qcol;
         const auto& qmid = site1.info.qmid;
         assert(qcol1 == qcol2 and qcol1.size() == 1);
         // rather than direct product, we need direct sum of two
         auto qrow12 = qrow1.directsum(qrow2);
         assert(site1.info.dir == dir_RCF);
         qtensor3<Qm::ifabelian,Tm> site12(site1.info.sym, qrow12, qcol1, qmid, dir_RCF);
         int br, bc, bm;
         for(int i=0; i<site12.info._nnzaddr.size(); i++){
            int idx = site12.info._nnzaddr[i];
            site12.info._addr_unpack(idx,br,bc,bm);
            auto blk = site12(br,bc,bm);
            qsym qr = site12.info.qrow.get_sym(br);
            qsym qc = site12.info.qcol.get_sym(bc);
            qsym qm = site12.info.qmid.get_sym(bm);
            int br1 = qrow1.existQ(qr);
            int br2 = qrow2.existQ(qr);
            if(br1 != -1 and br2 == -1){
               const auto blk1 = site1(br1,bc,bm);
               assert(blk1.size() != 0);
               assert(blk1.size() == blk.size());
               linalg::xcopy(blk1.size(), blk1.data(), blk.data());
            }else if(br1 == -1 and br2 != -1){
               const auto blk2 = site2(br2,bc,bm);
               assert(blk2.size() != 0);
               assert(blk2.size() == blk.size());
               linalg::xcopy(blk2.size(), blk2.data(), blk.data());
            }else if(br1 != -1 and br2 != -1){
               const auto blk1 = site1(br1,bc,bm);
               assert(blk1.size() != 0);
               const auto blk2 = site2(br2,bc,bm);
               assert(blk2.size() != 0);
               for(int im=0; im<blk.dim2; im++){
                  for(int ic=0; ic<blk.dim1; ic++){
                     for(int ir=0; ir<blk1.dim0; ir++){
                        blk(ir,ic,im) = blk1(ir,ic,im);
                     }
                  }
               }
               int dim0 = blk1.dim0;
               for(int im=0; im<blk.dim2; im++){
                  for(int ic=0; ic<blk.dim1; ic++){
                     for(int ir=0; ir<blk2.dim0; ir++){
                        blk(dim0+ir,ic,im) = blk2(ir,ic,im);
                     }
                  }
               }
            }
         } // i
         mps12.sites[0] = std::move(site12);
      }
   template <typename Qm, typename Tm, std::enable_if_t<!Qm::ifabelian,int> = 0>
      void rmps_addition_rsite0(const comb<Qm,Tm>& mps1,
            const comb<Qm,Tm>& mps2,
            comb<Qm,Tm>& mps12,
            const bool debug){
         if(debug) std::cout << "ctns::rmps_addition_rsite0(su2)" << std::endl;
         const auto& site1 = mps1.sites[0];
         const auto& site2 = mps2.sites[0];
         const auto& qrow1 = site1.info.qrow;
         const auto& qrow2 = site2.info.qrow;
         const auto& qcol1 = site1.info.qcol;
         const auto& qcol2 = site2.info.qcol;
         const auto& qmid = site1.info.qmid;
         assert(qcol1 == qcol2 and qcol1.size() == 1);
         // rather than direct product, we need direct sum of two
         auto qrow12 = qrow1.directsum(qrow2);
         assert(site1.info.dir == dir_RCF);
         qtensor3<Qm::ifabelian,Tm> site12(site1.info.sym, qrow12, qcol1, qmid, dir_RCF, CRcouple);
         int br, bc, bm, tsi;
         for(int i=0; i<site12.info._nnzaddr.size(); i++){
            auto key = site12.info._nnzaddr[i];
            br = std::get<0>(key);
            bc = std::get<1>(key);
            bm = std::get<2>(key);
            tsi = std::get<3>(key);
            auto blk = site12(br,bc,bm,tsi);
            qsym qr = site12.info.qrow.get_sym(br);
            qsym qc = site12.info.qcol.get_sym(bc);
            qsym qm = site12.info.qmid.get_sym(bm);
            int br1 = qrow1.existQ(qr);
            int br2 = qrow2.existQ(qr);
            if(br1 != -1 and br2 == -1){
               const auto blk1 = site1(br1,bc,bm,tsi);
               assert(blk1.size() != 0);
               assert(blk1.size() == blk.size());
               linalg::xcopy(blk1.size(), blk1.data(), blk.data());
            }else if(br1 == -1 and br2 != -1){
               const auto blk2 = site2(br2,bc,bm,tsi);
               assert(blk2.size() != 0);
               assert(blk2.size() == blk.size());
               linalg::xcopy(blk2.size(), blk2.data(), blk.data());
            }else if(br1 != -1 and br2 != -1){
               const auto blk1 = site1(br1,bc,bm,tsi);
               assert(blk1.size() != 0);
               const auto blk2 = site2(br2,bc,bm,tsi);
               assert(blk2.size() != 0);
               for(int im=0; im<blk.dim2; im++){
                  for(int ic=0; ic<blk.dim1; ic++){
                     for(int ir=0; ir<blk1.dim0; ir++){
                        blk(ir,ic,im) = blk1(ir,ic,im);
                     }
                  }
               }
               int dim0 = blk1.dim0;
               for(int im=0; im<blk.dim2; im++){
                  for(int ic=0; ic<blk.dim1; ic++){
                     for(int ir=0; ir<blk2.dim0; ir++){
                        blk(dim0+ir,ic,im) = blk2(ir,ic,im);
                     }
                  }
               }
            }
         } // i
         mps12.sites[0] = std::move(site12);
      }

   template <typename Qm, typename Tm, std::enable_if_t<Qm::ifabelian,int> = 0>
      void rmps_addition_sites(const int isite,
            const comb<Qm,Tm>& mps1,
            const comb<Qm,Tm>& mps2,
            comb<Qm,Tm>& mps12,
            const bool debug){
         if(debug) std::cout << "ctns::rmps_addition_sites isite=" << isite << std::endl;
         const auto& site1 = mps1.sites[isite];
         const auto& site2 = mps2.sites[isite];
         const auto& qrow1 = site1.info.qrow;
         const auto& qrow2 = site2.info.qrow;
         const auto& qcol1 = site1.info.qcol;
         const auto& qcol2 = site2.info.qcol;
         const auto& qmid = site1.info.qmid;
         // rather than direct product, we need direct sum of two
         auto qrow12 = qrow1.directsum(qrow2);
         auto qcol12 = qcol1.directsum(qcol2);
         assert(site1.info.dir == dir_RCF);
         qtensor3<Qm::ifabelian,Tm> site12(site1.info.sym, qrow12, qcol12, qmid, dir_RCF);
         int br, bc, bm;
         for(int i=0; i<site12.info._nnzaddr.size(); i++){
            int idx = site12.info._nnzaddr[i];
            site12.info._addr_unpack(idx,br,bc,bm);
            auto blk = site12(br,bc,bm);
            qsym qr = site12.info.qrow.get_sym(br);
            qsym qc = site12.info.qcol.get_sym(bc);
            qsym qm = site12.info.qmid.get_sym(bm);
            int br1 = qrow1.existQ(qr);
            int br2 = qrow2.existQ(qr);
            int bc1 = qcol1.existQ(qc);
            int bc2 = qcol2.existQ(qc);
            bool exist1 = (br1 != -1) and (bc1 != -1);
            bool exist2 = (br2 != -1) and (bc2 != -1);
            if(exist1 and !exist2){
               const auto blk1 = site1(br1,bc1,bm);
               assert(blk1.size() != 0);
               for(int im=0; im<blk.dim2; im++){
                  for(int ic=0; ic<blk1.dim1; ic++){
                     for(int ir=0; ir<blk1.dim0; ir++){
                        blk(ir,ic,im) = blk1(ir,ic,im);
                     }
                  }
               }
            }else if(!exist1 and exist2){
               const auto blk2 = site2(br2,bc2,bm);
               assert(blk2.size() != 0);
               int dim0 = (br1==-1)? 0 : qrow1.get_dim(br1);
               int dim1 = (bc1==-1)? 0 : qcol1.get_dim(bc1);
               for(int im=0; im<blk.dim2; im++){
                  for(int ic=0; ic<blk2.dim1; ic++){
                     for(int ir=0; ir<blk2.dim0; ir++){
                        blk(dim0+ir,ic+dim1,im) = blk2(ir,ic,im);
                     }
                  }
               }
            }else if(exist1 and exist2){
               const auto blk1 = site1(br1,bc1,bm);
               assert(blk1.size() != 0);
               const auto blk2 = site2(br2,bc2,bm);
               assert(blk2.size() != 0);
               for(int im=0; im<blk.dim2; im++){
                  for(int ic=0; ic<blk1.dim1; ic++){
                     for(int ir=0; ir<blk1.dim0; ir++){
                        blk(ir,ic,im) = blk1(ir,ic,im);
                     }
                  }
               }
               int dim0 = blk1.dim0;
               int dim1 = blk1.dim1;
               for(int im=0; im<blk.dim2; im++){
                  for(int ic=0; ic<blk2.dim1; ic++){
                     for(int ir=0; ir<blk2.dim0; ir++){
                        blk(dim0+ir,ic+dim1,im) = blk2(ir,ic,im);
                     }
                  }
               }
            }
         } // i
         mps12.sites[isite] = std::move(site12);
      }
   template <typename Qm, typename Tm, std::enable_if_t<!Qm::ifabelian,int> = 0>
      void rmps_addition_sites(const int isite,
            const comb<Qm,Tm>& mps1,
            const comb<Qm,Tm>& mps2,
            comb<Qm,Tm>& mps12,
            const bool debug){
         if(debug) std::cout << "ctns::rmps_addition_sites(su2) isite=" << isite << std::endl;
         const auto& site1 = mps1.sites[isite];
         const auto& site2 = mps2.sites[isite];
         const auto& qrow1 = site1.info.qrow;
         const auto& qrow2 = site2.info.qrow;
         const auto& qcol1 = site1.info.qcol;
         const auto& qcol2 = site2.info.qcol;
         const auto& qmid = site1.info.qmid;
         // rather than direct product, we need direct sum of two
         auto qrow12 = qrow1.directsum(qrow2);
         auto qcol12 = qcol1.directsum(qcol2);
         assert(site1.info.dir == dir_RCF);
         qtensor3<Qm::ifabelian,Tm> site12(site1.info.sym, qrow12, qcol12, qmid, dir_RCF, CRcouple);
         int br, bc, bm, tsi;
         for(int i=0; i<site12.info._nnzaddr.size(); i++){
            auto key = site12.info._nnzaddr[i];
            br = std::get<0>(key);
            bc = std::get<1>(key);
            bm = std::get<2>(key);
            tsi = std::get<3>(key);
            auto blk = site12(br,bc,bm,tsi);
            qsym qr = site12.info.qrow.get_sym(br);
            qsym qc = site12.info.qcol.get_sym(bc);
            qsym qm = site12.info.qmid.get_sym(bm);
            int br1 = qrow1.existQ(qr);
            int br2 = qrow2.existQ(qr);
            int bc1 = qcol1.existQ(qc);
            int bc2 = qcol2.existQ(qc);
            bool exist1 = (br1 != -1) and (bc1 != -1);
            bool exist2 = (br2 != -1) and (bc2 != -1);
            if(exist1 and !exist2){
               const auto blk1 = site1(br1,bc1,bm,tsi);
               assert(blk1.size() != 0);
               for(int im=0; im<blk.dim2; im++){
                  for(int ic=0; ic<blk1.dim1; ic++){
                     for(int ir=0; ir<blk1.dim0; ir++){
                        blk(ir,ic,im) = blk1(ir,ic,im);
                     }
                  }
               }
            }else if(!exist1 and exist2){
               const auto blk2 = site2(br2,bc2,bm,tsi);
               assert(blk2.size() != 0);
               int dim0 = (br1==-1)? 0 : qrow1.get_dim(br1);
               int dim1 = (bc1==-1)? 0 : qcol1.get_dim(bc1);
               for(int im=0; im<blk.dim2; im++){
                  for(int ic=0; ic<blk2.dim1; ic++){
                     for(int ir=0; ir<blk2.dim0; ir++){
                        blk(dim0+ir,ic+dim1,im) = blk2(ir,ic,im);
                     }
                  }
               }
            }else if(exist1 and exist2){
               const auto blk1 = site1(br1,bc1,bm,tsi);
               assert(blk1.size() != 0);
               const auto blk2 = site2(br2,bc2,bm,tsi);
               assert(blk2.size() != 0);
               for(int im=0; im<blk.dim2; im++){
                  for(int ic=0; ic<blk1.dim1; ic++){
                     for(int ir=0; ir<blk1.dim0; ir++){
                        blk(ir,ic,im) = blk1(ir,ic,im);
                     }
                  }
               }
               int dim0 = blk1.dim0;
               int dim1 = blk1.dim1;
               for(int im=0; im<blk.dim2; im++){
                  for(int ic=0; ic<blk2.dim1; ic++){
                     for(int ir=0; ir<blk2.dim0; ir++){
                        blk(dim0+ir,ic+dim1,im) = blk2(ir,ic,im);
                     }
                  }
               }
            }
         } // i
         mps12.sites[isite] = std::move(site12);
      }

   // Both Abelian and Non-Abelian cases 
   template <typename Qm, typename Tm>
      void rmps_addition_rwfuns(const comb<Qm,Tm>& mps1,
            const comb<Qm,Tm>& mps2,
            comb<Qm,Tm>& mps12,
            const bool debug){
         if(debug) std::cout << "ctns::rmps_addition_rwfuns" << std::endl;
         assert(mps1.rwfuns.size()==1 and mps2.rwfuns.size()==1);
         const auto& rwfun1 = mps1.rwfuns[0];
         const auto& rwfun2 = mps2.rwfuns[0];
         const auto& qrow = mps1.rwfuns[0].info.qrow;
         const auto& qcol1 = mps1.rwfuns[0].info.qcol;
         const auto& qcol2 = mps2.rwfuns[0].info.qcol;
         assert(qrow.size()==1 and qcol1.size()==1 and qcol2.size()==1);
         assert(qcol1.get_sym(0) == qcol2.get_sym(0));
         int dim1 = qcol1.get_dim(0);
         int dim2 = qcol2.get_dim(0);
         int dim12 = dim1+dim2;
         qbond qcol12({{qcol1.get_sym(0),dim12}});
         qtensor2<Qm::ifabelian,Tm> rwfun(rwfun1.info.sym, qrow, qcol12, rwfun1.info.dir);
         linalg::xcopy(dim1, rwfun1.data(), rwfun.data());
         linalg::xcopy(dim2, rwfun2.data(), rwfun.data()+dim1);
         mps12.rwfuns.resize(1);
         mps12.rwfuns[0] = std::move(rwfun);
      }

   // MPS12 = MPS1 + MPS2
   template <typename Qm, typename Tm>
      comb<Qm,Tm> rmps_addition(const comb<Qm,Tm>& mps1,
            const comb<Qm,Tm>& mps2,
            const bool debug=false){
         assert(mps1.topo.ifmps and mps2.topo.ifmps);
         // 0. init
         comb<Qm,Tm> mps12;
         mps12.topo = mps1.topo;
         mps12.sites.resize(mps1.get_nphysical());
         // 1. add the right boundary
         rmps_addition_rsite0(mps1, mps2, mps12, debug);
         // 2. add the middle sites
         for(int isite=1; isite<mps1.get_nphysical(); isite++){
            rmps_addition_sites(isite, mps1, mps2, mps12, debug);
         }
         // 3. add rwfuns
         rmps_addition_rwfuns(mps1, mps2, mps12, debug);
         return mps12;
      }

   template <typename Qm, typename Tm, std::enable_if_t<Qm::ifabelian,int> = 0>
      void rmps_formation(const comb<Qm,Tm>& icomb,
            const std::vector<std::string>& confs,
            const std::vector<Tm>& coeffs,
            fock::onspace& space,
            comb<Qm,Tm>& mps){
         // loop over space
         size_t size = confs.size();
         for(int i=0; i<size; i++){
            fock::onstate state(confs[i], 1);
            space[i] = state;
            auto detmps = det2mps<Qm,Tm>(icomb.topo, state, false);
            detmps.rwfuns[0] *= coeffs[i]; // save coefficients here
            if(i == 0){
               mps = std::move(detmps);
            }else{
               mps = rmps_addition(mps, detmps);
            }
         }
      }
   template <typename Qm, typename Tm, std::enable_if_t<!Qm::ifabelian,int> = 0>
      void rmps_formation(const comb<Qm,Tm>& icomb,
            const std::vector<std::string>& confs,
            const std::vector<Tm>& coeffs,
            fock::csfspace& space,
            comb<Qm,Tm>& mps){
         // loop over space
         size_t size = confs.size();
         for(int i=0; i<size; i++){
            fock::csfstate state(confs[i]);
            space[i] = state;
            auto csfmps = csf2samps<Tm>(icomb.topo, state, false);
            csfmps.rwfuns[0] *= coeffs[i]; // save coefficients here
            if(i == 0){
               mps = std::move(csfmps);
            }else{
               mps = rmps_addition(mps, csfmps);
            }
         }
      }

   // Abelian case
   template <typename Qm, typename Tm>
      void rcanon_loadconfs(comb<Qm,Tm>& icomb,
            const std::string fname){
         using spacetype = typename std::conditional<Qm::ifabelian, fock::onspace, fock::csfspace>::type; 
         const bool ifab = Qm::ifabelian;
         std::cout << "\nctns::rcanon_loadconfs ifab=" << ifab << " fname=" << fname << std::endl;
         assert(!fname.empty());
         assert(icomb.topo.ifmps); // only for MPS
         auto t0 = tools::get_time();
         // load
         std::vector<std::string> confs;
         std::vector<Tm> coeffs;
         double psum = parse_confs(fname, confs, coeffs);
         auto t1 = tools::get_time();
         // form MPS
         size_t size = confs.size();
         spacetype space(size);
         comb<Qm,Tm> mps;
         rmps_formation(icomb, confs, coeffs, space, mps);
         auto t2 = tools::get_time();
         // canonicalize
         const bool ifortho = true;
         const bool debug = false; 
         rcanon_canonicalize(mps, ifortho, debug);
         auto t3 = tools::get_time();
         // check cocfficients
         linalg::matrix<Tm> vs(size,1);
         linalg::xcopy(size, coeffs.data(), vs.data());
         auto smat = rcanon_CIovlp(mps, space, vs, false);
         std::cout << "<CI|MPS[normalized]>=" << smat(0,0) 
            << " std::sqrt(<CI|CI>)=" << std::sqrt(psum) 
            << " diff=" << smat(0,0)-std::sqrt(psum)
            << std::endl; 
         if(std::abs(smat(0,0) - std::sqrt(psum)) > 1.e-10){
            tools::exit("error: deviation is large!");
         }
         auto t4 = tools::get_time();
         icomb = std::move(mps);
         std::cout << "----- TIMING FOR rcanon_loadconfs: "
            << tools::get_duration(t4-t0) << " S"
            << " T(parse/rmpsform/canon/check)="
            << tools::get_duration(t1-t0) << ","
            << tools::get_duration(t2-t1) << ","
            << tools::get_duration(t3-t2) << ","
            << tools::get_duration(t4-t3) << " -----"
            << std::endl;
      }

} // ctns

#endif
