#ifndef CTNS_OODMRG_DISENTANGLE_H
#define CTNS_OODMRG_DISENTANGLE_H

#include "oodmrg_entropy.h"

namespace ctns{

   //        [ c   -s ]
   // (ui,uj)[        ] = (ui*c+uj*s,-ui*s+uj*c)
   //        [ s    c ]
   template <typename Tm>   
      void givens_rotatation(linalg::matrix<Tm>& urot, 
            const int i,
            const int j,
            const double theta){
         int norb = urot.rows();
         std::cout << "norb=" << norb 
            << " i=" << i 
            << " j=" << j
            << std::endl;
         std::vector<Tm> ui(norb,0), uj(norb,0);
         urot.print("urot");
         double c = std::cos(theta);
         double s = std::sin(theta);
         linalg::xaxpy(norb,  c, urot.col(i), ui.data());
         linalg::xaxpy(norb,  s, urot.col(j), ui.data());
         linalg::xaxpy(norb, -s, urot.col(i), uj.data());
         linalg::xaxpy(norb,  c, urot.col(j), uj.data());
         linalg::xcopy(norb, ui.data(), urot.col(i));
         linalg::xcopy(norb, uj.data(), urot.col(j));
         urot.print("urot");
      }

   template <typename Qm, typename Tm>
      void reduce_entropy_single(comb<Qm,Tm>& icomb,
            linalg::matrix<Tm>& urot,
            const int dmax,
            const double alpha,
            const bool debug){
         if(debug){
            std::cout << "reduce_entropy_single" << std::endl;
         }

         // generate sweep sequence
         auto sweep_seq = icomb.topo.get_mps_sweeps();
         const int nroots = 1;
         const int dots = 2;

         // initialization: generate cpsi
         const bool singlet = false;
         auto sym_state = icomb.get_qsym_state();
         const auto& rindex = icomb.topo.rindex;
         const auto& site0 = icomb.sites[rindex.at(std::make_pair(0,0))]; // will be updated
         const auto wf2 = get_boundary_coupling<Qm,Tm>(sym_state, singlet); // env*C[0]
         auto icomb_tmp = icomb;
         icomb_tmp.cpsi.resize(nroots);
         for(int iroot=0; iroot<nroots; iroot++){
            // qt2(1,r): ->-*->-
            auto qt2 = contract_qt2_qt2(wf2,icomb.rwfuns[iroot]);
            // qt2(1,r)*site0(r,r0,n0) = qt3(1,r0,n0)[CRcouple]
            auto qt3 = contract_qt3_qt2("l",site0,qt2);
            // recouple to qt3(1,r0,n0)[LCcouple]
            icomb_tmp.cpsi[iroot] = qt3.recouple_lc();
            icomb_tmp.cpsi[iroot].print("cwf");
         } // iroot

         // start forward sweep
         const int nphysical = icomb.get_nphysical();
         double maxdwt = -1.0;
         for(int ibond=0; ibond<sweep_seq.size(); ibond++){
            const auto& dbond = sweep_seq[ibond];
            auto tp0 = icomb.topo.get_type(dbond.p0);
            auto tp1 = icomb.topo.get_type(dbond.p1);
            std::string superblock;
            if(dbond.forward){
               superblock = dbond.is_cturn()? "lr" : "lc1";
            }else{
               superblock = dbond.is_cturn()? "c1c2" : "c2r";
            }
            if(debug){
               std::cout << "\nibond=" << ibond << "/seqsize=" << sweep_seq.size()
                  << " dots=" << dots << " dbond=" << dbond
                  << std::endl;
            }

            // construct twodot wavefunction: wf4
            const auto& cpsi0 = icomb_tmp.cpsi[0];
            const auto& site1 = icomb_tmp.sites[rindex.at(std::make_pair(ibond+1,0))];
            const auto& ql  = cpsi0.info.qrow;
            const auto& qr  = site1.info.qcol;
            const auto& qc1 = cpsi0.info.qmid;
            const auto& qc2 = site1.info.qmid;
            qtensor4<Qm::ifabelian,Tm> wf(sym_state, ql, qr, qc1, qc2);
            size_t ndim = wf.size();
            if(debug){
               std::cout << "wf4(diml,dimr,dimc1,dimc2)=(" 
                  << ql.get_dimAll() << ","
                  << qr.get_dimAll() << ","
                  << qc1.get_dimAll() << ","
                  << qc2.get_dimAll() << ")"
                  << " nnz=" << ndim << ":"
                  << tools::sizeMB<Tm>(ndim) << "MB"
                  << std::endl;
               wf.print("wf4");
            }
            if(ndim == 0){
               std::cout << "error: symmetry is inconsistent as ndim=0" << std::endl;
               exit(1);
            }
            
            // generate wavefunction;
            std::vector<Tm> v0;
            twodot_guess_v0(icomb_tmp, dbond, ndim, nroots, wf, v0);
            assert(v0.size() == ndim*nroots);
            linalg::matrix<Tm> vsol(ndim,nroots,v0.data());
            
            //------------------------------------------
            // reduce entanglement via orbital rotation
            //------------------------------------------

            // apply u to wf4

            // perform decimation
            qtensor2<Qm::ifabelian,Tm> rot;
            std::vector<qtensor2<Qm::ifabelian,Tm>> wfs2(nroots);
            for(int i=0; i<nroots; i++){
               wf.from_array(vsol.col(i));
               auto wf2 = wf.merge_lc1_c2r();
               wfs2[i] = std::move(wf2);
            }
            const bool iftrunc = true;
            const double rdm_svd = 1.5;
            const int alg_decim = 0;
            std::string fname;
            std::vector<double> sigs2full; 
            double dwt;
            int deff;
            decimation_row(icomb_tmp, wf.info.qrow, wf.info.qmid, 
                  iftrunc, dmax, rdm_svd, alg_decim,
                  wfs2, sigs2full, rot, dwt, deff, fname,
                  debug);


            // propagate to the next site
            twodot_guess_psi(superblock, icomb_tmp, dbond, vsol, wf, rot);
            vsol.clear();

            // update u
            double theta = 3.1415926535897932384626/2;
            givens_rotatation(urot, dbond.p0.first, dbond.p1.first, -theta);
            exit(1);

         } // ibond
      }

   template <typename Qm, typename Tm>
      void reduce_entropy_multi(comb<Qm,Tm>& icomb,
            linalg::matrix<Tm>& urot,
            const int dmax,
            const int microiter,
            const double alpha,
            const bool debug){
         const double thresh = 1.e-8;
         double totaldiff = 0.0;
         bool ifconv = false;
         double s_old = sum_of_entropy(icomb, alpha);
         std::cout << "s_old=" << s_old << std::endl;
         for(int imicro=0; imicro<microiter; imicro++){
            if(debug){
               std::cout << "\n=== imicro=" << imicro << " ===" << std::endl;
            }
            reduce_entropy_single(icomb, urot, dmax, alpha, debug);
            exit(1);
            double s_new = sum_of_entropy(icomb, alpha);
            double diff = s_new - s_old;
            if(debug){
               std::cout << "imicro=" << imicro << " s_old=" << s_old
                  << " s_new=" << s_new << " diff=" << diff
                  << std::endl;
            }
            if(std::abs(diff) < thresh){
               if(debug) std::cout << "reduce_entropy_multi converges!" << std::endl;
               ifconv = true; 
               break;          
            }else{
               s_old = s_new;
            }
         } // imicro
         if(not ifconv){
            std::cout << "Warning: reduce_entropy_multi does not converge!" << std::endl;
         }
      }

} // ctns

#endif
