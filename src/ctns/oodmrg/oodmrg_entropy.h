#ifndef CTNS_OODMRG_ENTROPY_H
#define CTNS_OODMRG_ENTROPY_H

#include "../sweep_twodot_guess.h"

namespace ctns{

   template <typename Qm, typename Tm>
      std::vector<std::vector<double>> get_schmidt_values(const comb<Qm,Tm>& icomb,
            const bool debug=true){

         // generate sweep sequence
         const int dmax = icomb.get_dmax();
         auto sweep_seq = icomb.topo.get_mps_fsweeps();
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
         std::vector<std::vector<double>> svalues(nphysical-1);
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
            // perform decimation
            qtensor2<Qm::ifabelian,Tm> rot;
            std::vector<qtensor2<Qm::ifabelian,Tm>> wfs2(nroots);
            for(int i=0; i<nroots; i++){
               wf.from_array(&v0[i*ndim]);
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
            svalues[ibond] = std::move(sigs2full);
            // propagate to the next site
            linalg::matrix<Tm> vsol(ndim,nroots,v0.data());
            twodot_guess_psi(superblock, icomb_tmp, dbond, vsol, wf, rot);
            vsol.clear();
         } // ibond
         return svalues;
      }

   inline double renyi_entropy(const std::vector<double>& p,
         const double alpha, 
         const double cutoff=1.e-100){
      double psum = 0.0, ssum = 0.0;
      for(const auto& pi : p){
         if(pi < cutoff) continue;
         psum += pi;
         if(abs(alpha-1.0) < 1.e-8){
            ssum -= pi*log(pi);
         }else{
            ssum += 1.0/(1.0-alpha)*std::pow(pi,alpha);
         }
      }
      assert(abs(psum-1.0) < 1.e-8);
      return ssum;
   }

   template <typename Qm, typename Tm>
      double sum_of_entropy(const comb<Qm,Tm>& icomb,
            const double alpha){
         const bool debug = true;
         auto svalues = get_schmidt_values(icomb);
         double s_sum = 0.0;
         for(int i=0; i<svalues.size(); i++){
            s_sum += renyi_entropy(svalues[i], alpha);
         }
         return s_sum;
      }

} // ctns

#endif
