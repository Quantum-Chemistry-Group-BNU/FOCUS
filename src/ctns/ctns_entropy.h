#ifndef CTNS_ENTROPY_H
#define CTNS_ENTROPY_H

#include "sweep_twodot_guess.h"

namespace ctns{
        
   // r*R0*R1*R2 to C0*R1*R2 
   template <typename Qm, typename Tm>
      void init_cpsi_dot0(comb<Qm,Tm>& icomb,
            const int iroot=-1,
            const bool singlet=true){
         const auto& rindex = icomb.topo.rindex;
         const auto& site0 = icomb.sites[rindex.at(std::make_pair(0,0))]; // will be updated
         const auto sym_state = icomb.get_qsym_state();
         const auto wf2 = get_boundary_coupling<Qm,Tm>(sym_state, singlet); // env*C[0]
         int nroots = (iroot==-1)? icomb.get_nroots() : 1;
         icomb.cpsi.resize(nroots);
         if(iroot == -1){
            for(int iroot=0; iroot<nroots; iroot++){
               // qt2(1,r): ->-*->-
               auto qt2 = contract_qt2_qt2(wf2,icomb.rwfuns[iroot]);
               // qt2(1,r)*site0(r,r0,n0) = qt3(1,r0,n0)[CRcouple]
               auto qt3 = contract_qt3_qt2("l",site0,qt2);
               // recouple to qt3(1,r0,n0)[LCcouple]
               icomb.cpsi[iroot] = qt3.recouple_lc();
            } // iroot
         }else{
            // get an MPS for a single state
            if(iroot != 0) icomb.rwfuns[0] = std::move(icomb.rwfuns[iroot]);
            icomb.rwfuns.resize(1);
            // qt2(1,r): ->-*->-
            auto qt2 = contract_qt2_qt2(wf2,icomb.rwfuns[0]);
            // qt2(1,r)*site0(r,r0,n0) = qt3(1,r0,n0)[CRcouple]
            auto qt3 = contract_qt3_qt2("l",site0,qt2);
            // recouple to qt3(1,r0,n0)[LCcouple]
            icomb.cpsi[0] = qt3.recouple_lc();
         }
      }

   template <typename Qm, typename Tm>
      std::vector<std::vector<double>> get_schmidt_values(const comb<Qm,Tm>& icomb,
            const int iroot=0,
            const bool singlet=true,
            const bool debug=false){
         const bool ifab = Qm::ifabelian;

         // generate cpsi for dot0 to init the sweep
         auto icomb_tmp = icomb;
         init_cpsi_dot0(icomb_tmp, iroot, singlet);

         // holder for schdmidt values
         const int nphysical = icomb_tmp.get_nphysical();
         std::vector<std::vector<double>> svalues(nphysical-1);
         
         // generate forward sweep sequence
         const int nroots = 1;
         const auto& rindex = icomb_tmp.topo.rindex;
         const int dmax = icomb_tmp.get_dmax();
         auto fsweep_seq = icomb_tmp.topo.get_mps_fsweeps();
         for(int ibond=0; ibond<fsweep_seq.size(); ibond++){
            const auto& dbond = fsweep_seq[ibond];
            auto tp0 = icomb_tmp.topo.get_type(dbond.p0);
            auto tp1 = icomb_tmp.topo.get_type(dbond.p1);
            std::string superblock;
            if(dbond.forward){
               superblock = dbond.is_cturn()? "lr" : "lc1";
            }else{
               superblock = dbond.is_cturn()? "c1c2" : "c2r";
            }
            if(debug){
               const int dots = 2;
               std::cout << "\nibond=" << ibond << "/seqsize=" << fsweep_seq.size()
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
            const auto sym_state = icomb_tmp.get_qsym_state();
            auto wfsym = get_qsym_state(Qm::isym, sym_state.ne(),
                  (ifab? sym_state.tm() : sym_state.ts()), singlet);
            qtensor4<ifab,Tm> wf(wfsym, ql, qr, qc1, qc2);
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
            twodot_guess_v0(icomb_tmp, dbond, ndim, nroots, wf, v0, debug);
            assert(v0.size() == ndim*nroots);
            // perform decimation
            qtensor2<ifab,Tm> rot;
            std::vector<qtensor2<ifab,Tm>> wfs2(nroots);
            for(int i=0; i<nroots; i++){
               wf.from_array(&v0[i*ndim]);
               auto wf2 = wf.merge_lc1_c2r();
               wfs2[i] = std::move(wf2);
            }
            const bool iftrunc = true;
            const double rdm_svd = 1.5;
            const int alg_decim = 0; // serial version
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
            linalg::matrix<Tm> vsol(ndim, nroots, v0.data());
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

   // compute the Schmidt decomposition of an MPS wavefunction
   template <typename Qm, typename Tm>
      void rcanon_schmidt(const comb<Qm,Tm>& icomb, // initial comb wavefunction
            const int iroot,
            const std::string schmidt_file){
         std::cout << "\nctns::rcanon_schmidt:"
           << " iroot=" << iroot 
           << " fname=" << schmidt_file << ".txt"
           << std::endl;
         auto t0 = tools::get_time();

         // compute the Schmidt decomposition
         const bool debug = false;
         const bool singlet = true; // use singlet embedding by default
         auto svalues = get_schmidt_values(icomb, iroot, singlet, debug);

         // compute the entropy
         std::cout << "von Neumann entropies across all bonds:" << std::endl;
         double s_sum = 0.0;
         for(int i=0; i<svalues.size(); i++){
            auto s_val = renyi_entropy(svalues[i], 1.0); // von Neumann entropy
            s_sum += s_val; 
            std::cout << " ibond=" << i 
               << " SvN=" << std::setprecision(6) << s_val 
               << std::endl;
         }
         std::cout << "sum of all = " << s_sum << std::endl;

         // save into file
         std::cout << "save schdmidt values into file" << std::endl;
         std::ofstream file(schmidt_file+".txt");
         file << icomb.get_nphysical() << std::endl;
         file << std::scientific << std::setprecision(10);
         for(int i=0; i<svalues.size(); i++){
            for(int j=0; j<svalues[i].size(); j++){
               file << svalues[i][j] << " ";
            }
            file << std::endl;
         }
         file.close();

         auto t1 = tools::get_time();
         tools::timing("ctns::rcanon_schmidt", t0, t1);
      }

} // ctns

#endif
