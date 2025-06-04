#ifndef CTNS_ENTROPY_H
#define CTNS_ENTROPY_H

#include "sweep_twodot_guess.h"
#include "ctns_rcanon.h"

namespace ctns{

   // Schmidt decomposition for MPS[iroot]   
   template <typename Qm, typename Tm>
      std::vector<std::vector<double>> get_schmidt_values(const comb<Qm,Tm>& icomb,
            const int iroot=0,
            const bool singlet=true,
            const bool debug=false){
         const bool ifab = Qm::ifabelian;

         // currently only support MPS
         assert(icomb.topo.ifmps); 

         // generate cpsi for dot0 to init the sweep
         auto icomb_tmp = icomb;
         init_cpsi_dot0(icomb_tmp, iroot, singlet);

         // holder for schdmidt values
         const int nphysical = icomb_tmp.get_nphysical();
         std::vector<std::vector<double>> svalues(nphysical-1);
         
         // generate forward sweep sequence
         const int nroots = 1;
         const auto& rindex = icomb_tmp.topo.rindex;
         auto fsweep_seq = icomb_tmp.topo.get_mps_fsweeps();
         const int dmax = icomb_tmp.get_dmax();
         for(int ibond=0; ibond<fsweep_seq.size(); ibond++){
            const auto& dbond = fsweep_seq[ibond];
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
               std::cout << tools::line_separator << std::endl;
               icomb.topo.check_partition(dots, dbond, debug, 0);
            }
            
            // construct twodot wavefunction: wf4
            const auto sym_state = icomb_tmp.get_qsym_state();
            auto wfsym = get_qsym_state(Qm::isym, sym_state.ne(),
                  (ifab? sym_state.tm() : sym_state.ts()), singlet);
            const auto& cpsi0 = icomb_tmp.cpsi[0];
            const auto& site1 = icomb_tmp.sites[rindex.at(std::make_pair(ibond+1,0))];
            const auto& ql  = cpsi0.info.qrow;
            const auto& qr  = site1.info.qcol;
            const auto& qc1 = cpsi0.info.qmid;
            const auto& qc2 = site1.info.qmid;
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
            
            //---------------------------------------------------------
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
            const int svd_iop = 3;
            const int alg_decim = 0; // serial version
            std::string fname;
            std::vector<double> sigs2full; 
            double dwt;
            int deff;
            decimation_row(icomb_tmp, wf.info.qrow, wf.info.qmid, 
                  iftrunc, dmax, rdm_svd, svd_iop, alg_decim,
                  wfs2, sigs2full, rot, dwt, deff, fname,
                  debug);
            svalues[ibond] = std::move(sigs2full);
            //---------------------------------------------------------
            
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
         if(std::abs(alpha-1.0) < 1.e-8){
            ssum -= pi*log(pi);
         }else{
            ssum += std::pow(pi,alpha);
         }
      }
      // Correct formula for Renyi entropy
      if(std::abs(alpha-1.0) > 1.e-8) ssum = 1.0/(1.0-alpha)*std::log(ssum);
      assert(std::abs(psum-1.0) < 1.e-8);
      return ssum;
   }

   // Warning: this function used in OO-DMRG,
   // which works only for iroot=0 & singlet=true
   template <typename Qm, typename Tm>
      double rcanon_entropysum(const comb<Qm,Tm>& icomb,
            const double alpha){
         const int iroot = 0;
         const bool singlet = true;
         auto svalues = get_schmidt_values(icomb, iroot, singlet);
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
            const std::string schmidt_file,
            const bool save_schdmit,
            const bool debug=false){
         std::cout << "\nctns::rcanon_schmidt:"
           << " iroot=" << iroot 
           << " fname=" << schmidt_file << ".txt"
           << " save_schmidt=" << save_schmidt
           << std::endl;
         auto t0 = tools::get_time();

         // compute the Schmidt decomposition
         const bool singlet = true; // use singlet embedding by default
         auto svalues = get_schmidt_values(icomb, iroot, singlet, debug);

         // compute the entropy
         std::cout << "von Neumann / Renyi[0.5] entropies across all bonds:" << std::endl;
         double s_sum = 0.0, sh_sum = 0.0;
         double s_max = -1.0, sh_max = -1.0;
         for(int i=0; i<svalues.size(); i++){
            //tools::print_vector(svalues[i], "sval"+std::to_string(i), 10);
            auto s_val = renyi_entropy(svalues[i], 1.0); // von Neumann entropy
            auto sh_val = renyi_entropy(svalues[i], 0.5); // renyi entropy [0.5]
            s_sum += s_val;
            sh_sum += sh_val;
            s_max = std::max(s_max,s_val);
            sh_max = std::max(sh_max,sh_val); 
            std::cout << " ibond=" << i 
               << " SvN=" << std::scientific << std::setprecision(3) << s_val
               << " Sr=" << sh_val 
               << std::endl;
         }
         std::cout << "SvN[sum]=" << s_sum 
            << " Sr[sum]=" << sh_sum << std::endl;
         std::cout << "SvN[max]=" << s_max
            << " Sr[max]=" << sh_max << std::endl; 

         // save into file
         if(save_schmidt){
            std::cout << "save schdmidt values into file=" << schmidt_file << std::endl;
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
         }

         auto t1 = tools::get_time();
         tools::timing("ctns::rcanon_schmidt", t0, t1);
      }

} // ctns

#endif
