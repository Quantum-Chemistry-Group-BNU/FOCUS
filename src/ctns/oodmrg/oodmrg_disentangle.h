#ifndef CTNS_OODMRG_DISENTANGLE_H
#define CTNS_OODMRG_DISENTANGLE_H

#include "../../io/input_ctns.h"
#include "../ctns_entropy.h"
#include "oodmrg_rotate.h"
#include <nlopt.hpp>

namespace ctns{

   template <typename Qm, typename Tm>
      double twodot_wavefun_entropy(const std::vector<double>& theta,
            const comb<Qm,Tm>& icomb,
            const std::vector<Tm>& v0,
            std::vector<Tm>& vr,
            qtensor4<Qm::ifabelian,Tm>& wf,
            qtensor2<Qm::ifabelian,Tm>& rot,
            std::vector<qtensor2<Qm::ifabelian,Tm>>& wfs2,
            const bool& forward,
            const int& dmax,
            const double& alpha,
            double& dwt,
            int& deff,
            int& ncall,
            const bool& debug){
         ncall += 1;
         // rotate wavefunction
         assert(theta.size() == 1);
         twodot_rotate<Qm::ifabelian,Tm>(v0, vr, wf, theta[0]);
         // decimation
         const bool iftrunc = true;
         const double rdm_svd = 1.5;
         const int alg_decim = 0;
         std::string fname;
         std::vector<double> sigs2full; 
         // see sweep_twodot_decim.h
         wf.from_array(vr.data());
         if(forward){
            wfs2[0] = wf.merge_lc1_c2r();
            decimation_row(icomb, wf.info.qrow, wf.info.qmid, // lc1=(row,mid) 
                  iftrunc, dmax, rdm_svd, alg_decim,
                  wfs2, sigs2full, rot, dwt, deff, fname,
                  debug);
         }else{
            wfs2[0] = wf.merge_lc1_c2r().P();
            decimation_row(icomb, wf.info.qver, wf.info.qcol, // c2r=(ver,col)
                  iftrunc, dmax, rdm_svd, alg_decim,
                  wfs2, sigs2full, rot, dwt, deff, fname,
                  debug);
            rot = rot.P(); 
         }
         double entropy = renyi_entropy(sigs2full, alpha);
         return entropy;
      }

   using optfun = std::function<double(const std::vector<double>&)>;

   // interface to nlopt
   inline double nlopt_vfun_entropy(const std::vector<double>& theta, 
         std::vector<double>& grad, 
         void* my_func_data){
      optfun* fun = (optfun*)my_func_data;
      double result = (*fun)(theta);
      return result;
   }

   template <typename Qm, typename Tm>
      double reduce_entropy_single(comb<Qm,Tm>& icomb,
            linalg::matrix<Tm>& urot,
            const std::string scheme,
            const int dmax,
            const input::params_oodmrg& ooparams,
            std::vector<int> gates={}){
         const bool debug_check = false;
         const bool ifab = Qm::ifabelian;
         const int& iprt = ooparams.iprt;
         const double& alpha = ooparams.alpha;
         const double& thrdloc = ooparams.thrdloc;
         const int& nptloc = ooparams.nptloc;
         if(iprt > 0){
            std::cout << "reduce_entropy_single"
               << " ifab=" << ifab
               << " scheme=" << scheme
               << " dmax=" << dmax
               << " alpha=" << alpha
               << " thrdloc=" << thrdloc
               << " nptloc=" << nptloc
               << std::endl;
         }
         auto t0 = tools::get_time();

         // initialization: generate cpsi
         const int iroot = 0;
         const bool singlet = true;
         init_cpsi_dot0(icomb, iroot, singlet);

         // generate sweep sequence
         const int nroots = 1;
         const auto& rindex = icomb.topo.rindex;
         auto sweep_seq = icomb.topo.get_mps_sweeps();
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
            if(debug_check){
               const int dots = 2;
               std::cout << "\nibond=" << ibond << "/seqsize=" << sweep_seq.size()
                  << " dots=" << dots << " dbond=" << dbond
                  << std::endl;
            }

            // construct twodot wavefunction: wf4
            const auto sym_state = icomb.get_qsym_state();
            auto wfsym = get_qsym_state(Qm::isym, sym_state.ne(),
                  (ifab? sym_state.tm() : sym_state.ts()), singlet);
            const auto& site0 = dbond.forward? icomb.cpsi[0] : icomb.sites[rindex.at(dbond.p0)];
            const auto& site1 = dbond.forward? icomb.sites[rindex.at(dbond.p1)] : icomb.cpsi[0];
            const auto& ql  = site0.info.qrow;
            const auto& qr  = site1.info.qcol;
            const auto& qc1 = site0.info.qmid;
            const auto& qc2 = site1.info.qmid;
            qtensor4<ifab,Tm> wf(wfsym, ql, qr, qc1, qc2);
            size_t ndim = wf.size();
            if(debug_check){
               std::cout << "wf4(diml,dimr,dimc1,dimc2)=(" 
                  << wf.info.qrow.get_dimAll() << ","
                  << wf.info.qcol.get_dimAll() << ","
                  << wf.info.qmid.get_dimAll() << ","
                  << wf.info.qver.get_dimAll() << ")"
                  << " nnz=" << ndim << ":"
                  << tools::sizeMB<Tm>(ndim) << "MB"
                  << std::endl;
               wf.print("wf4");
            }
            if(ndim == 0){
               std::cout << "error: symmetry is inconsistent as ndim=0" << std::endl;
               std::cout << "ibond=" << ibond << std::endl;
               wf.print("wf4");
               exit(1);
            }

            // generate wavefunction
            std::vector<Tm> v0;
            twodot_guess_v0(icomb, dbond, ndim, nroots, wf, v0, debug_check);
            double norm = linalg::xnrm2(v0.size(), v0.data());
            linalg::xscal(v0.size(), 1.0/norm, v0.data());
            assert(v0.size() == ndim*nroots);

            //---------------------------------------------------------
            // reduce entanglement via orbital rotation
            qtensor2<ifab,Tm> rot;
            std::vector<qtensor2<ifab,Tm>> wfs2(nroots);
            std::vector<Tm> vr(ndim);
            double dwt;
            int deff;
            int ncall = 0;
            bool debug = false;
            optfun fun;
            using std::placeholders::_1;
            fun = bind(&ctns::twodot_wavefun_entropy<Qm,Tm>, _1, std::cref(icomb), 
                  std::cref(v0), std::ref(vr), std::ref(wf), std::ref(rot), std::ref(wfs2),
                  std::cref(dbond.forward), std::cref(dmax), std::cref(alpha),
                  std::ref(dwt), std::ref(deff), std::ref(ncall), std::cref(debug)); 

            // start optimization
            const double pi = 4.0*std::atan(1.0);
            int npt;
            std::vector<double> anglst, funlst;
            std::vector<double> x(1);
            double fmin;
            if(scheme == "opt"){
            
               // repare a good initial guess by scanning 
               npt = nptloc;
               anglst.resize(npt);
               funlst.resize(npt);
               for(int i=0; i<npt; i++){
                  std::vector<double> theta = {pi*i/(npt-1)};
                  anglst[i] = theta[0];
                  funlst[i] = fun(theta); 
               }
               auto index = tools::sort_index(funlst); 

               // nlopt: optimize
               nlopt::opt opt(nlopt::LN_BOBYQA, 1);
               std::vector<double> lb = {0};
               std::vector<double> ub = {pi};
               opt.set_lower_bounds(lb);
               opt.set_upper_bounds(ub);
               opt.set_xtol_rel(thrdloc);
               opt.set_ftol_rel(thrdloc);
               opt.set_min_objective(nlopt_vfun_entropy, &fun);
               try{
                  // initial guess
                  x[0] = {index[0]*pi/(npt-1)};
                  nlopt::result result = opt.optimize(x, fmin);
               }
               catch(std::exception &e) {
                  std::cout << "nlopt failed: " << e.what() << std::endl;
               }

            // randomly apply SWAP gates
            }else if(scheme == "randomswap"){
           
               npt = 1;
               anglst.resize(npt);
               funlst.resize(npt);
               std::vector<double> theta = {0};
               anglst[0] = theta[0];
               funlst[0] = fun(theta);
               if(dbond.forward){
                  std::uniform_int_distribution<> dist(0,1);
                  x[0] = pi/2.0*dist(tools::generator);
               }else{
                  x[0] = 0.0;
               }
           
            // randomly apply rotation gates
            }else if(scheme == "random"){
           
               npt = 1;
               anglst.resize(npt);
               funlst.resize(npt);
               std::vector<double> theta = {0};
               anglst[0] = theta[0];
               funlst[0] = fun(theta);
               if(dbond.forward){
                  std::uniform_real_distribution<double> dist(0,1);
                  if(gates.size() == 0){
                     x[0] = 2*pi*dist(tools::generator);
                  }else{
                     auto result = std::find(gates.begin(), gates.end(), dbond.p0.first);
                     bool ifexist = !(result == gates.end());
                     if(ifexist) x[0] = pi/2; //2*pi*dist(tools::generator);                     
                  }
               }else{
                  x[0] = 0.0;
               }
 
            }else{
               std::cout << "error: no such scheme=" << scheme << std::endl;
               exit(1); 
            }

            // re-evaluate the function to output {vr, rot} correctly. 
            fmin = twodot_wavefun_entropy(x, icomb, v0, vr, wf, rot, wfs2,
                  dbond.forward, dmax, alpha, dwt, deff, ncall, debug_check);
            maxdwt = std::max(maxdwt,dwt);
            if(iprt > 1){
               std::cout << " i=" << ibond
                  << " (" << dbond.p0.first << ","
                  << dbond.p1.first << ")[" 
                  << dbond.forward << "]"
                  << std::scientific << std::setprecision(3)
                  << " f0=" << funlst[0]
                  << " fx=" << fmin 
                  << " diff=" << fmin-funlst[0]
                  << " x=" << x[0]
                  << " deff=" << deff 
                  << " dwt=" << dwt
                  << " ncall=" << ncall 
                  << std::endl;
            }
            
            // update urot
            double theta = x[0];
            givens_rotation(urot, dbond.p0.first, dbond.p1.first, theta);

            // save the current site
            const auto p = dbond.get_current();
            const auto& pdx = rindex.at(p); 
            if(superblock == "lc1"){
               icomb.sites[pdx] = rot.split_lc(wf.info.qrow, wf.info.qmid);
            }else if(superblock == "c2r"){
               icomb.sites[pdx] = rot.split_cr(wf.info.qver, wf.info.qcol);
            }else{
               std::cout << "error: no such option for superblock=" << superblock << std::endl;
               exit(1);
            }
            //---------------------------------------------------------

            // propagate to the next site
            linalg::matrix<Tm> vsol(ndim, nroots, vr.data());
            twodot_guess_psi(superblock, icomb, dbond, vsol, wf, rot);
            vsol.clear();
         } // ibond

         // compute rwfuns for the next call of reduce_entropy_single
         const double rdm_svd = 1.5;
         std::string fname;
         sweep_final_CR2cRR(icomb, rdm_svd, fname, debug_check);
        
         if(iprt > 0){
            auto t1 = tools::get_time();
            tools::timing("ctns::reduce_entropy_single", t0, t1);
         } 
         return maxdwt;
      }

} // ctns

#endif
