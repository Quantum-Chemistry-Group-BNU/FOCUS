#ifndef CTNS_OODMRG_DISENTANGLE_H
#define CTNS_OODMRG_DISENTANGLE_H

#include "nelder_mead.h"
#include "oodmrg_entropy.h"
#include "oodmrg_rotate.h"
#include <nlopt.hpp>

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
         std::vector<Tm> ui(norb,0), uj(norb,0);
         double c = std::cos(theta);
         double s = std::sin(theta);
         linalg::xaxpy(norb,  c, urot.col(i), ui.data());
         linalg::xaxpy(norb,  s, urot.col(j), ui.data());
         linalg::xaxpy(norb, -s, urot.col(i), uj.data());
         linalg::xaxpy(norb,  c, urot.col(j), uj.data());
         linalg::xcopy(norb, ui.data(), urot.col(i));
         linalg::xcopy(norb, uj.data(), urot.col(j));
      }

   template <typename Qm, typename Tm>
      double twodot_wavefun_entropy(const std::array<double,1> theta_array,
            const comb<Qm,Tm>& icomb,
            const std::vector<Tm>& v0,
            std::vector<Tm>& vr,
            qtensor4<Qm::ifabelian,Tm>& wf,
            qtensor2<Qm::ifabelian,Tm>& rot,
            std::vector<qtensor2<Qm::ifabelian,Tm>>& wfs2,
            const bool& forward,
            const int& dmax,
            const double& alpha,
            int& counter){
         counter += 1;
         const bool debug = false;
         // rotate wavefunction
         twodot_rotate<Qm::ifabelian,Tm>(v0, vr, wf, theta_array[0]);
         // decimation
         const bool iftrunc = true;
         const double rdm_svd = 1.5;
         const int alg_decim = 0;
         std::string fname;
         std::vector<double> sigs2full; 
         double dwt;
         int deff;
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
         // entropy
         double entropy = renyi_entropy(sigs2full, alpha);
         return entropy;
      }

   using optfun = std::function<double(const std::array<double,1>)>;

   inline double nlopt_vfun_entropy(const std::vector<double> &theta, std::vector<double> &grad, void *my_func_data){
      optfun* fun = (optfun*)my_func_data;
      std::array<double,1> theta_array = {theta[0]};
      double result = (*fun)(theta_array);
      return result;
   }

   template <typename Qm, typename Tm>
      void reduce_entropy_single(comb<Qm,Tm>& icomb,
            linalg::matrix<Tm>& urot,
            const int dmax,
            const double alpha,
            const bool debug){
         if(debug){
            std::cout << "reduce_entropy_single dmax=" << dmax
               << " alpha=" << alpha 
               << std::endl;
         }
         const bool debug_check = false;

         auto svalues = get_schmidt_values(icomb);
         for(int i=0; i<svalues.size(); i++){
            std::cout << "i=" << i 
               << " sval=" << renyi_entropy(svalues[i], alpha)
               << std::endl;
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
         icomb.cpsi.resize(nroots);
         for(int iroot=0; iroot<nroots; iroot++){
            // qt2(1,r): ->-*->-
            auto qt2 = contract_qt2_qt2(wf2,icomb.rwfuns[iroot]);
            // qt2(1,r)*site0(r,r0,n0) = qt3(1,r0,n0)[CRcouple]
            auto qt3 = contract_qt3_qt2("l",site0,qt2);
            // recouple to qt3(1,r0,n0)[LCcouple]
            icomb.cpsi[iroot] = qt3.recouple_lc();
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
            qtensor4<Qm::ifabelian,Tm> wf;
            if(dbond.forward){
               const auto& site0 = icomb.cpsi[0];
               const auto& site1 = icomb.sites[rindex.at(dbond.p1)];
               const auto& ql  = site0.info.qrow;
               const auto& qr  = site1.info.qcol;
               const auto& qc1 = site0.info.qmid;
               const auto& qc2 = site1.info.qmid;
               wf.init(sym_state, ql, qr, qc1, qc2);
            }else{
               const auto& site0 = icomb.sites[rindex.at(dbond.p0)];
               const auto& site1 = icomb.cpsi[0];
               const auto& ql  = site0.info.qrow;
               const auto& qr  = site1.info.qcol;
               const auto& qc1 = site0.info.qmid;
               const auto& qc2 = site1.info.qmid;
               wf.init(sym_state, ql, qr, qc1, qc2);
            }
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
               exit(1);
            }

            // generate wavefunction;
            std::vector<Tm> v0;
            twodot_guess_v0(icomb, dbond, ndim, nroots, wf, v0, debug_check);
            assert(v0.size() == ndim*nroots);

            //------------------------------------------
            // reduce entanglement via orbital rotation
            //------------------------------------------
            std::vector<Tm> vr(ndim);
            qtensor2<Qm::ifabelian,Tm> rot;
            std::vector<qtensor2<Qm::ifabelian,Tm>> wfs2(nroots);
            // interface for optimizer
            std::function<double(const std::array<double,1>)> fun;
            using std::placeholders::_1;
            int counter;
            fun = bind(&ctns::twodot_wavefun_entropy<Qm,Tm>, _1, std::cref(icomb), 
                  std::cref(v0), std::ref(vr), std::ref(wf), std::ref(rot), std::ref(wfs2),
                  std::cref(dbond.forward), std::cref(dmax), std::cref(alpha),
                  std::ref(counter)); 
            
            // prepare a good initial guess by scanning 
            counter = 0;
            const int npt = 5;
            const double pi = 4.0*std::atan(1.0);
            std::vector<double> funlst(npt);
            for(int i=0; i<npt; i++){
               std::array<double,1> theta = {pi*i/(npt-1)};
               funlst[i] = fun(theta); 
            }
            auto index = tools::sort_index(funlst); 
            tools::print_vector(funlst,"funlst");
            // optimize
            std::array<double,1> theta0 = {index[0]*pi/(npt-1)};
            std::array<double,1> step = {0.1};
            nelder_mead_result<double,1> result = nelder_mead<double,1>(
                  fun, theta0, 1.e-10, step);
            if(result.ifault != 0){
               std::cout << "warning: minimization fails in optimizer!" << std::endl;
               exit(1);
            }
            std::cout << "lzd thetamin=" << result.xmin[0] 
               << " ynewlo=" << result.ynewlo
               << " icount=" << result.icount
               << " ifault=" << result.ifault 
               << std::endl;

            counter = 0;
            
            //nlopt::opt opt(nlopt::LN_COBYLA, 1);
            //nlopt::opt opt(nlopt::LN_BOBYQA, 1);
            //nlopt::opt opt(nlopt::LN_NELDERMEAD, 1);
            nlopt::opt opt(nlopt::LN_SBPLX, 1);
            
            std::vector<double> lb = {0};
            std::vector<double> ub = {pi};
            opt.set_lower_bounds(lb);
            opt.set_upper_bounds(ub);
            opt.set_xtol_rel(1.e-8);
            opt.set_ftol_rel(1.e-8);
            std::vector<double> x = {index[0]*pi/(npt-1)};
            opt.set_min_objective(nlopt_vfun_entropy, &fun);
            double minf;
            try{
               nlopt::result result = opt.optimize(x, minf);
               std::cout << "counter=" << counter 
                  << " found minimum at f(" << x[0] << ") = "
                  << std::setprecision(10) << minf << std::endl;
            }
            catch(std::exception &e) {
               std::cout << "nlopt failed: " << e.what() << std::endl;
            }

            // re-evaluate the function to output {vr, rot} correctly. 
            fun({result.xmin[0]});
            
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
            
            // propagate to the next site
            linalg::matrix<Tm> vsol(ndim,nroots,vr.data());
            twodot_guess_psi(superblock, icomb, dbond, vsol, wf, rot);
            vsol.clear();

            // update u
            double theta = 0.0; //result.xmin[0];
            givens_rotatation(urot, dbond.p0.first, dbond.p1.first, theta);

         } // ibond
         exit(1);

         // compute rwfuns for the next call of reduce_entropy_single
         const double rdm_svd = 1.5;
         std::string fname;
         sweep_final_CR2cRR(icomb, rdm_svd, fname, debug);
      }

   template <typename Qm, typename Tm>
      void reduce_entropy_multi(comb<Qm,Tm>& icomb,
            linalg::matrix<Tm>& urot,
            const int dmax,
            const int microiter,
            const double alpha,
            const bool debug){
         if(debug){
            std::cout << "reduce_entropy_multi:" 
               << " dmax=" << dmax
               << " microiter=" << microiter
               << " alpha=" << alpha
               << std::endl;
         }
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
