#ifndef CTNS_INIT_H
#define CTNS_INIT_H

#ifdef _OPENMP
#include <omp.h>
#endif

#include "../io/input.h"
#include "../core/tools.h"
#include "../core/onspace.h"
#include "../ci/fci_util.h"
#include "ctns_comb.h"
#include "init_phys.h"
#include "init_bipart.h"
#include "oper_rbasis.h" // debug_oper_rbasis

const bool debug_init = true;
extern const bool debug_init;

namespace ctns{

   // initialize RCF from SCI wavefunctions
   template <typename Qm, typename Tm>
      void rcanon_init(comb<Qm,Tm>& icomb,
            const fock::onspace& space0,
            const linalg::matrix<Tm>& vs0,
            const input::schedule& schd){
         std::cout << "\nctns::rcanon_init qkind=" << qkind::get_name<Qm>() << std::endl;
         auto t0 = tools::get_time();

         // 0. truncate ci wavefunction if necessary
         fock::onspace space = space0;
         linalg::matrix<Tm> vs;
         int nroots_selected = schd.ctns.ciroots.size();
         if(nroots_selected == 0){
            vs = vs0;
         }else{
            int nroots = vs0.cols();
            size_t dim = space.size();
            vs.resize(dim, nroots_selected);
            for(int i=0; i<nroots_selected; i++){
               int idx = schd.ctns.ciroots[i];
               if(idx > nroots-1){
                  std::cout << "error: ciroots exceed nroots=" << nroots << std::endl;
                  tools::print_vector(schd.ctns.ciroots, "ciroots");
                  exit(1); 
               }
               linalg::xcopy(dim, vs0.col(idx), vs.col(i)); 
            }
         }
         fci::ci_truncate(space, vs, schd.ctns.maxdets);

         // ZL@20241020: check symmetry
         std::set<qsym> sym_sectors;
         for(int i=0; i<space.size(); i++){
            const auto& state = space[i];
            auto sym = get_qsym_onstate(Qm::isym, state);
            sym_sectors.insert(sym);
         }
         if(sym_sectors.size()>1){
            std::cout << "error: more than one symmetry sectors in CI space for Qm=" 
               << qkind::get_name<Qm>() << " : ";
            for(const auto& sym : sym_sectors){
               std::cout << sym << " ";
            }
            std::cout << std::endl;
            exit(1);
         }

         // 1. compute renormalized bases {|r>} from SCI wavefunctions
         init_rbases(icomb, space, vs, schd.ctns.rdm_svd, schd.ctns.svd_iop, schd.ctns.thresh_proj);

         // 2. build sites from rbases
         init_rsites(icomb, schd.ctns.thresh_ortho);

         // 3. compute wave functions at the start for right canonical form 
         init_rwfuns(icomb, space, vs, schd.ctns.thresh_ortho);

         // 4. canonicalization
         init_rcanon(icomb, space, vs, schd.ctns.thresh_ortho);

         auto t1 = tools::get_time();
         tools::timing("ctns::rcanon_init", t0, t1);
      }

   // compute renormalized bases {|r>} from SCI wavefunctions 
   template <typename Qm, typename Tm>
      void init_rbases(comb<Qm,Tm>& icomb,
            const fock::onspace& space,
            const linalg::matrix<Tm>& vs,
            const double rdm_svd,
            const int svd_iop,
            const double thresh_proj){
         std::cout << "\nctns::init_rbases" << std::scientific << std::setprecision(3) 
            << " rdm_svd=" << rdm_svd
            << " svd_iop=" << svd_iop
            << " thresh_proj=" << thresh_proj 
            << std::endl;
         auto t0 = tools::get_time();

         // loop over bond - parallelizable
         const auto& topo = icomb.topo;
         icomb.rbases.resize(topo.ntotal);
         std::vector<double> popBc(topo.ntotal,1.0);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
         for(int idx=0; idx<topo.ntotal; idx++){
            auto p = topo.rcoord[idx];
            int i = p.first, j = p.second;
            const auto& node = topo.nodes[i][j];
            if(debug_init){
               std::cout << "\nidx=" << idx << " node=" << p << " type=" << node.type; 
               std::cout << " rsupport=";
               for(int k : node.rsupport) std::cout << k << " ";
               std::cout << std::endl;
            }
            renorm_basis<Tm> rbasis;
            // for boundary site, we simply choose to use identity
            if(node.type == 0 && p != std::make_pair(0,0)){

               rbasis = get_rbasis_phys<Tm>(Qm::isym);

               // Generate {|r>} at the internal nodes
            }else{

               // 1. generate 1D ordering
               const auto& rsupp = node.rsupport; 
               auto order = node.lsupport;
               int bpos = order.size(); // must be put here to account bipartition position
               copy(rsupp.begin(), rsupp.end(), back_inserter(order));

               // 2. transform SCI coefficient to this ordering
               fock::onspace space2;
               linalg::matrix<Tm> vs2;
               fock::transform_coeff(space, vs, order, space2, vs2); 

               // 3. bipartition of space and compute renormalized states [time-consuming part!]
               popBc[idx] = right_projection<Qm,Tm>(rbasis, 2*bpos, space2, vs2, 
                     thresh_proj, rdm_svd, svd_iop, debug_init);

            } // node type
//#ifdef _OPENMP
//#pragma omp critical
//#endif
            icomb.rbases[idx] = rbasis; // std::move(rbasis) may cause memory error sometimes
            auto shape = get_shape(icomb.rbases[idx]); 
            assert(shape.first > 0 && shape.second > 0);
         } // idx

         // print information for all renormalized basis {|r>} at each bond
         if(debug_init){
            std::cout << "\ngenerated rbases with thresh_proj=" << thresh_proj << std::endl;
            int Dmax = 0;
            for(int idx=0; idx<topo.ntotal; idx++){
               auto p = topo.rcoord[idx];
               int i = p.first, j = p.second;
               // shape can be different from dim(rspace) if associated weight is zero!
               auto shape = get_shape(icomb.rbases[idx]);
               std::cout << " idx=" << idx << " node=" << p << " type=" << topo.get_type(p)
                  << " shape=" << shape.first << "," << shape.second
                  << " 1-popBc=" << std::scientific << std::setprecision(3) << 1.0-popBc[idx]
                  << std::endl;
               Dmax = std::max(Dmax,shape.second);
            } // idx
            std::cout << "maximum bond dimension = " << Dmax << std::endl;
         }

         auto t1 = tools::get_time();
         tools::timing("ctns::init_rbases", t0, t1);
      }

   // build site tensor from {|r>} bases
   template <typename Qm, typename Tm>
      void init_rsites(comb<Qm,Tm>& icomb,
            const double thresh_ortho){
         std::cout << "\nctns::init_rsites qkind=" << qkind::get_name<Qm>() 
            << " thresh_ortho=" << thresh_ortho
            << std::endl;
         auto t0 = tools::get_time();

         // loop over sites - parallelizable
         const auto& topo = icomb.topo;
         icomb.sites.resize(topo.ntotal);
         for(int idx=0; idx<topo.ntotal; idx++){
            auto p = topo.rcoord[idx];
            int i = p.first, j = p.second;
            const auto& node = topo.nodes[i][j];
            if(debug_init){ 
               std::cout << " idx=" << idx << " node=" << p << "[" << node.type << "]";   
            } 
            auto ti = tools::get_time();

            // type=0: end or leaves
            if(node.type == 0 && p != std::make_pair(0,0)){

               icomb.sites[idx] = get_right_bsite<Qm,Tm>();

            // physical/internal on backbone/branch
            }else{

               //  node.type == 3: internal site on backbone
               //    |u>(0)      
               //     |
               //  ---*---|r>(1)
               //
               //  node.type = 1/2: physical site on backbone/branch
               //     n            |u> 
               //     |             |
               //  ---*---|r>   n---*
               //                   |
               const auto& rbasis_l = icomb.rbases[idx];
               const auto& rbasis_c = (node.type==3)? icomb.rbases[topo.rindex.at(node.center)] : \
                                      get_rbasis_phys<Tm>(Qm::isym); 
               const auto& rbasis_r = icomb.rbases[topo.rindex.at(node.right)];
               auto qrow = get_qbond(rbasis_l); 
               auto qcol = get_qbond(rbasis_r);
               auto qmid = get_qbond(rbasis_c);
               stensor3<Tm> qt3(qsym(Qm::isym), qrow, qcol, qmid);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) collapse(3)
#endif
               for(int kl=0; kl<rbasis_l.size(); kl++){ // left
                  for(int kr=0; kr<rbasis_r.size(); kr++){ // right 
                     for(int kc=0; kc<rbasis_c.size(); kc++){ // upper 
                        auto blk = qt3(kl,kr,kc);
                        if(blk.empty()) continue;
                        //
                        // construct site R[c][lr] = <qc,qr|ql> 
                        // 			     = W*[c'c] W*[r'r] <D[c'],D[r']|D[l']> W[l',l]
                        //
                        auto Wc = rbasis_c[kc].coeff.H();
                        auto Wr = rbasis_r[kr].coeff.conj();
                        auto Wl = rbasis_l[kl].coeff;
                        // 
                        // 20210902: new implementation using hash table for constructing <D[c'],D[r']|D[l']>
                        //
                        const auto& rspace = rbasis_r[kr].space;
                        const auto& lspace = rbasis_l[kl].space;
                        int dimr = rspace.size(), diml = lspace.size(), dimlc = Wl.cols();
                        std::unordered_map<fock::onstate,size_t> helper;
                        for(int j=0; j<diml; j++){
                           helper[lspace[j]] = j;
                        }
                        for(int dc=0; dc<Wc.cols(); dc++){
                           auto state_c = rbasis_c[kc].space[dc];
                           //
                           // old implementation:
                           //
                           // // tmp1[c'][r'l'] = <D[c'],D[r']|[l']>
                           // auto tmp1 = fock::get_Bcouple<Tm>(state_c,rbasis_r[kr].space,rbasis_l[kl].space);
                           // // tmp2[c'][r'l] = tmp1[c'][r'l']Wl[l'l]
                           // auto tmp2 = linalg::xgemm("N","N",tmp1,Wl);
                           // 
                           // 20210902: new implementation using hash table & sparse matrix multiplication
                           // 	  given D[c'], tmp1(r',l')=<D[c'],D[r']|[l']> is extremely sparse!
                           //
                           linalg::matrix<Tm> tmp12(dimr,dimlc);
                           for(int i=0; i<dimr; i++){
                              auto state = state_c.join(rspace[i]);
                              auto search = helper.find(state);
                              if(search != helper.end()){
                                 int j = search->second;
                                 for(int k=0; k<dimlc; k++){
                                    tmp12(i,k) += Wl(j,k);
                                 } // k
                              } // j
                           } // i
                           // tmp3[c'](l,r)= Wr*[r'r]tmp2[c'][r'l] = tmp2^T*Wr.conj() 
                           auto tmp3 = linalg::xgemm("T","N",tmp12,Wr);
                           int N = tmp3.size();
                           // R[c][lr] = sum_c' Wc*[c'c]tmp3[c'][lr]
                           for(int ic=0; ic<Wc.rows(); ic++){
                              linalg::xaxpy(N, Wc(ic,dc), tmp3.data(), blk.get(ic).data());
                           } // ic
                        } // ibas
                     } // kc
                  } // kr
               } // kl
               icomb.sites[idx] = std::move(qt3); // save
            } // if

            auto tf = tools::get_time(); 
            if(debug_init){ 
               auto dt = tools::get_duration(tf-ti);
               auto ova = contract_qt3_qt3("cr", icomb.sites[idx], icomb.sites[idx]);
               double maxdiff = ova.check_identityMatrix(thresh_ortho, false);
               std::cout << " shape(l,r,c)=("
                  << icomb.sites[idx].info.qrow.get_dimAll() << ","
                  << icomb.sites[idx].info.qcol.get_dimAll() << ","
                  << icomb.sites[idx].info.qmid.get_dimAll() << ")"
                  << " maxdiff=" << std::scientific << maxdiff 
                  << " deviate=" << (maxdiff>thresh_ortho)
                  << " TIMING=" << dt << " S"
                  << std::endl;
               //if(maxdiff>thresh_ortho) tools::exit("error: deviate from identity matrix!");
            }
         } // idx

         auto t1 = tools::get_time();
         tools::timing("ctns::init_rsites", t0, t1);
      }

   // compute wave function at the start for right canonical form
   template <typename Qm, typename Tm>
      void init_rwfuns(comb<Qm,Tm>& icomb,
            const fock::onspace& space,
            const linalg::matrix<Tm>& vs,
            const double thresh_ortho){
         std::cout << "\nctns::init_rwfuns qkind=" << qkind::get_name<Qm>() 
            << " thresh_ortho=" << thresh_ortho 
            << std::endl;
         auto t0 = tools::get_time();

         // determine symmetry of rwfuns
         auto sym_state = get_qsym_onstate(Qm::isym, space[0]);
         // check symmetry: we assume all the dets are of the same symmetry!
         for(int i=0; i<space.size(); i++){
            auto sym = get_qsym_onstate(Qm::isym, space[i]);
            if(sym != sym_state){
               std::cout << "sym_state=" << sym_state 
                  << " det=" << space[i] << " sym=" << sym
                  << std::endl;
               tools::exit("error: symmetry is different in space!");
            }
         }
         // setup wavefunction: map vs2 to the correct position
         fock::onspace space2;
         linalg::matrix<Tm> vs2;
         const auto& order = icomb.topo.nodes[0][0].rsupport;
         fock::transform_coeff(space, vs, order, space2, vs2);
         //
         // Needs to locate position of space2[i] in rbasis[0].space
         // NOTE: for isym=1, ndet can be larger than space.size()
         //       due to the possible reorder of basis in init_bipart.h
         //
         const auto& rbasis = icomb.rbases[icomb.topo.ntotal-1];
         int nroots = vs.cols();
         int ndet = rbasis[0].space.size();
         std::map<fock::onstate,int> index; // index of a state
         for(int i=0; i<ndet; i++){
            const auto& state = rbasis[0].space[i];
            index[state] = i;
         }
         //
         // construct the boundary matrix: |psi[i]> = \sum_a |rbas[a]>(<rbas[a]|psi[i]>)
         // In RCF the site is defined as 
         //    W[i,a] =  <rbas[a]|psi[i]> = (rbas^+*wfs)^T = wfs^T*rbas.conj()
         // such that W*[i,a]W[j,a] = delta[i,j]
         //
         qbond qrow({{sym_state, 1}});
         auto qcol = get_qbond(rbasis);
         if(qcol.size() != 1) tools::exit("error: multiple symmetries in qcol!"); 
         icomb.rwfuns.resize(nroots);
         for(int iroot=0; iroot<nroots; iroot++){
            linalg::matrix<Tm> wf(ndet,1);
            for(int i=0; i<space2.size(); i++){
               int ir = index.at(space2[i]);
               wf(ir,0) = vs2(i,iroot);
            } // i
            // rwfuns[l,r] for RCF: ->-*->- 
            stensor2<Tm> rwfun(qsym(Qm::isym), qrow, qcol, {0,1}); 
            xgemm("T","N",1.0,wf,rbasis[0].coeff.conj(),0.0,rwfun(0,0));
            icomb.rwfuns[iroot] = std::move(rwfun);
         } // iroot
         
         // check overlaps
         if(debug_init){
            auto wf2 = icomb.get_wf2();
            wf2.print("wf2");
            auto wfmat = wf2.to_matrix();
            std::cout << "\ncheck state overlaps ..." << std::endl;
            // ova = <CTNS[i]|CTNS[j]>
            auto ova = linalg::xgemm("N","C",wfmat,wfmat).conj();
            ova.print("ova_rwfuns");
            // ova0 = <CI[i]|CI[j]>
            linalg::matrix<Tm> ova0(nroots,nroots);
            for(int i=0; i<nroots; i++){
               for(int j=0; j<nroots; j++){
                  ova0(i,j) = linalg::xdot(vs.rows(),vs.col(i),vs.col(j));
               }
            }
            ova0.print("ova0_vs");
            auto diff = (ova-ova0).normF();
            std::cout << "diff of ova matrices = " << diff << std::endl;
            if(diff > thresh_ortho){ 
               std::string msg = "error: too large diff=";
               tools::exit(msg+std::to_string(diff)+" with thresh_ortho="+std::to_string(thresh_ortho));
            }
         } // debug_init

         auto t1 = tools::get_time();
         tools::timing("ctns::init_rwfuns", t0, t1);
      }

   // canonicalize
   template <typename Qm, typename Tm>
      void init_rcanon(comb<Qm,Tm>& icomb,
            const fock::onspace& space,
            const linalg::matrix<Tm>& vs,
            const double thresh_ortho){
         std::cout << "\nctns::init_rcanon" 
            << " thresh_ortho=" << thresh_ortho
            << std::endl; 

         auto smat0a = rcanon_CIovlp(icomb, space, vs);
         smat0a.print("<CI|CTNS[0]>");
         auto smat1a = get_Smat(icomb);
         std::cout << std::endl;
         smat1a.print("<CTNS[0]|CTNS[0]>");

         // canonicalization
         const bool ifortho = true;
         const bool debug = false;
         auto icomb_new = icomb;
         // performing canonicalization can change renormalized basis,
         // and hence should not be used when performing debug_oper_rbasis
         if(!debug_oper_rbasis){
            rcanon_canonicalize(icomb_new, ifortho, debug);
         }

         auto smat0b = get_Smat(icomb_new);
         std::cout << std::endl;
         smat0b.print("<CTNS[new]|CTNS[new]>");
         auto smat1b = get_Smat(icomb,icomb_new);
         std::cout << std::endl;
         smat1b.print("<CTNS[0]|CTNS[new]>");
         auto smat2b = rcanon_CIovlp(icomb_new, space, vs);
         smat2b.print("<CI|CTNS[new]>");

         // check canonical form
         rcanon_check(icomb_new, thresh_ortho);
         
         icomb = std::move(icomb_new);
      }

} // ctns

#endif
