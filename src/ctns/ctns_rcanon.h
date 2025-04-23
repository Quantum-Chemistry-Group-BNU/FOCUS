#ifndef CTNS_RCANON_H
#define CTNS_RCANON_H

namespace ctns{

   // --<--*-->--
   //     /|\
   // boundary_coupling: |tot>=|vaccum>*|physical>
   template <typename Qm, typename Tm>
      qtensor2<Qm::ifabelian,Tm> get_boundary_coupling(const qsym& sym_state, const bool singlet){
         const int isym = sym_state.isym();
         qbond qcol({{sym_state,1}});
         qtensor2<Qm::ifabelian,Tm> wf2;
         if(isym == 3 && singlet){
            qbond qrow({{qsym(3,sym_state.ts(),sym_state.ts()),1}}); // couple to fictious site
            wf2.init(qsym(3,sym_state.ne()+sym_state.ts(),0),qrow,qcol,dir_WF2);
         }else{ 
            qbond qrow({{qsym(isym,0,0),1}}); // couple to vacuum
            wf2.init(sym_state,qrow,qcol,dir_WF2);
         }
         assert(wf2.size() == 1);
         wf2._data[0] = 1.0;
         return wf2;
      }

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

   //     | qs |
   //  ---*----*---(0,0)
   //    r1  r0
   template <typename Qm, typename Tm>
      void rcanon_lastdots(comb<Qm,Tm>& icomb){
         // select boundary sites
         // topo:
         //  i=0 : 4
         //  i=1 : -1 5 1
         //  i=2 : -1 0 3
         //  i=3 : 2
         std::vector<std::pair<int,int>> bpairs;
         for(int i=1; i<icomb.topo.nbackbone; i++){ // exclude the first site
            for(int j=0; j<icomb.topo.nodes[i].size(); j++){
               const auto& node = icomb.topo.nodes[i][j];
               if(node.type != 0) continue;
               bpairs.push_back(std::make_pair(i,j));
            }
         }
         // loop over boundaries
         for(const auto& pr : bpairs){
            int i = pr.first;
            int j = pr.second;
            int pdx0, pdx1;
            if(j == 0){
               pdx0 = icomb.topo.rindex.at(std::make_pair(i,0));
               pdx1 = icomb.topo.rindex.at(std::make_pair(i-1,0));
            }else{
               pdx0 = icomb.topo.rindex.at(std::make_pair(i,j));
               pdx1 = icomb.topo.rindex.at(std::make_pair(i,j-1));
            }
            auto& rsite_last0 = icomb.sites[pdx0];
            auto& rsite_last1 = icomb.sites[pdx1];
            auto& qs = rsite_last0.info.qrow;
            auto qmid = get_qbond_phys(Qm::isym);
            if(qs == qmid) return; // no need to do anything
            assert(qs.size() <= qmid.size());
            // We simply use computation to avoid explicitly deal
            // with the formation of the last two sites.
            auto rmat = rsite_last0.merge_cr();
            // P is due to definition in contract_qt3_qt2
            rsite_last1 = contract_qt3_qt2("r",rsite_last1,rmat.P());
            rsite_last0 = get_right_bsite<Qm,Tm>();
         }
      }

   template <typename Qm, typename Tm>
      void rcanon_rwfuns(comb<Qm,Tm>& icomb, const bool debug){
         const int nroots = icomb.get_nroots();
         auto rwfunW = icomb.get_wf2().to_matrix();
         std::vector<double> s;
         linalg::matrix<Tm> U, Vt;
         linalg::svd_solver(rwfunW, s, U, Vt, 13);
         if(debug){
            std::cout << "\nrcanon_rwfuns: Lowdin orthonormalization" << std::endl;
            tools::print_vector(s,"sigs");
         }
         // lowdin orthonormalization: wf2new = U*Vt 
         auto wf2new = linalg::xgemm("N","N",U,Vt).T(); // T() for convenience
         // save the data back
         for(int i=0; i<nroots; i++){
            assert(wf2new.rows() == icomb.rwfuns[i].size());
            linalg::xcopy(wf2new.rows(), wf2new.col(i), icomb.rwfuns[i].data());
         }
      }

   template <>
      inline void rcanon_rwfuns(comb<qkind::qNK,std::complex<double>>& icomb, const bool debug){
         const int nroots = icomb.get_nroots();
         auto sym_state = icomb.get_qsym_state();
         std::vector<double> sigs2;
         linalg::matrix<std::complex<double>> U;
         std::vector<double> phases;
         int nbas = icomb.rwfuns[0].info.qcol.get_dimAll();
         if(sym_state.parity()){
            assert(nbas%2 == 0);
            phases.resize(nbas/2,1.0);
         }
         std::vector<linalg::matrix<std::complex<double>>> blks(nroots);
         for(int iroot=0; iroot<nroots; iroot++){
            blks[iroot] = icomb.rwfuns[iroot].to_matrix();
         }
         const double rdm_svd = 1.5;
         const int svd_iop = 3;
         kramers::get_renorm_states_kr(sym_state, phases, blks, sigs2, U, rdm_svd, svd_iop, debug);
         assert(U.cols() == nroots);
         if(debug){
            std::cout << "\nrcanon_rwfuns: Kramers projection" << std::endl;
            tools::print_vector(sigs2,"sigs2");
         }
         // save the data back
         for(int i=0; i<nroots; i++){
            linalg::xcopy(U.rows(), U.col(i), icomb.rwfuns[i].data());
         } 
      }

   // rcanonicalize MPS: this function use twodot [!!!] algorithm to perform canonicalization of MPS
   template <typename Qm, typename Tm>
      void rcanon_canonicalize(comb<Qm,Tm>& icomb,
            const int dmax,
            const bool ifortho=true,
            const bool debug=true){
         const bool ifab = Qm::ifabelian;
         const int nroots = icomb.get_nroots();
         std::cout << "\nrcanon::rcanon_canonicalize"
              << " ifortho=" << ifortho
              << " ifab=" << ifab
              << " nroots=" << nroots
              << " dmax=" << dmax
              << std::endl;
         auto t0 = tools::get_time();

         // initialization: generate cpsi
         const int iroot = -1;
         const bool singlet = true;
         init_cpsi_dot0(icomb, iroot, singlet);

         // generate sweep sequence
         const auto& rindex = icomb.topo.rindex;
         auto sweep_seq = icomb.topo.get_sweeps(true); // include boundary
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
               const int dots = 2;
               std::cout << "\nibond=" << ibond << "/seqsize=" << sweep_seq.size()
                  << " dots=" << dots << " dbond=" << dbond
                  << std::endl;
               std::cout << tools::line_separator << std::endl;
            }

            // construct twodot wavefunction: wf4
            const auto sym_state = icomb.get_qsym_state();
            auto wfsym = get_qsym_state(Qm::isym, sym_state.ne(),
                  (ifab? sym_state.tm() : sym_state.ts()), singlet);
            const auto& site0 = dbond.forward? icomb.cpsi[0] : icomb.sites[rindex.at(dbond.p0)];
            const auto& site1 = dbond.forward? icomb.sites[rindex.at(dbond.p1)] : icomb.cpsi[0];
            qbond ql, qr, qc1, qc2;
            if(!dbond.is_cturn()){
               //        c1   c2
               //        |    |
               //    l---p0---p1---r
               //
               assert(site0.info.qcol == site1.info.qrow);
               ql = site0.info.qrow;
               qr = site1.info.qcol;
               qc1 = site0.info.qmid;
               qc2 = site1.info.qmid;
            }else{
               //       c2
               //       |
               //  c1---p1
               //       |
               //   l---p0---r
               //
               assert(site0.info.qmid == site1.info.qrow);
               ql = site0.info.qrow;
               qr = site0.info.qcol;
               qc1 = site1.info.qmid;
               qc2 = site1.info.qcol;
            }
            qtensor4<ifab,Tm> wf(wfsym, ql, qr, qc1, qc2);
            size_t ndim = wf.size();
            if(debug){
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
            twodot_guess_v0(icomb, dbond, ndim, nroots, wf, v0, debug);
            assert(v0.size() == ndim*nroots);

            //---------------------------------------------------------
            // perform decimation: see sweep_twodot_decim.h
            const bool iftrunc = true;
            const double rdm_svd = 1.5;
            const int svd_iop = 3;
            const int alg_decim = 0; // serial version
            std::string fname;
            std::vector<double> sigs2full; 
            double dwt;
            int deff; 
            qtensor2<ifab,Tm> rot;
            std::vector<qtensor2<ifab,Tm>> wfs2(nroots);
            if(superblock == "lc1"){

               for(int i=0; i<nroots; i++){
                  wf.from_array(&v0[i*ndim]);
                  auto wf2 = wf.merge_lc1_c2r();
                  wfs2[i] = std::move(wf2);
               }
               decimation_row(icomb, wf.info.qrow, wf.info.qmid, // lc1=(row,mid) 
                     iftrunc, dmax, rdm_svd, svd_iop, alg_decim,
                     wfs2, sigs2full, rot, dwt, deff, fname,
                     debug);

            }else if(superblock == "c2r"){

               for(int i=0; i<nroots; i++){
                  wf.from_array(&v0[i*ndim]);
                  auto wf2 = wf.merge_lc1_c2r().P();
                  wfs2[i] = std::move(wf2);
               }
               decimation_row(icomb, wf.info.qver, wf.info.qcol, // c2r=(ver,col)
                     iftrunc, dmax, rdm_svd, svd_iop, alg_decim,
                     wfs2, sigs2full, rot, dwt, deff, fname,
                     debug);
               rot = rot.P();

            }else if(superblock == "lr"){

               assert(Qm::ifabelian);
               for(int i=0; i<nroots; i++){
                  wf.from_array(&v0[i*ndim]);
                  wf.permCR_signed();
                  auto wf2 = wf.merge_lr_c1c2();
                  wfs2[i] = std::move(wf2);
               }
               decimation_row(icomb, wf.info.qrow, wf.info.qcol, 
                     iftrunc, dmax, rdm_svd, svd_iop, alg_decim,
                     wfs2, sigs2full, rot, dwt, deff, fname,
                     debug);

            }else if(superblock == "c1c2"){

               assert(Qm::ifabelian);
               for(int i=0; i<nroots; i++){
                  wf.from_array(&v0[i*ndim]);
                  wf.permCR_signed();
                  auto wf2 = wf.merge_lr_c1c2().P();
                  wfs2[i] = std::move(wf2);
               } // i
               decimation_row(icomb, wf.info.qmid, wf.info.qver, 
                     iftrunc, dmax, rdm_svd, svd_iop, alg_decim,
                     wfs2, sigs2full, rot, dwt, deff, fname,
                     debug);
               rot = rot.P(); // permute two lines for RCF

            }

            // save the current site
            const auto p = dbond.get_current();
            const auto& pdx = rindex.at(p); 
            if(superblock == "lc1"){
               icomb.sites[pdx] = rot.split_lc(wf.info.qrow, wf.info.qmid);
            }else if(superblock == "c2r"){
               icomb.sites[pdx] = rot.split_cr(wf.info.qver, wf.info.qcol);
            }else if(superblock == "lr"){
               assert(Qm::ifabelian);
               icomb.sites[pdx]= rot.split_lr(wf.info.qrow, wf.info.qcol);
            }else if(superblock == "c1c2"){
               assert(Qm::ifabelian);
               icomb.sites[pdx] = rot.split_cr(wf.info.qmid, wf.info.qver);
            }
            //---------------------------------------------------------

            // propagate to the next site
            linalg::matrix<Tm> vsol(ndim, nroots, v0.data());
            twodot_guess_psi(superblock, icomb, dbond, vsol, wf, rot);
            vsol.clear();
         } // ibond

         // compute rwfuns for the next call of reduce_entropy_single
         const double rdm_svd = 1.5;
         const int svd_iop = 3;
         std::string fname;
         sweep_final_CR2cRR(icomb, rdm_svd, svd_iop, fname, debug);

         // Orthonormalize rwfuns[iroot,alpha] (->-*->-) via SVD
         if(ifortho) rcanon_rwfuns(icomb, debug);

         // canonicalize last dot to identity
         rcanon_lastdots(icomb);

         auto t1 = tools::get_time();
         tools::timing("ctns::rcanon_canonicalize", t0, t1);
      }

} // ctns

#endif
