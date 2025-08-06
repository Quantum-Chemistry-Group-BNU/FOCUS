#ifndef CTNS_LCANON_H
#define CTNS_LCANON_H

#include "../io/io.h"
#include "../core/serialization.h"

namespace ctns{

   template <typename Qm, typename Tm>
      void sweep_final_LC2LLc(comb<Qm,Tm>& icomb,
            const double rdm_svd,
            const int svd_iop,
            const std::string fname,
            const bool debug){
         if(debug) std::cout << "\nctns::sweep_final_LC2LLc: convert L[C] => L[Lc]" << std::endl;
         int nroots = icomb.cpsi.size();
         // 1. setup 
         const auto& wfinfo = icomb.cpsi[0].info;
         qtensor2<Qm::ifabelian,Tm> rot;
         std::vector<qtensor2<Qm::ifabelian,Tm>> wfs2(nroots);
         for(int i=0; i<nroots; i++){
            auto wf2 = icomb.cpsi[i].recouple_lc().merge_lc();
            wfs2[i] = std::move(wf2);
         }
         const int dcut = nroots; // psi[1,n,r,i] => U[1,i,a]sigma[a]Vt[a,n,r]
         double dwt;
         int deff;
         const bool iftrunc = true;
         const int alg_decim = 0;
         std::vector<double> sigs2full;
         // 2. decimation
         decimation_row(icomb, wfinfo.qrow, wfinfo.qmid,
               iftrunc, dcut, rdm_svd, svd_iop, alg_decim,
               wfs2, sigs2full, rot, dwt, deff, fname,
               debug);
         // 3. save site0
         const auto& pdx0 = icomb.topo.rindex.at(std::make_pair(icomb.topo.nphysical-1,0));
         icomb.sites[pdx0] = rot.split_lc(wfinfo.qrow, wfinfo.qmid);
         // 4. form rwfuns(iroot,irbas)
         //const auto& sym_state = icomb.get_qsym_state(); // not wfinfo.sym in singlet embedding (wfinfo.sym=0) 
         const auto& sym_state = wfinfo.sym; // in this case auxilliary site is included in the left boundary
         const auto& qrow = rot.info.qcol;
         qbond qcol({{sym_state, 1}});
         icomb.rwfuns.resize(nroots);
         for(int i=0; i<nroots; i++){
            auto cwf = rot.H().dot(wfs2[i]); // <-W[alpha,1]->
            // change the carrier of sym_state from center to left
            qtensor2<Qm::ifabelian,Tm> rwfun(qsym(Qm::isym), qrow, qcol, dir_OPER);
            assert(cwf.size() == rwfun.size());
            linalg::xcopy(cwf.size(), cwf.data(), rwfun.data());
            icomb.rwfuns[i] = std::move(rwfun);
            
            // TO BE IMPROVED IN FUTURE; FOR INTERFACING WITH BLOCK2, WE ONLY CONSIDER SINGLE STATE 
            if(nroots != 1 or !check_identityMatrix(cwf.to_matrix())){
               std::cout << "error: lcanon only supports nroots=1 and cwf=Id currently!" << std::endl;
               exit(1); 
            }else{
               icomb.rwfuns[i].print("lwfun",2);
            }

         } // iroot
      }

   // left canonicalization
   template <typename Qm, typename Tm>
      void lcanon(const comb<Qm,Tm>& icomb0,
            const input::schedule& schd,
            const std::string rcanon_file,
            const bool debug=true){
         const bool ifab = Qm::ifabelian;
         const size_t dmax = icomb0.get_dmax();
         const bool singlet = schd.ctns.singlet;
         std::cout << "\nctns::lcanon"
              << " ifab=" << ifab
              << " dmax=" << dmax
              << " singlet=" << singlet
              << std::endl;
         auto t0 = tools::get_time();
             
         const int nroots = 1; 
         auto icomb = icomb0;
         init_cpsi_dot0(icomb, schd.ctns.iroot, singlet);
         
         // generate sweep sequence
         const auto& rindex = icomb.topo.rindex;
         auto sweep_seq = icomb.topo.get_mps_fsweeps(true); // include boundary
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
            maxdwt = std::max(maxdwt,dwt);

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
         sweep_final_LC2LLc(icomb, rdm_svd, svd_iop, fname, debug);

         auto t1 = tools::get_time();
         std::cout << "\nmaxdwt during canonicalization = " << maxdwt << std::endl;
         icomb.display_shape();

         // --- savebin ---
         fname = schd.scratch + "/" + rcanon_file + ".lcanon";
         if(singlet) fname += ".singlet";
         std::cout << "\nsave lcanon into file = " << fname << ".bin" << std::endl;
         std::ofstream ofs2(fname+".bin", std::ios::binary);
         ofs2.write((char*)(&icomb.topo.ntotal), sizeof(int));
         // save all sites
         for(int idx=0; idx<icomb.topo.ntotal; idx++){
            icomb.sites[idx].dump(ofs2);
         }
         ofs2.close();

         tools::timing("ctns::lcanon", t0, t1);
      }

} // ctns

#endif
