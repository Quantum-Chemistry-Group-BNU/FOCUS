#ifndef PREPROCESS_DISTRIBUTION_H
#define PREPROCESS_DISTRIBUTION_H

#include <numeric>
#include "../core/tools.h"
#include "oper_ab2pq.h"

namespace ctns{

   inline void analyze_distribution(const std::vector<int>& sizes,
         const std::string name){
      //tools::print_vector(sizes,name);
      int mpisize = sizes.size();
      std::vector<double> fsizes(mpisize);
      std::transform(sizes.cbegin(), sizes.cend(), fsizes.begin(),
            [](const int& x){ return double(x); }); 
      int max = *std::max_element(fsizes.begin(), fsizes.end());
      int min = *std::min_element(fsizes.begin(), fsizes.end());
      double sum = std::accumulate(fsizes.begin(), fsizes.end(), 0.0);
      double mean = sum/mpisize;
      std::transform(fsizes.begin(), fsizes.end(), fsizes.begin(),
            [&mean](const double& x){ return (x-mean)*(x-mean); });
      double stdev = std::sqrt(std::accumulate(fsizes.begin(), fsizes.end(), 0.0)/mpisize);
      std::cout << " " << name
         << " mpisize=" << mpisize
         << " sum=" << int(sum)
         << " mean=" << int(mean) 
         << " max=" << max 
         << " min=" << min
         << " diff=" << (max-min)
         << " stdev=" << std::setprecision(1) << stdev
         << std::endl;
   }

   inline void analyze_distribution2(const std::vector<std::map<std::string,int>>& counters,
         std::vector<std::string> classes){
      int mpisize = counters.size();
      for(int i=0; i<classes.size(); i++){
         auto cls = classes[i];
         std::vector<int> sizes(mpisize);
         for(int j=0; j<mpisize; j++){
            sizes[j] = counters[j].at(cls);
         }
         analyze_distribution(sizes, " formulae "+cls);
      }
   }

   // analyze the distribution of operators along a sweep
   template <typename Qm, typename Tm>
      void preprocess_distribution(const comb<Qm,Tm>& icomb,
            const input::schedule& schd){	
         int size = 1, rank = 0;
#ifndef SERIAL
         size = icomb.world.size();
         rank = icomb.world.rank();
#endif  
         // settings 
         const int isym = Qm::isym;
         const bool ifkr = Qm::ifkr;
         const bool ifab = Qm::ifabelian;
         const std::string qname = qkind::get_name<Qm>();
         if(rank == 0){
            std::cout << "\nctns::preprocess_distribution" 
               << " qkind=" << qname
               << " isym=" << isym
               << " ifkr=" << ifkr
               << std::endl;
         }
         if(rank != 0) return;
         auto t0 = tools::get_time();

         integral::two_body<Tm> int2e;
         int2e.sorb = 2*icomb.topo.nphysical;
         int2e.init_mem();
         std::fill_n(int2e.data.begin(), int2e.data.size(), 1.0); 

         // twodot/ondot case   
         const bool ifmps = true;
         const int nsite = icomb.topo.nphysical;
         const int ndots = schd.ctns.ctrls[0].dots;
         const int mpisize = schd.ctns.mpisize_debug;
         const bool& ifdist1 = schd.ctns.ifdist1;
         const bool& ifdistc = schd.ctns.ifdistc;
         const bool& ifab2pq = schd.ctns.ifab2pq;
         const bool ifhermi = true;
         const bool ifsave = true;

         const bool ifboundary = false;
         const bool debug = true;
         auto sweeps = icomb.topo.get_sweeps(ifboundary, debug);
         int mid0 = icomb.topo.nphysical/2-2;
         int mid1 = sweeps.size()-1-mid0;
         for(int ibond=0; ibond<sweeps.size(); ibond++){
            const auto& dbond = sweeps[ibond];
            const auto& p0 = dbond.p0;
            const auto& p1 = dbond.p1;
            const auto& forward = dbond.forward;
            std::cout << "\nibond=" << ibond << " dbond=" << dbond 
               << " mpisize=" << mpisize
               << " ifdist1=" << ifdist1
               << " ifdistc=" << ifdistc
               << " ifab2pq=" << ifab2pq
               << " ifkr=" << ifkr 
               << std::endl;
            std::cout << tools::line_separator2 << std::endl;

            std::vector<int> suppl, suppr, suppc1, suppc2;
            auto node0 = icomb.topo.get_node(p0);
            auto node1 = icomb.topo.get_node(p1);
            if(!dbond.is_cturn()){
               suppl = node0.lorbs;
               suppr = node1.rorbs;
               suppc1 = node0.corbs;
               suppc2 = node1.corbs;
            }else{
               suppl = node0.lorbs;
               suppr = node0.rorbs;
               suppc1 = node1.corbs;
               suppc2 = node1.rorbs;
            }
            int sl = suppl.size();
            int sr = suppr.size();
            int sc1 = suppc1.size();
            int sc2 = suppc2.size();
            assert(sc1+sc2+sl+sr == icomb.topo.nphysical);

            std::string lblock = "lc";
            std::string rblock = "cr";
            auto coplist = "CSHABPQ"; 
            std::string loplist, roplist;
            if(p0.first == 1){
               loplist = coplist; // at the left boundary 
            }else{
               loplist = oper_renorm_oplist(lblock, ifmps, nsite, p0, ifab2pq, ndots);
            }
            if(p1.first == nsite-2){
               roplist = coplist; // at the right boundary
            }else{
               roplist = oper_renorm_oplist(rblock, ifmps, nsite, p1, ifab2pq, ndots);
            }
            std::vector<int> cindex_l, cindex_r, cindex_c1, cindex_c2, cindex_c;
            int csize_l, csize_r;
            if(ndots == 2){
               cindex_l = oper_index_opC(suppl, ifkr);
               cindex_r = oper_index_opC(suppr, ifkr);
               cindex_c1 = oper_index_opC(suppc1, ifkr);
               cindex_c2 = oper_index_opC(suppc2, ifkr);
               csize_l = sl+sc1;
               csize_r = sc2+sr; 
            }else{
               if(forward){ // l|c1|c2r
                  cindex_l = oper_index_opC(suppl, ifkr);
                  cindex_c1 = oper_index_opC(suppc1, ifkr);
                  cindex_c = cindex_c1;
                  auto suppc2r = suppc2;
                  std::copy(suppr.begin(), suppr.end(), std::back_inserter(suppc2r));
                  cindex_r = oper_index_opC(suppc2r, ifkr);
                  csize_l = sl;
                  csize_r = sc2+sr;
               }else{ // lc1|c2|r
                  auto supplc1 = suppl;
                  std::copy(suppc1.begin(), suppc1.end(), std::back_inserter(supplc1));
                  cindex_l = oper_index_opC(supplc1, ifkr);
                  cindex_c2 = oper_index_opC(suppc2, ifkr);
                  cindex_c = cindex_c2;
                  cindex_r = oper_index_opC(suppr, ifkr);
                  csize_l = sl+sc1;
                  csize_r = sr;
               }
            }
            bool ifNC = determine_NCorCN_Ham(loplist, roplist, csize_left, csize_right);
            auto key1 = ifNC? "AP" : "PA";
            auto key2 = ifNC? "BQ" : "QB";
            std::string strdots = (ndots==2)? "twodot" : "onedot";

            std::string scratch = "analysis_"+qname+"_ibond"+std::to_string(ibond);
            io::remove_scratch(scratch);
            io::create_scratch(scratch); 

            //-----------------------------------------------
            // H-formulae
            //-----------------------------------------------
            std::cout << "\nno. of hformulae" << std::endl;
            std::cout << "----------------" << std::endl;
            std::cout << "(sl,sr,sc1,sc2)="
               << sl << "," << sr << "," << sc1 << "," << sc2
               << " ifNC=" << ifNC
               << " loplist=" << loplist
               << " roplist=" << roplist
               << std::endl;
            std::string hscratch = scratch+"/hformulae_mpisize"+std::to_string(mpisize);
            io::create_scratch(hscratch); 
            std::vector<int> hsizes(mpisize,0.0);
            std::vector<std::map<std::string,int>> hcounters(mpisize);
            for(int rank=0; rank<mpisize; rank++){
               std::string fname = hscratch+"/hformulae_rank"+std::to_string(rank)+".txt";
               std::streambuf *psbuf, *backup;
               std::ofstream file;
               if(ifsave){
                  file.open(fname);
                  backup = std::cout.rdbuf();
                  psbuf = file.rdbuf();
                  std::cout.rdbuf(psbuf);
                  std::cout << "gen_formulae_"+strdots
                     << " qkind=" << qname
                     << " isym=" << isym
                     << " ifkr=" << ifkr
                     << " mpisize=" << mpisize
                     << " mpirank=" << rank
                     << std::endl;
               }

               std::map<std::string,int> counter;
               symbolic_task<Tm> formulae;
               if(ndots == 2){
                  if(ifab){
                     formulae = gen_formulae_twodot(loplist, roplist, coplist, coplist,
                           cindex_l, cindex_r, cindex_c1, cindex_c2,
                           isym, ifkr, int2e, mpisize, rank, ifdist1, ifdistc, 
                           ifsave, counter);
                  }else{
                     formulae = gen_formulae_twodot_su2(loplist, roplist, coplist, coplist,
                           cindex_l, cindex_r, cindex_c1, cindex_c2,
                           isym, ifkr, int2e, mpisize, rank, ifdist1, ifdistc, 
                           ifsave, counter);
                  }
               }else{
                  if(ifab){
                     formulae = gen_formulae_onedot(loplist, roplist, coplist, 
                           cindex_l, cindex_r, cindex_c, 
                           isym, ifkr, int2e, mpisize, rank, ifdist1, ifdistc, 
                           ifsave, counter);
                  }else{
                     formulae = gen_formulae_onedot_su2(loplist, roplist, coplist,
                           cindex_l, cindex_r, cindex_c,
                           isym, ifkr, int2e, mpisize, rank, ifdist1, ifdistc, 
                           ifsave, counter);
                  }
               } // ndots

               if(ifsave){
                  std::cout << "\nSUMMARY["+strdots+"] ifNC=" << ifNC << " size=" << formulae.size()
                     << " H1:" << counter["H1"] << " H2:" << counter["H2"];
                  if(ifNC){
                     std::cout << " CS:" << counter["CS"] << " SC:" << counter["SC"]
                        << " AP:" << counter["AP"] << " BQ:" << counter["BQ"]
                        << std::endl;
                  }else{
                     std::cout << " SC:" << counter["SC"] << " CS:" << counter["CS"]
                        << " PA:" << counter["PA"] << " QB:" << counter["QB"]
                        << std::endl;
                  }
                  formulae.display("total");
                  std::cout.rdbuf(backup);
                  file.close();
               }

               // statistics
               std::cout << " rank=" << rank
                  << " size(hformulae)=" << formulae.size()
                  << " H1:" << counter["H1"]
                  << " H2:" << counter["H2"]
                  << " CS:" << counter["CS"]
                  << " SC:" << counter["SC"]
                  << " " << key1 << ":" << counter[key1]
                  << " " << key2 << ":" << counter[key2]
                  << std::endl;
               hsizes[rank] = formulae.size();
               hcounters[rank] = counter;
            } // rank
            analyze_distribution(hsizes,"H"+strdots);
            analyze_distribution2(hcounters,{"H1","H2","CS","SC",key1,key2});

            //-----------------------------------------------
            // Renormalized operators
            //-----------------------------------------------
            std::cout << "\ndistribution of renormalized operators" << std::endl;
            std::cout << "----------------\n" << std::endl;
            auto fbond = icomb.topo.get_fbond(dbond,".");
            auto frop = fbond.first;
            auto fdel = fbond.second;
            std::cout << " renormalization: frop=" << frop << " fdel=" << fdel << std::endl;
            auto node = icomb.topo.get_node(dbond.p1);
            auto lsupp = node.lsupport;
            auto rsupp = node.rsupport;
            tools::print_vector(lsupp,"lsupp");
            tools::print_vector(rsupp,"rsupp");
            std::vector<int> ksupp, krest;
            if(forward){
               ksupp = lsupp;
               krest = rsupp;
            }else{
               ksupp = rsupp;
               krest = lsupp;
            }
            auto cindex = oper_index_opC(ksupp, ifkr);
            auto sindex = oper_index_opS(krest, ifkr);
            auto aindex = oper_index_opA(cindex, ifkr, isym);
            auto bindex = oper_index_opB(cindex, ifkr, isym);
            auto pindex = oper_index_opP(krest, ifkr, isym);
            auto qindex = oper_index_opQ(krest, ifkr, isym);
            std::cout << " renormalized operators: C:" << cindex.size() 
               << " S:" << sindex.size()
               << " A:" << aindex.size()
               << " B:" << bindex.size()
               << " P:" << pindex.size()
               << " Q:" << qindex.size()
               << std::endl;
            // s
            std::vector<int> sstat(mpisize,0);  
            for(int idx : sindex){
               int iproc = distribute1(ifkr,mpisize,idx);
               sstat[iproc] += 1;
            }
            analyze_distribution(sstat,"opS stat");
            // a
            std::vector<int> astat(mpisize,0);  
            for(int idx : aindex){
               int iproc = distribute2('A',ifkr,mpisize,idx,int2e.sorb);
               astat[iproc] += 1;
            }
            analyze_distribution(astat,"opA stat");
            // b
            std::vector<int> bstat(mpisize,0);  
            for(int idx : bindex){
               int iproc = distribute2('B',ifkr,mpisize,idx,int2e.sorb);
               bstat[iproc] += 1;
            }
            analyze_distribution(bstat,"opB stat");
            // p
            std::vector<int> pstat(mpisize,0);  
            for(int idx : pindex){
               int iproc = distribute2('P',ifkr,mpisize,idx,int2e.sorb);
               pstat[iproc] += 1;
            }
            analyze_distribution(pstat,"opP stat");
            // q
            std::vector<int> qstat(mpisize,0);  
            for(int idx : qindex){
               int iproc = distribute2('Q',ifkr,mpisize,idx,int2e.sorb);
               qstat[iproc] += 1;
            }
            analyze_distribution(qstat,"opQ stat");

            std::string superblock, block1, block2, oplist, oplist1, oplist2;
            std::vector<int> cindex1, cindex2;
            if(forward){
               superblock = lblock;
               block1 = "l"; cindex1 = cindex_l;
               block2 = "c"; cindex2 = cindex_c1;
               oplist1 = loplist;
               oplist2 = coplist;
               oplist = oper_renorm_oplist(lblock, ifmps, nsite, p0, ifab2pq, ndots);
            }else{
               superblock = rblock;
               block1 = "c"; cindex1 = cindex_c2;
               block2 = "r"; cindex2 = cindex_r;
               oplist1 = coplist;
               oplist2 = roplist;
               oplist = oper_renorm_oplist(rblock, ifmps, nsite, p0, ifab2pq, ndots);
            }

            std::cout << "\nno. of rformulae" << std::endl;
            std::cout << "----------------" << std::endl;
            std::cout << "superblock=" << superblock 
               << " oplist=" << oplist
               << " oplist1=" << oplist1
               << " oplist2=" << oplist2
               << std::endl;
            std::string rscratch = scratch+"/rformulae_mpisize"+std::to_string(mpisize);
            io::create_scratch(rscratch); 
            std::vector<int> rsizes(mpisize,0);
            std::vector<std::map<std::string,int>> rcounters(mpisize);
            for(int rank=0; rank<mpisize; rank++){
               std::string fname = rscratch+"/rformulae_rank"+std::to_string(rank)+".txt";
               std::streambuf *psbuf, *backup;
               std::ofstream file;
               if(ifsave){
                  file.open(fname);
                  backup = std::cout.rdbuf();
                  psbuf = file.rdbuf();
                  std::cout.rdbuf(psbuf);
                  std::cout << "gen_formulae_renorm" 
                     << " qkind=" << qname
                     << " isym=" << isym
                     << " ifkr=" << ifkr
                     << " mpisize=" << mpisize
                     << " mpirank=" << rank
                     << std::endl;
               }
               std::map<std::string,int> counter;
               renorm_tasks<Tm> formulae;
               if(ifab){
                  formulae = gen_formulae_renorm(oplist, oplist1, oplist2,
                        block1, block2, cindex1, cindex2, krest, isym, ifkr, ifhermi, 
                        int2e, int2e.sorb, mpisize, rank, ifdist1, ifdistc, 
                        ifsave, counter);
               }else{
                  formulae = gen_formulae_renorm_su2(oplist, oplist1, oplist2,
                        block1, block2, cindex1, cindex2, krest, isym, ifkr, ifhermi, 
                        int2e, int2e.sorb, mpisize, rank, ifdist1, ifdistc, 
                        ifsave, counter);
               }
               if(ifsave){
                  formulae.display("total");
                  std::cout.rdbuf(backup);
                  file.close();
               }

               // statistics
               std::cout << " rank=" << rank
                  << " size(rformulae)=" << formulae.size()
                  << " sizetot=" << formulae.sizetot()
                  << " C:" << counter["C"]
                  << " S:" << counter["S"]
                  << " H:" << counter["H"]
                  << " A:" << counter["A"]
                  << " B:" << counter["B"]
                  << " P:" << counter["P"]
                  << " Q:" << counter["Q"]
                  << std::endl;
               rsizes[rank] = formulae.sizetot();
               rcounters[rank] = counter;
            } // rank
            analyze_distribution(rsizes,"renorm");
            analyze_distribution2(rcounters,{"C","S","H","A","B","P","Q"});

         } // ibond

         auto t1 = tools::get_time();
         tools::timing("preprocess", t0, t1);
      }

} // ctns

#endif
