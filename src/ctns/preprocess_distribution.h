#ifndef PREPROCESS_DISTRIBUTION_H
#define PREPROCESS_DISTRIBUTION_H

#include <numeric>
#include "../core/tools.h"

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
         const std::string qname = qkind::get_name<Qm>();
         if(rank == 0){
            std::cout << "\nctns::preprocess_distribution" 
               << " qkind=" << qname
               << " isym=" << isym
               << " ifkr=" << ifkr
               << std::endl;
         }
         if(rank != 0) return;

         integral::two_body<Tm> int2e;
         int2e.sorb = 2*icomb.topo.nphysical;
         int2e.init_mem();
         std::fill_n(int2e.data.begin(), int2e.data.size(), 1.0); 

         auto sweeps = icomb.topo.get_sweeps(false,true);
         int mid0 = icomb.topo.nphysical/2-2;
         int mid1 = sweeps.size()-1-mid0;
         for(int ibond=0; ibond<sweeps.size(); ibond++){

            //if(ibond != mid0 and ibond != mid1) continue;
            if(ibond != mid0) continue;
            //if(ibond != mid0-3 && ibond != mid0+3) continue;

            const auto& dbond = sweeps[ibond];
            const auto& p0 = dbond.p0;
            const auto& p1 = dbond.p1;
            const auto& forward = dbond.forward;
            std::cout << '\n' << tools::line_separator2 << std::endl;
            std::cout << "ibond=" << ibond << " dbond=" << dbond << std::endl;
            std::cout << tools::line_separator2 << std::endl;

            // twodot Hx
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
            size_t csize_lc1 = sl+sc1;
            size_t csize_c2r = sc2+sr;
            std::string oplist = "CABPQSH";
            bool ifNC = determine_NCorCN_opH(oplist, oplist, csize_lc1, csize_c2r);
            auto key1 = ifNC? "AP" : "PA";
            auto key2 = ifNC? "BQ" : "QB";
            std::cout << " (sl,sr,sc1,sc2)="
               << sl << "," << sr << "," << sc1 << "," << sc2
               << " ifNC=" << ifNC
               << std::endl;
            auto cindex_l = oper_index_opC(suppl, ifkr);
            auto cindex_r = oper_index_opC(suppr, ifkr);
            auto cindex_c1 = oper_index_opC(suppc1, ifkr);
            auto cindex_c2 = oper_index_opC(suppc2, ifkr);
            const bool& ifdist1 = schd.ctns.ifdist1;
            const bool& ifdistc = schd.ctns.ifdistc;
            
            // renormalization
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
            auto aindex = oper_index_opA(cindex, ifkr);
            auto bindex = oper_index_opB(cindex, ifkr);
            auto pindex = oper_index_opP(krest, ifkr); 
            auto qindex = oper_index_opQ(krest, ifkr);
            std::cout << " renormalized operators: C:" << cindex.size() 
               << " S:" << sindex.size()
               << " A:" << aindex.size()
               << " B:" << bindex.size()
               << " P:" << pindex.size()
               << " Q:" << qindex.size()
               << std::endl;

            std::string scratch = "analysis_"+qname;
            io::remove_scratch(scratch);
            io::create_scratch(scratch); 

            std::vector<int> mpisizes({1,2,4,8,16,32,64,128});
            for(int idx=0; idx<mpisizes.size(); idx++){
               int mpisize = mpisizes[idx];
               std::cout << '\n' << tools::line_separator << std::endl;
               std::cout << "mpisize=" << mpisize 
                  << " ifdist1=" << ifdist1
                  << " ifkr=" << ifkr 
                  << std::endl;
               std::cout << tools::line_separator << std::endl;

               std::cout << "\n1. analyze the distribution of renormalized operators:" << std::endl;
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

               std::cout << "\n2. sizes of renorm formulae" << std::endl;
               std::string rscratch = scratch+"/rformulae_mpisize"+std::to_string(mpisize);
               io::create_scratch(rscratch); 
               std::vector<int> rsizes(mpisize,0.0);
               std::vector<std::map<std::string,int>> rcounters(mpisize);
               for(int rank=0; rank<mpisize; rank++){
                  std::string fname = rscratch+"/rformulae_rank"+std::to_string(rank)+".txt";
                  std::streambuf *psbuf, *backup;
                  std::ofstream file;
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

                  std::string block1, block2;
                  std::vector<int> cindex1, cindex2;
                  if(forward){
                     block1 = "l"; cindex1 = cindex_l;
                     block2 = "c"; cindex2 = cindex_c1;
                  }else{
                     block1 = "c"; cindex1 = cindex_c2;
                     block2 = "r"; cindex2 = cindex_r;
                  }
                  bool ifhermi = false;
                  bool ifsave = true;
                  std::map<std::string,int> counter;
                  auto formulae = gen_formulae_renorm(oplist, oplist, oplist,
                        block1, block2, 
                        cindex1, cindex2, krest,
                        isym, ifkr, ifhermi, int2e, int2e.sorb,
                        mpisize, rank, ifdist1, ifdistc, 
                        ifsave, counter);

                  formulae.display("total");
                  std::cout.rdbuf(backup);
                  file.close();

                  // statistics
                  std::cout << " rank=" << rank
                     << " size=" << formulae.size()
                     << " sizetot=" << formulae.sizetot()
                     << " C:" << counter["C"]
                     << " A:" << counter["A"]
                     << " B:" << counter["B"]
                     << " P:" << counter["P"]
                     << " Q:" << counter["Q"]
                     << " S:" << counter["S"]
                     << " H:" << counter["H"]
                     << std::endl;
                  rsizes[rank] = formulae.sizetot();
                  rcounters[rank] = counter;
               } // rank
               analyze_distribution(rsizes,"renorm");
               analyze_distribution2(rcounters,{"C","A","B","P","Q","S","H"});

               std::cout << "\n3. sizes of Hx formulae" << std::endl;
               std::string hscratch = scratch+"/hformulae_mpisize"+std::to_string(mpisize);
               io::create_scratch(hscratch); 
               std::vector<int> hsizes(mpisize,0.0);
               std::vector<std::map<std::string,int>> hcounters(mpisize);
               for(int rank=0; rank<mpisize; rank++){
                  std::string fname = hscratch+"/hformulae_rank"+std::to_string(rank)+".txt";
                  std::streambuf *psbuf, *backup;
                  std::ofstream file;
                  file.open(fname);
                  backup = std::cout.rdbuf();
                  psbuf = file.rdbuf();
                  std::cout.rdbuf(psbuf);
                  std::cout << "gen_formulae_twodot" 
                     << " qkind=" << qname
                     << " isym=" << isym
                     << " ifkr=" << ifkr
                     << " mpisize=" << mpisize
                     << " mpirank=" << rank
                     << std::endl;

                  bool ifsave = true;
                  std::map<std::string,int> counter;
                  auto formulae = gen_formulae_twodot(oplist, oplist, oplist, oplist,
                        cindex_l, cindex_r, cindex_c1, cindex_c2,
                        isym, ifkr, int2e, mpisize, rank, ifdist1, ifdistc, 
                        ifsave, counter);

                  formulae.display("total");
                  std::cout.rdbuf(backup);
                  file.close();

                  // statistics
                  std::cout << " rank=" << rank
                     << " size(formulae)=" << formulae.size()
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
               analyze_distribution(hsizes,"Htwodot");
               analyze_distribution2(hcounters,{"H1","H2","CS","SC",key1,key2});

            } // idx

/*
            // Classification of Hx
            bool ifsave = false;
            std::map<std::string,int> counter;
            auto formulae = gen_formulae_twodot(oplist, oplist, oplist, oplist,
                  cindex_l, cindex_r, cindex_c1, cindex_c2,
                  isym, ifkr, int2e, 1, 0, ifdist1, ifdistc,
                  ifsave, counter);
            std::map<std::string,std::vector<int>> maps;
            for(int i=0; i<formulae.size(); i++){
               auto symbol = formulae.tasks[i].symbol();
               maps[symbol].push_back(i);
            }	     
            std::cout << "Hx_twodot: " << std::endl;
            int nterm = 0;
            int htype = 0;
            for(const auto& pr : maps){
               std::cout << "idx=" << htype 
                  << " htype=" << pr.first
                  << " size=" << pr.second.size()
                  << std::endl;
               nterm += pr.second.size(); 
               htype++;
            } 
            std::cout << "Total types=" << htype
               << " nterm=" << nterm
               << std::endl;
            assert(formulae.size() == nterm);

            // classification based on types
            for(int idx=0; idx<sizes.size(); idx++){
               int mpisize = sizes[idx];
               std::cout << "\nmpisize=" << mpisize 
                  << " ifdist1=" << ifdist1
                  << " ifkr=" << ifkr 
                  << std::endl;
               std::vector<std::map<std::string,int>> imaps(mpisize);
               for(int i=0; i<mpisize; i++){
                  for(const auto& pr : maps){
                     const auto& key = pr.first;
                     imaps[i][key] = 0;
                  }
               }
               for(int rank=0; rank<mpisize; rank++){
                  bool ifsave = false;
                  std::map<std::string,int> counter;
                  auto formulae = gen_formulae_twodot(oplist, oplist, oplist, oplist,
                        cindex_l, cindex_r, cindex_c1, cindex_c2,
                        isym, ifkr, int2e, mpisize, rank, ifdist1, ifdistc, 
                        ifsave, counter);
                  for(int i=0; i<formulae.size(); i++){
                     auto symbol = formulae.tasks[i].symbol();
                     imaps[rank][symbol] += 1;
                  }
               } // rank
               std::cout << "Hx_twodot: " << std::endl;
               int i = 0;
               for(const auto& pr : maps){
                  std::cout << "idx=" << i 
                     << " htype=" << pr.first
                     << " total=" << pr.second.size() 
                     << " : ";
                  for(int rank=0; rank<mpisize; rank++){
                     std::cout << imaps[rank][pr.first] << " ";
                  }
                  std::cout << std::endl;
                  i++;
               }
            } // idx
*/

         } // ibond
      }

} // ctns

#endif
