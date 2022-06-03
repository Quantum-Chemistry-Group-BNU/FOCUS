#ifndef PREPROCESS_H
#define PREPROCESS_H

#include <numeric>

namespace ctns{

// primitive form (without factorization)
template <typename Tm>
symbolic_task<Tm> preprocess_formulae_twodot(const std::vector<int>& cindex_l,
					     const std::vector<int>& cindex_r,
					     const std::vector<int>& cindex_c1,
					     const std::vector<int>& cindex_c2,
	                                     const integral::two_body<Tm>& int2e,
   					     const int isym,
   					     const bool ifkr,
	                                     const int size,
	                                     const int rank,
   					     const bool debug = false){
   auto t0 = tools::get_time();
   const int print_level = 1;
   int slc1 = cindex_l.size() + cindex_c1.size();
   int sc2r = cindex_c2.size() + cindex_r.size();
   const bool ifNC = (slc1 <= sc2r);

   symbolic_task<Tm> formulae;
   
   int idx = 0;
   std::map<std::string,int> counter;
   counter["CS"] = 0;
   counter["SC"] = 0;
   
   // Local terms:
   // H[lc1]
   auto Hlc1 = symbolic_compxwf_opH<Tm>("l", "c1", cindex_l, cindex_c1,
		                        ifkr, size, rank);
   formulae.join(Hlc1);
   if(debug){ 
      std::cout << "idx=" << idx++; 
      Hlc1.display("Hlc1", print_level);
   }
   // H[c2r]
   auto Hc2r = symbolic_compxwf_opH<Tm>("c2", "r", cindex_c2, cindex_r, 
		                        ifkr, size, rank);
   formulae.join(Hc2r);
   if(debug){ 
      std::cout << "idx=" << idx++;
      Hc2r.display("Hc2r", print_level);
   }

   // One-index terms:
   // 3. sum_p1 p1^+[LC1]*Sp1^[C2R] + h.c.
   auto infoC1 = oper_combine_opC(cindex_l, cindex_c1);
   for(const auto& pr : infoC1){
      int index = pr.first;
      int iformula = pr.second;
      // p1^L1C1+*Sp1^C2R & -p1^L1C1*Sp1^C2R+
      auto Clc1 = symbolic_normxwf_opC<Tm>("l", "c1", index, iformula);
      auto Sc2r = symbolic_compxwf_opS<Tm>("c2", "r", cindex_c2, cindex_r,
		                           int2e, index, isym, ifkr, size, rank);
      auto Clc1_Sc2r = Clc1.outer_product(Sc2r);
      formulae.join(Clc1_Sc2r);
      if(debug){ 
	 std::cout << "idx=" << idx++;
	 Clc1_Sc2r.display("Clc1_Sc2r["+std::to_string(index)+"]", print_level);
	 counter["CS"] += 1;
      }
   }
   // 4. sum_q2 q2^+[C2R]*Sq2^[LC1] + h.c. = -Sq2^[LC1]*q2^+[C2R] + h.c.
   auto infoC2 = oper_combine_opC(cindex_c2, cindex_r);
   for(const auto& pr : infoC2){
      int index = pr.first;
      int iformula = pr.second;
      // q2^C2R+*Sq2^LC1 = -Sq2^LC1*q2^C2R+ & Sq2^LC1+*q2^C2R
      auto Slc1 = symbolic_compxwf_opS<Tm>("l", "c1", cindex_l, cindex_c1,
		                           int2e, index, isym, ifkr, size, rank);
      auto Cc2r = symbolic_normxwf_opC<Tm>("c2", "r", index, iformula);
      Cc2r.scale(-1.0);
      auto Slc1_Cc2r = Slc1.outer_product(Cc2r);
      formulae.join(Slc1_Cc2r);
      if(debug){ 
	 std::cout << "idx=" << idx++;
	 Slc1_Cc2r.display("Slc1_Cc2r["+std::to_string(index)+"]", print_level);
	 counter["SC"] += 1;
      }
   }

   // Two-index terms:
   if(ifNC){
      auto ainfo = oper_combine_opA(cindex_l, cindex_c1, ifkr);
      auto binfo = oper_combine_opB(cindex_l, cindex_c1, ifkr);
      counter["AP"] = 0;
      counter["BQ"] = 0;
      // 5. Apq^LC1*Ppq^C2R + h.c. or Ars^C2R*Prs^LC1 + h.c.
      for(const auto pr : ainfo){
         int index = pr.first;
         int iformula = pr.second;
         int iproc = distribute2(index,size);
         if(iproc == rank){
            // Apq*Ppq + Apq^+*Ppq^+
            auto Alc1 = symbolic_normxwf_opA<Tm>("l", "c1", index, iformula, ifkr);
            auto Pc2r = symbolic_compxwf_opP<Tm>("c2", "r", cindex_c2, cindex_r,
            				         int2e, index, isym, ifkr);
            const double wt = ifkr? wfacAP(index) : 1.0;
            Pc2r.scale(wt);
            auto Alc1_Pc2r = Alc1.outer_product(Pc2r);
            formulae.join(Alc1_Pc2r);
            if(debug){ 
               std::cout << "idx=" << idx++;
               Alc1_Pc2r.display("Alc1_Pc2r["+std::to_string(index)+"]", print_level);
	       counter["AP"] += 1;
            }
         } // iproc
      }
      // 6. Bps^LC1*Qps^C2R + h.c. or Qqr^LC1*Bqr^C2R
      for(const auto pr : binfo){
         int index = pr.first;
         int iformula = pr.second;
         int iproc = distribute2(index,size);
         if(iproc == rank){
            auto Blc1 = symbolic_normxwf_opB<Tm>("l", "c1", index, iformula, ifkr);
            auto Qc2r = symbolic_compxwf_opQ<Tm>("c2", "r", cindex_c2, cindex_r,
           		                         int2e, index, isym, ifkr);
            // Bpq*Qpq + Bpq^+*Qpq^+
            const double wt = ifkr? wfacBQ(index) : wfac(index);
            Qc2r.scale(wt);
            auto Blc1_Qc2r = Blc1.outer_product(Qc2r);
            formulae.join(Blc1_Qc2r);
            if(debug){
               std::cout << "idx=" << idx++;
               Blc1_Qc2r.display("Blc1_Qc2r["+std::to_string(index)+"]", print_level);
	       counter["BQ"] += 1;
            }
         } // iproc
      }
   }else{
      auto ainfo = oper_combine_opA(cindex_c2, cindex_r, ifkr);
      auto binfo = oper_combine_opB(cindex_c2, cindex_r, ifkr);
      counter["PA"] = 0;
      counter["QB"] = 0;
      // 5. Apq^LC1*Ppq^C2R + h.c. or Ars^C2R*Prs^LC1 + h.c.
      for(const auto pr : ainfo){
         int index = pr.first;
         int iformula = pr.second;
         int iproc = distribute2(index,size);
         if(iproc == rank){
            // Apq*Ppq + Apq^+*Ppq^+
            auto Plc1 = symbolic_compxwf_opP<Tm>("l", "c1", cindex_l, cindex_c1,
            				         int2e, index, isym, ifkr);
            auto Ac2r = symbolic_normxwf_opA<Tm>("c2", "r", index, iformula, ifkr);
            const double wt = ifkr? wfacAP(index) : 1.0;
            Plc1.scale(wt);
            auto Plc1_Ac2r = Plc1.outer_product(Ac2r);
            formulae.join(Plc1_Ac2r);
            if(debug){
               std::cout << "idx=" << idx++;
               Plc1_Ac2r.display("Plc1_Ac2r["+std::to_string(index)+"]", print_level);
	       counter["PA"] += 1;
            }
         } // iproc
      }
      // 6. Bps^LC1*Qps^C2R + h.c. or Qqr^LC1*Bqr^C2R
      for(const auto pr : binfo){
         int index = pr.first;
         int iformula = pr.second;
         int iproc = distribute2(index,size);
         if(iproc == rank){
            auto Qlc1 = symbolic_compxwf_opQ<Tm>("l", "c1", cindex_l, cindex_c1,
           		                         int2e, index, isym, ifkr);
            auto Bc2r = symbolic_normxwf_opB<Tm>("c2", "r", index, iformula, ifkr);
            // Bpq*Qpq + Bpq^+*Qpq^+
            const double wt = ifkr? wfacBQ(index) : wfac(index);
            Qlc1.scale(wt);
            auto Qlc1_Bc2r = Qlc1.outer_product(Bc2r);
            formulae.join(Qlc1_Bc2r);
            if(debug){
               std::cout << "idx=" << idx++;
               Qlc1_Bc2r.display("Qlc1_Bc2r["+std::to_string(index)+"]", print_level);
	       counter["QB"] += 1;
            }
         } // iproc
      }
   } // ifNC

   if(debug){
      std::cout << "\nSUMMARY size=" << idx;
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
      auto t1 = tools::get_time();
      int size = formulae.size();
      tools::timing("symbolic_formulae_twodot with size="+std::to_string(size), t0, t1);
   }
   return formulae;
}

// bipartite form (with factorization)
template <typename Tm>
bipart_task<Tm> preprocess_formulae_twodot2(const std::vector<int>& cindex_l,
					    const std::vector<int>& cindex_r,
					    const std::vector<int>& cindex_c1,
					    const std::vector<int>& cindex_c2,
	                                    const integral::two_body<Tm>& int2e,
   					    const int isym,
   					    const bool ifkr,
	                                    const int size,
	                                    const int rank,
   					    const bool debug = false){
   auto t0 = tools::get_time();
   const int print_level = 1;
   int slc1 = cindex_l.size() + cindex_c1.size();
   int sc2r = cindex_c2.size() + cindex_r.size();
   const bool ifNC = (slc1 <= sc2r);

   bipart_task<Tm> formulae;
   
   int idx = 0;
   std::map<std::string,int> counter;
   counter["CS"] = 0;
   counter["SC"] = 0;
   
   // Local terms:
   // H[lc1]
   auto Hlc1 = symbolic_compxwf_opH<Tm>("l", "c1", cindex_l, cindex_c1, 
		                        ifkr, size, rank);
   auto Hlc1_Ic2r = bipart_oper('l',Hlc1,"Hlc1_Ic2r");
   assert(Hlc1_Ic2r.parity == 0);
   formulae.push_back(Hlc1_Ic2r);
   if(debug){ 
      std::cout << "idx=" << idx++; 
      Hlc1_Ic2r.display(print_level);
   }
   // H[c2r]
   auto Hc2r = symbolic_compxwf_opH<Tm>("c2", "r", cindex_c2, cindex_r, 
		                        ifkr, size, rank);
   auto Ilc1_Hc2r = bipart_oper('r',Hc2r,"Ilc1_Hc2r");
   assert(Ilc1_Hc2r.parity == 0);
   formulae.push_back(Ilc1_Hc2r);
   if(debug){ 
      std::cout << "idx=" << idx++;
      Ilc1_Hc2r.display(print_level);
   }

   // One-index terms:
   // 3. sum_p1 p1^+[LC1]*Sp1^[C2R] + h.c.
   auto infoC1 = oper_combine_opC(cindex_l, cindex_c1);
   for(const auto& pr : infoC1){
      int index = pr.first;
      int iformula = pr.second;
      // p1^L1C1+*Sp1^C2R & -p1^L1C1*Sp1^C2R+
      auto Clc1 = symbolic_normxwf_opC<Tm>("l", "c1", index, iformula);
      auto Sc2r = symbolic_compxwf_opS<Tm>("c2", "r", cindex_c2, cindex_r,
		                           int2e, index, isym, ifkr, size, rank);
      auto Clc1_Sc2r = bipart_oper(Clc1,Sc2r,"Clc1_Sc2r["+std::to_string(index)+"]");
      assert(Clc1_Sc2r.parity == 0);
      formulae.push_back(Clc1_Sc2r);
      if(debug){ 
	 std::cout << "idx=" << idx++;
	 Clc1_Sc2r.display(print_level);
	 counter["CS"] += 1;
      }
   }
   // 4. sum_q2 q2^+[C2R]*Sq2^[LC1] + h.c. = -Sq2^[LC1]*q2^+[C2R] + h.c.
   auto infoC2 = oper_combine_opC(cindex_c2, cindex_r);
   for(const auto& pr : infoC2){
      int index = pr.first;
      int iformula = pr.second;
      // q2^C2R+*Sq2^LC1 = -Sq2^LC1*q2^C2R+ & Sq2^LC1+*q2^C2R
      auto Slc1 = symbolic_compxwf_opS<Tm>("l", "c1", cindex_l, cindex_c1,
		                           int2e, index, isym, ifkr, size, rank);
      auto Cc2r = symbolic_normxwf_opC<Tm>("c2", "r", index, iformula);
      Cc2r.scale(-1.0);
      auto Slc1_Cc2r = bipart_oper(Slc1,Cc2r,"Slc1_Cc2r["+std::to_string(index)+"]");
      assert(Slc1_Cc2r.parity == 0);
      formulae.push_back(Slc1_Cc2r);
      if(debug){ 
	 std::cout << "idx=" << idx++;
	 Slc1_Cc2r.display(print_level);
	 counter["SC"] += 1;
      }
   }

   // Two-index terms:
   if(ifNC){
      auto ainfo = oper_combine_opA(cindex_l, cindex_c1, ifkr);
      auto binfo = oper_combine_opB(cindex_l, cindex_c1, ifkr);
      counter["AP"] = 0;
      counter["BQ"] = 0;
      // 5. Apq^LC1*Ppq^C2R + h.c. or Ars^C2R*Prs^LC1 + h.c.
      for(const auto pr : ainfo){
         int index = pr.first;
         int iformula = pr.second;
         int iproc = distribute2(index,size);
         if(iproc == rank){
            // Apq*Ppq + Apq^+*Ppq^+
            auto Alc1 = symbolic_normxwf_opA<Tm>("l", "c1", index, iformula, ifkr);
            auto Pc2r = symbolic_compxwf_opP<Tm>("c2", "r", cindex_c2, cindex_r,
            				         int2e, index, isym, ifkr);
            const double wt = ifkr? wfacAP(index) : 1.0;
            Pc2r.scale(wt);
            auto Alc1_Pc2r = bipart_oper(Alc1,Pc2r,"Alc1_Pc2r["+std::to_string(index)+"]");
            assert(Alc1_Pc2r.parity == 0);
            formulae.push_back(Alc1_Pc2r);
            if(debug){ 
               std::cout << "idx=" << idx++;
               Alc1_Pc2r.display(print_level);
	       counter["AP"] += 1;
            }
         } // iproc
      }
      // 6. Bps^LC1*Qps^C2R + h.c. or Qqr^LC1*Bqr^C2R
      for(const auto pr : binfo){
         int index = pr.first;
         int iformula = pr.second;
         int iproc = distribute2(index,size);
         if(iproc == rank){
            auto Blc1 = symbolic_normxwf_opB<Tm>("l", "c1", index, iformula, ifkr);
            auto Qc2r = symbolic_compxwf_opQ<Tm>("c2", "r", cindex_c2, cindex_r,
           		                         int2e, index, isym, ifkr);
            // Bpq*Qpq + Bpq^+*Qpq^+
            const double wt = ifkr? wfacBQ(index) : wfac(index);
            Qc2r.scale(wt);
            auto Blc1_Qc2r = bipart_oper(Blc1,Qc2r,"Blc1_Qc2r["+std::to_string(index)+"]");
            assert(Blc1_Qc2r.parity == 0);
            formulae.push_back(Blc1_Qc2r);
            if(debug){
               std::cout << "idx=" << idx++;
               Blc1_Qc2r.display(print_level);
	       counter["BQ"] += 1;
            }
         } // iproc
      }
   }else{
      auto ainfo = oper_combine_opA(cindex_c2, cindex_r, ifkr);
      auto binfo = oper_combine_opB(cindex_c2, cindex_r, ifkr);
      counter["PA"] = 0;
      counter["QB"] = 0;
      // 5. Apq^LC1*Ppq^C2R + h.c. or Ars^C2R*Prs^LC1 + h.c.
      for(const auto pr : ainfo){
         int index = pr.first;
         int iformula = pr.second;
         int iproc = distribute2(index,size);
         if(iproc == rank){
            // Apq*Ppq + Apq^+*Ppq^+
            auto Plc1 = symbolic_compxwf_opP<Tm>("l", "c1", cindex_l, cindex_c1,
            				         int2e, index, isym, ifkr);
            auto Ac2r = symbolic_normxwf_opA<Tm>("c2", "r", index, iformula, ifkr);
            const double wt = ifkr? wfacAP(index) : 1.0;
            Plc1.scale(wt);
            auto Plc1_Ac2r = bipart_oper(Plc1,Ac2r,"Plc1_Ac2r["+std::to_string(index)+"]");
            assert(Plc1_Ac2r.parity == 0);
            formulae.push_back(Plc1_Ac2r);
            if(debug){
               std::cout << "idx=" << idx++;
               Plc1_Ac2r.display(print_level);
	       counter["PA"] += 1;
            }
         } // iproc
      }
      // 6. Bps^LC1*Qps^C2R + h.c. or Qqr^LC1*Bqr^C2R
      for(const auto pr : binfo){
         int index = pr.first;
         int iformula = pr.second;
         int iproc = distribute2(index,size);
         if(iproc == rank){
            auto Qlc1 = symbolic_compxwf_opQ<Tm>("l", "c1", cindex_l, cindex_c1,
           		                         int2e, index, isym, ifkr);
            auto Bc2r = symbolic_normxwf_opB<Tm>("c2", "r", index, iformula, ifkr);
            // Bpq*Qpq + Bpq^+*Qpq^+
            const double wt = ifkr? wfacBQ(index) : wfac(index);
            Qlc1.scale(wt);
            auto Qlc1_Bc2r = bipart_oper(Qlc1,Bc2r,"Qlc1_Bc2r["+std::to_string(index)+"]");
            assert(Qlc1_Bc2r.parity == 0);
            formulae.push_back(Qlc1_Bc2r);
            if(debug){
               std::cout << "idx=" << idx++;
               Qlc1_Bc2r.display(print_level);
	       counter["QB"] += 1;
            }
         } // iproc
      }
   } // ifNC

   if(debug){
      std::cout << "\nSUMMARY size=" << idx;
      if(ifNC){
	 std::cout << " CS:" << counter["CS"] << " SC:" << counter["SC"]
             	   << " AP:" << counter["AP"] << " BQ:" << counter["BQ"]
           	   << std::endl;
      }else{
         std::cout << " SC:" << counter["SC"] << " CS:" << counter["CS"]
           	   << " PA:" << counter["PA"] << " QB:" << counter["QB"]
           	   << std::endl;
      }
      display(formulae, "total");
      auto t1 = tools::get_time();
      int size = formulae.size();
      tools::timing("symbolic_formulae_twodot2 with size="+std::to_string(size), t0, t1);
   }
   return formulae;
}

// analyze the distribution of operators along a sweep
template <typename Km>
void preprocess_oper(const comb<Km>& icomb){
   int size = 1, rank = 0;
#ifndef SERIAL
   size = icomb.world.size();
   rank = icomb.world.rank();
#endif  
   if(rank == 0){
      std::cout << "\nctns::preprocess_oper" << std::endl;
   }
   if(rank != 0) return;
   auto sweeps = icomb.topo.get_sweeps(true);
   int mid0 = icomb.topo.nphysical/2-2;
   int mid1 = mid0 + icomb.topo.nphysical-2;
   std::cout << "mid0,mid1=" << mid0 << "," << mid1 << std::endl; 
   for(int ibond=0; ibond<sweeps.size(); ibond++){
      //if(ibond != mid0 && ibond != mid1) continue;
      if(ibond != mid0) continue;
      const auto& dbond = sweeps[ibond];
      const auto& p0 = dbond.p0;
      const auto& p1 = dbond.p1;
      const auto& forward = dbond.forward;
      std::cout << "\nibond=" << ibond 
	        << " dbond=" << dbond
		<< std::endl;
      auto fbond = icomb.topo.get_fbond(dbond,".");
      auto frop = fbond.first;
      auto fdel = fbond.second;
      std::cout << " frop=" << frop 
	        << " fdel=" << fdel
		<< std::endl;
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
      bool ifkr = false;
      auto cindex = oper_index_opC(ksupp, ifkr);
      auto sindex = oper_index_opS(krest, ifkr);
      auto aindex = oper_index_opA(cindex, ifkr);
      auto bindex = oper_index_opB(cindex, ifkr);
      auto pindex = oper_index_opP(krest, ifkr); 
      auto qindex = oper_index_opQ(krest, ifkr);
      std::cout << " C:" << cindex.size() 
                << " S:" << sindex.size()
                << " A:" << aindex.size()
                << " B:" << bindex.size()
                << " P:" << pindex.size()
                << " Q:" << qindex.size()
		<< std::endl;

      // twodot Hx
      const int isym = 2;
      ifkr = false;
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
      bool ifNC = (sl+sc1 <= sc2+sr);
      std::cout << "(sl,sr,sc1,sc2)="
	        << sl << "," << sr << "," << sc1 << "," << sc2
		<< " ifNC=" << ifNC
		<< endl;
      auto cindex_l = oper_index_opC(suppl, ifkr);
      auto cindex_r = oper_index_opC(suppr, ifkr);
      auto cindex_c1 = oper_index_opC(suppc1, ifkr);
      auto cindex_c2 = oper_index_opC(suppc2, ifkr);

      using DTYPE = double;
      integral::two_body<DTYPE> int2e;
      int2e.sorb = 2*icomb.topo.nphysical;
      int2e.init_mem();
      std::fill_n(int2e.data.begin(), int2e.data.size(), 1.0); 
      
      std::vector<int> sizes({1,8,32,64,128,1024}); //,2,4,8,16,32,64});
      for(int i=0; i<sizes.size(); i++){
	 int mpisize = sizes[i];
	 std::cout << "\nmpisize=" << mpisize 
		   << " ifkr=" << ifkr 
		   << std::endl;
	 // a
	 std::vector<int> astat(mpisize,0);  
         for(int idx : aindex){
            int iproc = distribute2(idx,mpisize);
            astat[iproc] += 1;
	 }
         tools::print_vector(astat,"astat");
	 // b
	 std::vector<int> bstat(mpisize,0);  
         for(int idx : bindex){
            int iproc = distribute2(idx,mpisize);
            bstat[iproc] += 1;
	 }
         tools::print_vector(bstat,"bstat");
	 // p
	 std::vector<int> pstat(mpisize,0);  
         for(int idx : pindex){
            int iproc = distribute2(idx,mpisize);
            pstat[iproc] += 1;
	 }
         tools::print_vector(pstat,"pstat");
	 // q
	 std::vector<int> qstat(mpisize,0);  
         for(int idx : qindex){
            int iproc = distribute2(idx,mpisize);
            qstat[iproc] += 1;
	 }
         tools::print_vector(qstat,"qstat");

	 std::vector<double> fsizes(mpisize,0.0);
	 for(int rank=0; rank<mpisize; rank++){
            bool debug = mpisize==1 && rank==0;		 
            auto formulae = preprocess_formulae_twodot2(cindex_l, cindex_r, cindex_c1, cindex_c2,
		      	  	 		       int2e, isym, ifkr, mpisize, rank,
						       debug);
/*
	    std::cout << "rank=" << rank
		      << " size(formulae)=" << formulae.size()
		      << std::endl;
*/
	    fsizes[rank] = formulae.size();
	 } // rank
	 double max = *std::max_element(fsizes.begin(), fsizes.end());
	 double min = *std::min_element(fsizes.begin(), fsizes.end());
	 double mean = std::accumulate(fsizes.begin(), fsizes.end(), 0.0)/mpisize;
	 double sq_sum = std::inner_product(fsizes.begin(), fsizes.end(), fsizes.begin(), 0.0);
         double stdev = std::sqrt(sq_sum/mpisize - mean*mean);
	 std::cout << "mpisize=" << mpisize
		   << " max=" << max 
		   << " min=" << min
		   << " mean=" << mean 
		   << " stdev=" << stdev
		   << std::endl;

      } // i 

   } // ibond
   exit(1);
}

} // ctns

#endif
