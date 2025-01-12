#ifndef OPER_DOT_SU2_H
#define OPER_DOT_SU2_H

/*
   Dot operators: HCSABPQ

   The local basis is {|0>,|2>,|a>,|b>} in consistent with ctns_phys.h

   We use the convention that p1+*p2+*q2*q1 where p1<p2 and q2>q1, i.e., 
   The index in the middle is larger than that close to the boundary.
   This is different from the ordering used in onstate.h
*/

#include "../init_phys.h"
#include "../oper_dict.h"

namespace ctns{

   const bool debug_oper_dot_su2 = false;
   extern const bool debug_oper_dot_su2;

   // return the spin and parity of operator (key,index)
   inline std::pair<int,int> get_spin_parity(const char key, const int index){
       int tk, pk;
       if(key == 'H' || key == 'I' || key == 'F'){
          tk = 0;
          pk = 0;
       }else if(key == 'C' || key == 'D' || key == 'S' || key == 'T'){
          tk = 1;
          pk = 1;
       }else if(key == 'A' || key == 'P' || key == 'M'){
          auto pq = oper_unpack(index);
          int p = pq.first, kp = p/2, sp = p%2;
          int q = pq.second, kq = q/2, sq = q%2;
          tk = (sp!=sq)? 0 : 2;   
          pk = 0;
       }else if(key == 'B' || key == 'Q'){
          auto pq = oper_unpack(index);
          int p = pq.first, kp = p/2, sp = p%2;
          int q = pq.second, kq = q/2, sq = q%2;
          tk = (sp!=sq)? 2 : 0;
          pk = 0;
       }else{
          std::cout << "error: get_spin_parity does not have key=" << key << std::endl;
          exit(1); 
       }
       return std::make_pair(tk,pk);
   }
               
   template <typename Tm>
      void oper_dotSE(const int ts,
            const int tk,
            const int pk,
            const stensor2su2<Tm>& op, 
            stensor2su2<Tm>& op2){
         // map from b[EC] of op2 to b of op[C] (E=env,C=dot) 
         // see get_qbond_embed in ../init_phys.h
         std::vector<int> cmap({0,1,2,2});
         for(int i=0; i<op2.info._nnzaddr.size(); i++){
            auto key = op2.info._nnzaddr[i];
            int br = std::get<0>(key);
            int bc = std::get<1>(key);
            auto blk2 = op2(br,bc);
            if(blk2.empty()) continue;
            int cbr = cmap[br];
            int cbc = cmap[bc];
            const auto blk = op(cbr,cbc);
            if(blk.empty()) continue;
            int tsr = op2.info.qrow.get_sym(br).ts();
            int tsc = op2.info.qcol.get_sym(bc).ts();
            int ctsr = op.info.qrow.get_sym(cbr).ts();
            int ctsc = op.info.qcol.get_sym(cbc).ts();
            double fac = std::sqrt((ts+1.0)*(ctsr+1.0)*(tsc+1.0)*(tk+1.0))*
               fock::wigner9j(ts,ctsr,tsr,ts,ctsc,tsc,0,tk,tk);
            if(ts%2==1 and pk==1) fac *= -1;
            assert(blk2.size() == 1 and blk.size() == 1);
            blk2._data[0] = fac*blk._data[0];
         }
      }

   // init local operators on dot: singlet embedding case
   template <typename Tm>
      void oper_init_dotSE(const oper_dict<Tm>& qops,
            oper_dict<Tm>& qops2,
            const int ts){
         std::cout << "error: oper_init_dotSE only work for su2 case!" << std::endl;
         exit(1);
      } 
   template <typename Tm>
      void oper_init_dotSE(const opersu2_dict<Tm>& qops,
            opersu2_dict<Tm>& qops2,
            const int ts){
         // setup basic information
         qops2.sorb = qops.sorb;
         qops2.isym = qops.isym;
         qops2.ifkr = qops.ifkr;
         qops2.cindex = qops.cindex;
         qops2.krest = qops.krest;
         qops2.oplist = qops.oplist;
         auto qembed = get_qbond_embed(ts);
         qops2.qbra = qembed;
         qops2.qket = qembed;
         // initialize memory
         qops2.init(true);
         // transform operators
         for(const auto& key : qops.oplist){
            for(const auto& pr : qops(key)){
               const auto& index = pr.first;
               const auto& op = pr.second;
               auto spin_parity = get_spin_parity(key, index);
               int tk = spin_parity.first;
               int pk = spin_parity.second;
               auto& op2 = qops2(key).at(index);
               oper_dotSE(ts, tk, pk, op, op2);
            }
         }
      }

   // kA^+
   template <typename Tm>
      void oper_dot_opC(opersu2_dict<Tm>& qops,
            const int k0){
         if(debug_oper_dot_su2) std::cout << "ctns::oper_dot_opC" << std::endl; 
         int ka = 2*k0, kb = ka+1;
         // c[0] = kA^+
         // [[0. 0. 0.]
         //  [0. 0. -sqrt(2)]
         //  [1. 0. 0.]]
         linalg::matrix<Tm> mat(3,3);
         mat(1,2) = -std::sqrt(2.0);
         mat(2,0) = 1;
         qops('C')[ka].from_matrix(mat);
         if(debug_oper_dot_su2) qops('C')[ka].to_matrix().print("cp+");
      }

   // A[kA,kB] = kA^+kB^+
   template <typename Tm>
      void oper_dot_opA(opersu2_dict<Tm>& qops,
            const int k0){
         if(debug_oper_dot_su2) std::cout << "ctns::oper_dot_opA" << std::endl; 
         int ka = 2*k0, kb = ka+1;
         // c[0].dot(c[1])
         // [[0. 0. 0.]
         //  [sqrt(2) 0. 0.]
         //  [0. 0. 0.]]
         linalg::matrix<Tm> mat(3,3);
         mat(1,0) = std::sqrt(2.0);
         qops('A')[oper_pack(ka,kb)].from_matrix(mat);
         if(debug_oper_dot_su2) qops('A')[oper_pack(ka,kb)].to_matrix().print("App0");
      }

   // B[kA,kA] = kA^+kA, B[kA,kB] = kA^+kB
   // B[kB,kA] = kB^+kA = B[kA,kB].K(1)
   // B[kB,kB] = kB^+kB = B[kA,kA].K(2)
   template <typename Tm>
      void oper_dot_opB(opersu2_dict<Tm>& qops,
            const int k0){
         if(debug_oper_dot_su2) std::cout << "ctns::oper_dot_opB" << std::endl; 
         int ka = 2*k0, kb = ka+1;
         // [Bpp]^0
         // [[0. 0. 0. ]
         //  [0. sqrt(2) 0. ]
         //  [0. 0. 1/sqrt(2)]]
         linalg::matrix<Tm> matBpp0(3,3);
         matBpp0(1,1) = std::sqrt(2);
         matBpp0(2,2) = 1.0/std::sqrt(2);
         qops('B')[oper_pack(ka,ka)].from_matrix(matBpp0);
         if(debug_oper_dot_su2) qops('B')[oper_pack(ka,ka)].to_matrix().print("Bpp0");
         // [Bpp]^1
         // [[0. 0. 0. ]
         //  [0. 0. 0. ]
         //  [0. 0. sqrt(3/2)]]
         linalg::matrix<Tm> matBpp1(3,3);
         matBpp1(2,2) = std::sqrt(3.0/2.0);
         qops('B')[oper_pack(ka,kb)].from_matrix(matBpp1);
         if(debug_oper_dot_su2) qops('B')[oper_pack(ka,kb)].to_matrix().print("Bpp1");
      }

   // build local H^C = hpq ap^+aq + <pq||sr> ap^+aq^+aras [p<q,r>s]
   template <typename Tm>
      void oper_dot_opH(opersu2_dict<Tm>& qops,
            const int k0,
            const integral::two_body<Tm>& int2e,
            const integral::one_body<Tm>& int1e){
         if(debug_oper_dot_su2) std::cout << "ctns::oper_dot_opH" << std::endl; 
         int ka = 2*k0, kb = ka+1;
         // 0110 (*<01||01>)
         // c[0].dot(c[1].dot(a[1].dot(a[0])))
         // [[0. 0. 0.]
         //  [0. 1. 0.]
         //  [0. 0. 0.]]
         linalg::matrix<Tm> mat(3,3);
         mat(1,1) = 1;
         stensor2su2<Tm> qt2nanb(qsym(3,0,0), qops.qbra, qops.qket);
         qt2nanb.from_matrix(mat);
         const auto& qBpp0 = qops('B')[oper_pack(ka,ka)];
         int N = qops('H')[0].size();
         assert(N == qBpp0.size() and N == qt2nanb.size());
         Tm* ptr = qops('H')[0].data();
         linalg::xaxpy(N, int2e.get(ka,kb,ka,kb), qt2nanb.data(), ptr);
         linalg::xaxpy(N, std::sqrt(2.0)*int1e.get(ka,ka), qBpp0.data(), ptr);
         if(debug_oper_dot_su2) qops('H')[0].to_matrix().print("H");
      }

   // build local S_{p}^C = 1/2 hpq aq + <pq||sr> aq^+aras [r>s]
   template <typename Tm>
      void oper_dot_opS(opersu2_dict<Tm>& qops, 
            const int k0,
            const integral::two_body<Tm>& int2e,
            const integral::one_body<Tm>& int1e){
         if(debug_oper_dot_su2) std::cout << "ctns::oper_dot_opS" << std::endl; 
         int ka = 2*k0, kb = ka+1;
         // [Ts]^1/2
         // [[0. 0. 0. ]
         //  [0. 0. 0. ]
         //  [0. 1. 0. ]]
         linalg::matrix<Tm> mat(3,3);
         mat(2,1) = 1;
         stensor2su2<Tm> qt2Ts(qsym(3,-1,1), qops.qbra, qops.qket); // ka^+ kb ka
         qt2Ts.from_matrix(mat);
         // S_{p}^C = 1/2 hpq aq + <pq||sr> aq^+aras [r>s]
         const auto akA = qops('C')[ka].H(true);
         for(auto& pr : qops('S')){
            int p = pr.first;
            auto& opS = pr.second;
            int N = opS.size();
            assert(p%2 == 0);
            linalg::xaxpy(N, 0.5*int1e.get(p,ka), akA.data(), opS.data());
            linalg::xaxpy(N, int2e.get(p,kb,ka,kb), qt2Ts.data(), opS.data());
         }
      }

   // Ppq = <pq||sr> aras [p<q,r>s] = <pq||sr> A[sr]^+
   // 				   <pq||ab> ba = (a^+b^+)^+
   template <typename Tm>
      void oper_dot_opP(opersu2_dict<Tm>& qops,
            const int k0,
            const integral::two_body<Tm>& int2e){
         if(debug_oper_dot_su2) std::cout << "ctns::oper_dot_opP" << std::endl;   
         int ka = 2*k0, kb = ka+1;
         // [[0. -sqrt(2) 0.]
         //  [0. 0. 0.]
         //  [0. 0. 0.]]
         linalg::matrix<Tm> mat(3,3);
         mat(0,1) = -std::sqrt(2.0);
         stensor2su2<Tm> qt2Css0(qsym(3,-2,0), qops.qbra, qops.qket);
         qt2Css0.from_matrix(mat);
         for(auto& pr : qops('P')){
            int pq = pr.first;
            auto& opP = pr.second;
            auto upq = oper_unpack(pq);
            int p = upq.first, q = upq.second; 
            int N = opP.size();
            if(p%2 == q%2) continue; // only for Ppq0[os]
            linalg::xaxpy(N, int2e.get(p,q,ka,kb), qt2Css0.data(), opP.data());
         }
      }

   // Qps = <pq||sr> aq^+ar
   template <typename Tm>
      void oper_dot_opQ(opersu2_dict<Tm>& qops, 
            const int k0,
            const integral::two_body<Tm>& int2e){
         if(debug_oper_dot_su2) std::cout << "ctns::oper_dot_opQ" << std::endl; 
         int ka = 2*k0, kb = ka+1;
         const auto& qt2Bpp0 = qops('B')[oper_pack(ka,ka)];
         const auto& qt2Bpp1 = qops('B')[oper_pack(ka,kb)];
         for(auto& pr : qops('Q')){
            int ps = pr.first;
            auto& opQ = pr.second;
            auto ups = oper_unpack(ps);
            int p = ups.first, s = ups.second;
            int N = opQ.size();
            if(p%2 == s%2){ // Qps0
               // <pk||sk> + <pk'||sk'> = 2[ps|kk] - [pk|ks]
               auto erifac = int2e.get(p,ka,s,ka) + int2e.get(p,kb,s,kb);
               linalg::xaxpy(N, erifac, qt2Bpp0.data(), opQ.data());
            }else{ // Qps1
               // <pk'||s'k> = -[pk|ks]
               auto erifac = int2e.get(p,kb,s,ka);
               linalg::xaxpy(N, erifac, qt2Bpp1.data(), opQ.data());
            }
         }
      }

   // --- for RDMs ---
   // Identity
   template <typename Tm>
      void oper_dot_opI(opersu2_dict<Tm>& qops){
         if(debug_oper_dot_su2) std::cout << "ctns::oper_dot_opI" << std::endl; 
         // [[1. 0. 0.]
         //  [0. 1. 0.]
         //  [0. 0. 1.]]
         linalg::matrix<Tm> mat = linalg::identity_matrix<Tm>(3);
         qops('I')[0].from_matrix(mat);
      }

   // F = a^+b^+ba 
   template <typename Tm>
      void oper_dot_opF(opersu2_dict<Tm>& qops,
            const int k0){
         if(debug_oper_dot_su2) std::cout << "ctns::oper_dot_opF" << std::endl; 
         int ka = 2*k0, kb = ka+1;
         // 0110 (*<01||01>)
         // c[0].dot(c[1].dot(a[1].dot(a[0])))
         // [[0. 0. 0.]
         //  [0. 1. 0.]
         //  [0. 0. 0.]]
         linalg::matrix<Tm> mat(3,3);
         mat(1,1) = 1;
         qops('F')[ka].from_matrix(mat);
         if(debug_oper_dot_su2) qops('F')[ka].to_matrix().print("F");
      }

   // T = {a^+ba,b^+ba} in my note
   template <typename Tm>
      void oper_dot_opT(opersu2_dict<Tm>& qops, 
            const int k0){
         if(debug_oper_dot_su2) std::cout << "ctns::oper_dot_opT" << std::endl; 
         int ka = 2*k0, kb = ka+1;
         // [Ts]^1/2
         // [[0. 0. 0. ]
         //  [0. 0. 0. ]
         //  [0. 1. 0. ]]
         linalg::matrix<Tm> mat(3,3);
         mat(2,1) = 1;
         qops('T')[ka].from_matrix(mat);
      }

   // kA
   template <typename Tm>
      void oper_dot_opD(opersu2_dict<Tm>& qops,
            const int k0){
         if(debug_oper_dot_su2) std::cout << "ctns::oper_dot_opD" << std::endl; 
         int ka = 2*k0, kb = ka+1;
         // [[0. 0. sqrt(2)]
         //  [0. 0. 0.]
         //  [0. 1. 0.]]
         linalg::matrix<Tm> mat(3,3);
         mat(0,2) = std::sqrt(2.0);
         mat(2,1) = 1;
         qops('D')[ka].from_matrix(mat);
         if(debug_oper_dot_su2) qops('D')[ka].to_matrix().print("cp");
      }

   // M[kA,kB] = kA*kB
   template <typename Tm>
      void oper_dot_opM(opersu2_dict<Tm>& qops,
            const int k0){
         if(debug_oper_dot_su2) std::cout << "ctns::oper_dot_opM" << std::endl; 
         int ka = 2*k0, kb = ka+1;
         // [[0. -sqrt(2) 0.]
         //  [0. 0. 0.]
         //  [0. 0. 0.]]
         linalg::matrix<Tm> mat(3,3);
         mat(0,1) = -std::sqrt(2.0);
         qops('M')[oper_pack(ka,kb)].from_matrix(mat);
         if(debug_oper_dot_su2) qops('M')[oper_pack(ka,kb)].to_matrix().print("Mpp0");
      }

} // ctns

#endif
