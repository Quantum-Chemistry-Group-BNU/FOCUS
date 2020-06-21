#ifndef FCI_UTIL_H
#define FCI_UTIL_H

#include <string>
#include <vector>
#include <map>
#include <set>
#include <complex>
#include "../core/serialization.h"
#include "../core/integral.h"
#include "../core/matrix.h"
#include "../core/onspace.h"
#include "../core/hamiltonian.h"

namespace fci{

// type information
template <typename Tm>
inline bool is_complex(){ return false; }
template <>
inline bool is_complex<std::complex<double>>(){ return true; }

// represent space of dets by direct product structure
struct product_space{
   public:
      void get_pspace(const fock::onspace& space,
		      const int istart=0);
   public:
      // second int is used for indexing in constructing rowA, colB 
      std::map<fock::onstate,int> umapA, umapB;
      fock::onspace spaceA, spaceB; // ordered by appearance
      std::vector<std::vector<std::pair<int,int>>> rowA, colB;  
      int dimA0 = 0, dimB0 = 0, dimA, dimB;
};

// Coupling pattern between states as defined by diff_type,
// which partition the cartesian space {(I,J)} into disjoint sets!
// Note: for two-body H, only S & D matter such that 0<=m<=2, 0<=n<=2, 0<m+n<=2.
struct coupling_table{
   public:
      void get_Cmn(const fock::onspace& space, 
		   const bool Htype,
		   const int istart=0);
      void update_Cmn(const fock::onspace& space,
		      const int istart,
		      const std::pair<int,int>& key,
		      std::vector<std::set<int>>& Cmn);
   public:
      std::vector<std::set<int>> C11; // differ by single (sorted, binary_search)
      std::vector<std::set<int>> C10, C01; // <I|p^+|J>, <I|p|J>
      std::vector<std::set<int>> C20, C02; // <I|p^+q^+|J>, <I|rs|J>
};

// linked list - store each row H[i] as a list
template <typename Tm>
struct sparse_hamiltonian{
   public:

      // construct Hij	   
      void get_hamiltonian(const fock::onspace& space,
			   const integral::two_body<Tm>& int2e,
			   const integral::one_body<Tm>& int1e,
			   const double ecore,
		   	   const bool Htype,
			   const int istart=0){
         const bool debug = true;
         auto t0 = tools::get_time();
         std::cout << "\nsparse_hamiltonian::get_hamiltonian" 
                   << " dim0 = " << istart << " dim = " << space.size() 
	           << std::endl; 
         // 1. setup product_space
         _pspace.get_pspace(space, istart);
         auto ta = tools::get_time();
         if(debug) std::cout << "timing for pspace : " << std::setprecision(2) 
		             << tools::get_duration(ta-t0) << " s" << std::endl;
         // 2. setupt coupling_table
         _ctabA.get_Cmn(_pspace.spaceA, Htype, _pspace.dimA0);
         auto tb = tools::get_time();
         _ctabB.get_Cmn(_pspace.spaceB, Htype, _pspace.dimB0);
         auto tc = tools::get_time();
         if(debug) std::cout << "timing for ctabA/B : " << std::setprecision(2) 
      		             << tools::get_duration(tb-ta) << " s" << " "
      		             << tools::get_duration(tc-tb) << " s" << std::endl; 
         // 3. compute sparse_hamiltonian
   	 make_hamiltonian(space, _pspace, _ctabA, _ctabB, int2e, int1e, ecore, Htype, istart);
         auto t1 = tools::get_time();
         if(debug) std::cout << "timing for get_hamiltonian : " << std::setprecision(2) 
		             << tools::get_duration(t1-t0) << " s" << std::endl;
      }

      // construct Hij	   
      void make_hamiltonian(const fock::onspace& space,
		           const product_space& pspace,
		           const coupling_table& ctabA,
			   const coupling_table& ctabB,
			   const integral::two_body<Tm>& int2e,
			   const integral::one_body<Tm>& int1e,
			   const double ecore,
		   	   const bool Htype,
			   const int istart=0){
         const bool debug = true;
         auto t0 = tools::get_time();
         std::cout << "sparse_hamiltonian::make_hamiltonian" << std::endl; 
         // initialization for the first use
         if(istart == 0){
            diag.clear();
            connect.clear();
            value.clear();
            diff.clear();
         }
         // diagonal 
         dim = space.size();
         diag.resize(dim);
         for(size_t i=istart; i<dim; i++){
            diag[i] = fock::get_Hii(space[i],int2e,int1e) + ecore;
         }
         auto ta = tools::get_time();
         if(debug) std::cout << " timing for diagonal : " << std::setprecision(2) 
              	        << tools::get_duration(ta-t0) << " s" << std::endl;
         // off-diagonal 
         connect.resize(dim);
         value.resize(dim);
         diff.resize(dim);
         get_HIJ_A1122_B00(space, pspace, ctabA, ctabB, int2e, int1e, istart, debug);
	 get_HIJ_A00_B1122(space, pspace, ctabA, ctabB, int2e, int1e, istart, debug);
	 get_HIJ_A11_B11(space, pspace, ctabA, ctabB, int2e, int1e, istart, debug);
         if(Htype){
	    get_HIJ_ABmixed1(space, pspace, ctabA, ctabB, int2e, int1e, istart, debug);
	    get_HIJ_ABmixed2(space, pspace, ctabA, ctabB, int2e, int1e, istart, debug);
	 }
         auto t1 = tools::get_time();
         std::cout << "timing for sparse_hamiltonian::get_hamiltonian : " 
              << std::setprecision(2) << tools::get_duration(t1-t0) << " s" << std::endl;
      }

      // --- HIJ construction by coupling pattern between two states (I & J) ---
      
      // <I_A,I_B|H|J_A,J_B> = (C11+C22)_A*C00_B
      // 		       {I_A,J_A} differ by single/double
      // 	 	       {I_B,J_B} differ by zero (I_B=J_B)
      void get_HIJ_A1122_B00(const fock::onspace& space,
                             const product_space& pspace,
                             const coupling_table& ctabA,
                             const coupling_table& ctabB,
                             const integral::two_body<Tm>& int2e,
                             const integral::one_body<Tm>& int1e,
                             const int istart,
		             const bool debug){
         auto t0 = tools::get_time();
	 // (C11+C22)_A*C00_B
         for(int ia=0; ia<pspace.dimA; ia++){
            for(const auto& pib : pspace.rowA[ia]){
               int ib = pib.first;
               int i = pib.second;
               if(i < istart) continue; // incremental build
               for(const auto& pja : pspace.colB[ib]){
                  int ja = pja.first;
                  int j = pja.second;
                  if(j >= i) continue; 
                  // check connectivity <I_A|H|J_A>
                  auto pr = pspace.spaceA[ia].diff_type(pspace.spaceA[ja]);
                  if(pr == std::make_pair(1,1)){
                     auto pr = fock::get_HijS(space[i], space[j], int2e, int1e);
		     connect[i].push_back(j);
                     value[i].push_back(pr.first);
                     diff[i].push_back(pr.second);
                  }else if(pr == std::make_pair(2,2)){
                     auto pr = fock::get_HijD(space[i], space[j], int2e, int1e); 
                     connect[i].push_back(j);
                     value[i].push_back(pr.first);
                     diff[i].push_back(pr.second);
                  }
               } // ja
            } // ib
         } // ia
         auto t1 = tools::get_time();
         if(debug) std::cout << " timing for get_HIJ_A1122_B00 : " << std::setprecision(2) 
              	        << tools::get_duration(t1-t0) << " s" << std::endl;
      }

      // <I_A,I_B|H|J_A,J_B> = C00_A*(C11+C22)_B
      // 		       {I_A,J_A} differ by zero (I_A=J_A)
      // 		       {I_B,J_B} differ by single/double
      void get_HIJ_A00_B1122(const fock::onspace& space,
                             const product_space& pspace,
                             const coupling_table& ctabA,
                             const coupling_table& ctabB,
                             const integral::two_body<Tm>& int2e,
                             const integral::one_body<Tm>& int1e,
                             const int istart,
		             const bool debug){
         auto t0 = tools::get_time();
	 // C00_A*(C11+C22)_B
         for(int ia=0; ia<pspace.dimA; ia++){
            for(const auto& pib : pspace.rowA[ia]){
      	       int ib = pib.first;
      	       int i = pib.second;
      	       if(i < istart) continue; // incremental build
               for(const auto& pjb : pspace.rowA[ia]){
      	          int jb = pjb.first;
      	          int j = pjb.second;
      	          if(j >= i) continue; 
      	          // check connectivity <I_B|H|J_B>
      	          auto pr = pspace.spaceB[ib].diff_type(pspace.spaceB[jb]);
      	          if(pr == std::make_pair(1,1)){
      	             auto pr = fock::get_HijS(space[i], space[j], int2e, int1e);
      	             connect[i].push_back(j);
      	             value[i].push_back(pr.first);
      	             diff[i].push_back(pr.second);
      	          }else if(pr == std::make_pair(2,2)){
      	             auto pr = fock::get_HijD(space[i], space[j], int2e, int1e); 
      	             connect[i].push_back(j);
      	             value[i].push_back(pr.first);
      	             diff[i].push_back(pr.second);
      	          }
      	       } // jb
            } // ib
         } // ia
         auto t1 = tools::get_time();
         if(debug) std::cout << " timing for get_HIJ_A00_B1122 : " << std::setprecision(2) 
      		        << tools::get_duration(t1-t0) << " s" << std::endl;
      }
      
      // <I_A,I_B|H|J_A,J_B> = C11_A*C11_B
      // 		       {I_A,J_A} differ by single
      // 		       {I_B,J_B} differ by single
      void get_HIJ_A11_B11(const fock::onspace& space,
                           const product_space& pspace,
                           const coupling_table& ctabA,
                           const coupling_table& ctabB,
                           const integral::two_body<Tm>& int2e,
                           const integral::one_body<Tm>& int1e,
                           const int istart,
		           const bool debug){
         auto t0 = tools::get_time();
         for(int ia=0; ia<pspace.dimA; ia++){
            for(const auto& pib : pspace.rowA[ia]){
               int ib = pib.first;
               int i = pib.second;
               if(i < istart) continue; // incremental build
               for(int ja : ctabA.C11[ia]){
                  for(const auto& pjb : pspace.rowA[ja]){
                     int jb = pjb.first;
                     int j = pjb.second;	       
         	     if(j >=i) continue;
                     auto search = ctabB.C11[ib].find(jb);
                     if(search != ctabB.C11[ib].end()){
                        auto pr = fock::get_HijD(space[i], space[j], int2e, int1e);
                        connect[i].push_back(j);
                        value[i].push_back(pr.first);
                        diff[i].push_back(pr.second);
                     } // j>0
                  } // jb
               } // ja
            } // ia
         } // ia
         auto t1 = tools::get_time();
         if(debug) std::cout << " timing for get_HIJ_A11_B11 : " << std::setprecision(2) 
         		<< tools::get_duration(t1-t0) << " s" << std::endl;
      }

      // --- relativistic case ---

      // <I_A,I_B|H|J_A,J_B> = C01_A*C10_B
      // 		     + C01_A*C21_B
      // 		     + C10_A*C01_B
      // 		     + C10_A*C12_B
      // 		     + C12_A*C10_B 
      // 		     + C21_A*C01_B
      void get_HIJ_ABmixed1(const fock::onspace& space,
                            const product_space& pspace,
                            const coupling_table& ctabA,
                            const coupling_table& ctabB,
                            const integral::two_body<Tm>& int2e,
                            const integral::one_body<Tm>& int1e,
                            const int istart,
		            const bool debug){
         auto t0 = tools::get_time();
         // C01_A*(C10_B+C21_B)
	 for(int ia=0; ia<pspace.dimA; ia++){
            for(const auto& pib : pspace.rowA[ia]){
               int ib = pib.first;
               int i = pib.second;
               if(i < istart) continue; // incremental build
	       for(int ja : ctabA.C01[ia]){
	          for(const auto& pjb : pspace.rowA[ja]){
                     int jb = pjb.first;
                     int j = pjb.second;	       
         	     if(j >=i) continue;
                     // check connectivity 
                     auto pr = pspace.spaceB[ib].diff_type(pspace.spaceB[jb]);
                     if(pr == std::make_pair(1,0)){
                        auto pr = fock::get_HijS(space[i], space[j], int2e, int1e);
		        connect[i].push_back(j);
                        value[i].push_back(pr.first);
                        diff[i].push_back(pr.second);
                     }else if(pr == std::make_pair(2,1)){
                        auto pr = fock::get_HijD(space[i], space[j], int2e, int1e); 
                        connect[i].push_back(j);
                        value[i].push_back(pr.first);
                        diff[i].push_back(pr.second);
                     }
		  } // jb
               } // ja
            } // ib
         } // ia
         // C10_A*(C01_B+C12_B)
	 for(int ia=0; ia<pspace.dimA; ia++){
            for(const auto& pib : pspace.rowA[ia]){
               int ib = pib.first;
               int i = pib.second;
               if(i < istart) continue; // incremental build
	       for(int ja : ctabA.C10[ia]){
	          for(const auto& pjb : pspace.rowA[ja]){
                     int jb = pjb.first;
                     int j = pjb.second;	       
         	     if(j >=i) continue;
                     // check connectivity 
                     auto pr = pspace.spaceB[ib].diff_type(pspace.spaceB[jb]);
                     if(pr == std::make_pair(0,1)){
                        auto pr = fock::get_HijS(space[i], space[j], int2e, int1e);
		        connect[i].push_back(j);
                        value[i].push_back(pr.first);
                        diff[i].push_back(pr.second);
                     }else if(pr == std::make_pair(1,2)){
                        auto pr = fock::get_HijD(space[i], space[j], int2e, int1e); 
                        connect[i].push_back(j);
                        value[i].push_back(pr.first);
                        diff[i].push_back(pr.second);
                     }
		  } // jb
               } // ja
            } // ib
         } // ia
         // C12_A*C10_B + C21_A*C01_B
         for(int ia=0; ia<pspace.dimA; ia++){
            for(const auto& pib : pspace.rowA[ia]){
               int ib = pib.first;
               int i = pib.second;
               if(i < istart) continue; // incremental build
	       // C12_A*C10_B
               for(int jb : ctabB.C10[ib]){
                  for(const auto& pja : pspace.colB[jb]){
                     int ja = pja.first;
                     int j = pja.second;	       
         	     if(j >=i) continue;
                     // check connectivity 
                     auto pr = pspace.spaceA[ia].diff_type(pspace.spaceA[ja]);
                     if(pr == std::make_pair(1,2)){
                        auto pr = fock::get_HijD(space[i], space[j], int2e, int1e);
		        connect[i].push_back(j);
                        value[i].push_back(pr.first);
                        diff[i].push_back(pr.second);
		     }
                  } // ja
               } // jb
	       // C21_A*C01_B
               for(int jb : ctabB.C01[ib]){
                  for(const auto& pja : pspace.colB[jb]){
                     int ja = pja.first;
                     int j = pja.second;	       
         	     if(j >=i) continue;
                     // check connectivity 
                     auto pr = pspace.spaceA[ia].diff_type(pspace.spaceA[ja]);
                     if(pr == std::make_pair(2,1)){
                        auto pr = fock::get_HijD(space[i], space[j], int2e, int1e);
		        connect[i].push_back(j);
                        value[i].push_back(pr.first);
                        diff[i].push_back(pr.second);
		     }
                  } // ja
               } // jb
            } // ja
         } // ia
         auto t1 = tools::get_time();
         if(debug) std::cout << " timing for get_HIJ_ABmixed1 : " << std::setprecision(2) 
              	        << tools::get_duration(t1-t0) << " s" << std::endl;
      }

      // <I_A,I_B|H|J_A,J_B> = C02_A*C20_B + C20_A*C02_B
      void get_HIJ_ABmixed2(const fock::onspace& space,
                            const product_space& pspace,
                            const coupling_table& ctabA,
                            const coupling_table& ctabB,
                            const integral::two_body<Tm>& int2e,
                            const integral::one_body<Tm>& int1e,
                            const int istart,
		            const bool debug){
         auto t0 = tools::get_time();
	 // C02_A*C20_B = <I_A|rs|J_A><I_B|p+q+|J_B>
         for(int ia=0; ia<pspace.dimA; ia++){
            for(const auto& pib : pspace.rowA[ia]){
               int ib = pib.first;
               int i = pib.second;
               if(i < istart) continue; // incremental build
               for(int ja : ctabA.C02[ia]){
                  for(const auto& pjb : pspace.rowA[ja]){
                     int jb = pjb.first;
                     int j = pjb.second;	       
         	     if(j >=i) continue;
                     auto search = ctabB.C20[ib].find(jb);
                     if(search != ctabB.C20[ib].end()){
                        auto pr = fock::get_HijD(space[i], space[j], int2e, int1e);
                        connect[i].push_back(j);
                        value[i].push_back(pr.first);
                        diff[i].push_back(pr.second);
                     } // j>0
                  } // jb
               } // ib
            } // ja
         } // ia
	 // C20_A*C02_B = <I_A|p+q+|J_A><I_B|rs|J_B>
         for(int ia=0; ia<pspace.dimA; ia++){
            for(const auto& pib : pspace.rowA[ia]){
               int ib = pib.first;
               int i = pib.second;
               if(i < istart) continue; // incremental build
               for(int ja : ctabA.C20[ia]){
                  for(const auto& pjb : pspace.rowA[ja]){
                     int jb = pjb.first;
                     int j = pjb.second;	       
         	     if(j >=i) continue;
                     auto search = ctabB.C02[ib].find(jb);
                     if(search != ctabB.C02[ib].end()){
                        auto pr = fock::get_HijD(space[i], space[j], int2e, int1e);
                        connect[i].push_back(j);
                        value[i].push_back(pr.first);
                        diff[i].push_back(pr.second);
                     } // j>0
                  } // jb
               } // ib
            } // ja
         } // ia
         auto t1 = tools::get_time();
         if(debug) std::cout << " timing for get_HIJ_ABmixed2 : " << std::setprecision(2) 
         		<< tools::get_duration(t1-t0) << " s" << std::endl;
      }

      // compare with full construction
      void check(const fock::onspace& space,
	 	 const integral::two_body<Tm>& int2e,
		 const integral::one_body<Tm>& int1e,
		 const double ecore,
		 const double thresh=1.e-10){
   	 std::cout << "\nsparse_hamiltonian::check" << std::endl;
         int dim = connect.size();
	 linalg::matrix<Tm> H1(dim,dim);
         for(int i=0; i<dim; i++){
            for(int jdx=0; jdx<connect[i].size(); jdx++){
               int j = connect[i][jdx];
               H1(i,j) = value[i][jdx];
	       H1(j,i) = tools::conjugate(value[i][jdx]); 
            }
	    H1(i,i) += diag[i];
         }
         // compared againts construction by Slater-Condon rule
         auto H0 = fock::get_Ham(space,int2e,int1e,ecore);
	 std::cout << std::setprecision(6);
         for(int i=0; i<dim; i++){
            for(int j=0; j<dim; j++){
               //if(abs(H1(i,j))<1.e-8 && abs(H2(i,j))<1.e-8) continue;
               if(abs(H1(i,j)-H0(i,j))<1.e-8) continue;
               std::cout << "i,j=" << i << "," << j 
                    << " val1=" << H1(i,j)  
		    << " val0=" << H0(i,j)  
                    << " diff=" << H1(i,j)-H0(i,j) 
                    << " pair=" << space[i] << " " << space[j]  
                    << " num=" << space[i].diff_num(space[j]) 
                    << std::endl;
            }
         } 
	 double diff = normF(H1-H0);
         std::cout << "|H1-H0|=" << diff << std::endl;
	 if(diff > thresh){
	    std::cout << "error: difference is greater than thresh=" << thresh << std::endl;
	    exit(1); 
	 }
      }

      // analyze the magnitude of Hij
      void analysis(){
   	 std::cout << "\nsparse_hamiltonian::analysis" << std::endl;
         const double thresh = 1.e-8;
	 std::map<int,int,std::greater<int>> bucket;
         double size = 1.e-20; // avoid divide zero in the special case H=0;
         double Hsum = 0;
         for(int i=0; i<dim; i++){
            size += connect[i].size();
            for(int jdx=0; jdx<connect[i].size(); jdx++){
      	       int j = connect[i][jdx];
      	       double aval = abs(value[i][jdx]);
      	       if(aval > thresh){
      	          int n = floor(log10(aval));
      	          bucket[n] += 1;
      	          Hsum += aval;
      	       }
            }
         }
	 // averaged connection per row
         double avc = 2.0*size/dim; 
         std::cout << "dim = " << dim
         	<< "  avc = " << std::defaultfloat << std::fixed << avc
         	<< "  per = " << std::defaultfloat << std::setprecision(3) << avc/(dim-1)*100 << std::endl; 
         std::cout << "average size |Hij| = " << std::scientific << std::setprecision(1) << Hsum/size << std::endl;
         // print statistics by magnitude 
         double accum = 0.0;
         for(const auto& pr : bucket){
            double per = pr.second/size*100;
            int n = pr.first;
            accum += per;
            std::cout << "|Hij| in 10^" << std::showpos << n+1 << "-10^" << n << " : " 
      	         << " per=" << std::defaultfloat << std::noshowpos << std::fixed << std::setw(5) << std::setprecision(1) << per << " " 
      	         << " accum=" << std::defaultfloat << std::noshowpos << std::fixed << std::setw(5) << std::setprecision(1) << accum 
      	         << std::endl;
         }
      }
   private:
      product_space _pspace;
      coupling_table _ctabA, _ctabB;
   public:
      int dim;
      std::vector<double> diag; // H[i,i]
      // lower-riangular part: H[i,j] (i>j)
      std::vector<std::vector<int>> connect; // connected by H
      std::vector<std::vector<Tm>> value;    // H[i][j] 
      std::vector<std::vector<long>> diff;   // packed orbital difference 
};

} // fci

#endif
